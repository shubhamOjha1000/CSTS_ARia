#!/usr/bin/env python3
"""
Progressive cross-modal distillation training loop.

Trains the lightweight StudentGazeModel on the Aria dataset using
knowledge distilled from the frozen CSTS teacher, with a four-stage
curriculum that activates losses progressively.

Curriculum
----------
  epoch  0-4  : V1 + V2          (output + feature alignment)
  epoch  5-9  : V1 + V2 + V3     (add attention transfer)
  epoch 10+   : V1 + V2 + V3 + V4 (add progressive CRD)

Loss weights
------------
  V1=1.0  V2=1.0  V3=0.5  V4=0.5

Run on Colab
------------
  cd /content/CSTS_ARia
  python tools/train_distillation.py \\
      --cfg     configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \\
      --teacher /content/drive/MyDrive/checkpoints/teacher.pth \\
      --output  /content/drive/MyDrive/distillation_run \\
      --epochs  20 \\
      NUM_GPUS 0 \\
      DATA.PATH_PREFIX /content/drive/MyDrive/Aria_eg_dataset/clips \\
      TRAIN.BATCH_SIZE 4 \\
      TEST.BATCH_SIZE  4

  # Resume from a student checkpoint:
  python tools/train_distillation.py \\
      --cfg    configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \\
      --teacher /content/drive/MyDrive/checkpoints/teacher.pth \\
      --student /content/drive/MyDrive/distillation_run/best.pth \\
      --output  /content/drive/MyDrive/distillation_run \\
      --epochs  20 \\
      NUM_GPUS 0 \\
      DATA.PATH_PREFIX /content/drive/MyDrive/Aria_eg_dataset/clips
"""

import argparse
import os
import sys
import time
import csv
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.models.student_model import StudentGazeModel
from slowfast.models.distillation_losses import (
    OutputDistillationLoss,
    FeatureDistillationLoss,
    AttentionTransferLoss,
    ProgressiveCRDLoss,
)
from slowfast.datasets import decoder
from slowfast.datasets import utils as ds_utils
from slowfast.datasets import video_container as container

# ── student audio shape (must match student_model.py) ─────────────────────────
F_STU = 64
L_STU = 128

# ── video normalisation (ImageNet stats) ──────────────────────────────────────
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset — returns both teacher and student audio from the same .npy
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_student(npy_path: str) -> torch.Tensor:
    """Load full-clip log-STFT and bilinearly resize to [1, F_STU, L_STU]."""
    spec = np.load(npy_path).astype(np.float32)          # [256, N_time]
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, N_time]
    spec_t = F.interpolate(spec_t, size=(F_STU, L_STU),
                           mode='bilinear', align_corners=False)
    return spec_t.squeeze(0)  # [1, 64, 128]


def _load_audio_teacher(npy_path: str,
                         frames_idx: np.ndarray,
                         frame_length: int) -> torch.Tensor:
    """
    Load per-frame STFT windows for the teacher — [1, T, 256, 256].

    Mirrors the logic in aria_avgaze.py: for each decoded frame index,
    extract the corresponding 256-wide time slice from the full STFT.
    """
    spec = np.load(npy_path).astype(np.float32)          # [256, N_time]
    audio_idx = (frames_idx / max(frame_length, 1)) * spec.shape[1]
    audio_idx = np.round(audio_idx).astype(int)
    audio_idx = np.clip(audio_idx, 128, spec.shape[1] - 1 - 128)
    windows   = np.stack([spec[:, idx - 128: idx + 128]
                          for idx in audio_idx], axis=0)  # [T, 256, 256]
    return torch.from_numpy(windows[np.newaxis])          # [1, T, 256, 256]


def _make_heatmap(label: np.ndarray, T: int, H4: int, W4: int,
                  kernel_size: int = 19) -> torch.Tensor:
    """Build Gaussian gaze heatmap [T, H4, W4] from (x,y) labels in [0,1]."""
    hm = np.zeros((T, H4, W4), dtype=np.float32)
    for i in range(T):
        cx = round(float(label[i, 0]) * W4)
        cy = round(float(label[i, 1]) * H4)
        left   = max(cx - (kernel_size - 1) // 2, 0)
        right  = min(cx + (kernel_size - 1) // 2, W4 - 1)
        top    = max(cy - (kernel_size - 1) // 2, 0)
        bottom = min(cy + (kernel_size - 1) // 2, H4 - 1)
        if left < right and top < bottom:
            k1d = cv2.getGaussianKernel(kernel_size, -1, cv2.CV_32F)
            k2d = k1d @ k1d.T
            k_left   = (kernel_size - 1) // 2 - cx + left
            k_right  = (kernel_size - 1) // 2 + right - cx
            k_top    = (kernel_size - 1) // 2 - cy + top
            k_bottom = (kernel_size - 1) // 2 + bottom - cy
            hm[i, top:bottom + 1, left:right + 1] = \
                k2d[k_top:k_bottom + 1, k_left:k_right + 1]
        d = hm[i].sum()
        hm[i] = hm[i] / d if d > 0 else np.full_like(hm[i], 1.0 / (H4 * W4))
    return torch.from_numpy(hm)


class DistillationAriaDataset(torch.utils.data.Dataset):
    """
    Aria dataset for distillation training.

    Returns per-sample:
      video         [3, T, H, W]       — normalised RGB frames
      audio_teacher [1, T, 256, 256]   — per-frame STFT windows (teacher input)
      audio_student [1, F_STU, L_STU]  — full-clip STFT resized (student input)
      label_hm      [T, H/4, W/4]      — GT Gaussian gaze heatmap
      label         [T, 2]             — (x, y) gaze in [0,1]
      index         int
    """

    def __init__(self, cfg, mode: str, num_retries: int = 10):
        assert mode in ("train", "val", "test")
        self.mode         = mode
        self.cfg          = cfg
        self._num_retries = num_retries
        self._video_meta  = {}

        self._num_clips = (
            1 if mode in ("train", "val")
            else cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        )
        self._construct_loader()

    def _construct_loader(self):
        csv_path = ("data/train_aria_gaze.csv"
                    if self.mode == "train"
                    else "data/test_aria_gaze.csv")
        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

        self._path_to_videos = []
        self._path_to_audios = []
        self._spatial_temporal_idx = []
        self._labels = {}

        with open(csv_path) as f:
            paths = [l.strip() for l in f if l.strip()]

        for clip_idx, rel_path in enumerate(paths):
            for idx in range(self._num_clips):
                full = os.path.join(self.cfg.DATA.PATH_PREFIX, rel_path)
                self._path_to_videos.append(full)
                self._spatial_temporal_idx.append(idx)
                self._video_meta[clip_idx * self._num_clips + idx] = {}

        for vp in self._path_to_videos:
            ap = (vp.replace("clips", "clips.audio_24kHz_stft")
                    .replace(".mp4", ".npy"))
            self._path_to_audios.append(ap)

        for vp in self._path_to_videos:
            video_name = vp.split("/")[-2]
            if video_name in self._labels:
                continue
            prefix     = os.path.dirname(self.cfg.DATA.PATH_PREFIX)
            label_file = os.path.join(prefix, "gaze_frame_label",
                                      f"{video_name}.csv")
            with open(label_file) as f:
                rows = [list(map(float, r))
                        for i, r in enumerate(csv.reader(f)) if i > 0]
            self._labels[video_name] = np.array(rows)[:, 2:]  # (x, y, ...)

        assert len(self._path_to_videos) > 0, f"No videos from {csv_path}"

    def __len__(self):
        return len(self._path_to_videos)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, _ = index

        if self.mode == "train":
            temporal_idx = -1
            spatial_idx  = -1
            min_scale    = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale    = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size    = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            temporal_idx = 1
            spatial_idx  = (
                self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1
            )
            min_scale = max_scale = crop_size = self.cfg.DATA.TEST_CROP_SIZE

        sampling_rate = self.cfg.DATA.SAMPLING_RATE

        for _ in range(self._num_retries):
            vid_container = None
            try:
                vid_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception:
                pass

            if vid_container is None:
                if self.mode != "test":
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frame_length = vid_container.streams.video[0].frames

            frames, frames_idx = decoder.decode(
                container=vid_container,
                sampling_rate=sampling_rate,
                num_frames=self.cfg.DATA.NUM_FRAMES,
                clip_idx=temporal_idx,
                num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                get_frame_idx=True,
            )

            if frames is None:
                if self.mode != "test":
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # ── gaze label ───────────────────────────────────────────────────
            video_path  = self._path_to_videos[index]
            video_name  = video_path.split("/")[-2]
            clip_name   = os.path.basename(video_path)[:-4]
            parts       = clip_name.split("_")
            clip_tstart = int(parts[-2][1:])
            clip_fstart = clip_tstart * self.cfg.DATA.TARGET_FPS
            frames_global_idx = frames_idx.numpy() + clip_fstart

            if (self.mode != "test" and
                    frames_global_idx[-1] >= self._labels[video_name].shape[0]):
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[video_name][frames_global_idx, :]   # [T, >=2]

            # ── audio ─────────────────────────────────────────────────────────
            audio_path = self._path_to_audios[index]
            try:
                audio_stu = _load_audio_student(audio_path)
                audio_tea = _load_audio_teacher(audio_path,
                                                 frames_global_idx,
                                                 frame_length)
            except Exception:
                T_ = self.cfg.DATA.NUM_FRAMES
                audio_stu = torch.zeros(1, F_STU, L_STU)
                audio_tea = torch.zeros(1, T_, 256, 256)

            # ── spatial transforms ────────────────────────────────────────────
            frames = ds_utils.tensor_normalize(frames, MEAN, STD)
            frames = frames.permute(3, 0, 1, 2)  # T H W C → C T H W

            frames, label = ds_utils.spatial_sampling(
                frames,
                gaze_loc=label,
                spatial_idx=spatial_idx,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            T_  = frames.size(1)
            H4  = frames.size(2) // 4
            W4  = frames.size(3) // 4
            label_hm = _make_heatmap(label, T_, H4, W4,
                                     self.cfg.DATA.GAUSSIAN_KERNEL)

            label_tensor = torch.as_tensor(label[:, :2]).float()   # [T, 2]

            return (frames, audio_tea, audio_stu,
                    label_hm, label_tensor, index)

        raise RuntimeError(
            f"Failed to load a valid sample after {self._num_retries} retries."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Distillation Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DistillationTrainer:
    """
    Progressive cross-modal distillation trainer.

    Curriculum
    ----------
    Stage 1  epochs  0-4  : V1 + V2
    Stage 2  epochs  5-9  : V1 + V2 + V3
    Stage 3  epochs 10+   : V1 + V2 + V3 + V4

    All learnable parameters (student + loss adapters) are updated by a
    single AdamW optimiser with CosineAnnealing LR decay.
    """

    CURRICULUM = {"v1": 0, "v2": 0, "v3": 5, "v4": 10}
    WEIGHTS    = {"v1": 1.0, "v2": 1.0, "v3": 0.5, "v4": 0.5}

    def __init__(self, teacher, student, device, lr=1e-4,
                 weight_decay=1e-4, n_epochs=20, output_dir="."):
        self.teacher    = teacher.to(device).eval()
        self.student    = student.to(device)
        self.device     = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Freeze teacher completely
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ── Loss modules ─────────────────────────────────────────────────────
        self.loss_v1 = OutputDistillationLoss().to(device)
        self.loss_v2 = FeatureDistillationLoss().to(device)
        self.loss_v3 = AttentionTransferLoss().to(device)
        self.loss_v4 = ProgressiveCRDLoss().to(device)

        # ── Optimise student + adapter layers (V2, V4 have trainable params) ─
        trainable = (
            list(self.student.parameters()) +
            list(self.loss_v2.parameters()) +
            list(self.loss_v4.parameters())
        )
        self.optim     = torch.optim.AdamW(trainable,
                                           lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=n_epochs, eta_min=1e-6
        )

        self.history = {k: [] for k in ["total", "v1", "v2", "v3", "v4",
                                         "val_total"]}
        self.best_val = float("inf")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_active(self, version: str, epoch: int) -> bool:
        return epoch >= self.CURRICULUM[version]

    def _active_stages(self, epoch: int) -> list:
        return [v for v in ["v1", "v2", "v3", "v4"] if self._is_active(v, epoch)]

    def _stage_label(self, epoch: int) -> str:
        stages = self._active_stages(epoch)
        if len(stages) <= 2:
            return "Stage-1 (V1+V2)"
        if len(stages) == 3:
            return "Stage-2 (V1+V2+V3)"
        return "Stage-3 (full)"

    # ── training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, loader, epoch: int) -> dict:
        self.student.train()
        self.loss_v2.train()
        self.loss_v4.train()

        sums = {k: 0.0 for k in ["total", "v1", "v2", "v3", "v4"]}
        n = 0

        for batch in loader:
            video, audio_tea, audio_stu, hm_gt, _, _ = batch

            video     = video.to(self.device, non_blocking=True)
            audio_tea = audio_tea.to(self.device, non_blocking=True)
            audio_stu = audio_stu.to(self.device, non_blocking=True)
            hm_gt     = hm_gt.to(self.device, non_blocking=True)

            # ── Teacher forward (frozen) ──────────────────────────────────────
            with torch.no_grad():
                t_out = self.teacher(video, audio_tea, return_feats=True)

            # ── Student forward ───────────────────────────────────────────────
            s_out = self.student(video, audio_stu)

            # ── Compute losses according to curriculum ────────────────────────
            total = torch.zeros(1, device=self.device)

            if self._is_active("v1", epoch):
                l1 = self.loss_v1(s_out["heatmap"],
                                   t_out["heatmap"], hm_gt)["total"]
                total = total + self.WEIGHTS["v1"] * l1
                sums["v1"] += l1.item()

            if self._is_active("v2", epoch):
                l2 = self.loss_v2(
                    t_out["vis_feat"], t_out["aud_feat"],
                    s_out["sv_feat"],  s_out["sa_feat"],
                    s_out["sfused"],
                )["total"]
                total = total + self.WEIGHTS["v2"] * l2
                sums["v2"] += l2.item()

            if self._is_active("v3", epoch):
                l3 = self.loss_v3(
                    t_out["av_attn"],
                    s_out["fusion_attn"],
                    s_out["spatial_feats"],
                )["total"]
                total = total + self.WEIGHTS["v3"] * l3
                sums["v3"] += l3.item()

            if self._is_active("v4", epoch):
                l4 = self.loss_v4(t_out, s_out)["total"]
                total = total + self.WEIGHTS["v4"] * l4
                sums["v4"] += l4.item()

            # ── Backward ──────────────────────────────────────────────────────
            self.optim.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optim.step()

            sums["total"] += total.item()
            n += 1

        self.scheduler.step()
        return {k: v / max(n, 1) for k, v in sums.items()}

    # ── validation epoch ─────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> float:
        """
        Validation: compute V1 output loss (KLD+MSE) on the val split.
        Returns mean total V1 loss — lower is better.
        """
        self.student.eval()
        total_loss = 0.0
        n = 0

        for batch in loader:
            video, audio_tea, audio_stu, hm_gt, _, _ = batch
            video     = video.to(self.device, non_blocking=True)
            audio_tea = audio_tea.to(self.device, non_blocking=True)
            audio_stu = audio_stu.to(self.device, non_blocking=True)
            hm_gt     = hm_gt.to(self.device, non_blocking=True)

            t_out = self.teacher(video, audio_tea, return_feats=True)
            s_out = self.student(video, audio_stu)

            l1 = self.loss_v1(s_out["heatmap"],
                               t_out["heatmap"], hm_gt)["total"]
            total_loss += l1.item()
            n += 1

        return total_loss / max(n, 1)

    # ── checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, train_losses: dict,
                        val_loss: float, is_best: bool):
        state = {
            "epoch"         : epoch,
            "train_losses"  : train_losses,
            "val_loss"      : val_loss,
            "model_state"   : self.student.state_dict(),
            "adapter_v2"    : self.loss_v2.state_dict(),
            "adapter_v4"    : self.loss_v4.state_dict(),
            "optimizer"     : self.optim.state_dict(),
            "scheduler"     : self.scheduler.state_dict(),
        }
        path = os.path.join(self.output_dir, f"epoch_{epoch:03d}.pth")
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_dir, "best.pth")
            torch.save(state, best_path)
            print(f"    ** New best val={val_loss:.6f}  →  {best_path}")

    def load_checkpoint(self, path: str):
        """Resume from a previously saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["model_state"], strict=False)
        if "adapter_v2" in ckpt:
            self.loss_v2.load_state_dict(ckpt["adapter_v2"], strict=False)
        if "adapter_v4" in ckpt:
            self.loss_v4.load_state_dict(ckpt["adapter_v4"], strict=False)
        if "optimizer" in ckpt:
            self.optim.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from {path}  (epoch {start_epoch})")
        return start_epoch

    # ── main training loop ────────────────────────────────────────────────────

    def run(self, train_loader, val_loader,
            n_epochs: int = 20, start_epoch: int = 0):
        header = (f"{'Epoch':>6}  {'Total':>8}  {'V1':>8}  "
                  f"{'V2':>8}  {'V3':>8}  {'V4':>8}  "
                  f"{'Val-V1':>8}  {'LR':>9}  Stage")
        print(header)
        print("─" * len(header))

        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()

            # ── train ────────────────────────────────────────────────────────
            tr = self.train_epoch(train_loader, epoch)

            # ── validate ─────────────────────────────────────────────────────
            val_loss = self.validate(val_loader, epoch)

            # ── record ───────────────────────────────────────────────────────
            for k in ["total", "v1", "v2", "v3", "v4"]:
                self.history[k].append(tr[k])
            self.history["val_total"].append(val_loss)

            # ── checkpoint ───────────────────────────────────────────────────
            is_best = val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
            self.save_checkpoint(epoch, tr, val_loss, is_best)

            # ── log ──────────────────────────────────────────────────────────
            lr  = self.optim.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"{epoch:>6d}  "
                f"{tr['total']:>8.4f}  {tr['v1']:>8.4f}  "
                f"{tr['v2']:>8.4f}  {tr['v3']:>8.4f}  {tr['v4']:>8.4f}  "
                f"{val_loss:>8.4f}  {lr:>9.2e}  "
                f"{self._stage_label(epoch)}  [{elapsed:.0f}s]"
            )

        print("\nTraining complete.")
        print(f"Best val V1-loss : {self.best_val:.6f}")
        print(f"Checkpoints in   : {self.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Progressive cross-modal distillation for Aria gaze"
    )
    p.add_argument("--cfg",     required=True,
                   help="Path to YAML config (CSTS_Aria_Gaze_Estimation.yaml)")
    p.add_argument("--teacher", required=True,
                   help="Path to frozen teacher .pth checkpoint")
    p.add_argument("--student", default=None,
                   help="(optional) Student checkpoint to resume from")
    p.add_argument("--output",  default="distillation_out",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--epochs",  type=int, default=20,
                   help="Total training epochs (default: 20)")
    p.add_argument("--lr",      type=float, default=1e-4,
                   help="Initial learning rate (default: 1e-4)")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Extra cfg key=value overrides")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── config ────────────────────────────────────────────────────────────────
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.TRAIN.ENABLE = False   # we manage training ourselves

    # ── device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Device : cuda  ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device : cpu")

    # ── teacher ───────────────────────────────────────────────────────────────
    print(f"\nLoading teacher from  {args.teacher}")
    teacher = build_model(cfg)
    ckpt    = torch.load(args.teacher, map_location="cpu")
    state   = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    n_tea = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"Teacher parameters : {n_tea:.1f} M  (frozen)")

    # ── student ───────────────────────────────────────────────────────────────
    student = StudentGazeModel()
    n_stu   = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student parameters : {n_stu:.2f} M\n")

    # ── datasets ──────────────────────────────────────────────────────────────
    print("Building datasets …")
    train_ds = DistillationAriaDataset(cfg, mode="train")
    val_ds   = DistillationAriaDataset(cfg, mode="val")
    print(f"  Train clips : {len(train_ds)}")
    print(f"  Val   clips : {len(val_ds)}")

    num_workers = min(cfg.DATA_LOADER.NUM_WORKERS, 4)
    pin_mem     = device.type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = DistillationTrainer(
        teacher    = teacher,
        student    = student,
        device     = device,
        lr         = args.lr,
        weight_decay = 1e-4,
        n_epochs   = args.epochs,
        output_dir = args.output,
    )

    # Resume if a student checkpoint was provided
    start_epoch = 0
    if args.student and os.path.isfile(args.student):
        start_epoch = trainer.load_checkpoint(args.student)

    # ── curriculum summary ────────────────────────────────────────────────────
    print("\nProgressive curriculum")
    print("  epochs  0-4   → V1 + V2          (Stage 1)")
    print("  epochs  5-9   → V1 + V2 + V3     (Stage 2)")
    print("  epochs 10+    → V1 + V2 + V3 + V4 (Stage 3)")
    print(f"\nLoss weights : {trainer.WEIGHTS}")
    print(f"Epochs       : {start_epoch} → {args.epochs}")
    print(f"Output dir   : {args.output}\n")

    # ── run ───────────────────────────────────────────────────────────────────
    trainer.run(train_loader, val_loader,
                n_epochs=args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
