#!/usr/bin/env python3
"""
Gaze heatmap visualisation — Teacher vs Student vs Ground Truth.

For each clip in the CSV, produces a 3-row × T-col grid:
  Row 0 — Ground Truth heatmap (Gaussian from label)
  Row 1 — Teacher (CSTS)       heatmap
  Row 2 — Student (AR/VR)      heatmap

Additionally saves:
  • Per-clip PNG grids   →  <output>/grids/<clip_name>.png
  • Per-frame overlays   →  <output>/overlays/<clip_name>/frame_XXX_[gt|teacher|student].jpg
  • Summary mosaic       →  <output>/mosaic.png  (first N_MOSAIC clips × first 4 frames)
  • Loss curves          →  <output>/loss_curves.png  (if --checkpoint given)

Run on Colab
------------
  cd /content/CSTS_ARia

  # Minimal — random student weights (shape check)
  python tools/visualize_distillation.py \\
      --cfg      configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \\
      --teacher  /content/drive/MyDrive/Aria_eg_dataset/csts_ego4d_forecast.pyth \\
      --csv      data/distill_test_aria.csv \\
      --output   /content/drive/MyDrive/vis_out \\
      NUM_GPUS 0 \\
      DATA.PATH_PREFIX /content/drive/MyDrive/Aria_eg_dataset/clips

  # With trained student checkpoint
  python tools/visualize_distillation.py \\
      --cfg        configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \\
      --teacher    /content/drive/MyDrive/Aria_eg_dataset/csts_ego4d_forecast.pyth \\
      --student    /content/drive/MyDrive/distillation_run/best.pth \\
      --csv        data/distill_test_aria.csv \\
      --output     /content/drive/MyDrive/vis_out \\
      --n-clips    6 \\
      NUM_GPUS 0 \\
      DATA.PATH_PREFIX /content/drive/MyDrive/Aria_eg_dataset/clips
"""

import argparse
import os
import sys
import csv
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # no display needed on Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.models.student_model import StudentGazeModel
from slowfast.datasets import decoder
from slowfast.datasets import utils as ds_utils
from slowfast.datasets import video_container as container

# ── constants ─────────────────────────────────────────────────────────────────
F_STU  = 64
L_STU  = 128
MEAN   = [0.45, 0.45, 0.45]
STD    = [0.225, 0.225, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_student(npy_path: str) -> torch.Tensor:
    spec   = np.load(npy_path).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    spec_t = F.interpolate(spec_t, size=(F_STU, L_STU),
                            mode='bilinear', align_corners=False)
    return spec_t.squeeze(0)


def _load_audio_teacher(npy_path: str,
                         frames_idx: np.ndarray,
                         frame_length: int) -> torch.Tensor:
    spec      = np.load(npy_path).astype(np.float32)
    audio_idx = (frames_idx / max(frame_length, 1)) * spec.shape[1]
    audio_idx = np.clip(np.round(audio_idx).astype(int), 128,
                        spec.shape[1] - 1 - 128)
    windows   = np.stack([spec[:, i - 128: i + 128] for i in audio_idx])
    return torch.from_numpy(windows[np.newaxis])


def _make_heatmap(label: np.ndarray, T: int, H4: int, W4: int,
                  kernel_size: int = 19) -> np.ndarray:
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
    return hm


def _norm_hm(hm: np.ndarray) -> np.ndarray:
    """Normalise a [T,H,W] heatmap to [0,1] per frame."""
    out = np.zeros_like(hm)
    for t in range(hm.shape[0]):
        lo, hi = hm[t].min(), hm[t].max()
        out[t] = (hm[t] - lo) / (hi - lo + 1e-6)
    return out


def _to_colormap(hm_t: np.ndarray) -> np.ndarray:
    """Convert a [H,W] float [0,1] heatmap to [H,W,3] RGB uint8 via JET."""
    u8  = (hm_t * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return bgr[:, :, ::-1]   # BGR → RGB


def _overlay(frame_rgb: np.ndarray, hm_t: np.ndarray,
             alpha: float = 0.45) -> np.ndarray:
    """Overlay heatmap on frame (both [H,W,3] uint8)."""
    h, w = frame_rgb.shape[:2]
    hm_up = cv2.resize(hm_t.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR)
    lo, hi = hm_up.min(), hm_up.max()
    hm_up  = (hm_up - lo) / (hi - lo + 1e-6)
    heat   = _to_colormap(hm_up)
    return (frame_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-clip inference
# ─────────────────────────────────────────────────────────────────────────────

def run_clip(clip_path: str, cfg, teacher, student, device,
             labels_dict: dict):
    """
    Returns dict with keys:
      frames_rgb   np [T, H, W, 3]  uint8 (unnormalised, original scale)
      gt_hm        np [T, H4, W4]   normalised to [0,1]
      teacher_hm   np [T, H4, W4]   normalised to [0,1]
      student_hm   np [T, H4, W4]   normalised to [0,1]
      clip_name    str
    """
    vid_container = container.get_video_container(
        clip_path,
        cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
        cfg.DATA.DECODING_BACKEND,
    )
    frame_length = vid_container.streams.video[0].frames

    frames, frames_idx = decoder.decode(
        container=vid_container,
        sampling_rate=cfg.DATA.SAMPLING_RATE,
        num_frames=cfg.DATA.NUM_FRAMES,
        clip_idx=1,          # centre clip
        num_clips=1,
        video_meta={},
        target_fps=cfg.DATA.TARGET_FPS,
        backend=cfg.DATA.DECODING_BACKEND,
        max_spatial_scale=cfg.DATA.TEST_CROP_SIZE,
        use_offset=cfg.DATA.USE_OFFSET_SAMPLING,
        get_frame_idx=True,
    )

    # ── gaze label ────────────────────────────────────────────────────────────
    video_name  = clip_path.split("/")[-2]
    clip_name   = os.path.basename(clip_path)[:-4]
    parts       = clip_name.split("_")
    clip_tstart = int(parts[-2][1:])
    clip_fstart = clip_tstart * cfg.DATA.TARGET_FPS
    frames_global_idx = frames_idx.numpy() + clip_fstart

    label = labels_dict[video_name][frames_global_idx, :]   # [T, >=2]

    # ── keep unnormalised RGB for overlay ─────────────────────────────────────
    frames_unnorm = frames.numpy()                # [T, H, W, C]  float32 0-255
    frames_rgb    = frames_unnorm.clip(0, 255).astype(np.uint8)   # [T, H, W, 3]

    # ── normalise + crop → model input ───────────────────────────────────────
    f_norm = ds_utils.tensor_normalize(frames, MEAN, STD)
    f_norm = f_norm.permute(3, 0, 1, 2)       # [C, T, H, W]
    f_norm, label = ds_utils.spatial_sampling(
        f_norm, gaze_loc=label,
        spatial_idx=1,
        min_scale=cfg.DATA.TEST_CROP_SIZE,
        max_scale=cfg.DATA.TEST_CROP_SIZE,
        crop_size=cfg.DATA.TEST_CROP_SIZE,
        random_horizontal_flip=False,
        inverse_uniform_sampling=False,
    )

    T_  = f_norm.size(1)
    H4  = f_norm.size(2) // 4
    W4  = f_norm.size(3) // 4

    # crop the raw RGB frames to the same spatial crop
    # (spatial_sampling crops from centre for spatial_idx=1)
    crop_size = cfg.DATA.TEST_CROP_SIZE
    H_raw, W_raw = frames_rgb.shape[1], frames_rgb.shape[2]
    cy = (H_raw - crop_size) // 2
    cx = (W_raw - crop_size) // 2
    cy = max(cy, 0); cx = max(cx, 0)
    frames_rgb = frames_rgb[:,
                            cy: cy + crop_size,
                            cx: cx + crop_size, :]

    # ── GT heatmap ───────────────────────────────────────────────────────────
    gt_hm = _make_heatmap(label, T_, H4, W4, cfg.DATA.GAUSSIAN_KERNEL)
    gt_hm = _norm_hm(gt_hm)

    # ── audio ─────────────────────────────────────────────────────────────────
    audio_path = (clip_path
                  .replace("clips", "clips.audio_24kHz_stft")
                  .replace(".mp4", ".npy"))
    try:
        a_stu = _load_audio_student(audio_path).unsqueeze(0).to(device)
        a_tea = _load_audio_teacher(audio_path,
                                     frames_global_idx,
                                     frame_length).unsqueeze(0).to(device)
    except Exception:
        a_stu = torch.zeros(1, 1, F_STU, L_STU, device=device)
        a_tea = torch.zeros(1, 1, T_, 256, 256, device=device)

    video_t = f_norm.unsqueeze(0).to(device)   # [1, C, T, H, W]

    with torch.no_grad():
        t_out = teacher([video_t], a_tea, return_feats=True)
        s_out = student(video_t, a_stu)

    teacher_hm = torch.sigmoid(t_out["heatmap"]).squeeze(0).squeeze(0)  # [T, H4, W4]
    student_hm = torch.sigmoid(s_out["heatmap"]).squeeze(0).squeeze(0)  # [T, H4, W4]

    teacher_hm = _norm_hm(teacher_hm.cpu().numpy())
    student_hm = _norm_hm(student_hm.cpu().numpy())

    return {
        "frames_rgb" : frames_rgb,
        "gt_hm"      : gt_hm,
        "teacher_hm" : teacher_hm,
        "student_hm" : student_hm,
        "clip_name"  : clip_name,
        "T"          : T_,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

ROW_META = [
    ("Ground Truth",    "gt_hm",      "#2ecc71"),
    ("Teacher (CSTS)",  "teacher_hm", "#3498db"),
    ("Student (AR/VR)", "student_hm", "#e74c3c"),
]


def save_grid(result: dict, out_path: str, max_frames: int = 8):
    """3-row × T-col grid of heatmaps for a single clip."""
    T      = min(result["T"], max_frames)
    fig    = plt.figure(figsize=(T * 2.4, 8), dpi=110)
    gs     = gridspec.GridSpec(3, T, hspace=0.04, wspace=0.04)

    for row, (title, key, colour) in enumerate(ROW_META):
        hm   = result[key]            # [T, H4, W4]
        for t in range(T):
            ax = fig.add_subplot(gs[row, t])

            # heatmap coloured
            img = _to_colormap(hm[t])
            ax.imshow(img)

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2)

            if t == 0:
                ax.set_ylabel(title, fontsize=9, color=colour,
                              labelpad=4, fontweight="bold")
            if row == 0:
                ax.set_title(f"t={t}", fontsize=8, pad=3)

    fig.suptitle(result["clip_name"], fontsize=11, y=1.01)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overlay_frames(result: dict, out_dir: str, max_frames: int = 8):
    """
    Save per-frame side-by-side overlays:
      [frame | GT overlay | teacher overlay | student overlay]
    """
    T   = min(result["T"], max_frames)
    rgb = result["frames_rgb"]          # [T, H, W, 3]
    H4, W4 = result["gt_hm"].shape[1:]

    os.makedirs(out_dir, exist_ok=True)
    for t in range(T):
        frame = cv2.resize(rgb[t], (W4 * 4, H4 * 4))   # scale up to match input

        panels = [frame]
        for _, key, _ in ROW_META:
            panels.append(_overlay(frame, result[key][t]))

        row_img = np.concatenate(panels, axis=1)           # side by side

        # write labels at top
        labels = ["Frame", "GT", "Teacher", "Student"]
        colours_bgr = [(255,255,255), (82,211,116), (52,152,219), (231,76,60)]
        panel_w = frame.shape[1]
        for i, (lbl, col) in enumerate(zip(labels, colours_bgr)):
            x0 = i * panel_w + 4
            cv2.putText(row_img, lbl, (x0, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                        cv2.LINE_AA)
            cv2.putText(row_img, lbl, (x0, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2,
                        cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, f"frame_{t:03d}.jpg"),
                    row_img[:, :, ::-1])   # RGB → BGR for cv2


def save_mosaic(results: list, out_path: str,
                n_clips: int = 4, n_frames: int = 4):
    """
    n_clips × n_frames comparison mosaic (all three rows per cell).
    Layout: rows = clips, cols = frames, sub-cell = GT/teacher/student.
    """
    n_clips  = min(n_clips, len(results))
    n_frames = min(n_frames, min(r["T"] for r in results))

    cell_h, cell_w = 3, n_frames   # 3 rows per clip
    fig = plt.figure(figsize=(n_frames * 2.2, n_clips * 6), dpi=100)
    outer = gridspec.GridSpec(n_clips, 1, hspace=0.35)

    for ci, res in enumerate(results[:n_clips]):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, n_frames, subplot_spec=outer[ci],
            hspace=0.06, wspace=0.06
        )
        for row, (title, key, colour) in enumerate(ROW_META):
            hm = res[key]
            for t in range(n_frames):
                ax = fig.add_subplot(inner[row, t])
                ax.imshow(_to_colormap(hm[t]))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(colour)
                    spine.set_linewidth(1.5)
                if t == 0:
                    ax.set_ylabel(title, fontsize=7, color=colour,
                                  labelpad=3, fontweight="bold")
                if row == 0 and t == 0:
                    ax.set_title(res["clip_name"][:28], fontsize=8,
                                 loc="left", pad=3)
                if row == 0:
                    ax.set_title(f"t={t}", fontsize=7, pad=2)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mosaic     → {out_path}")


def save_loss_curves(ckpt_path: str, out_path: str):
    """Plot training history saved inside a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("train_losses", None)
    if history is None:
        print("  No history found in checkpoint — skipping loss curves.")
        return

    # Re-build epoch series from a single epoch snapshot
    # If the checkpoint contains cumulative history (saved by run()), use it.
    # Otherwise we have only the last epoch's losses — skip.
    val_loss = ckpt.get("val_loss", None)
    epoch    = ckpt.get("epoch", 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=110)

    # — Left: bar chart of last-epoch losses
    ax = axes[0]
    keys   = ["v1", "v2", "v3", "v4"]
    values = [history.get(k, 0.0) for k in keys]
    colours = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    bars = ax.bar(keys, values, color=colours, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.set_title(f"Epoch {epoch} — per-loss breakdown", fontsize=11)
    ax.set_ylabel("Loss")
    ax.set_ylim(0, max(values) * 1.3 if values else 1)
    ax.grid(axis="y", alpha=0.3)

    # — Right: stage annotation
    ax2 = axes[1]
    ax2.axis("off")
    info = (
        f"Epoch       : {epoch}\n"
        f"Train total : {history.get('total', 0.0):.6f}\n"
        f"Val   V1    : {val_loss:.6f}\n\n" if val_loss else ""
        f"Curriculum\n"
        f"  Stage 1 (0–4)  : V1 + V2\n"
        f"  Stage 2 (5–9)  : V1 + V2 + V3\n"
        f"  Stage 3 (10+)  : V1 + V2 + V3 + V4\n\n"
        f"Loss weights\n"
        f"  V1=1.0  V2=1.0  V3=0.5  V4=0.5"
    )
    ax2.text(0.05, 0.95, info, transform=ax2.transAxes,
             fontsize=10, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss info  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def kld(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence KLD(p || q), both [H,W] or [T,H,W] normalised."""
    eps = 1e-10
    return float((p * np.log(p / (q + eps) + eps)).sum())


def l2_px(hm_pred: np.ndarray, label: np.ndarray,
          H4: int, W4: int) -> float:
    """Mean pixel-L2 between heatmap argmax and GT label."""
    T = hm_pred.shape[0]
    errs = []
    for t in range(T):
        flat_idx = hm_pred[t].argmax()
        py, px   = divmod(flat_idx, W4)
        gx = label[t, 0] * W4
        gy = label[t, 1] * H4
        errs.append(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
    return float(np.mean(errs))


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise gaze heatmaps: GT vs Teacher vs Student"
    )
    p.add_argument("--cfg",      required=True,
                   help="Path to YAML config (CSTS_Aria_Gaze_Estimation.yaml)")
    p.add_argument("--teacher",  required=True,
                   help="Path to frozen teacher .pth checkpoint")
    p.add_argument("--student",  default=None,
                   help="Path to trained student checkpoint (optional)")
    p.add_argument("--csv",      default="data/distill_test_aria.csv",
                   help="CSV of clips to visualise (default: distill_test_aria.csv)")
    p.add_argument("--output",   default="vis_out",
                   help="Output directory for all saved images")
    p.add_argument("--n-clips",  type=int, default=None,
                   help="Max clips to process (default: all clips in CSV)")
    p.add_argument("--n-frames", type=int, default=8,
                   help="Max frames per clip to visualise (default: 8)")
    p.add_argument("--no-overlays", action="store_true",
                   help="Skip per-frame overlay images (saves disk space)")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Extra cfg overrides e.g. NUM_GPUS 0")
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
    cfg.TRAIN.ENABLE = False

    # ── device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Device : cuda  ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device : cpu")

    # ── teacher ───────────────────────────────────────────────────────────────
    print(f"\nLoading teacher  ← {args.teacher}")
    teacher = build_model(cfg)
    ckpt_t  = torch.load(args.teacher, map_location="cpu")
    state_t = ckpt_t.get("model_state", ckpt_t.get("state_dict", ckpt_t))
    teacher.load_state_dict(state_t, strict=False)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── student ───────────────────────────────────────────────────────────────
    student = StudentGazeModel().to(device).eval()
    if args.student and os.path.isfile(args.student):
        print(f"Loading student  ← {args.student}")
        ckpt_s  = torch.load(args.student, map_location=device)
        state_s = ckpt_s.get("model_state", ckpt_s.get("state_dict", ckpt_s))
        student.load_state_dict(state_s, strict=False)
    else:
        print("No student checkpoint — using random weights (shape check only).")
    for p in student.parameters():
        p.requires_grad_(False)

    # ── clip list ─────────────────────────────────────────────────────────────
    assert os.path.exists(args.csv), f"CSV not found: {args.csv}"
    with open(args.csv) as f:
        rel_paths = [l.strip() for l in f if l.strip()]
    if args.n_clips:
        rel_paths = rel_paths[: args.n_clips]
    print(f"\nClips to visualise: {len(rel_paths)}\n")

    # ── gaze labels ───────────────────────────────────────────────────────────
    labels_dict = {}
    for rel in rel_paths:
        video_name = rel.split("/")[0]
        if video_name in labels_dict:
            continue
        prefix     = os.path.dirname(cfg.DATA.PATH_PREFIX)
        label_file = os.path.join(prefix, "gaze_frame_label",
                                  f"{video_name}.csv")
        with open(label_file) as f:
            rows = [list(map(float, r))
                    for i, r in enumerate(csv.reader(f)) if i > 0]
        labels_dict[video_name] = np.array(rows)[:, 2:]

    # ── output directories ────────────────────────────────────────────────────
    grids_dir    = os.path.join(args.output, "grids")
    overlays_dir = os.path.join(args.output, "overlays")
    os.makedirs(grids_dir,    exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    # ── per-clip processing ───────────────────────────────────────────────────
    all_results  = []
    all_metrics  = []

    for idx, rel in enumerate(rel_paths):
        clip_path = os.path.join(cfg.DATA.PATH_PREFIX, rel)
        video_name = rel.split("/")[0]
        clip_name  = os.path.basename(clip_path)[:-4]
        print(f"  [{idx+1:>3}/{len(rel_paths)}]  {clip_name}", end="  ")

        try:
            res = run_clip(clip_path, cfg, teacher, student,
                           device, labels_dict)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        # ── grid PNG ─────────────────────────────────────────────────────────
        grid_path = os.path.join(grids_dir, f"{clip_name}.png")
        save_grid(res, grid_path, max_frames=args.n_frames)

        # ── overlay JPGs ──────────────────────────────────────────────────────
        if not args.no_overlays:
            ov_dir = os.path.join(overlays_dir, clip_name)
            save_overlay_frames(res, ov_dir, max_frames=args.n_frames)

        # ── metrics (teacher vs student vs GT) ───────────────────────────────
        H4, W4   = res["gt_hm"].shape[1:]
        parts    = clip_name.split("_")
        tstart   = int(parts[-2][1:])
        fstart   = tstart * cfg.DATA.TARGET_FPS
        T_       = res["T"]
        # Re-derive label slice for metric (already aligned in run_clip)
        label_np = labels_dict[video_name]
        label_slice = label_np[fstart: fstart + T_, :2]
        if label_slice.shape[0] < T_:
            label_slice = np.pad(label_slice,
                                 ((0, T_ - label_slice.shape[0]), (0, 0)),
                                 mode="edge")

        kld_tea  = kld(res["gt_hm"].mean(0), res["teacher_hm"].mean(0))
        kld_stu  = kld(res["gt_hm"].mean(0), res["student_hm"].mean(0))
        l2_tea   = l2_px(res["teacher_hm"], label_slice, H4, W4)
        l2_stu   = l2_px(res["student_hm"], label_slice, H4, W4)

        all_metrics.append({
            "clip":    clip_name,
            "KLD_tea": kld_tea, "KLD_stu": kld_stu,
            "L2_tea":  l2_tea,  "L2_stu":  l2_stu,
        })
        print(f"KLD  tea={kld_tea:.3f}  stu={kld_stu:.3f}  |  "
              f"L2  tea={l2_tea:.1f}px  stu={l2_stu:.1f}px")

        all_results.append(res)

    # ── mosaic ────────────────────────────────────────────────────────────────
    if all_results:
        mosaic_path = os.path.join(args.output, "mosaic.png")
        save_mosaic(all_results, mosaic_path,
                    n_clips=min(6, len(all_results)),
                    n_frames=min(4, args.n_frames))

    # ── loss curves from student checkpoint ───────────────────────────────────
    if args.student and os.path.isfile(args.student):
        save_loss_curves(args.student,
                         os.path.join(args.output, "loss_info.png"))

    # ── summary table ─────────────────────────────────────────────────────────
    if all_metrics:
        print("\n" + "=" * 70)
        print(f"  {'Clip':<30} {'KLD_tea':>8} {'KLD_stu':>8} "
              f"{'L2_tea':>8} {'L2_stu':>8}")
        print("─" * 70)
        for m in all_metrics:
            print(f"  {m['clip']:<30} {m['KLD_tea']:>8.3f} {m['KLD_stu']:>8.3f} "
                  f"{m['L2_tea']:>8.1f} {m['L2_stu']:>8.1f}")

        avg_kld_tea = np.mean([m["KLD_tea"] for m in all_metrics])
        avg_kld_stu = np.mean([m["KLD_stu"] for m in all_metrics])
        avg_l2_tea  = np.mean([m["L2_tea"]  for m in all_metrics])
        avg_l2_stu  = np.mean([m["L2_stu"]  for m in all_metrics])
        print("─" * 70)
        print(f"  {'MEAN':<30} {avg_kld_tea:>8.3f} {avg_kld_stu:>8.3f} "
              f"{avg_l2_tea:>8.1f} {avg_l2_stu:>8.1f}")
        print("=" * 70)

    print(f"\nAll outputs saved to : {args.output}/")
    print(f"  grids/             — per-clip 3×T heatmap grids")
    if not args.no_overlays:
        print(f"  overlays/          — per-frame side-by-side overlays")
    print(f"  mosaic.png         — summary mosaic")


if __name__ == "__main__":
    main()
