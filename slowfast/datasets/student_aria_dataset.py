"""
Aria Dataset for the Lightweight Student Gaze Model.

Produces per-sample tensors:
  video  : [3, T, H, W]       = [3, 8, 256, 256]  — normalised RGB frames
  audio  : [1, F_stu, L_stu]  = [1, 64, 128]       — half-res log-STFT patch
  label  : [T, 2]              — (x, y) gaze in [0,1]
  label_hm: [T, H/4, W/4]     — Gaussian gaze heatmap  [T, 64, 64]

Expected folder layout (same as teacher pipeline):
  <PATH_PREFIX>/
    clips/
      v1/
        v1_t30_t34.mp4
        ...
    clips.audio_24kHz_stft/
      v1/
        v1_t30_t34.npy        # shape [256, N_time] log-STFT
    gaze_frame_label/
      v1.csv                  # produced by preprocess.get_aria_frame_label()

CSV file (data/test_aria_gaze.csv):
  v1/v1_t30_t34.mp4
  v2/v2_t38_t42.mp4
  ...
"""

import csv
import os
import random

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms

from slowfast.datasets import decoder
from slowfast.datasets import utils as ds_utils
from slowfast.datasets import video_container as container
from slowfast.datasets.build import DATASET_REGISTRY

# ─── student audio shape ──────────────────────────────────────────────────────
F_STU  = 64   # frequency bins  (teacher uses 256)
L_STU  = 128  # time steps      (teacher uses 256 per frame)

# ─── video normalisation (ImageNet stats, same as teacher) ────────────────────
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]


def _load_audio_student(npy_path: str) -> torch.Tensor:
    """
    Load a full-clip log-STFT .npy and resize to [1, F_STU, L_STU].

    The teacher .npy has shape [256, N_time] where N_time ≈ 800 for a 4-s clip
    at 24 kHz with hop=5 ms.  We bilinearly resize to [F_STU, L_STU] and add
    the channel dim → [1, F_STU, L_STU].
    """
    spec = np.load(npy_path).astype(np.float32)   # [256, N_time]
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, N_time]
    spec_t = F.interpolate(spec_t, size=(F_STU, L_STU),
                           mode='bilinear', align_corners=False)
    return spec_t.squeeze(0)  # [1, F_STU, L_STU]


def _make_heatmap(label: np.ndarray, T: int, H4: int, W4: int,
                  kernel_size: int = 19) -> torch.Tensor:
    """
    Build a Gaussian heatmap from (x,y) gaze labels.

    label  : [T, >=2]  — first two columns are (x, y) in [0,1]
    Returns: [T, H4, W4]  float32, sums to 1 per frame
    """
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
            hm[i, top:bottom+1, left:right+1] = k2d[k_top:k_bottom+1, k_left:k_right+1]
        d = hm[i].sum()
        if d == 0:
            hm[i] += 1.0 / (H4 * W4)
        else:
            hm[i] /= d
    return torch.from_numpy(hm)


@DATASET_REGISTRY.register()
class Student_Aria_Gaze(torch.utils.data.Dataset):
    """
    Aria video dataset wrapper for the lightweight student gaze model.

    Returns
    -------
    video     : [3, T, H, W]       torch.float32
    audio     : [1, F_STU, L_STU]  torch.float32
    label     : [T, 2]             torch.float32   (x, y) in [0,1]
    label_hm  : [T, H/4, W/4]     torch.float32
    index     : int
    meta      : dict  with keys 'path', 'index'
    """

    def __init__(self, cfg, mode: str, num_retries: int = 10):
        assert mode in ("train", "val", "test"), \
            f"Mode '{mode}' not supported. Choose train / val / test."
        self.mode        = mode
        self.cfg         = cfg
        self._num_retries = num_retries
        self._video_meta  = {}

        self._num_clips = (1 if mode in ("train", "val")
                           else cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)

        self._construct_loader()

    # ── loader ────────────────────────────────────────────────────────────────

    def _construct_loader(self):
        if self.mode == "train":
            csv_path = "data/train_aria_gaze.csv"
        else:
            csv_path = "data/test_aria_gaze.csv"

        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

        self._path_to_videos = []
        self._path_to_audios = []
        self._spatial_temporal_idx = []
        self._labels = {}

        with open(csv_path, "r") as f:
            paths = [l.strip() for l in f.read().splitlines() if l.strip()]

        for clip_idx, rel_path in enumerate(paths):
            for idx in range(self._num_clips):
                full_path = os.path.join(self.cfg.DATA.PATH_PREFIX, rel_path)
                self._path_to_videos.append(full_path)
                self._spatial_temporal_idx.append(idx)
                self._video_meta[clip_idx * self._num_clips + idx] = {}

        # Audio: teacher STFT stored alongside clips in clips.audio_24kHz_stft/
        for vp in self._path_to_videos:
            ap = vp.replace("clips", "clips.audio_24kHz_stft").replace(".mp4", ".npy")
            self._path_to_audios.append(ap)

        # Gaze labels
        for vp in self._path_to_videos:
            video_name = vp.split("/")[-2]  # e.g. "v1"
            if video_name in self._labels:
                continue
            prefix = os.path.dirname(self.cfg.DATA.PATH_PREFIX)
            label_file = os.path.join(prefix, "gaze_frame_label", f"{video_name}.csv")
            with open(label_file, "r") as f:
                rows = [list(map(float, row)) for i, row in enumerate(csv.reader(f)) if i > 0]
            self._labels[video_name] = np.array(rows)[:, 2:]  # [x, y, gaze_type, ...]

        assert len(self._path_to_videos) > 0, \
            f"No videos loaded from {csv_path}"

    # ── getitem ───────────────────────────────────────────────────────────────

    def __getitem__(self, index):
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        # ── sampling params ──────────────────────────────────────────────────
        if self.mode == "train":
            temporal_sample_index = -1   # random
            spatial_sample_index  = -1
            min_scale  = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale  = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size  = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            temporal_sample_index = 1
            spatial_sample_index  = (
                self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1
            )
            min_scale = max_scale = crop_size = self.cfg.DATA.TEST_CROP_SIZE

        sampling_rate = self.cfg.DATA.SAMPLING_RATE  # 4

        for i_try in range(self._num_retries):
            # ── decode video ─────────────────────────────────────────────────
            vid_container = None
            try:
                vid_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                pass

            if vid_container is None:
                if self.mode != "test" and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frame_length = vid_container.streams.video[0].frames

            frames, frames_idx = decoder.decode(
                container=vid_container,
                sampling_rate=sampling_rate,
                num_frames=self.cfg.DATA.NUM_FRAMES,
                clip_idx=temporal_sample_index,
                num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                get_frame_idx=True,
            )

            if frames is None:
                if self.mode != "test" and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # ── gaze label ───────────────────────────────────────────────────
            video_path = self._path_to_videos[index]
            video_name = video_path.split("/")[-2]
            clip_name  = os.path.basename(video_path)[:-4]  # strip .mp4
            parts      = clip_name.split("_")
            # clip name format: {vid}_t{start}_t{end}
            clip_tstart = int(parts[-2][1:])  # remove leading 't'
            clip_fstart = clip_tstart * self.cfg.DATA.TARGET_FPS
            frames_global_idx = frames_idx.numpy() + clip_fstart

            if (self.mode != "test" and
                    frames_global_idx[-1] >= self._labels[video_name].shape[0]):
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[video_name][frames_global_idx, :]  # [T, >=2]

            # ── student audio: resize full-clip STFT to [1, 64, 128] ─────────
            audio_path = self._path_to_audios[index]
            try:
                audio = _load_audio_student(audio_path)  # [1, F_STU, L_STU]
            except Exception:
                # fallback: silence (model still runs, just degrades quality)
                audio = torch.zeros(1, F_STU, L_STU)

            # ── spatial transforms ───────────────────────────────────────────
            frames = ds_utils.tensor_normalize(frames, MEAN, STD)
            frames = frames.permute(3, 0, 1, 2)  # T H W C -> C T H W

            frames, label = ds_utils.spatial_sampling(
                frames,
                gaze_loc=label,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            # frames: [C, T, H, W]

            # ── heatmap ──────────────────────────────────────────────────────
            T_  = frames.size(1)
            H4  = frames.size(2) // 4   # 64
            W4  = frames.size(3) // 4   # 64
            label_hm = _make_heatmap(label, T_, H4, W4,
                                     self.cfg.DATA.GAUSSIAN_KERNEL)

            label_tensor = torch.as_tensor(label[:, :2]).float()  # [T, 2]

            meta = {
                "path" : video_path,
                "index": frames_global_idx,
            }

            return frames, audio, label_tensor, label_hm, index, meta

        raise RuntimeError(
            f"Failed to load a valid sample after {self._num_retries} retries."
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._path_to_videos)

    @property
    def num_videos(self):
        return len(self._path_to_videos)
