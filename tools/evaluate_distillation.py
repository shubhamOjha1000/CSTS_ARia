#!/usr/bin/env python3
"""
Evaluate Teacher vs Student gaze models on F1, Precision, and Recall.

Two complementary metric families are computed for each model vs. GT:

  1.  Spatial hit-rate
      — argmax of predicted heatmap vs. GT (x,y) point
      — reported at radii: 5, 10, 20, 30 px on the quarter-resolution heatmap
      — since there is one prediction and one GT per frame, Precision = Recall = F1 = hit-rate

  2.  Heatmap pixel-level F1  (distinct P and R)
      — predicted heatmap is binarized by activating only the top-k% brightest pixels
      — GT Gaussian heatmap is binarized the same way
      — micro TP / FP / FN are accumulated over *all* pixels in *all* frames
      — k is swept over [1, 2, 5, 10, 20] % so you see the full P–R trade-off
      — a model that predicts a broad/diffuse heatmap scores high R but low P
      — a model that is precise but slightly off scores low R but high P

Outputs
-------
  <output>/metrics.json          — all numbers machine-readable
  <output>/spatial_f1.png        — bar chart: hit-rate at each radius
  <output>/heatmap_pr_curve.png  — P/R/F1 vs threshold % for teacher and student
  <output>/per_clip_table.txt    — clip-level breakdown

Run on Colab
------------
  cd /content/CSTS_ARia

  python tools/evaluate_distillation.py \\
      --cfg      configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \\
      --teacher  /content/drive/MyDrive/Aria_eg_dataset/csts_ego4d_forecast.pyth \\
      --student  /content/drive/MyDrive/distillation_run/best.pth \\
      --csv      data/distill_test_aria.csv \\
      --output   /content/drive/MyDrive/eval_out \\
      NUM_GPUS 0 \\
      DATA.PATH_PREFIX /content/drive/MyDrive/Aria_eg_dataset/clips
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Radius thresholds (pixels at quarter-resolution, e.g. 56×56 heatmap)
SPATIAL_RADII = [5, 10, 20, 30]

# Top-k% thresholds for binary heatmap F1
HM_TOPK_PCT = [1, 2, 5, 10, 20]


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers  (identical to train/visualize scripts)
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_student(npy_path: str) -> torch.Tensor:
    spec   = np.load(npy_path).astype(np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    spec_t = F.interpolate(spec_t, size=(F_STU, L_STU),
                            mode="bilinear", align_corners=False)
    return spec_t.squeeze(0)   # [1, F_STU, L_STU]


def _load_audio_teacher(npy_path: str,
                         frames_idx: np.ndarray,
                         frame_length: int) -> torch.Tensor:
    spec      = np.load(npy_path).astype(np.float32)
    audio_idx = (frames_idx / max(frame_length, 1)) * spec.shape[1]
    audio_idx = np.clip(np.round(audio_idx).astype(int), 128,
                        spec.shape[1] - 1 - 128)
    windows   = np.stack([spec[:, i - 128: i + 128] for i in audio_idx])
    return torch.from_numpy(windows[np.newaxis])  # [1, T, 256, 256]


def _make_heatmap(label: np.ndarray, T: int, H4: int, W4: int,
                  kernel_size: int = 19) -> np.ndarray:
    """Gaussian GT heatmap [T, H4, W4] from (x,y) in [0,1]."""
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
    """Per-frame normalise [T,H,W] to [0,1]."""
    out = np.zeros_like(hm)
    for t in range(hm.shape[0]):
        lo, hi = hm[t].min(), hm[t].max()
        out[t] = (hm[t] - lo) / (hi - lo + 1e-6)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _argmax_xy(hm: np.ndarray) -> tuple:
    """Return (px, py) pixel coordinates of heatmap peak. hm is [H, W]."""
    flat = hm.argmax()
    py, px = divmod(flat, hm.shape[1])
    return float(px), float(py)


def _topk_mask(hm: np.ndarray, pct: float) -> np.ndarray:
    """
    Binary mask: True for the top `pct`% of pixels by activation value.
    hm: [H, W]  float
    Returns: [H, W] bool
    """
    n_active = max(1, int(round(hm.size * pct / 100.0)))
    threshold = np.partition(hm.ravel(), -n_active)[-n_active]
    return hm >= threshold


def _prf1(tp: int, fp: int, fn: int) -> tuple:
    """Precision, Recall, F1 from raw counts."""
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ─────────────────────────────────────────────────────────────────────────────
#  Per-clip inference
# ─────────────────────────────────────────────────────────────────────────────

def run_clip(clip_path: str, cfg, teacher, student, device, labels_dict: dict):
    """
    Returns:
      frames_rgb   np [T, H, W, 3] uint8
      gt_hm        np [T, H4, W4]  normalised [0,1]
      teacher_hm   np [T, H4, W4]  normalised [0,1]
      student_hm   np [T, H4, W4]  normalised [0,1]
      label        np [T, 2]       GT (x,y) in [0,1]
      clip_name    str
      T            int
      H4, W4       int
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
        clip_idx=1,
        num_clips=1,
        video_meta={},
        target_fps=cfg.DATA.TARGET_FPS,
        backend=cfg.DATA.DECODING_BACKEND,
        max_spatial_scale=cfg.DATA.TEST_CROP_SIZE,
        use_offset=cfg.DATA.USE_OFFSET_SAMPLING,
        get_frame_idx=True,
    )

    video_name  = clip_path.split("/")[-2]
    clip_name   = os.path.basename(clip_path)[:-4]
    parts       = clip_name.split("_")
    clip_tstart = int(parts[-2][1:])
    clip_fstart = clip_tstart * cfg.DATA.TARGET_FPS
    frames_global_idx = frames_idx.numpy() + clip_fstart

    label = labels_dict[video_name][frames_global_idx, :]   # [T, >=2]

    f_norm = ds_utils.tensor_normalize(frames, MEAN, STD)
    f_norm = f_norm.permute(3, 0, 1, 2)          # [C, T, H, W]
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

    gt_hm = _make_heatmap(label, T_, H4, W4, cfg.DATA.GAUSSIAN_KERNEL)
    gt_hm = _norm_hm(gt_hm)

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

    video_t = f_norm.unsqueeze(0).to(device)

    with torch.no_grad():
        t_out = teacher([video_t], a_tea, return_feats=True)
        s_out = student(video_t, a_stu)

    teacher_hm = torch.sigmoid(t_out["heatmap"]).squeeze(0).squeeze(0).cpu().numpy()
    student_hm = torch.sigmoid(s_out["heatmap"]).squeeze(0).squeeze(0).cpu().numpy()

    teacher_hm = _norm_hm(teacher_hm)
    student_hm = _norm_hm(student_hm)

    return dict(
        gt_hm      = gt_hm,
        teacher_hm = teacher_hm,
        student_hm = student_hm,
        label      = np.array(label[:, :2], dtype=np.float32),
        clip_name  = clip_name,
        T          = T_,
        H4         = H4,
        W4         = W4,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Metric accumulators
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAccumulator:
    """Hit-rate at each radius threshold (px on quarter-res heatmap)."""

    def __init__(self, radii=SPATIAL_RADII):
        self.radii  = radii
        # [model][radius] → (hits, total)
        self._hits  = {m: {r: 0 for r in radii} for m in ("teacher", "student")}
        self._total = 0

    def update(self, res: dict):
        T, H4, W4 = res["T"], res["H4"], res["W4"]
        label      = res["label"]  # [T, 2]  (x,y) in [0,1]

        for t in range(T):
            gx = label[t, 0] * W4
            gy = label[t, 1] * H4

            for model, key in [("teacher", "teacher_hm"),
                                ("student", "student_hm")]:
                px, py = _argmax_xy(res[key][t])
                dist   = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                for r in self.radii:
                    if dist <= r:
                        self._hits[model][r] += 1

        self._total += T

    def results(self) -> dict:
        """Returns hit-rate = Precision = Recall = F1 for each model×radius."""
        out = {}
        for model in ("teacher", "student"):
            out[model] = {}
            for r in self.radii:
                hr = self._hits[model][r] / max(self._total, 1)
                out[model][r] = dict(hit_rate=hr, precision=hr, recall=hr, f1=hr)
        return out


class HeatmapF1Accumulator:
    """
    Pixel-level F1 using top-k% binarisation, accumulated micro over all frames.

    For each k in HM_TOPK_PCT, we track:
      TP, FP, FN  for teacher vs GT
      TP, FP, FN  for student vs GT
    """

    def __init__(self, topk_pcts=HM_TOPK_PCT):
        self.topk_pcts = topk_pcts
        self._tp = {m: {k: 0 for k in topk_pcts} for m in ("teacher", "student")}
        self._fp = {m: {k: 0 for k in topk_pcts} for m in ("teacher", "student")}
        self._fn = {m: {k: 0 for k in topk_pcts} for m in ("teacher", "student")}

    def update(self, res: dict):
        T = res["T"]
        for t in range(T):
            gt_frame = res["gt_hm"][t]

            for k in self.topk_pcts:
                gt_bin = _topk_mask(gt_frame, k)

                for model, key in [("teacher", "teacher_hm"),
                                    ("student", "student_hm")]:
                    pred_bin = _topk_mask(res[key][t], k)

                    tp = int((pred_bin & gt_bin).sum())
                    fp = int((pred_bin & ~gt_bin).sum())
                    fn = int((~pred_bin & gt_bin).sum())

                    self._tp[model][k] += tp
                    self._fp[model][k] += fp
                    self._fn[model][k] += fn

    def results(self) -> dict:
        out = {}
        for model in ("teacher", "student"):
            out[model] = {}
            for k in self.topk_pcts:
                tp = self._tp[model][k]
                fp = self._fp[model][k]
                fn = self._fn[model][k]
                p, r, f1 = _prf1(tp, fp, fn)
                out[model][k] = dict(precision=p, recall=r, f1=f1,
                                     tp=tp, fp=fp, fn=fn)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _color(model: str) -> str:
    return "#3498db" if model == "teacher" else "#e74c3c"


def plot_spatial(spatial_res: dict, out_path: str):
    """Bar chart: hit-rate (F1) at each radius for teacher vs student."""
    radii   = sorted(next(iter(spatial_res.values())).keys())
    models  = list(spatial_res.keys())
    x       = np.arange(len(radii))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    for i, model in enumerate(models):
        vals = [spatial_res[model][r]["f1"] for r in radii]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=model.capitalize(), color=_color(model),
                      edgecolor="white", alpha=0.9)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"r={r}px" for r in radii])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Hit-rate  (= Precision = Recall = F1)", fontsize=10)
    ax.set_title("Spatial Hit-rate at Multiple Radii\n"
                 "(argmax of heatmap vs. GT gaze point)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Radius threshold on quarter-resolution heatmap")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spatial plot  → {out_path}")


def plot_heatmap_prf(hm_res: dict, out_path: str):
    """3-panel plot: Precision / Recall / F1 vs top-k% threshold."""
    topk  = sorted(next(iter(hm_res.values())).keys())
    models = list(hm_res.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=110)
    metrics   = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]

    for ax, (metric, title) in zip(axes, metrics):
        for model in models:
            vals = [hm_res[model][k][metric] for k in topk]
            ax.plot(topk, vals, marker="o", label=model.capitalize(),
                    color=_color(model), linewidth=2, markersize=6)
            for x, y in zip(topk, vals):
                ax.annotate(f"{y:.3f}", (x, y),
                            textcoords="offset points", xytext=(0, 7),
                            fontsize=7, ha="center",
                            color=_color(model))

        ax.set_xlabel("Top-k% active pixels threshold", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f"Heatmap {title} vs. Threshold", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(topk)
        ax.set_xticklabels([f"{k}%" for k in topk])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Heatmap Pixel-level Precision / Recall / F1\n"
                 "(top-k% binarisation of predicted and GT heatmaps)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap P/R/F1 plot → {out_path}")


def plot_summary_bars(spatial_res: dict, hm_res: dict, out_path: str):
    """
    Single summary figure: side-by-side bars for Teacher vs Student.
    Columns:
      Spatial F1 @ r=20px | Heatmap P @ k=5% | Heatmap R @ k=5% | Heatmap F1 @ k=5%
    """
    models  = ["teacher", "student"]
    labels  = ["Spatial F1\n(r=20px)", "Heatmap P\n(k=5%)",
               "Heatmap R\n(k=5%)", "Heatmap F1\n(k=5%)"]

    tea_vals = [
        spatial_res["teacher"].get(20, {}).get("f1", 0.0),
        hm_res["teacher"].get(5, {}).get("precision", 0.0),
        hm_res["teacher"].get(5, {}).get("recall", 0.0),
        hm_res["teacher"].get(5, {}).get("f1", 0.0),
    ]
    stu_vals = [
        spatial_res["student"].get(20, {}).get("f1", 0.0),
        hm_res["student"].get(5, {}).get("precision", 0.0),
        hm_res["student"].get(5, {}).get("recall", 0.0),
        hm_res["student"].get(5, {}).get("f1", 0.0),
    ]

    x      = np.arange(len(labels))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)

    b1 = ax.bar(x - width / 2, tea_vals, width, label="Teacher",
                color="#3498db", edgecolor="white", alpha=0.9)
    b2 = ax.bar(x + width / 2, stu_vals, width, label="Student",
                color="#e74c3c", edgecolor="white", alpha=0.9)
    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=3)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Teacher vs. Student — Key Metrics Summary", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary plot   → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Teacher vs Student: F1 / Precision / Recall"
    )
    p.add_argument("--cfg",     required=True,
                   help="Path to YAML config (CSTS_Aria_Gaze_Estimation.yaml)")
    p.add_argument("--teacher", required=True,
                   help="Path to frozen teacher .pth checkpoint")
    p.add_argument("--student", default=None,
                   help="Path to trained student checkpoint (optional)")
    p.add_argument("--csv",     default="data/distill_test_aria.csv",
                   help="CSV of clips to evaluate (default: distill_test_aria.csv)")
    p.add_argument("--output",  default="eval_out",
                   help="Output directory for plots and JSON (default: eval_out)")
    p.add_argument("--n-clips", type=int, default=None,
                   help="Max clips to evaluate (default: all clips in CSV)")
    p.add_argument("--radii",   type=int, nargs="+", default=SPATIAL_RADII,
                   help=f"Spatial radius thresholds in px "
                        f"(default: {SPATIAL_RADII})")
    p.add_argument("--topk",    type=float, nargs="+", default=HM_TOPK_PCT,
                   help=f"Top-k%% active pixel thresholds for heatmap F1 "
                        f"(default: {HM_TOPK_PCT})")
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
    n_clips = len(rel_paths)
    print(f"\nClips to evaluate : {n_clips}\n")

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

    # ── accumulators ──────────────────────────────────────────────────────────
    sp_accum = SpatialAccumulator(radii=args.radii)
    hm_accum = HeatmapF1Accumulator(topk_pcts=args.topk)

    per_clip_rows = []

    # ── per-clip loop ─────────────────────────────────────────────────────────
    for idx, rel in enumerate(rel_paths):
        clip_path  = os.path.join(cfg.DATA.PATH_PREFIX, rel)
        clip_name  = os.path.basename(clip_path)[:-4]
        print(f"  [{idx + 1:>3}/{n_clips}]  {clip_name}", end="  ", flush=True)

        try:
            res = run_clip(clip_path, cfg, teacher, student, device, labels_dict)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        # ── clip-level spatial hit-rate @ r=20px ──────────────────────────────
        T, H4, W4 = res["T"], res["H4"], res["W4"]
        label     = res["label"]

        hit_tea = hit_stu = 0
        r_ref   = 20  # reference radius for per-clip reporting

        for t in range(T):
            gx = label[t, 0] * W4
            gy = label[t, 1] * H4
            for model_key, hm_key in [("tea", "teacher_hm"),
                                       ("stu", "student_hm")]:
                px, py = _argmax_xy(res[hm_key][t])
                dist   = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                if model_key == "tea" and dist <= r_ref:
                    hit_tea += 1
                elif model_key == "stu" and dist <= r_ref:
                    hit_stu += 1

        sp_hr_tea = hit_tea / max(T, 1)
        sp_hr_stu = hit_stu / max(T, 1)

        # ── clip-level heatmap F1 @ k=5% ─────────────────────────────────────
        k_ref = 5
        tp_t = fp_t = fn_t = 0
        tp_s = fp_s = fn_s = 0

        for t in range(T):
            gt_bin   = _topk_mask(res["gt_hm"][t],      k_ref)
            tea_bin  = _topk_mask(res["teacher_hm"][t], k_ref)
            stu_bin  = _topk_mask(res["student_hm"][t], k_ref)

            tp_t += int((tea_bin & gt_bin).sum())
            fp_t += int((tea_bin & ~gt_bin).sum())
            fn_t += int((~tea_bin & gt_bin).sum())

            tp_s += int((stu_bin & gt_bin).sum())
            fp_s += int((stu_bin & ~gt_bin).sum())
            fn_s += int((~stu_bin & gt_bin).sum())

        _, _, f1_tea = _prf1(tp_t, fp_t, fn_t)
        _, _, f1_stu = _prf1(tp_s, fp_s, fn_s)

        print(f"hit@r20: tea={sp_hr_tea:.3f}  stu={sp_hr_stu:.3f}  |  "
              f"hm-F1@k5: tea={f1_tea:.3f}  stu={f1_stu:.3f}")

        per_clip_rows.append(dict(
            clip       = clip_name,
            sp_tea     = sp_hr_tea,
            sp_stu     = sp_hr_stu,
            hm_f1_tea  = f1_tea,
            hm_f1_stu  = f1_stu,
        ))

        # ── update global accumulators ────────────────────────────────────────
        sp_accum.update(res)
        hm_accum.update(res)

    # ── global results ────────────────────────────────────────────────────────
    spatial_res = sp_accum.results()
    hm_res      = hm_accum.results()

    # ── output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # ── save plots ────────────────────────────────────────────────────────────
    print()
    plot_spatial(spatial_res,   os.path.join(args.output, "spatial_f1.png"))
    plot_heatmap_prf(hm_res,    os.path.join(args.output, "heatmap_pr_curve.png"))
    plot_summary_bars(spatial_res, hm_res,
                      os.path.join(args.output, "summary_bars.png"))

    # ── save JSON ─────────────────────────────────────────────────────────────
    results_json = dict(
        spatial   = {
            m: {str(r): v for r, v in d.items()}
            for m, d in spatial_res.items()
        },
        heatmap   = {
            m: {str(k): v for k, v in d.items()}
            for m, d in hm_res.items()
        },
        per_clip  = per_clip_rows,
    )
    json_path = os.path.join(args.output, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved metrics JSON  → {json_path}")

    # ── per-clip text table ───────────────────────────────────────────────────
    tbl_path = os.path.join(args.output, "per_clip_table.txt")
    with open(tbl_path, "w") as f:
        hdr = (f"{'Clip':<32} {'Hit@r20_tea':>12} {'Hit@r20_stu':>12} "
               f"{'HmF1@k5_tea':>12} {'HmF1@k5_stu':>12}\n")
        f.write(hdr)
        f.write("─" * len(hdr) + "\n")
        for row in per_clip_rows:
            f.write(f"{row['clip']:<32} {row['sp_tea']:>12.4f} {row['sp_stu']:>12.4f} "
                    f"{row['hm_f1_tea']:>12.4f} {row['hm_f1_stu']:>12.4f}\n")
        if per_clip_rows:
            f.write("─" * len(hdr) + "\n")
            f.write(f"{'MEAN':<32} "
                    f"{np.mean([r['sp_tea']    for r in per_clip_rows]):>12.4f} "
                    f"{np.mean([r['sp_stu']    for r in per_clip_rows]):>12.4f} "
                    f"{np.mean([r['hm_f1_tea'] for r in per_clip_rows]):>12.4f} "
                    f"{np.mean([r['hm_f1_stu'] for r in per_clip_rows]):>12.4f}\n")
    print(f"  Saved per-clip table → {tbl_path}")

    # ── console summary tables ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SPATIAL HIT-RATE  (argmax vs GT point — P = R = F1 = hit-rate)")
    print("─" * 72)
    print(f"  {'Radius':>10}  {'Teacher F1':>12}  {'Student F1':>12}  {'Δ (stu-tea)':>12}")
    print("─" * 72)
    for r in sorted(args.radii):
        t_f1 = spatial_res["teacher"][r]["f1"]
        s_f1 = spatial_res["student"][r]["f1"]
        print(f"  {f'r={r}px':>10}  {t_f1:>12.4f}  {s_f1:>12.4f}  {s_f1 - t_f1:>+12.4f}")
    print("=" * 72)

    print("\n" + "=" * 90)
    print("  HEATMAP PIXEL-LEVEL  F1  (top-k% binarisation, micro-aggregated)")
    print("─" * 90)
    print(f"  {'k%':>6}  {'Tea-P':>8}  {'Tea-R':>8}  {'Tea-F1':>8}  "
          f"{'Stu-P':>8}  {'Stu-R':>8}  {'Stu-F1':>8}  {'ΔF1':>8}")
    print("─" * 90)
    for k in sorted(args.topk):
        tk = hm_res["teacher"][k]
        sk = hm_res["student"][k]
        delta = sk["f1"] - tk["f1"]
        print(f"  {k:>5}%  {tk['precision']:>8.4f}  {tk['recall']:>8.4f}  {tk['f1']:>8.4f}  "
              f"{sk['precision']:>8.4f}  {sk['recall']:>8.4f}  {sk['f1']:>8.4f}  "
              f"{delta:>+8.4f}")
    print("=" * 90)

    # ── per-clip table to console ─────────────────────────────────────────────
    if per_clip_rows:
        print("\n" + "=" * 72)
        print("  PER-CLIP  (hit@r=20px  |  heatmap-F1 @ k=5%)")
        print("─" * 72)
        print(f"  {'Clip':<30} {'Hit_tea':>8} {'Hit_stu':>8} "
              f"{'F1_tea':>8} {'F1_stu':>8}")
        print("─" * 72)
        for row in per_clip_rows:
            print(f"  {row['clip']:<30} {row['sp_tea']:>8.4f} {row['sp_stu']:>8.4f} "
                  f"{row['hm_f1_tea']:>8.4f} {row['hm_f1_stu']:>8.4f}")
        print("─" * 72)
        print(f"  {'MEAN':<30} "
              f"{np.mean([r['sp_tea']    for r in per_clip_rows]):>8.4f} "
              f"{np.mean([r['sp_stu']    for r in per_clip_rows]):>8.4f} "
              f"{np.mean([r['hm_f1_tea'] for r in per_clip_rows]):>8.4f} "
              f"{np.mean([r['hm_f1_stu'] for r in per_clip_rows]):>8.4f}")
        print("=" * 72)

    print(f"\nAll outputs saved to : {args.output}/")
    print(f"  spatial_f1.png        — hit-rate bar chart")
    print(f"  heatmap_pr_curve.png  — P / R / F1 vs threshold")
    print(f"  summary_bars.png      — headline comparison")
    print(f"  metrics.json          — all numbers (machine-readable)")
    print(f"  per_clip_table.txt    — per-clip breakdown")


if __name__ == "__main__":
    main()
