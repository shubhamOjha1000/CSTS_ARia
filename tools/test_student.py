#!/usr/bin/env python3
"""
Test the lightweight StudentGazeModel on the Aria dataset.

Run on Colab:
    cd /content/CSTS_ARia
    python tools/test_student.py \
        --cfg    configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \
        --weights /content/drive/MyDrive/checkpoints/student.pth \
        NUM_GPUS 0 \
        DATA.PATH_PREFIX /content/drive/MyDrive/Aria_eg_dataset/clips \
        TEST.BATCH_SIZE 4

    # To run with random weights (shape/smoke test only, no checkpoint needed):
    python tools/test_student.py \
        --cfg    configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \
        NUM_GPUS 0 \
        DATA.PATH_PREFIX /content/drive/MyDrive/Aria_eg_dataset/clips

Output
------
Per-batch metrics are printed to stdout.
Heatmap visualisations are saved to  <OUTPUT_DIR>/student_vis/
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowfast.config.defaults import get_cfg
from slowfast.models.student_model import StudentGazeModel
from slowfast.datasets.student_aria_dataset import Student_Aria_Gaze

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test StudentGazeModel on Aria")
    p.add_argument("--cfg",     required=True, help="Path to YAML config")
    p.add_argument("--weights", default=None,  help="Path to student .pth checkpoint")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Extra cfg overrides e.g. NUM_GPUS 0")
    return p.parse_args()


# ── metrics ───────────────────────────────────────────────────────────────────

def adaptive_threshold_f1(pred_flat, gt_hm_flat, gt_loc, img_w, img_h, n_thresh=100):
    """
    Compute best-F1 over N_THRESH adaptive thresholds.
    pred_flat / gt_hm_flat : [N] tensors (already flattened per sample)
    """
    best_f1 = 0.0
    for t in np.linspace(pred_flat.min().item(), pred_flat.max().item(), n_thresh):
        pred_bin = (pred_flat >= t).float()
        tp = (pred_bin * gt_hm_flat).sum().item()
        fp = (pred_bin * (1 - gt_hm_flat)).sum().item()
        fn = ((1 - pred_bin) * gt_hm_flat).sum().item()
        if tp + fp + fn == 0:
            continue
        prec  = tp / (tp + fp + 1e-8)
        rec   = tp / (tp + fn + 1e-8)
        f1    = 2 * prec * rec / (prec + rec + 1e-8)
        best_f1 = max(best_f1, f1)
    return best_f1


def euclidean_error(pred_hm, gt_xy, img_w=64, img_h=64):
    """
    Angular-proxy: L2 distance between argmax of pred heatmap and GT gaze point.
    pred_hm : [T, H, W]
    gt_xy   : [T, 2]   (x, y) in [0,1]
    """
    T = pred_hm.shape[0]
    errors = []
    for t in range(T):
        flat_idx = pred_hm[t].view(-1).argmax().item()
        py = flat_idx // img_w
        px = flat_idx  % img_w
        gx = gt_xy[t, 0].item() * img_w
        gy = gt_xy[t, 1].item() * img_h
        errors.append(np.sqrt((px - gx)**2 + (py - gy)**2))
    return float(np.mean(errors))


# ── visualisation ─────────────────────────────────────────────────────────────

def save_heatmap_overlay(frame_np, heatmap_np, save_path):
    """
    Overlay a heatmap on an RGB frame and write to disk.
    frame_np  : [H, W, 3]  uint8
    heatmap_np: [h, w]     float32 in [0,1]
    """
    h, w = frame_np.shape[:2]
    hm_up = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_up = (hm_up - hm_up.min()) / (hm_up.max() - hm_up.min() + 1e-6)
    heat  = cv2.applyColorMap((hm_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    frame_bgr = frame_np[:, :, ::-1].copy()   # RGB → BGR for cv2
    overlay   = cv2.addWeighted(frame_bgr, 0.6, heat, 0.4, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)


# ── main test loop ─────────────────────────────────────────────────────────────

def test(cfg, weights_path, device):
    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = Student_Aria_Gaze(cfg, mode="test")
    num_workers = min(cfg.DATA_LOADER.NUM_WORKERS, 2)   # Colab has limited CPUs
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"Dataset: {len(dataset)} clips  |  {len(loader)} batches")

    # ── model ─────────────────────────────────────────────────────────────────
    model = StudentGazeModel().to(device)

    if weights_path and os.path.isfile(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        print("No checkpoint provided — running with random weights (shape check only).")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Student parameters: {n_params:.2f} M\n")

    # ── inference ─────────────────────────────────────────────────────────────
    vis_dir   = os.path.join(cfg.OUTPUT_DIR, "student_vis")
    all_f1    = []
    all_l2    = []

    with torch.no_grad():
        for batch_idx, (frames, audio, label, label_hm, vid_idx, meta) in enumerate(loader):
            # frames   : [B, C, T, H, W]  (pack_pathway returns a list; here we use frames directly)
            # The Student_Aria_Gaze returns frames as [C, T, H, W] per sample;
            # DataLoader stacks to [B, C, T, H, W].
            if isinstance(frames, (list, tuple)):
                frames = frames[0]   # unpack single-pathway

            frames   = frames.to(device, non_blocking=True)    # [B, C, T, H, W]
            audio    = audio.to(device, non_blocking=True)    # [B, 1, F_STU, L_STU]
            label    = label.to(device, non_blocking=True)    # [B, T, 2]
            label_hm = label_hm.to(device, non_blocking=True) # [B, T, H/4, W/4]

            out = model(frames, audio)

            # heatmap: [B, 1, T, H/4, W/4]
            hm_raw = out["heatmap"].squeeze(1)  # [B, T, 64, 64]
            # Normalise to [0,1] per sample
            B, T_, H_, W_ = hm_raw.shape
            hm_flat = hm_raw.view(B, T_, -1)
            hm_min  = hm_flat.min(-1, keepdim=True)[0]
            hm_max  = hm_flat.max(-1, keepdim=True)[0]
            hm_norm = ((hm_flat - hm_min) / (hm_max - hm_min + 1e-6)).view(B, T_, H_, W_)

            # ── metrics per sample ────────────────────────────────────────────
            for b in range(B):
                for t in range(T_):
                    pred_f = hm_norm[b, t].cpu().flatten()
                    gt_f   = label_hm[b, t].cpu().flatten()
                    f1     = adaptive_threshold_f1(pred_f, gt_f,
                                                   label[b, t].cpu(), W_, H_)
                    all_f1.append(f1)

                l2 = euclidean_error(hm_norm[b].cpu(), label[b].cpu(), W_, H_)
                all_l2.append(l2)

            # ── visualise first batch ─────────────────────────────────────────
            if batch_idx == 0:
                frames_np = frames.cpu().numpy()  # [B, C, T, H, W]
                for b in range(min(2, B)):
                    clip_name = os.path.basename(meta["path"][b])[:-4]
                    for t in range(T_):
                        frame = frames_np[b, :, t, :, :].transpose(1, 2, 0)  # H W C
                        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
                        frame = (frame * 255).astype(np.uint8)
                        hm_t  = hm_norm[b, t].cpu().numpy()
                        save_heatmap_overlay(
                            frame, hm_t,
                            os.path.join(vis_dir, clip_name, f"frame_{t:03d}.jpg")
                        )

            # ── progress ──────────────────────────────────────────────────────
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                cur_f1 = float(np.mean(all_f1)) if all_f1 else 0.0
                cur_l2 = float(np.mean(all_l2)) if all_l2 else 0.0
                print(f"  Batch [{batch_idx+1:>3}/{len(loader)}]  "
                      f"F1={cur_f1:.4f}  L2-err={cur_l2:.2f} px")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"  Final F1 (adaptive thresh) : {np.mean(all_f1):.4f}")
    print(f"  Final L2 error (px @ 64x64): {np.mean(all_l2):.2f}")
    print(f"  Visualisations saved to    : {vis_dir}")
    print("="*50)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.TRAIN.ENABLE = False

    # Always prefer GPU if available, regardless of NUM_GPUS config value
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: cpu")

    test(cfg, args.weights, device)
