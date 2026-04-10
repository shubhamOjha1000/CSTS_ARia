#!/usr/bin/env python3
"""
Sanity-check for CSTS return_feats=True feature extraction.

Run on Colab:
    cd /content/CSTS_ARia
    python tools/test_return_feats.py \
        --cfg configs/Aria/CSTS_Aria_Gaze_Estimation.yaml \
        NUM_GPUS 0 \
        TRAIN.ENABLE False
"""

import argparse
import sys
import torch

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True, help="Path to YAML config")
parser.add_argument("opts", nargs=argparse.REMAINDER,
                    help="Extra cfg key=value overrides (e.g. NUM_GPUS 0)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
from slowfast.config.defaults import get_cfg
cfg = get_cfg()
cfg.merge_from_file(args.cfg)
if args.opts:
    cfg.merge_from_list(args.opts)

cfg.NUM_GPUS   = 0
cfg.TRAIN.ENABLE = False

# ── Model ─────────────────────────────────────────────────────────────────────
from slowfast.models import build_model
model = build_model(cfg)
model.eval()
print(f"Model built:  {cfg.MODEL.MODEL_NAME}\n")

# ── Dummy inputs ──────────────────────────────────────────────────────────────
B  = 2
T  = cfg.DATA.NUM_FRAMES          # 8
S  = cfg.DATA.TRAIN_CROP_SIZE     # 256

video = torch.randn(B, 3, T, S, S)   # [B, C, T, H, W]
audio = torch.randn(B, 1, T, S, S)   # [B, 1, T, H, W]

# ── 1. Normal forward (must not break) ───────────────────────────────────────
with torch.no_grad():
    out_normal = model([video], audio)

assert out_normal.shape == torch.Size([B, 1, T, S // 4, S // 4]), \
    f"Normal forward shape mismatch: {out_normal.shape}"
print(f"[PASS] normal forward : {tuple(out_normal.shape)}")

# ── 2. return_feats forward ───────────────────────────────────────────────────
with torch.no_grad():
    out_feats = model([video], audio, return_feats=True)

assert isinstance(out_feats, dict), "return_feats should return a dict"

expected = {
    "heatmap":    (B, 1,     T,    S // 4, S // 4),  # (B,1,8,64,64)
    "vis_stage1": (B, 16384, 192),
    "vis_stage2": (B, 4096,  384),
    "vis_feat":   (B, 256,   768),
    "aud_feat":   (B, 256,   768),
    "fused_feat": (B, T // 2, S // 32, S // 32, 768),  # (B,4,8,8,768)
    "av_attn":    None,                               # checked separately below
}

print("\nreturn_feats shapes:")
all_ok = True

for key in ["heatmap", "vis_stage1", "vis_stage2",
            "vis_feat", "aud_feat", "fused_feat"]:
    tensor = out_feats[key]
    got    = tuple(tensor.shape)
    exp    = expected[key]
    ok     = got == exp
    all_ok = all_ok and ok
    mark   = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {key:<12}: expected {exp}  got {got}")

# av_attn may be None if spatial_audio_attn=True in config
av_attn = out_feats.get("av_attn")
if av_attn is None:
    print("  [INFO] av_attn     : None  (spatial_audio_attn branch — expected)")
else:
    N_aud_pooled = av_attn.shape[-1]   # 4 after audio_pool
    exp_attn = (B, 256, N_aud_pooled)
    ok = tuple(av_attn.shape) == exp_attn
    all_ok = all_ok and ok
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] av_attn      : expected {exp_attn}  got {tuple(av_attn.shape)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if all_ok:
    print("All checks passed — return_feats is working correctly.")
else:
    print("Some checks FAILED — see above.")
    sys.exit(1)
