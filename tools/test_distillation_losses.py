#!/usr/bin/env python3
"""
Isolated tests for all four distillation losses.

Run on Colab:
    cd /content/CSTS_ARia
    python tools/test_distillation_losses.py

Tests:
  1. OutputDistillationLoss  (V1) — output shapes, sub-loss keys, gradient flow
  2. FeatureDistillationLoss (V2) — output shapes, sub-loss keys, gradient flow
  3. AttentionTransferLoss   (V3) — output shapes, sub-loss keys, gradient flow
  4. ProgressiveCRDLoss      (V4) — output shapes, sub-loss keys, gradient flow
  5. Combined loss           — sum of all four, single backward pass
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from slowfast.models.distillation_losses import (
    OutputDistillationLoss,
    FeatureDistillationLoss,
    AttentionTransferLoss,
    ProgressiveCRDLoss,
)

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}\n")
else:
    print()

# ── shared dimensions (must match teacher / student model configs) ─────────────
B      = 2       # batch size
T      = 8       # temporal frames
H, W   = 64, 64  # heatmap spatial size
N_VIS  = 256     # teacher visual tokens
N_AUD  = 4       # teacher audio tokens (after pooling)
D_T    = 768     # teacher embedding dim
D_SV   = 256     # student video embed dim
D_SA   = 256     # student audio embed dim
D_SF   = 512     # student fused embed dim


# ─────────────────────────────────────────────────────────────────────────────
#  Shared random inputs (created once, reused across all tests)
# ─────────────────────────────────────────────────────────────────────────────

def make_inputs():
    """
    Build random teacher and student feature dicts with requires_grad=True
    on the student tensors so we can verify gradient flow.
    """
    # ── teacher features (frozen — no grad) ──────────────────────────────────
    t_feats = {
        "heatmap"   : torch.randn(B, 1, T, H, W,    device=DEVICE),
        "vis_feat"  : torch.randn(B, N_VIS, D_T,     device=DEVICE),
        "aud_feat"  : torch.randn(B, N_AUD, D_T,     device=DEVICE),
        "fused_feat": torch.randn(B, T//2, 8, 8, D_T,device=DEVICE),
        "av_attn"   : torch.rand( B, N_VIS, N_AUD,   device=DEVICE),  # [0,1]
    }

    # ── student features (requires_grad so we can test .backward()) ───────────
    s_feats = {
        "heatmap"      : torch.randn(B, 1, T, H, W,       device=DEVICE, requires_grad=True),
        "sv_feat"      : torch.randn(B, D_SV,              device=DEVICE, requires_grad=True),
        "sa_feat"      : torch.randn(B, D_SA,              device=DEVICE, requires_grad=True),
        "sfused"       : torch.randn(B, D_SF,              device=DEVICE, requires_grad=True),
        "fusion_attn"  : torch.rand( B, 2, 2,              device=DEVICE, requires_grad=True),
        "spatial_feats": torch.randn(B, 256, T, 16, 16,   device=DEVICE, requires_grad=True),
    }

    # ── ground-truth heatmap (normalised to sum=1) ────────────────────────────
    hm_gt = torch.rand(B, T, H, W, device=DEVICE)
    hm_gt = hm_gt / hm_gt.sum(dim=(-2, -1), keepdim=True)

    return t_feats, s_feats, hm_gt


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: check a single loss result dict
# ─────────────────────────────────────────────────────────────────────────────

def check_loss_dict(result: dict, required_keys: list, label: str) -> bool:
    ok = True

    # 1. All expected keys present
    missing = [k for k in required_keys if k not in result]
    if missing:
        print(f"    [FAIL] {label} — missing keys: {missing}")
        ok = False

    # 2. 'total' must exist and be a scalar
    if "total" not in result:
        print(f"    [FAIL] {label} — 'total' key missing")
        return False

    total = result["total"]
    if total.shape != torch.Size([]):
        print(f"    [FAIL] {label} — 'total' is not a scalar, shape={total.shape}")
        ok = False

    # 3. No NaN / Inf in any loss component
    for k, v in result.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"    [FAIL] {label} — NaN/Inf in '{k}'")
            ok = False

    # 4. All values finite and >= 0
    for k, v in result.items():
        if v.item() < 0:
            print(f"    [FAIL] {label} — negative loss in '{k}': {v.item():.6f}")
            ok = False

    return ok


def check_grad(loss_tensor: torch.Tensor, s_feats: dict,
               keys: list, label: str) -> bool:
    """Run backward and check that gradients exist for the given student keys."""
    loss_tensor.backward()
    ok = True
    for k in keys:
        t = s_feats[k]
        if t.grad is None:
            print(f"    [FAIL] {label} — no gradient on s_feats['{k}']")
            ok = False
        elif torch.isnan(t.grad).any():
            print(f"    [FAIL] {label} — NaN gradient on s_feats['{k}']")
            ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_v1_output_distillation():
    print("── V1  OutputDistillationLoss ──────────────────────────────────────")
    t_feats, s_feats, hm_gt = make_inputs()
    loss_fn = OutputDistillationLoss(temperature=4.0,
                                     lam1=1.0, lam2=0.5, lam3=1.0).to(DEVICE)

    result = loss_fn(s_feats["heatmap"], t_feats["heatmap"], hm_gt)

    required = ["total", "kl", "mse", "gt_kl"]
    ok = check_loss_dict(result, required, "V1")

    # Print values
    for k in required:
        print(f"    {k:<8} = {result[k].item():.6f}")

    # Gradient check
    grad_ok = check_grad(result["total"], s_feats, ["heatmap"], "V1")
    ok = ok and grad_ok

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] V1 OutputDistillationLoss\n")
    return ok


def test_v2_feature_distillation():
    print("── V2  FeatureDistillationLoss ─────────────────────────────────────")
    t_feats, s_feats, _ = make_inputs()
    loss_fn = FeatureDistillationLoss(teacher_dim=D_T,
                                      student_dim=D_SV,
                                      proj_dim=256).to(DEVICE)

    result = loss_fn(
        t_feats["vis_feat"],
        t_feats["aud_feat"],
        s_feats["sv_feat"],
        s_feats["sa_feat"],
        s_feats["sfused"],
    )

    required = ["total", "vis", "aud", "cosine"]
    ok = check_loss_dict(result, required, "V2")

    for k in required:
        print(f"    {k:<8} = {result[k].item():.6f}")

    grad_ok = check_grad(result["total"], s_feats,
                         ["sv_feat", "sa_feat", "sfused"], "V2")
    ok = ok and grad_ok

    # Also verify adapter parameters received gradients
    for name, param in loss_fn.named_parameters():
        if param.grad is None:
            print(f"    [FAIL] V2 — no gradient on adapter param '{name}'")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] V2 FeatureDistillationLoss\n")
    return ok


def test_v3_attention_transfer():
    print("── V3  AttentionTransferLoss ────────────────────────────────────────")
    t_feats, s_feats, _ = make_inputs()
    loss_fn = AttentionTransferLoss(p=2).to(DEVICE)

    result = loss_fn(
        t_feats["av_attn"],
        s_feats["fusion_attn"],
        s_feats["spatial_feats"],
    )

    required = ["total", "modal", "spatial"]
    ok = check_loss_dict(result, required, "V3")

    for k in required:
        print(f"    {k:<8} = {result[k].item():.6f}")

    grad_ok = check_grad(result["total"], s_feats,
                         ["fusion_attn", "spatial_feats"], "V3")
    ok = ok and grad_ok

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] V3 AttentionTransferLoss\n")
    return ok


def test_v4_progressive_crd():
    print("── V4  ProgressiveCRDLoss ──────────────────────────────────────────")
    t_feats, s_feats, _ = make_inputs()
    loss_fn = ProgressiveCRDLoss(fused_proj_dim=256,
                                  modal_proj_dim=128,
                                  temperature=0.05).to(DEVICE)

    result = loss_fn(t_feats, s_feats)

    required = ["total", "crd", "nce_vis", "nce_aud"]
    ok = check_loss_dict(result, required, "V4")

    for k in required:
        print(f"    {k:<10} = {result[k].item():.6f}")

    grad_ok = check_grad(result["total"], s_feats,
                         ["sv_feat", "sa_feat", "sfused"], "V4")
    ok = ok and grad_ok

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] V4 ProgressiveCRDLoss\n")
    return ok


def test_combined_loss():
    print("── Combined (V1 + V2 + V3 + V4) ────────────────────────────────────")
    t_feats, s_feats, hm_gt = make_inputs()

    v1 = OutputDistillationLoss().to(DEVICE)
    v2 = FeatureDistillationLoss().to(DEVICE)
    v3 = AttentionTransferLoss().to(DEVICE)
    v4 = ProgressiveCRDLoss().to(DEVICE)

    l1 = v1(s_feats["heatmap"], t_feats["heatmap"], hm_gt)["total"]
    l2 = v2(t_feats["vis_feat"], t_feats["aud_feat"],
            s_feats["sv_feat"], s_feats["sa_feat"],
            s_feats["sfused"])["total"]
    l3 = v3(t_feats["av_attn"], s_feats["fusion_attn"],
            s_feats["spatial_feats"])["total"]
    l4 = v4(t_feats, s_feats)["total"]

    total = l1 + l2 + l3 + l4

    print(f"    V1={l1.item():.4f}  V2={l2.item():.4f}  "
          f"V3={l3.item():.4f}  V4={l4.item():.4f}")
    print(f"    Combined total = {total.item():.4f}")

    ok = not (torch.isnan(total) or torch.isinf(total))
    if not ok:
        print("    [FAIL] Combined total is NaN / Inf")
        return False

    # Single backward pass over combined loss
    total.backward()

    grad_keys = ["heatmap", "sv_feat", "sa_feat", "sfused",
                 "fusion_attn", "spatial_feats"]
    for k in grad_keys:
        t = s_feats[k]
        if t.grad is None:
            print(f"    [FAIL] Combined — no gradient on s_feats['{k}']")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Combined loss + single backward\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter count
# ─────────────────────────────────────────────────────────────────────────────

def print_param_counts():
    print("── Trainable adapter parameters ────────────────────────────────────")
    modules = {
        "OutputDistillationLoss (no adapters)": OutputDistillationLoss(),
        "FeatureDistillationLoss             ": FeatureDistillationLoss(),
        "AttentionTransferLoss  (no adapters)": AttentionTransferLoss(),
        "ProgressiveCRDLoss                  ": ProgressiveCRDLoss(),
    }
    for name, m in modules.items():
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"    {name}: {n:,} params")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  Distillation Loss Tests")
    print(f"  Batch={B}  T={T}  H×W={H}×{W}  "
          f"N_vis={N_VIS}  N_aud={N_AUD}  D_T={D_T}")
    print("=" * 68 + "\n")

    print_param_counts()

    results = {
        "V1": test_v1_output_distillation(),
        "V2": test_v2_feature_distillation(),
        "V3": test_v3_attention_transfer(),
        "V4": test_v4_progressive_crd(),
        "Combined": test_combined_loss(),
    }

    print("=" * 68)
    print("  Summary")
    print("=" * 68)
    all_ok = True
    for name, ok in results.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {name}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED — see above.")
        sys.exit(1)
