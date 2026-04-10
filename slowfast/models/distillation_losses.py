"""
Distillation losses for CSTS Teacher → Lightweight Student knowledge transfer.

Four progressive loss variants (use individually or combine):

  V1  OutputDistillationLoss  — match teacher's final gaze heatmap
  V2  FeatureDistillationLoss — match intermediate encoder embeddings
  V3  AttentionTransferLoss   — match teacher's AV cross-attention patterns
  V4  ProgressiveCRDLoss      — contrastive representation distillation (CRD)
                                + per-modality EgoNCE

Reference: CSTS_CrossModal_Distillation_Tutorial.ipynb — Section 5

Teacher feature dict keys (from custom_multimodal_builder.py return_feats=True):
  heatmap    : [B, 1, T, H/4, W/4]
  vis_stage1 : [B, 16384, 192]
  vis_stage2 : [B, 4096,  384]
  vis_feat   : [B, N_vis, 768]
  aud_feat   : [B, N_aud, 768]
  fused_feat : [B, T, 8, 8, 768]
  av_attn    : [B, N_vis, N_aud]

Student output dict keys (from student_model.py StudentGazeModel):
  heatmap      : [B, 1, T, 64, 64]
  sv_feat      : [B, 256]
  sa_feat      : [B, 256]
  sfused       : [B, 512]
  fusion_attn  : [B, 2, 2]
  spatial_feats: [B, 256, T, 16, 16]
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  V1  Output-Level Distillation
# ─────────────────────────────────────────────────────────────────────────────

class OutputDistillationLoss(nn.Module):
    """
    V1 — Output-level distillation.

    L = λ1·KLD(p_T ‖ p_S) + λ2·MSE(sigmoid(h_T), sigmoid(h_S)) + λ3·KLD(p_GT ‖ p_S)

    τ controls softness of teacher targets:
      τ=1 → exact match;  τ=4 → softer, more tolerance for student errors.

    Args:
        temperature : τ for spatial softmax  (default 4.0)
        lam1        : weight for teacher-KLD (default 1.0)
        lam2        : weight for MSE         (default 0.5)
        lam3        : weight for GT-KLD      (default 1.0)
    """

    def __init__(self, temperature: float = 4.0,
                 lam1: float = 1.0, lam2: float = 0.5, lam3: float = 1.0):
        super().__init__()
        self.tau              = temperature
        self.lam1, self.lam2, self.lam3 = lam1, lam2, lam3
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=False)

    @staticmethod
    def spatial_softmax(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Softmax over H×W per (B, C, T) slice."""
        B, C, T, H, W = x.shape
        return F.softmax(x.view(B, C, T, H * W) / tau, dim=-1).view(B, C, T, H, W)

    def forward(
        self,
        hm_student: torch.Tensor,  # [B, 1, T, H, W]  raw student logits
        hm_teacher: torch.Tensor,  # [B, 1, T, H, W]  teacher output (prob map)
        hm_gt     : torch.Tensor,  # [B,    T, H, W]  ground-truth heatmap
    ) -> Dict[str, torch.Tensor]:
        p_T = self.spatial_softmax(hm_teacher, self.tau)   # [B,1,T,H,W]
        p_S = self.spatial_softmax(hm_student, self.tau)

        B, _, T, H, W = p_T.shape

        # 1) KLD: teacher soft targets → student
        log_p_S  = torch.log(p_S.view(B, T, H * W) + 1e-10)
        kl_loss  = self.kl(log_p_S, p_T.view(B, T, H * W))

        # 2) MSE on sigmoid-normalised heatmaps
        mse_loss = F.mse_loss(torch.sigmoid(hm_student), torch.sigmoid(hm_teacher))

        # 3) KLD: GT heatmap → student
        p_GT = hm_gt.unsqueeze(1)                          # [B,1,T,H,W]
        p_GT = p_GT / (p_GT.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        gt_kl = self.kl(torch.log(p_S + 1e-10).view(B, T, H * W),
                        p_GT.view(B, T, H * W))

        total = self.lam1 * kl_loss + self.lam2 * mse_loss + self.lam3 * gt_kl
        return {"total": total, "kl": kl_loss, "mse": mse_loss, "gt_kl": gt_kl}


# ─────────────────────────────────────────────────────────────────────────────
#  V2  Intermediate Feature Distillation
# ─────────────────────────────────────────────────────────────────────────────

class FeatureDistillationLoss(nn.Module):
    """
    V2 — Intermediate feature distillation.

    Matches teacher token sequences to student pooled vectors via learnable
    projection adapters (only adapters are trained; teacher stays frozen).

    Adapters:
      proj_vis / proj_aud  : Linear(teacher_dim=768, proj_dim=256)
      proj_stu_v / _a      : Linear(student_dim=256, proj_dim=256)
      proj_fused           : Linear(student_dim*2=512, proj_dim=256)

    Args:
        teacher_dim : teacher token embedding dim  (default 768)
        student_dim : student pooled embedding dim (default 256)
        proj_dim    : shared projection space dim  (default 256)
    """

    def __init__(self, teacher_dim: int = 768,
                 student_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.proj_vis   = nn.Linear(teacher_dim,      proj_dim)
        self.proj_aud   = nn.Linear(teacher_dim,      proj_dim)
        self.proj_stu_v = nn.Linear(student_dim,      proj_dim)
        self.proj_stu_a = nn.Linear(student_dim,      proj_dim)
        self.proj_fused = nn.Linear(student_dim * 2,  proj_dim)

    def forward(
        self,
        t_vis  : torch.Tensor,  # [B, N_v, 768]  teacher visual tokens
        t_aud  : torch.Tensor,  # [B, N_a, 768]  teacher audio tokens
        s_vis  : torch.Tensor,  # [B, 256]        student video embedding
        s_aud  : torch.Tensor,  # [B, 256]        student audio embedding
        s_fused: torch.Tensor,  # [B, 512]        student fused embedding
    ) -> Dict[str, torch.Tensor]:
        # Visual alignment
        t_vis_proj = self.proj_vis(t_vis).mean(dim=1)    # [B, proj_dim]
        s_vis_proj = self.proj_stu_v(s_vis)
        loss_vis   = F.mse_loss(s_vis_proj, t_vis_proj.detach())

        # Audio alignment
        t_aud_proj = self.proj_aud(t_aud).mean(dim=1)    # [B, proj_dim]
        s_aud_proj = self.proj_stu_a(s_aud)
        loss_aud   = F.mse_loss(s_aud_proj, t_aud_proj.detach())

        # Cross-modal cosine alignment: teacher AV mean vs student fused
        t_av_mean    = (t_vis_proj + t_aud_proj) / 2.0   # [B, proj_dim]
        s_fused_proj = self.proj_fused(s_fused)
        cos_sim      = F.cosine_similarity(s_fused_proj, t_av_mean.detach(), dim=-1)
        loss_cos     = (1.0 - cos_sim).mean()

        total = loss_vis + loss_aud + 0.5 * loss_cos
        return {"total": total, "vis": loss_vis, "aud": loss_aud, "cosine": loss_cos}


# ─────────────────────────────────────────────────────────────────────────────
#  V3  Cross-Modal Attention Transfer
# ─────────────────────────────────────────────────────────────────────────────

class AttentionTransferLoss(nn.Module):
    """
    V3 — Cross-modal attention transfer.

    Forces the student's fusion attention to mimic the teacher's AV attention
    patterns (Zagoruyko & Komodakis, 2017 extended to cross-modal).

    Teacher: av_attn      [B, N_vis, N_aud]  — visual→audio attention weights
    Student: fusion_attn  [B, 2, 2]          — modality-level attention

    Two sub-losses:
      modal   : scalar video→audio attention score alignment
      spatial : ℓ₂-normalised spatial activation map alignment

    Args:
        p : power for attention map normalisation (default 2)
    """

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    @staticmethod
    def _attn_map(x: torch.Tensor, p: int = 2) -> torch.Tensor:
        """ℓ₂-normalised attention map, flattened to [B, -1]."""
        flat = x.abs().pow(p).view(x.shape[0], -1)
        return F.normalize(flat, p=2, dim=-1)

    def forward(
        self,
        t_av_attn     : torch.Tensor,  # [B, N_vis, N_aud]  teacher cross-attn
        s_fusion_attn : torch.Tensor,  # [B, 2, 2]          student modality attn
        s_spatial_feat: torch.Tensor,  # [B, C, T, H, W]    student spatial feats
    ) -> Dict[str, torch.Tensor]:
        # ── 1. Modality-level score alignment ────────────────────────────────
        t_va_score = t_av_attn.mean(dim=(-2, -1))          # [B]
        t_va_norm  = t_va_score / (t_va_score.abs().max() + 1e-8)

        s_va_score = s_fusion_attn[:, 0, 1]                # [B]  video→audio
        s_va_norm  = s_va_score / (s_va_score.abs().max() + 1e-8)

        loss_modal = F.mse_loss(s_va_norm, t_va_norm.detach())

        # ── 2. Spatial activation transfer ───────────────────────────────────
        # Teacher: row-sum gives importance of each visual token → [B, N_vis]
        t_row  = t_av_attn.sum(dim=-1)
        at_T   = self._attn_map(t_row, self.p)             # [B, N_vis]

        # Student: channel-mean spatial features → flatten → resize to N_vis
        s_spat = s_spatial_feat.mean(dim=1)                # [B, T, H, W]
        at_S   = self._attn_map(s_spat, self.p)            # [B, T*H*W]
        N_vis  = at_T.shape[-1]
        at_S_r = F.adaptive_avg_pool1d(at_S.unsqueeze(1), N_vis).squeeze(1)  # [B, N_vis]

        loss_spat = torch.norm(at_T.detach() - at_S_r, dim=-1).mean()

        total = loss_modal + 0.5 * loss_spat
        return {"total": total, "modal": loss_modal, "spatial": loss_spat}


# ─────────────────────────────────────────────────────────────────────────────
#  V4 helpers
# ─────────────────────────────────────────────────────────────────────────────

class _CRDLoss(nn.Module):
    """
    Contrastive Representation Distillation (Tian et al., 2020).

    InfoNCE over (teacher, student) embedding pairs — diagonal = positives,
    off-diagonal = negatives.  No memory bank required for batch-level CRD.

    Args:
        t_dim       : teacher embedding dimension
        s_dim       : student embedding dimension
        proj_dim    : shared projection dimension  (default 128)
        temperature : InfoNCE temperature τ        (default 0.07)
    """

    def __init__(self, t_dim: int, s_dim: int,
                 proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim,   proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.s_proj = nn.Sequential(
            nn.Linear(s_dim,   proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.tau = temperature

    def forward(self, z_teacher: torch.Tensor,
                z_student: torch.Tensor) -> torch.Tensor:
        z_T = F.normalize(self.t_proj(z_teacher), dim=-1)  # [B, proj_dim]
        z_S = F.normalize(self.s_proj(z_student), dim=-1)
        sim = torch.mm(z_S, z_T.T) / self.tau              # [B, B]
        B   = z_T.shape[0]
        lbl = torch.arange(B, device=z_T.device)
        return 0.5 * (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.T, lbl))


class _PerModalityEgoNCE(nn.Module):
    """
    Per-modality EgoNCE: align each student encoder independently to the
    corresponding teacher encoder.

    sv_feat ↔ vis_feat_T (pooled)
    sa_feat ↔ aud_feat_T (pooled)
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.vis_crd = _CRDLoss(t_dim=768, s_dim=256, proj_dim=128,
                                 temperature=temperature)
        self.aud_crd = _CRDLoss(t_dim=768, s_dim=256, proj_dim=128,
                                 temperature=temperature)

    def forward(
        self,
        t_vis: torch.Tensor,  # [B, N_v, 768]
        t_aud: torch.Tensor,  # [B, N_a, 768]
        s_vis: torch.Tensor,  # [B, 256]
        s_aud: torch.Tensor,  # [B, 256]
    ) -> Dict[str, torch.Tensor]:
        t_vis_p = t_vis.mean(dim=1)   # [B, 768]
        t_aud_p = t_aud.mean(dim=1)
        loss_vis = self.vis_crd(t_vis_p.detach(), s_vis)
        loss_aud = self.aud_crd(t_aud_p.detach(), s_aud)
        return {"total": loss_vis + loss_aud,
                "vis_nce": loss_vis, "aud_nce": loss_aud}


# ─────────────────────────────────────────────────────────────────────────────
#  V4  Progressive CRD
# ─────────────────────────────────────────────────────────────────────────────

class ProgressiveCRDLoss(nn.Module):
    """
    V4 — Progressive Multi-Modal CRD.

    L = CRD(fused_T, sfused_S)
      + EgoNCE(vis_T, sv_S)
      + EgoNCE(aud_T, sa_S)

    Maximises mutual information between teacher and student at three levels:
      - fused representation (holistic AV understanding)
      - visual encoder output
      - audio encoder output

    Args:
        fused_proj_dim  : projection dim for fused CRD  (default 256)
        modal_proj_dim  : projection dim for per-modal  (default 128)
        temperature     : InfoNCE τ                     (default 0.05)
    """

    def __init__(self, fused_proj_dim: int = 256,
                 modal_proj_dim: int = 128, temperature: float = 0.05):
        super().__init__()
        self.fused_crd = _CRDLoss(t_dim=768, s_dim=512,
                                   proj_dim=fused_proj_dim,
                                   temperature=temperature)
        self.per_modal = _PerModalityEgoNCE(temperature=temperature)

    def forward(
        self,
        t_feats: Dict[str, torch.Tensor],  # from CSTS  return_feats=True
        s_feats: Dict[str, torch.Tensor],  # from StudentGazeModel
    ) -> Dict[str, torch.Tensor]:
        # fused_feat: [B, T, 8, 8, 768] → mean pool → [B, 768]
        t_fused_pool = t_feats["fused_feat"].mean(dim=(1, 2, 3))
        loss_crd = self.fused_crd(t_fused_pool.detach(), s_feats["sfused"])

        modal = self.per_modal(
            t_feats["vis_feat"],   # [B, N_vis, 768]
            t_feats["aud_feat"],   # [B, N_aud, 768]
            s_feats["sv_feat"],    # [B, 256]
            s_feats["sa_feat"],    # [B, 256]
        )

        total = loss_crd + modal["total"]
        return {
            "total"  : total,
            "crd"    : loss_crd,
            "nce_vis": modal["vis_nce"],
            "nce_aud": modal["aud_nce"],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, H, W = 2, 8, 64, 64
    N_vis, N_aud = 256, 4

    # ── dummy teacher / student outputs ──────────────────────────────────────
    t_feats = {
        "heatmap"   : torch.randn(B, 1, T, H, W),
        "vis_feat"  : torch.randn(B, N_vis, 768),
        "aud_feat"  : torch.randn(B, N_aud, 768),
        "fused_feat": torch.randn(B, T // 2, 8, 8, 768),
        "av_attn"   : torch.rand(B, N_vis, N_aud),
    }
    s_feats = {
        "heatmap"      : torch.randn(B, 1, T, H, W),
        "sv_feat"      : torch.randn(B, 256),
        "sa_feat"      : torch.randn(B, 256),
        "sfused"       : torch.randn(B, 512),
        "fusion_attn"  : torch.rand(B, 2, 2),
        "spatial_feats": torch.randn(B, 256, T, 16, 16),
    }
    hm_gt = torch.rand(B, T, H, W)
    hm_gt = hm_gt / hm_gt.sum(dim=(-2, -1), keepdim=True)

    losses_to_test = {
        "V1 OutputDistillation ": OutputDistillationLoss(),
        "V2 FeatureDistillation": FeatureDistillationLoss(),
        "V3 AttentionTransfer  ": AttentionTransferLoss(),
        "V4 ProgressiveCRD     ": ProgressiveCRDLoss(),
    }

    print("Distillation loss sanity checks:\n")
    all_ok = True
    for name, loss_fn in losses_to_test.items():
        try:
            if "V1" in name:
                out = loss_fn(s_feats["heatmap"], t_feats["heatmap"], hm_gt)
            elif "V2" in name:
                out = loss_fn(t_feats["vis_feat"], t_feats["aud_feat"],
                              s_feats["sv_feat"], s_feats["sa_feat"], s_feats["sfused"])
            elif "V3" in name:
                out = loss_fn(t_feats["av_attn"], s_feats["fusion_attn"],
                              s_feats["spatial_feats"])
            else:
                out = loss_fn(t_feats, s_feats)

            total = out["total"].item()
            keys  = ", ".join(f"{k}={v.item():.4f}" for k, v in out.items())
            print(f"  [PASS]  {name}  total={total:.4f}  |  {keys}")
        except Exception as e:
            print(f"  [FAIL]  {name}  error: {e}")
            all_ok = False

    print()
    if all_ok:
        print("All loss checks passed.")
    else:
        import sys; sys.exit(1)
