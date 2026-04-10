"""
Lightweight Student Model for Egocentric Gaze Prediction (AR/VR deployment).

Architecture summary:
  Component       | Technique                          | Reason
  ----------------|------------------------------------|----------------------------------
  Video encoder   | Depthwise-Separable Conv3D + TSM   | 8-9x fewer multiply-adds
  Temporal model  | Temporal Shift Module (TSM)        | Zero extra parameters
  Audio encoder   | 4-layer Conv2D stack               | Spectrogram is 2-D
  Fusion          | 1-head cross-attention (linear)    | O(N) vs O(N^2) full attention
  Decoder         | Four ConvTranspose3D blocks        | Same structure as teacher

Input shapes:
  video         : [B, 3, T, H, W]           e.g. [B, 3, 8, 256, 256]
  audio_student : [B, 1, F_stu, L_stu]      e.g. [B, 1, 64, 128]

Output (dict):
  heatmap      : [B, 1, T, 64, 64]
  sv_feat      : [B, 256]
  sa_feat      : [B, 256]
  sfused       : [B, 512]
  fusion_attn  : [B, 2, 2]   (modality-to-modality attention weights)

Reference: CSTS_CrossModal_Distillation_Tutorial.ipynb — Section 4
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  4.1  Temporal Shift Module (TSM)
# ─────────────────────────────────────────────────────────────────────────────

class TSM(nn.Module):
    """
    Temporal Shift Module (Lin et al., 2019).

    Shifts a fraction of channels along the temporal axis — zero extra
    parameters, zero extra FLOPs.

    Input / Output: [B, C, T, H, W]
    """

    def __init__(self, n_div: int = 8):
        super().__init__()
        self.n_div = n_div  # shift 1/n_div of channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        fold = C // self.n_div
        out = x.clone()
        # Forward shift: channels [0:fold] <- x shifted one frame forward
        out[:, :fold,         1:, :, :] = x[:, :fold,         :-1, :, :]
        out[:, :fold,          0, :, :] = 0
        # Backward shift: channels [fold:2*fold] <- x shifted one frame back
        out[:, fold:2*fold, :-1, :, :] = x[:, fold:2*fold, 1:, :, :]
        out[:, fold:2*fold,  -1, :, :] = 0
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  4.1  Depthwise-Separable Conv3D (core building block)
# ─────────────────────────────────────────────────────────────────────────────

class DSConv3d(nn.Module):
    """
    Depthwise-Separable 3-D Convolution.

    Standard Conv3D : in_c * out_c * kT * kH * kW  params
    DS-Conv3D       : in_c * kT * kH * kW  +  in_c * out_c  params
    Speedup approx  : 1 / (1/out_c + 1/(kT*kH*kW))

    Used throughout the video encoder to reduce FLOPs for AR/VR deployment.
    """

    def __init__(self, in_c: int, out_c: int, stride: Tuple = (1, 1, 1)):
        super().__init__()
        self.dw = nn.Sequential(
            # Depthwise: one filter per input channel
            nn.Conv3d(in_c, in_c, kernel_size=3, stride=stride,
                      padding=1, groups=in_c, bias=False),
            nn.BatchNorm3d(in_c),
            nn.Hardswish(inplace=True),
        )
        self.pw = nn.Sequential(
            # Pointwise: 1x1x1 conv to mix channels
            nn.Conv3d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ─────────────────────────────────────────────────────────────────────────────
#  4.1  Lightweight Video Encoder (MobileNet3D style)
# ─────────────────────────────────────────────────────────────────────────────

class LightVideoEncoder(nn.Module):
    """
    Lightweight video encoder for AR/VR gaze prediction.

    Architecture:
      Stem    : Conv3D  3  -> 32,  stride (1,2,2)  -> [B, 32,  T, 128, 128]
      Stage-1 : TSM + DS-Conv3D  32 ->  64, stride (1,2,2) -> [B,  64, T,  64,  64]
      Stage-2 : TSM + DS-Conv3D  64 -> 128, stride (1,2,2) -> [B, 128, T,  32,  32]
      Stage-3 : TSM + DS-Conv3D 128 -> 256, stride (1,2,2) -> [B, 256, T,  16,  16]
      GlobalPool + Flatten -> sv_feat [B, 256]

    Input : video [B, 3, T, H, W]   (H=W=256)
    Output: sv_feat [B, 256],  spatial_feats [B, 256, T, 16, 16]
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.Hardswish(inplace=True),
        )
        self.tsm1   = TSM()
        self.stage1 = DSConv3d(32,  64,  stride=(1, 2, 2))
        self.tsm2   = TSM()
        self.stage2 = DSConv3d(64,  128, stride=(1, 2, 2))
        self.tsm3   = TSM()
        self.stage3 = DSConv3d(128, 256, stride=(1, 2, 2))
        self.pool   = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)              # [B,  32, T, 128, 128]
        x = self.stage1(self.tsm1(x)) # [B,  64, T,  64,  64]
        x = self.stage2(self.tsm2(x)) # [B, 128, T,  32,  32]
        x = self.stage3(self.tsm3(x)) # [B, 256, T,  16,  16]
        spatial_feats = x             # kept for attention transfer distillation
        sv_feat = self.pool(x).flatten(1)  # [B, 256]
        return sv_feat, spatial_feats


# ─────────────────────────────────────────────────────────────────────────────
#  4.2  Lightweight Audio Encoder
# ─────────────────────────────────────────────────────────────────────────────

class LightAudioEncoder(nn.Module):
    """
    4-layer Conv2D spectrogram encoder.

    Input : [B, 1, F_stu=64, L_stu=128]  -- half-res STFT
    Output: sa_feat [B, 256]

    Channel progression: 1 -> 32 -> 64 -> 128 -> 256
    Each layer: Conv2D -> BN -> GELU -> MaxPool(2,2)
    """

    def __init__(self):
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                nn.MaxPool2d(2),
            )

        self.conv_stack = nn.Sequential(
            block(1,   32),   # -> [B,  32, 32,  64]
            block(32,  64),   # -> [B,  64, 16,  32]
            block(64,  128),  # -> [B, 128,  8,  16]
            block(128, 256),  # -> [B, 256,  4,   8]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)         # [B, 256, 4, 8]
        sa_feat = self.pool(x).flatten(1)  # [B, 256]
        return sa_feat


# ─────────────────────────────────────────────────────────────────────────────
#  4.4  Cross-Modal Fusion (1-head attention, video + audio only)
# ─────────────────────────────────────────────────────────────────────────────

class LightFusion(nn.Module):
    """
    Fuses [sv_feat, sa_feat] via a single cross-attention layer,
    treating each modality as one token, then projects to 512-d.

    sv_feat [B, 256] -+
                      +--> stack -> [B, 2, 256] -> 1-head self-attn
    sa_feat [B, 256] -+                          -> flatten -> Linear -> sfused [B, 512]

    Returns sfused [B, 512] and attention weights [B, 2, 2] for V3 distillation.
    """

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.fc   = nn.Linear(2 * 256, 512)
        self.act  = nn.GELU()

    def forward(
        self,
        sv: torch.Tensor,  # [B, 256]
        sa: torch.Tensor,  # [B, 256]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([sv, sa], dim=1)                 # [B, 2, 256]
        attn_out, attn_w = self.attn(tokens, tokens, tokens)  # [B,2,256], [B,2,2]
        tokens = self.norm(tokens + attn_out)                 # residual + norm
        sfused = self.act(self.fc(tokens.flatten(1)))         # [B, 512]
        return sfused, attn_w  # attn_w returned for V3 distillation loss


# ─────────────────────────────────────────────────────────────────────────────
#  4.4  Gaze Head — ConvTranspose3D decoder
# ─────────────────────────────────────────────────────────────────────────────

class GazeHead(nn.Module):
    """
    ConvTranspose3D decoder: maps 512-d fused embedding -> gaze heatmap.

    Steps:
      1. Linear 512 -> 512*4*4*4   (spatial seed)
      2. Reshape  -> [B, 512, 4, 4, 4]
      3. up1: 512->256, spatial x2  -> [B, 256, 4,  8,  8]
      4. up2: 256->128, spatial x2  -> [B, 128, 4, 16, 16]
      5. up3: 128-> 64, spatial x2  -> [B,  64, 4, 32, 32]
      6. up4:  64-> 32, spatial x2, temporal x2 -> [B, 32, 8, 64, 64]
      7. 1x1 Conv -> heatmap [B, 1, 8, 64, 64]

    Input : sfused [B, 512]
    Output: heatmap [B, 1, T=8, 64, 64]
    """

    SEED_T, SEED_H, SEED_W = 4, 4, 4
    SEED_C = 512

    def __init__(self):
        super().__init__()
        S = self.SEED_C * self.SEED_T * self.SEED_H * self.SEED_W
        self.expand = nn.Sequential(
            nn.Linear(512, S),
            nn.GELU(),
        )

        def up_block(in_c: int, out_c: int,
                     s_stride: int, t_stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose3d(
                    in_c, out_c, kernel_size=3,
                    stride=(t_stride, s_stride, s_stride),
                    padding=1,
                    output_padding=(t_stride - 1, s_stride - 1, s_stride - 1),
                ),
                nn.BatchNorm3d(out_c),
                nn.GELU(),
            )

        self.up1  = up_block(512, 256, s_stride=2)              # -> [B, 256, 4,  8,  8]
        self.up2  = up_block(256, 128, s_stride=2)              # -> [B, 128, 4, 16, 16]
        self.up3  = up_block(128,  64, s_stride=2)              # -> [B,  64, 4, 32, 32]
        self.up4  = up_block( 64,  32, s_stride=2, t_stride=2)  # -> [B,  32, 8, 64, 64]
        self.head = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, sfused: torch.Tensor) -> torch.Tensor:
        x = self.expand(sfused)  # [B, 512*4*4*4]
        x = x.view(-1, self.SEED_C, self.SEED_T, self.SEED_H, self.SEED_W)
        x = self.up1(x)   # [B, 256, 4,  8,  8]
        x = self.up2(x)   # [B, 128, 4, 16, 16]
        x = self.up3(x)   # [B,  64, 4, 32, 32]
        x = self.up4(x)   # [B,  32, 8, 64, 64]
        return self.head(x)  # [B,   1, 8, 64, 64]


# ─────────────────────────────────────────────────────────────────────────────
#  4.5  Full Student Model (Video + Audio, no IMU)
# ─────────────────────────────────────────────────────────────────────────────

class StudentGazeModel(nn.Module):
    """
    AR/VR-deployable student for egocentric gaze prediction.

    Inputs:
      video         [B, 3, T, H, W]       e.g. [B, 3, 8, 256, 256]
      audio_student [B, 1, F_stu, L_stu]  e.g. [B, 1, 64, 128]

    Outputs (dict):
      'heatmap'      [B, 1, T, 64, 64]   — raw logits (apply sigmoid for prob)
      'sv_feat'      [B, 256]             — video embedding
      'sa_feat'      [B, 256]             — audio embedding
      'sfused'       [B, 512]             — fused AV embedding
      'fusion_attn'  [B, 2, 2]           — modality attention (for V3 distillation)
      'spatial_feats'[B, 256, T, 16, 16] — spatial feature maps (for attn transfer)
    """

    def __init__(self):
        super().__init__()
        self.video_enc = LightVideoEncoder()
        self.audio_enc = LightAudioEncoder()
        self.fusion    = LightFusion()
        self.gaze_head = GazeHead()

    def forward(
        self,
        video         : torch.Tensor,  # [B, 3, T, H, W]
        audio_student : torch.Tensor,  # [B, 1, F_stu, L_stu]
    ) -> Dict[str, torch.Tensor]:
        sv_feat, spatial_feats = self.video_enc(video)
        sa_feat                = self.audio_enc(audio_student)
        sfused, fusion_attn    = self.fusion(sv_feat, sa_feat)
        heatmap                = self.gaze_head(sfused)
        return {
            "heatmap"      : heatmap,        # [B, 1, T, 64, 64]
            "sv_feat"      : sv_feat,        # [B, 256]
            "sa_feat"      : sa_feat,        # [B, 256]
            "sfused"       : sfused,         # [B, 512]
            "fusion_attn"  : fusion_attn,    # [B, 2, 2]
            "spatial_feats": spatial_feats,  # [B, 256, T, 16, 16]
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, H, W = 2, 8, 256, 256
    F_stu, L_stu = 64, 128

    video = torch.randn(B, 3, T, H, W)
    audio = torch.randn(B, 1, F_stu, L_stu)

    model = StudentGazeModel()
    model.eval()

    with torch.no_grad():
        out = model(video, audio)

    expected = {
        "heatmap"      : (B, 1,   T,  64, 64),
        "sv_feat"      : (B, 256),
        "sa_feat"      : (B, 256),
        "sfused"       : (B, 512),
        "fusion_attn"  : (B, 2, 2),
        "spatial_feats": (B, 256, T, 16, 16),
    }

    print("StudentGazeModel output shapes:")
    all_ok = True
    for key, exp in expected.items():
        got = tuple(out[key].shape)
        ok  = got == exp
        all_ok = all_ok and ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {key:<14}: expected {exp}  got {got}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal parameters : {n_params:.2f} M")

    if all_ok:
        print("All shape checks passed.")
    else:
        import sys; sys.exit(1)
