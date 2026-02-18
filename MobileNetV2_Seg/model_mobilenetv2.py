"""
MobileNetV2-UNet for Brain Tumor Segmentation
==============================================
Pretrained MobileNetV2 encoder + lightweight UNet decoder.
Same I/O as DRUNetv2: [B, 3, 256, 256] → [B, 1, 256, 256] (logits).

~4.7 M params  (vs DRUNetv2's 33 M  → ~7× smaller)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """Upsample + concat skip + double conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2UNet(nn.Module):
    """
    Encoder : torchvision MobileNetV2 (pretrained on ImageNet)
    Decoder : lightweight UNet with skip connections

    For 256×256 input the encoder produces feature maps at:
        128×128  (16 ch)   ← features[0:2]
         64× 64  (24 ch)   ← features[2:4]
         32× 32  (32 ch)   ← features[4:7]
         16× 16  (96 ch)   ← features[7:14]
          8×  8  (320 ch)  ← features[14:18]   (bottleneck)
    """

    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()

        # --- Encoder (frozen-ready) ---
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights).features

        self.enc1 = backbone[0:2]    # → 128, 16 ch
        self.enc2 = backbone[2:4]    # →  64, 24 ch
        self.enc3 = backbone[4:7]    # →  32, 32 ch
        self.enc4 = backbone[7:14]   # →  16, 96 ch
        self.enc5 = backbone[14:18]  # →   8, 320 ch  (bottleneck)

        # If input has channels ≠ 3, swap the very first conv
        if in_channels != 3:
            old = self.enc1[0][0]  # Conv2dNormActivation → Conv2d
            self.enc1[0][0] = nn.Conv2d(
                in_channels, old.out_channels,
                old.kernel_size, old.stride, old.padding, bias=False,
            )

        # --- Decoder ---
        self.up5 = DecoderBlock(320, 96, 128)   #  8→16
        self.up4 = DecoderBlock(128, 32, 64)    # 16→32
        self.up3 = DecoderBlock(64, 24, 32)     # 32→64
        self.up2 = DecoderBlock(32, 16, 16)     # 64→128

        self.up1 = nn.Sequential(               # 128→256
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)   # 128×128, 16
        e2 = self.enc2(e1)  #  64× 64, 24
        e3 = self.enc3(e2)  #  32× 32, 32
        e4 = self.enc4(e3)  #  16× 16, 96
        e5 = self.enc5(e4)  #   8×  8, 320

        # --- Decoder ---
        d5 = self.up5(e5, e4)  # 16
        d4 = self.up4(d5, e3)  # 32
        d3 = self.up3(d4, e2)  # 64
        d2 = self.up2(d3, e1)  # 128
        d1 = self.up1(d2)      # 256

        return self.final(d1)  # raw logits


# ---- Quick test ----
if __name__ == "__main__":
    model = MobileNetV2UNet(in_channels=3, out_channels=1, pretrained=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")  # expect [2, 1, 256, 256]
