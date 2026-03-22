from __future__ import annotations
 
import torch
import torch.nn as nn
 
 
class DoubleConv(nn.Module):
    """(Conv3×3 → BN → ReLU) × 2"""
 
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)
 
 
class Down(nn.Module):
    """MaxPool → DoubleConv"""
 
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
 
    def forward(self, x):
        return self.block(x)
 
 
class Up(nn.Module):
    """ConvTranspose (×2) → concat skip → DoubleConv"""
 
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
 
    def forward(self, x, skip):
        x = self.up(x)
        # Si les tailles diffèrent (arrondi après pool), on crop le skip
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            skip = skip[
                :,
                :,
                dy // 2 : dy // 2 + x.size(2),
                dx // 2 : dx // 2 + x.size(3),
            ]
        return self.conv(torch.cat([skip, x], dim=1))
 
 
class UNet(nn.Module):
    """
    U-Net classique pour segmentation binaire.
 
    Architecture encoder-decoder avec skip connections.
 
    Parameters
    ----------
    in_channels : int
        Nombre de bandes en entrée (5 pour B, G, R, NIR, SWIR16).
    out_channels : int
        Nombre de classes en sortie (1 pour segmentation binaire).
    features : list of int
        Nombre de filtres à chaque niveau de l'encoder.
        Le bottleneck a features[-1] * 2 filtres.
    dropout : float
        Taux de dropout dans le bottleneck.
 
    La sortie est en **logits** (pas de sigmoid). C'est la loss
    (BCEWithLogitsLoss / DiceBCELoss) qui gère l'activation.
    """
 
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        features: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        f = features
 
        # Encoder
        self.enc0 = DoubleConv(in_channels, f[0])
        self.enc1 = Down(f[0], f[1])
        self.enc2 = Down(f[1], f[2])
        self.enc3 = Down(f[2], f[3])
 
        # Bottleneck
        self.bottleneck = nn.Sequential(
            Down(f[3], f[3] * 2),
            nn.Dropout2d(dropout),
        )
 
        # Decoder
        self.dec3 = Up(f[3] * 2, f[3])
        self.dec2 = Up(f[3], f[2])
        self.dec1 = Up(f[2], f[1])
        self.dec0 = Up(f[1], f[0])
 
        # Tête de segmentation
        self.head = nn.Conv2d(f[0], out_channels, kernel_size=1)
 
    def forward(self, x):
        # Encoder
        s0 = self.enc0(x)  # (B, f[0], H,    W)
        s1 = self.enc1(s0)  # (B, f[1], H/2,  W/2)
        s2 = self.enc2(s1)  # (B, f[2], H/4,  W/4)
        s3 = self.enc3(s2)  # (B, f[3], H/8,  W/8)
 
        b = self.bottleneck(s3)  # (B, f[3]*2, H/16, W/16)
 
        # Decoder + skip connections
        d3 = self.dec3(b, s3)  # (B, f[3], H/8,  W/8)
        d2 = self.dec2(d3, s2)  # (B, f[2], H/4,  W/4)
        d1 = self.dec1(d2, s1)  # (B, f[1], H/2,  W/2)
        d0 = self.dec0(d1, s0)  # (B, f[0], H,    W)
 
        return self.head(d0)  # (B, out_ch, H, W)  logits