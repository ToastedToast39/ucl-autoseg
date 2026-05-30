"""
Models for the UCL autosegmentation pipeline.

Two models, trained separately per Oscar's explicit guidance:

  UNet          — multi-class segmentation
                  Identical API to tendon/model.py UNet so train_seg.py
                  is a near-copy of scripts/train.py.
                  out_ch=4: bg + ucl + bone_humerus + bone_ulna

  HeatmapUNet   — landmark localization via heatmap regression
                  Same encoder-decoder backbone, final layer outputs N sigmoid
                  heatmaps (one per landmark). Peak of each = predicted (x,y).

dice_ce_loss and dice_bce_loss carried verbatim from tendon/model.py.
heatmap_mse_loss is new (no equivalent in the patellar reference).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks — identical to tendon/model.py
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class _Encoder(nn.Module):
    def __init__(self, in_ch, base):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base,   base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(base*8, base*16)
    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        return self.bott(self.pool(c4)), c1, c2, c3, c4


class _Decoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.u4  = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.u3  = DoubleConv(base*8,  base*4)
        self.up2 = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.u2  = DoubleConv(base*4,  base*2)
        self.up1 = nn.ConvTranspose2d(base*2,  base,   2, stride=2)
        self.u1  = DoubleConv(base*2,  base)
    def forward(self, b, c1, c2, c3, c4):
        x = self.u4(torch.cat([self.up4(b), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return x


# ---------------------------------------------------------------------------
# Model A: Segmentation U-Net  (drop-in replacement for tendon/model.py UNet)
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Multi-class segmentation.

    out_ch includes background (class 0):
        2 = bg + ucl                              (simplest start)
        4 = bg + ucl + bone_humerus + bone_ulna   (full UCL task)

    API identical to tendon/model.py UNet.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 4, base: int = 32):
        super().__init__()
        self.enc = _Encoder(in_ch, base)
        self.dec = _Decoder(base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        b, c1, c2, c3, c4 = self.enc(x)
        return self.out(self.dec(b, c1, c2, c3, c4))


def dice_ce_loss(logits, target, num_classes, eps=1e-6, ignore_bg_dice=True):
    """Combined multi-class Dice + cross-entropy.  Verbatim from tendon/model.py."""
    target = target.long()
    ce = F.cross_entropy(logits, target)
    probs  = torch.softmax(logits, 1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    start = 1 if (ignore_bg_dice and num_classes > 1) else 0
    p, t = probs[:, start:], onehot[:, start:]
    num  = 2*(p*t).sum(dim=(0,2,3)) + eps
    den  = p.sum(dim=(0,2,3)) + t.sum(dim=(0,2,3)) + eps
    return ce + 1 - (num/den).mean()


def dice_bce_loss(logits, target, eps=1e-6):
    """Backwards-compatible binary shim. Verbatim from tendon/model.py."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    num = 2*(probs*target).sum(dim=(1,2,3)) + eps
    den = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return bce + 1 - (num/den).mean()


# ---------------------------------------------------------------------------
# Model B: Landmark heatmap regression  (new — no patellar equivalent)
# ---------------------------------------------------------------------------

class HeatmapUNet(nn.Module):
    """Predicts one Gaussian heatmap per named landmark.

    Same backbone as UNet; final layer is sigmoid-activated (N outputs).
    Peak (x,y) of each heatmap = predicted landmark location.

    Default 4 landmarks for UCL task:
        0: ucl_humeral   — proximal attachment, medial epicondyle
        1: ucl_ulnar     — distal attachment, sublime tubercle
        2: gap_humerus   — medial joint line, humeral side
        3: gap_ulna      — medial joint line, ulnar side
    """
    def __init__(self, in_ch: int = 1, num_landmarks: int = 4, base: int = 32):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.enc = _Encoder(in_ch, base)
        self.dec = _Decoder(base)
        self.out = nn.Conv2d(base, num_landmarks, 1)
    def forward(self, x):
        b, c1, c2, c3, c4 = self.enc(x)
        return torch.sigmoid(self.out(self.dec(b, c1, c2, c3, c4)))


def make_gaussian_heatmaps(landmarks_xy, H, W, sigma=8.0):
    """Build (N,H,W) float32 Gaussian heatmap targets.

    landmarks_xy: list of (x,y) tuples or None if landmark absent.
    """
    N = len(landmarks_xy)
    hm = torch.zeros(N, H, W)
    ys = torch.arange(H).float().view(-1, 1)
    xs = torch.arange(W).float().view(1, -1)
    for i, pt in enumerate(landmarks_xy):
        if pt is None: continue
        cx, cy = float(pt[0]), float(pt[1])
        hm[i] = torch.exp(-((xs-cx)**2 + (ys-cy)**2) / (2*sigma**2))
    return hm


def heatmap_to_coords(heatmaps):
    """Decode (N,H,W) heatmaps → list of (x,y) or None (if max < 0.1)."""
    N, H, W = heatmaps.shape
    coords = []
    for i in range(N):
        h = heatmaps[i]
        if h.max().item() < 0.1:
            coords.append(None); continue
        idx = int(h.view(-1).argmax())
        coords.append((float(idx % W), float(idx // W)))
    return coords


def heatmap_mse_loss(pred, target, visible_mask=None):
    """MSE over heatmaps; visible_mask (B,N) bool masks absent landmarks."""
    loss = F.mse_loss(pred, target, reduction="none")
    if visible_mask is not None:
        m = visible_mask.float()[:, :, None, None]
        loss = loss * m
        return loss.sum() / m.sum().clamp(min=1.0)
    return loss.mean()
