"""
Data loading for the UCL pipeline.

Supports two mask formats — the pipeline accepts whichever is present:

  NIfTI masks (.nii.gz)  — saved directly from 3D Slicer Segment Editor
                           This is now the PRIMARY format. Label in Slicer,
                           save with the module, train directly. No conversion.

  PNG masks (.png)       — legacy indexed PNGs from Labelme workflow.
                           Still supported so old labels aren't lost.

UCLSegDataset checks for .nii.gz first, falls back to .png.
Everything else (augmentation, padding, U-Net input format) is unchanged.
"""
from __future__ import annotations
import json, os
from glob import glob
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    Dataset = object

try:
    import nibabel as nib
    _HAS_NIBABEL = True
except Exception:
    _HAS_NIBABEL = False


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

LANDMARK_NAMES: list[str] = [
    "ucl_humeral",
    "ucl_ulnar",
]

SEG_CLASS_MAP: dict[str, int] = {
    # Phase 1: bone surfaces first
    "humerus": 1,
    "ulna":    2,
}
NUM_SEG_CLASSES = 3   # 0=bg, 1=humerus, 2=ulna


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------

def pad_to_multiple(arr: np.ndarray, mult: int = 16):
    h, w = arr.shape[:2]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if arr.ndim == 2:
        out = np.pad(arr, ((0, ph), (0, pw)), mode="reflect")
    else:
        out = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return out, (ph, pw)


def unpad(arr: np.ndarray, pad: tuple[int, int]):
    ph, pw = pad
    return arr[:arr.shape[0]-ph, :arr.shape[1]-pw]


# ---------------------------------------------------------------------------
# Mask loading — NIfTI (Slicer) or PNG (Labelme)
# ---------------------------------------------------------------------------

def load_mask(stem: str, mask_dir: Path, num_classes: int) -> Optional[np.ndarray]:
    """Load a segmentation mask for a given image stem.

    Checks for NIfTI first (from Slicer), then PNG (from Labelme).
    Returns (H, W) int64 array with class IDs, or None if not found.
    """
    # NIfTI from Slicer Segment Editor
    nii_path = mask_dir / (stem + ".nii.gz")
    if nii_path.exists() and _HAS_NIBABEL:
        data = nib.load(str(nii_path)).get_fdata()
        # Slicer saves (X, Y, Z) — squeeze Z and transpose to (H, W)
        if data.ndim == 3:
            data = data[:, :, 0]
        m = np.array(data, dtype=np.int64).T   # transpose X,Y → row,col
        if m.max() > num_classes - 1:
            m = np.clip(m, 0, num_classes - 1)
        return m

    # PNG from Labelme (legacy)
    png_path = mask_dir / (stem + ".png")
    if png_path.exists():
        m = np.asarray(Image.open(png_path).convert("L"))
        if m.max() > num_classes - 1:
            m = (m > 127).astype(np.int64)
        else:
            m = m.astype(np.int64)
        return m

    return None


def mask_exists(stem: str, mask_dir: Path) -> bool:
    """Check if any mask format exists for this image stem."""
    return (mask_dir / (stem + ".nii.gz")).exists() or \
           (mask_dir / (stem + ".png")).exists()


# ---------------------------------------------------------------------------
# Dataset A: Segmentation
# ---------------------------------------------------------------------------

class UCLSegDataset(Dataset):
    """Image + mask pairs for UCL segmentation training.

    Accepts NIfTI masks (from Slicer) and PNG masks (from Labelme).
    NIfTI takes priority when both exist.

    Layout:
        root/images/   *.dcm  OR  *.png  OR  *.jpg
        root/masks/    *.nii.gz  (from Slicer)  OR  *.png  (from Labelme)
    """
    def __init__(self, root: str, augment: bool = False,
                 resize: Optional[tuple] = None,
                 num_classes: int = NUM_SEG_CLASSES):
        if not _HAS_TORCH: raise RuntimeError("PyTorch required")
        self.img_dir     = Path(root) / "images"
        self.msk_dir     = Path(root) / "masks"
        self.augment     = augment
        self.resize      = resize
        self.num_classes = num_classes

        # accept DICOMs, PNGs, or JPGs as source images
        all_imgs = (sorted(self.img_dir.glob("*.dcm")) +
                    sorted(self.img_dir.glob("*.png")) +
                    sorted(self.img_dir.glob("*.jpg")))
        self.files = sorted(
            p.name for p in all_imgs
            if mask_exists(p.stem, self.msk_dir)
        )

    def __len__(self): return len(self.files)

    def _load_image(self, name: str) -> np.ndarray:
        """Load image as grayscale float32 [0,1] for model input.
        Source can be RGB DICOM or PNG — always converted to grayscale for training.
        """
        p = self.img_dir / name
        if p.suffix.lower() == ".dcm":
            import pydicom
            ds  = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array
            while arr.ndim > 3:
                arr = arr[0] if arr.shape[0]==1 else arr.reshape(arr.shape[-3],arr.shape[-2],arr.shape[-1])
            # RGB → grayscale
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = (0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]).astype(np.uint8)
            elif arr.ndim == 3:
                arr = arr[:,:,0]
            img = arr.astype(np.float32) / 255.0
        else:
            # PNG may be RGB — convert to grayscale for model
            img_pil = Image.open(p).convert("L")
            img = np.asarray(img_pil, np.float32) / 255.0
        if self.resize:
            H, W = self.resize
            img_pil = Image.fromarray((img*255).astype(np.uint8)).resize((W,H), Image.BILINEAR)
            img = np.asarray(img_pil, np.float32) / 255.0
        return img

    def _augment(self, img, msk):
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
        if np.random.rand() < 0.5:
            img = np.clip(img * np.random.uniform(0.8,1.2) + np.random.uniform(-0.05,0.05), 0, 1)
        if np.random.rand() < 0.3:
            img = np.clip(img + np.random.randn(*img.shape) * 0.02, 0, 1)
        return img, msk

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = self._load_image(name)
        msk  = load_mask(Path(name).stem, self.msk_dir, self.num_classes)
        if msk is None:
            msk = np.zeros(img.shape[:2], dtype=np.int64)
        # resize mask if needed
        if self.resize and msk.shape != tuple(self.resize):
            H, W = self.resize
            msk = np.asarray(
                Image.fromarray(msk.astype(np.uint8)).resize((W,H), Image.NEAREST),
                np.int64)
        if self.augment: img, msk = self._augment(img, msk)
        img, _ = pad_to_multiple(img.astype(np.float32), 16)
        msk, _ = pad_to_multiple(msk.astype(np.int64),   16)
        return torch.from_numpy(img)[None], torch.from_numpy(msk).long()


# ---------------------------------------------------------------------------
# Dataset B: Landmark heatmap regression (unchanged)
# ---------------------------------------------------------------------------

class UCLLandmarkDataset(Dataset):
    def __init__(self, root: str, augment: bool = False,
                 resize: Optional[tuple] = None,
                 landmark_names: list[str] = LANDMARK_NAMES,
                 sigma: float = 8.0):
        if not _HAS_TORCH: raise RuntimeError("PyTorch required")
        self.img_dir        = Path(root) / "images"
        self.pts_dir        = Path(root) / "points"
        all_imgs = sorted(self.img_dir.glob("*.png")) + sorted(self.img_dir.glob("*.jpg"))
        self.files = sorted(p.name for p in all_imgs if (self.pts_dir/(p.stem+".json")).exists())
        self.augment        = augment
        self.resize         = resize
        self.landmark_names = landmark_names
        self.sigma          = sigma

    def __len__(self): return len(self.files)

    def _load(self, name):
        img = Image.open(self.img_dir/name).convert("L")
        W0, H0 = img.size
        with open(self.pts_dir/(Path(name).stem+".json")) as f:
            pts = json.load(f)
        if self.resize:
            H, W = self.resize; sx, sy = W/W0, H/H0
            img = img.resize((W,H), Image.BILINEAR)
        else:
            H, W, sx, sy = H0, W0, 1.0, 1.0
        img_arr   = np.asarray(img, np.float32) / 255.0
        landmarks = [(float(pts[nm][0])*sx, float(pts[nm][1])*sy) if pts.get(nm) else None
                     for nm in self.landmark_names]
        return img_arr, landmarks, H, W

    def __getitem__(self, idx):
        from ucl.model import make_gaussian_heatmaps
        img_arr, landmarks, H, W = self._load(self.files[idx])
        if self.augment:
            if np.random.rand() < 0.5:
                img_arr   = img_arr[:,::-1].copy()
                landmarks = [(W-1-pt[0],pt[1]) if pt else None for pt in landmarks]
            if np.random.rand() < 0.5:
                img_arr = np.clip(img_arr*np.random.uniform(0.8,1.2)+np.random.uniform(-0.05,0.05),0,1)
        img_arr, pad = pad_to_multiple(img_arr.astype(np.float32), 16)
        pH, pW   = img_arr.shape
        heatmaps = make_gaussian_heatmaps(landmarks, pH, pW, self.sigma)
        visible  = torch.tensor([pt is not None for pt in landmarks], dtype=torch.bool)
        return torch.from_numpy(img_arr)[None], heatmaps, visible
