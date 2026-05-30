"""
Data loading for the UCL pipeline.

UCLSegDataset      mirrors TendonSegDataset from tendon/data.py exactly
                   (same augmentation: hflip, gain/bias, speckle noise)

UCLLandmarkDataset new — reads companion point JSON files from points/
                   and builds (N,H,W) Gaussian heatmap tensors as targets

labelme_to_masks_and_points
                   extends labelme_to_masks from tendon/data.py to also
                   extract point annotations into a separate JSON per image

pad_to_multiple / unpad  — identical to tendon/data.py
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


# ---------------------------------------------------------------------------
# Landmark and class definitions  (change here if clinical spec changes)
# ---------------------------------------------------------------------------

LANDMARK_NAMES: list[str] = [
    # Kept minimal for Phase 1 — segmentation context model comes first.
    # Landmarks for gap distance, trochlea length etc. added in Phase 2
    # once segmentation is reliable.
    "ucl_humeral",   # proximal UCL attachment on medial epicondyle
    "ucl_ulnar",     # distal UCL attachment on sublime tubercle
]

SEG_CLASS_MAP: dict[str, int] = {
    # Phase 1: start with the two most visible structures — the bone surfaces.
    # Once humerus and ulna segment reliably, add ucl and flexor_pronator.
    "humerus": 1,   # humeral bone surface / trochlea
    "ulna":    2,   # ulnar bone surface / sublime tubercle
}
NUM_SEG_CLASSES = 3   # 0=bg, 1=humerus, 2=ulna


# ---------------------------------------------------------------------------
# Padding utilities — verbatim from tendon/data.py
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
# Dataset A: Segmentation  (mirrors TendonSegDataset)
# ---------------------------------------------------------------------------

class UCLSegDataset(Dataset):
    """Image/mask pairs for multi-class UCL segmentation.

    Layout:
        root/images/   *.png or *.jpg
        root/masks/    *.png  (indexed: 0=bg, 1=ucl, 2=bone_humerus, 3=bone_ulna)

    Augmentation mirrors TendonSegDataset: hflip, gain/bias jitter, speckle.
    """
    def __init__(self, root: str, augment: bool = False,
                 resize: Optional[tuple] = None,
                 num_classes: int = NUM_SEG_CLASSES):
        if not _HAS_TORCH: raise RuntimeError("PyTorch required")
        self.img_dir = Path(root) / "images"
        self.msk_dir = Path(root) / "masks"
        all_imgs = sorted(self.img_dir.glob("*.png")) + sorted(self.img_dir.glob("*.jpg"))
        self.files = sorted(
            p.name for p in all_imgs
            if (self.msk_dir / (p.stem + ".png")).exists()
        )
        self.augment = augment
        self.resize  = resize
        self.num_classes = num_classes

    def __len__(self): return len(self.files)

    def _load(self, name):
        img = Image.open(self.img_dir / name).convert("L")
        msk = Image.open(self.msk_dir / (Path(name).stem + ".png")).convert("L")
        if self.resize:
            H, W = self.resize
            img = img.resize((W, H), Image.BILINEAR)
            msk = msk.resize((W, H), Image.NEAREST)
        img = np.asarray(img, np.float32) / 255.0
        m   = np.asarray(msk)
        # accept both 0/255 binary masks and indexed label maps
        m = (m > 127).astype(np.int64) if m.max() > self.num_classes-1 else m.astype(np.int64)
        return img, m

    def _augment(self, img, msk):
        # horizontal flip — matches TendonSegDataset
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
        # gain/bias jitter (ultrasound gain varies — same as reference)
        if np.random.rand() < 0.5:
            img = np.clip(img * np.random.uniform(0.8, 1.2)
                          + np.random.uniform(-0.05, 0.05), 0, 1)
        # speckle noise — same as reference
        if np.random.rand() < 0.3:
            img = np.clip(img + np.random.randn(*img.shape) * 0.02, 0, 1)
        return img, msk

    def __getitem__(self, idx):
        img, msk = self._load(self.files[idx])
        if self.augment: img, msk = self._augment(img, msk)
        img, _ = pad_to_multiple(img.astype(np.float32), 16)
        msk, _ = pad_to_multiple(msk.astype(np.int64),   16)
        return torch.from_numpy(img)[None], torch.from_numpy(msk).long()


# ---------------------------------------------------------------------------
# Dataset B: Landmark heatmap regression  (new — no patellar equivalent)
# ---------------------------------------------------------------------------

class UCLLandmarkDataset(Dataset):
    """Image + Gaussian heatmap targets for landmark localization.

    Layout:
        root/images/   *.png or *.jpg
        root/points/   *.json  {landmark_name: [x,y] or null}

    Produces:
        img_tensor  (1, H, W) float32
        heatmaps    (N, H, W) float32 in [0,1]
        visible     (N,) bool — True where landmark IS labeled
    """
    def __init__(self, root: str, augment: bool = False,
                 resize: Optional[tuple] = None,
                 landmark_names: list[str] = LANDMARK_NAMES,
                 sigma: float = 8.0):
        if not _HAS_TORCH: raise RuntimeError("PyTorch required")
        self.img_dir   = Path(root) / "images"
        self.pts_dir   = Path(root) / "points"
        all_imgs = sorted(self.img_dir.glob("*.png")) + sorted(self.img_dir.glob("*.jpg"))
        self.files = sorted(
            p.name for p in all_imgs
            if (self.pts_dir / (p.stem + ".json")).exists()
        )
        self.augment = augment
        self.resize  = resize
        self.landmark_names = landmark_names
        self.sigma   = sigma

    def __len__(self): return len(self.files)

    def _load(self, name):
        img = Image.open(self.img_dir / name).convert("L")
        W0, H0 = img.size
        with open(self.pts_dir / (Path(name).stem + ".json")) as f:
            pts = json.load(f)
        if self.resize:
            H, W = self.resize
            sx, sy = W/W0, H/H0
            img = img.resize((W, H), Image.BILINEAR)
        else:
            H, W, sx, sy = H0, W0, 1.0, 1.0
        img_arr = np.asarray(img, np.float32) / 255.0
        landmarks = []
        for nm in self.landmark_names:
            pt = pts.get(nm)
            landmarks.append((float(pt[0])*sx, float(pt[1])*sy) if pt else None)
        return img_arr, landmarks, H, W

    def __getitem__(self, idx):
        from ucl.model import make_gaussian_heatmaps
        img_arr, landmarks, H, W = self._load(self.files[idx])
        if self.augment:
            if np.random.rand() < 0.5:
                img_arr = img_arr[:, ::-1].copy()
                landmarks = [
                    (W-1-pt[0], pt[1]) if pt else None for pt in landmarks
                ]
            if np.random.rand() < 0.5:
                img_arr = np.clip(
                    img_arr * np.random.uniform(0.8,1.2)
                    + np.random.uniform(-0.05, 0.05), 0, 1)
        img_arr, pad = pad_to_multiple(img_arr.astype(np.float32), 16)
        pH, pW = img_arr.shape
        heatmaps = make_gaussian_heatmaps(landmarks, pH, pW, self.sigma)
        visible  = torch.tensor([pt is not None for pt in landmarks], dtype=torch.bool)
        return torch.from_numpy(img_arr)[None], heatmaps, visible


# ---------------------------------------------------------------------------
# Labelme → masks + points  (extends labelme_to_masks from tendon/data.py)
# ---------------------------------------------------------------------------

def labelme_to_masks_and_points(
    json_dir: str,
    out_mask_dir: str,
    out_points_dir: str,
    seg_class_map: dict[str, int] = SEG_CLASS_MAP,
    landmark_names: list[str] = LANDMARK_NAMES,
    img_size: Optional[tuple] = None,
):
    """Convert Labelme JSON files to indexed masks + landmark coordinate JSON.

    Extends labelme_to_masks (tendon/data.py) to also extract point shapes.

    polygon shapes → indexed PNG masks (0=bg, 1=ucl, 2=bone_humerus, 3=bone_ulna)
    point shapes   → {landmark_name: [x,y] or null} JSON files
    """
    os.makedirs(out_mask_dir,   exist_ok=True)
    os.makedirs(out_points_dir, exist_ok=True)
    for jf in sorted(glob(os.path.join(json_dir, "*.json"))):
        with open(jf) as f:
            d = json.load(f)
        if img_size:
            H, W = img_size
        else:
            H, W = d["imageHeight"], d["imageWidth"]
        stem = Path(d.get("imagePath", jf)).stem

        # --- segmentation mask (polygon shapes) ---
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        for label, cid in sorted(seg_class_map.items(), key=lambda kv: kv[1]):
            for shp in d.get("shapes", []):
                if shp.get("label") != label or shp.get("shape_type") != "polygon":
                    continue
                pts = [tuple(p) for p in shp["points"]]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=int(cid))
        mask.save(os.path.join(out_mask_dir, f"{stem}.png"))

        # --- landmark points (point shapes) ---
        pts_out = {nm: None for nm in landmark_names}
        for shp in d.get("shapes", []):
            if shp.get("shape_type") != "point": continue
            nm = shp.get("label", "")
            if nm in pts_out:
                pts_out[nm] = [float(shp["points"][0][0]), float(shp["points"][0][1])]
        with open(os.path.join(out_points_dir, f"{stem}.json"), "w") as f:
            json.dump(pts_out, f, indent=2)
    print(f"Converted {len(glob(os.path.join(json_dir,'*.json')))} files → "
          f"{out_mask_dir}  +  {out_points_dir}")
