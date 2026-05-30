"""
Run the UCL segmentation model on a subject's DICOM images.

Mirrors process_neural_subject.sh from the spine project exactly:
  - change SUBJECT at the top
  - run the script
  - when it says Done!, load in Slicer with load_ucl_slicer.py

For each DICOM frame in the subject's session:
  1. Read pixel data → grayscale numpy array
  2. Run segmentation model → 5-class label map
  3. Save label map as NIfTI (.nii.gz) alongside the PNG
  4. Save overlay PNG for quick QC (no Slicer needed)

Outputs (in subjects/<SUBJECT>/sessions/<SESSION>/results/):
  <stem>_seg.nii.gz     segmentation mask loadable in Slicer
  <stem>_seg.png        quick QC overlay
  measurements.csv      measurements if px_per_mm is set

Usage:
    python scripts/process_ucl_subject.py

Or with arguments:
    python scripts/process_ucl_subject.py \
        --subject PL003 \
        --session session_01 \
        --seg_model models/ucl_seg.pt \
        --px_per_mm 12.5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

# ---- add pipeline root to path ----
PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch required. Run: conda activate ucl")

try:
    import pydicom
except ImportError:
    raise SystemExit("pydicom required. Run: pip install pydicom pylibjpeg pylibjpeg-libjpeg")

try:
    import nibabel as nib
    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False

from PIL import Image
from ucl.model   import UNet
from ucl.data    import pad_to_multiple, unpad, NUM_SEG_CLASSES
from ucl.project import (list_models, session_images_dir,
                          session_results_dir, effective_settings)

# colour map for QC overlay: 1=UCL(magenta) 2=humerus(orange) 3=ulna(green) 4=flexor(cyan)
PALETTE = {1: (255,0,255), 2: (0,165,255), 3: (0,255,0), 4: (0,255,255)}


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    nc = ck.get("num_classes", NUM_SEG_CLASSES)
    m  = UNet(in_ch=1, out_ch=nc, base=ck.get("base", 32)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("resize"), nc


def read_dicom_gray(path: Path) -> np.ndarray | None:
    try:
        ds   = pydicom.dcmread(str(path), force=True)
        arr  = ds.pixel_array
        if arr.ndim == 3:
            arr = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8)
        return arr.astype(np.uint8)
    except Exception as e:
        print(f"  skip {path.name}: {e}"); return None


@torch.no_grad()
def run_seg(model, gray, device, resize):
    H0, W0 = gray.shape
    img = gray.astype(np.float32) / 255.0
    if resize:
        rh, rw = resize
        img = np.asarray(Image.fromarray((img*255).astype(np.uint8))
                         .resize((rw,rh), Image.BILINEAR), np.float32) / 255.0
    img_p, pad = pad_to_multiple(img, 16)
    t      = torch.from_numpy(img_p)[None,None].to(device)
    labels = torch.softmax(model(t),1)[0].argmax(0).cpu().numpy().astype(np.int32)
    labels = unpad(labels, pad)
    if labels.shape != (H0,W0):
        labels = np.asarray(Image.fromarray(labels.astype(np.uint8))
                            .resize((W0,H0), Image.NEAREST), np.int32)
    return labels


def save_nifti(labels: np.ndarray, out_path: Path):
    """Save label map as NIfTI — loadable in 3D Slicer as a labelmap."""
    if not _HAS_NIBABEL:
        # fallback: save as PNG only
        print(f"  (nibabel not installed — saving PNG mask only, not NIfTI)")
        return False
    # NIfTI expects (X,Y,Z) — for a 2D image we add a Z dimension of 1
    data = labels.astype(np.int16)[..., np.newaxis]
    img  = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(out_path))
    return True


def save_overlay(gray, labels, out_path):
    """Quick QC overlay — same colour scheme as Slicer view."""
    rgb  = np.stack([gray, gray, gray], axis=-1)
    tint = np.zeros_like(rgb)
    for cid, col in PALETTE.items():
        tint[labels == cid] = col
    blended = np.clip(rgb.astype(np.float32)*0.7 + tint.astype(np.float32)*0.3,
                      0, 255).astype(np.uint8)
    Image.fromarray(blended).save(str(out_path))


def is_dicom(path: Path) -> bool:
    try:
        with open(path,"rb") as f:
            f.seek(128); return f.read(4) == b"DICM"
    except Exception: return False


def find_dicoms(folder: Path) -> list[Path]:
    found = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file(): continue
        if p.suffix.lower() == ".dcm" or (p.suffix == "" and is_dicom(p)):
            found.append(p)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",     default=None,
                    help="subject ID e.g. PL001 (prompted if omitted)")
    ap.add_argument("--session",     default="session_01")
    ap.add_argument("--seg_model",   default=None,
                    help="path to trained seg model .pt (auto-detects if omitted)")
    ap.add_argument("--px_per_mm",   type=float, default=None)
    ap.add_argument("--pl_data",     default=str(Path.home()/"Desktop"/"pl_data"),
                    help="path to pl_data folder (used if session images not yet exported)")
    args = ap.parse_args()

    # ---- resolve subject ----
    from ucl.project import list_subjects, session_images_dir as sid_dir
    subjects = list_subjects()
    if args.subject:
        sid = args.subject
    elif subjects:
        print("Available subjects:", ", ".join(subjects))
        sid = input("Subject ID: ").strip()
    else:
        raise SystemExit("No subjects found. Run export_for_labeling.py first.")

    sess = args.session
    s    = effective_settings(sid, sess)
    px   = args.px_per_mm or s.get("px_per_mm")

    # ---- resolve model ----
    seg_model_path = args.seg_model
    if not seg_model_path:
        models = list_models("ucl_seg")
        if not models:
            raise SystemExit("No trained seg model found. Train one first (ucl_tool.py option 6).")
        seg_model_path = models[0]
        print(f"Using model: {seg_model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, resize, nc = load_model(seg_model_path, device)
    print(f"model: {nc} classes")

    # ---- find images ----
    img_dir = sid_dir(sid, sess)
    pngs    = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

    if not pngs:
        raise SystemExit(
            f"No images found in {img_dir}.\n"
            f"Run export_for_labeling.py first to export DICOMs as PNGs."
        )

    # ---- output folder ----
    out_dir = session_results_dir(sid, sess)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"overlays").mkdir(exist_ok=True)

    # install nibabel if missing
    global _HAS_NIBABEL
    if not _HAS_NIBABEL:
        print("Installing nibabel for NIfTI output...")
        import subprocess
        subprocess.call([sys.executable, "-m", "pip", "install",
                         "nibabel", "-q"])
        try:
            import nibabel as nib; _HAS_NIBABEL = True
        except ImportError:
            pass

    print(f"\nProcessing {len(pngs)} images for {sid}/{sess}...")
    print("-" * 50)

    for fp in pngs:
        gray   = np.asarray(Image.open(fp).convert("L"))
        labels = run_seg(model, gray, device, resize)

        # save NIfTI segmentation (for Slicer)
        nii_path = out_dir / (fp.stem + "_seg.nii.gz")
        save_nifti(labels, nii_path)

        # save QC overlay
        save_overlay(gray, labels, out_dir/"overlays"/(fp.stem+"_overlay.png"))

        n_ucl = (labels==1).sum()
        print(f"  {fp.name}: UCL pixels={n_ucl}  → {fp.stem}_seg.nii.gz")

    print("-" * 50)
    print(f"\nDone! Results in {out_dir}/")
    print(f"\nLoad in 3D Slicer:")
    print(f"  1. Open 3D Slicer Python console")
    print(f"  2. Edit subject/session at top of scripts/load_ucl_slicer.py")
    print(f"  3. Paste the script and hit Enter")
    print(f"\nOr open overlays/ folder for a quick QC check without Slicer.")


if __name__ == "__main__":
    main()
