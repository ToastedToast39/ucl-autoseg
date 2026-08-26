"""
Collect labeled images from all subjects into a flat training dataset.

Scans subjects/*/sessions/*/images/ for images that have a corresponding
mask in sessions/*/masks/ (either .nii.gz from Slicer or .png from Labelme).
Copies them into _train_seg/images/ and _train_seg/masks/.

Run before training, or called automatically by train_seg.py.

Usage:
    python scripts/build_training_set.py
"""
import json, shutil, sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
OUT_IMGS  = PIPELINE / "_train_seg" / "images"
OUT_MASKS = PIPELINE / "_train_seg" / "masks"
HOLDOUT_FILE  = PIPELINE / "holdout_subjects.json"
EXCLUDED_FILE = PIPELINE / "excluded_images.json"


def load_holdout():
    """Subjects reserved for clinical validation — never included in training data."""
    if not HOLDOUT_FILE.exists(): return set()
    return set(json.loads(HOLDOUT_FILE.read_text()).get("holdout_subjects", []))


def load_excluded():
    """Images marked as wrong anatomy (e.g. supraspinatus) — never trained on."""
    if not EXCLUDED_FILE.exists(): return set()
    return set(json.loads(EXCLUDED_FILE.read_text()).keys())


def main():
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)

    # clear old training data
    for f in OUT_IMGS.glob("*"):  f.unlink()
    for f in OUT_MASKS.glob("*"): f.unlink()

    holdout  = load_holdout()
    excluded = load_excluded()
    total = 0
    skipped_holdout = 0
    skipped_excluded = 0
    subjects_dir = PIPELINE / "subjects"
    if not subjects_dir.exists():
        print("No subjects found."); return

    for subj in sorted(subjects_dir.iterdir()):
        if not subj.is_dir(): continue
        if subj.name in holdout:
            skipped_holdout += 1; continue
        for sess in sorted((subj/"sessions").iterdir() if (subj/"sessions").exists() else []):
            if not sess.is_dir(): continue
            img_dir  = sess / "images"
            mask_dir = sess / "masks"
            if not img_dir.exists() or not mask_dir.exists(): continue

            for img_path in sorted(list(img_dir.glob("*.dcm")) +
                                   list(img_dir.glob("*.png")) +
                                   list(img_dir.glob("*.jpg"))):
                stem = img_path.stem
                if f"{subj.name}/{sess.name}/{stem}" in excluded:
                    skipped_excluded += 1; continue
                tag  = f"{subj.name}__{sess.name}__{stem}"

                # find mask — NIfTI first, PNG fallback
                nii = mask_dir / (stem + ".nii.gz")
                png = mask_dir / (stem + ".png")

                if nii.exists():
                    shutil.copy(img_path, OUT_IMGS / (tag + img_path.suffix))
                    shutil.copy(nii,       OUT_MASKS / (tag + ".nii.gz"))
                    total += 1
                elif png.exists():
                    shutil.copy(img_path, OUT_IMGS / (tag + img_path.suffix))
                    shutil.copy(png,       OUT_MASKS / (tag + ".png"))
                    total += 1

    print(f"Training set built: {total} labeled images → {PIPELINE/'_train_seg'}/")
    if holdout:
        print(f"Holdout subjects excluded: {sorted(holdout)} ({skipped_holdout} skipped)")
    if skipped_excluded:
        print(f"Wrong-anatomy images excluded: {skipped_excluded}")
    return total


if __name__ == "__main__":
    n = main()
    if not n:
        print("No labeled images found. Label some images first.")
        sys.exit(1)
