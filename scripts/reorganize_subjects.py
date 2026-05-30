"""
Reorganize exported PNGs into correctly-named subject folders.

Reads PatientID from every DICOM in pl_data, finds the matching PNG
in the old PL001-PL004 folders, and copies it into the correct
subject folder (PLUCSD003, Chapman_Weber, etc.).

Also re-exports any DICOMs whose PNGs are missing.

Usage:
    python scripts/reorganize_subjects.py
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    raise SystemExit("pydicom required: pip install pydicom pylibjpeg pylibjpeg-libjpeg")

PIPELINE = Path.home() / "Desktop" / "ucl_pipeline"
PL_DATA  = Path.home() / "Desktop" / "pl_data"
SESSION  = "session_01"

sys.path.insert(0, str(PIPELINE))
from ucl.project import make_subject, make_session, session_images_dir


def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def get_subject_id(path: Path) -> str | None:
    try:
        ds  = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        pid = str(ds.get("PatientID", "")).strip()
        if pid and pid not in ("", "None"):
            return pid.replace(" ", "_")
        pname = str(ds.get("PatientName", "")).strip()
        if pname and pname not in ("", "None"):
            import re
            return re.sub(r"[^\w\-]", "_", pname).strip("_")
    except Exception:
        pass
    return None


def read_dicom_gray(path: Path) -> np.ndarray | None:
    try:
        ds  = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
        while arr.ndim > 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.reshape(arr.shape[-3], arr.shape[-2], arr.shape[-1])
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = (0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]).astype(np.uint8)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:,:,0]
        return arr.astype(np.uint8)
    except Exception as e:
        print(f"    skip {path.name}: {e}")
        return None


def find_dicoms(folder: Path) -> list[Path]:
    found = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file(): continue
        if p.suffix.lower() == ".dcm" or (p.suffix == "" and is_dicom(p)):
            found.append(p)
    return found


def main():
    if not PL_DATA.exists():
        raise SystemExit(f"pl_data not found: {PL_DATA}")

    subject_folders = sorted(
        p for p in PL_DATA.iterdir()
        if p.is_dir() and p.name.lower().startswith("pl")
    )

    print("Scanning all DICOMs and mapping to subject IDs...")
    print("=" * 60)

    # Build map: subject_id → list of dcm_paths
    subject_map: dict[str, list[Path]] = {}
    skipped = 0

    for sf in subject_folders:
        dicoms = find_dicoms(sf)
        print(f"\n{sf.name}: {len(dicoms)} DICOMs")
        for dcm in dicoms:
            sid = get_subject_id(dcm)
            if not sid:
                skipped += 1
                continue
            import re
            sid = re.sub(r"[^\w\-]", "_", sid).strip("_") or "UNKNOWN"
            subject_map.setdefault(sid, []).append(dcm)

    print(f"\n{'=' * 60}")
    print(f"Found {len(subject_map)} unique subjects:")
    for sid in sorted(subject_map):
        print(f"  {sid:20s}: {len(subject_map[sid])} images")

    print(f"\n{'=' * 60}")
    print("Exporting to correctly-named subject folders...")

    total_exported = 0
    total_skipped  = 0

    for sid in sorted(subject_map):
        make_subject(sid)
        make_session(sid, SESSION)
        out_dir = session_images_dir(sid, SESSION)
        exported = 0

        for dcm_path in subject_map[sid]:
            out_path = out_dir / (dcm_path.stem + ".png")
            if out_path.exists():
                total_skipped += 1
                continue
            arr = read_dicom_gray(dcm_path)
            if arr is None:
                continue
            Image.fromarray(arr).save(str(out_path))
            exported += 1
            total_exported += 1

        print(f"  {sid:20s}: {exported} new PNGs exported  ({total_skipped} already existed)")

    print(f"\n{'=' * 60}")
    print(f"Done. {total_exported} new PNGs exported, {total_skipped} already existed.")

    # Clean up old PL001-PL004 folders if they exist and are now redundant
    old_folders = [PIPELINE/"subjects"/f for f in ("PL001","PL002","PL003","PL004")
                   if (PIPELINE/"subjects"/f).exists()]
    if old_folders:
        print(f"\nOld placeholder folders found: {[f.name for f in old_folders]}")
        ans = input("Delete old PL001-PL004 folders? (y/n): ").strip().lower()
        if ans == "y":
            for f in old_folders:
                shutil.rmtree(f)
                print(f"  Deleted {f.name}")
        else:
            print("  Kept old folders — you can delete them manually later.")

    print("\nAll subjects now organized by real ID.")
    print("Open Slicer module → Refresh List to see the updated subject list.")


if __name__ == "__main__":
    main()
