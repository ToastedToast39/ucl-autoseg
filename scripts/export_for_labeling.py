"""
Export UCL DICOM frames as grayscale PNGs for labeling in Labelme.

Reads the real subject ID from each DICOM's PatientID or PatientName field,
so pipeline folders match actual study IDs (PLUCSD003, Chapman_Weber etc.)
instead of generic folder names.

Handles both:
  - GE GEMS format (extensionless files in pl data 1/2)
  - capture_*.dcm format (pl data 3/4)

Usage:
    python scripts/export_for_labeling.py
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False


def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def get_subject_id(path: Path) -> str | None:
    """Read PatientID or PatientName from DICOM header without loading pixels."""
    try:
        ds  = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        pid = str(ds.get("PatientID", "")).strip()
        if pid and pid not in ("", "None"):
            return pid.replace(" ", "_")
        pname = str(ds.get("PatientName", "")).strip()
        if pname and pname not in ("", "None"):
            return pname.replace(" ", "_").replace("^", "_").strip("_")
    except Exception:
        pass
    return None


def read_dicom_gray(path: Path) -> np.ndarray | None:
    try:
        ds  = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
        # handle unusual shapes e.g. (1,1,H,W,3)
        while arr.ndim > 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.reshape(arr.shape[-3], arr.shape[-2], arr.shape[-1])
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8)
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


def sanitize_id(raw: str) -> str:
    """Make a string safe for use as a folder name."""
    import re
    s = raw.strip().replace(" ", "_").replace("^", "_")
    s = re.sub(r"[^\w\-]", "", s)
    return s.strip("_") or "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pl_data",  default=str(Path.home()/"Desktop"/"pl_data"))
    ap.add_argument("--pipeline", default=str(Path.home()/"Desktop"/"ucl_pipeline"))
    ap.add_argument("--session",  default="session_01")
    args = ap.parse_args()

    if not _HAS_PYDICOM:
        raise SystemExit("pydicom required: pip install pydicom pylibjpeg pylibjpeg-libjpeg")

    pl_data  = Path(args.pl_data)
    pipeline = Path(args.pipeline)

    sys.path.insert(0, str(pipeline))
    from ucl.project import make_subject, make_session, session_images_dir

    if not pl_data.exists():
        raise SystemExit(f"pl_data not found: {pl_data}")

    # find all top-level pl data folders
    subject_folders = sorted(
        p for p in pl_data.iterdir()
        if p.is_dir() and p.name.lower().startswith("pl")
    )
    if not subject_folders:
        raise SystemExit(f"No 'pl data *' folders found in {pl_data}")

    total_exported = 0
    total_skipped  = 0
    # track which DICOMs belong to which subject ID
    # key: subject_id, value: list of (dcm_path, png_name)
    subject_map: dict[str, list] = {}

    print("Scanning DICOM headers to identify subjects…")
    for sf in subject_folders:
        dicoms = find_dicoms(sf)
        print(f"\n{sf.name}: found {len(dicoms)} DICOM files")
        for dcm_path in dicoms:
            sid = get_subject_id(dcm_path)
            if not sid:
                print(f"    skip {dcm_path.name}: no PatientID or PatientName")
                continue
            sid = sanitize_id(sid)
            if sid not in subject_map:
                subject_map[sid] = []
            subject_map[sid].append(dcm_path)

    print(f"\nFound {len(subject_map)} unique subjects:")
    for sid in sorted(subject_map):
        print(f"  {sid}: {len(subject_map[sid])} images")

    print("\nExporting…")
    for sid in sorted(subject_map):
        make_subject(sid)
        make_session(sid, args.session)
        out_dir = session_images_dir(sid, args.session)
        exported = 0

        for dcm_path in subject_map[sid]:
            png_name = dcm_path.stem + ".png"
            out_path = out_dir / png_name
            if out_path.exists():
                total_skipped += 1
                continue
            arr = read_dicom_gray(dcm_path)
            if arr is None:
                continue
            Image.fromarray(arr).save(str(out_path))
            exported += 1

        print(f"  {sid}: exported {exported} PNGs → {out_dir}")
        total_exported += exported

    print(f"\nDone. {total_exported} new PNGs exported, {total_skipped} already existed.")
    print("Next: open Labelme via ucl_tool.py option 4 or the Slicer module.")


if __name__ == "__main__":
    main()
