from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False


def is_dicom(path):
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def read_dicom_gray(path):
    try:
        ds  = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
        while arr.ndim > 3:
            arr = arr.squeeze(0)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = (0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]).astype(np.uint8)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:,:,0]
        return arr.astype(np.uint8)
    except Exception as e:
        print(f"    skip {path.name}: {e}")
        return None


def find_dicoms(folder):
    found = []
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file(): continue
        if p.suffix.lower() == ".dcm" or (p.suffix == "" and is_dicom(p)):
            found.append(p)
    return found


def subject_id(folder_name):
    digits = "".join(c for c in folder_name if c.isdigit())
    return f"PL{digits.zfill(3)}" if digits else folder_name.replace(" ","_").upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pl_data",  default=str(Path.home()/"Desktop"/"pl_data"))
    ap.add_argument("--pipeline", default=str(Path.home()/"Desktop"/"ucl_pipeline"))
    ap.add_argument("--session",  default="session_01")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    if not _HAS_PYDICOM:
        raise SystemExit("pydicom required")

    pl_data  = Path(args.pl_data)
    pipeline = Path(args.pipeline)

    import sys
    sys.path.insert(0, str(pipeline))
    from ucl.project import make_subject, make_session, session_images_dir

    subject_folders = sorted(
        p for p in pl_data.iterdir()
        if p.is_dir() and p.name.lower().startswith("pl")
    )
    if not subject_folders:
        raise SystemExit(f"No pl data folders found in {pl_data}")

    total_exported = 0
    total_skipped  = 0

    for sf in subject_folders:
        sid = subject_id(sf.name)
        print(f"\n{sf.name}  ->  subject {sid}")
        make_subject(sid)
        make_session(sid, args.session)
        out_dir = session_images_dir(sid, args.session)
        dicoms  = find_dicoms(sf)
        print(f"  found {len(dicoms)} DICOM files")
        exported = 0
        for dcm_path in dicoms:
            png_name = dcm_path.stem + ".png"
            out_path = out_dir / png_name
            if args.skip_existing and out_path.exists():
                total_skipped += 1
                continue
            arr = read_dicom_gray(dcm_path)
            if arr is None:
                continue
            Image.fromarray(arr).save(str(out_path))
            exported += 1
        print(f"  exported {exported} PNGs -> {out_dir}")
        total_exported += exported

    print(f"\nDone. {total_exported} PNGs exported, {total_skipped} skipped.")
    print("\nNext: python ucl_tool.py -> option 4 to label in Labelme.")


if __name__ == "__main__":
    main()
