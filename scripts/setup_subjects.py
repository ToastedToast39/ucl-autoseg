"""
Set up subject folders from pl_data.

Reads PatientID from every DICOM in pl_data and copies each file
into the correct subject/session/images folder. Run once after
authenticating, before labeling.

Works for both:
  - GE GEMS format (extensionless files in pl data 1/2)
  - capture_*.dcm format (pl data 3/4)

Usage:
    python scripts/setup_subjects.py
"""
import pydicom, shutil, re, sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
PL_DATA  = Path.home() / "Desktop" / "pl_data"


def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128); return f.read(4) == b"DICM"
    except Exception:
        return False


def get_subject_id(path: Path) -> str | None:
    try:
        ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        pid = str(ds.get("PatientID", "")).strip()
        if pid and pid not in ("", "None"):
            return re.sub(r"[^\w\-]", "_", pid).strip("_")
        pname = str(ds.get("PatientName", "")).strip()
        if pname and pname not in ("", "None"):
            return re.sub(r"[^\w\-]", "_", pname).strip("_")
    except Exception:
        pass
    return None


def main():
    if not PL_DATA.exists():
        print(f"pl_data not found at {PL_DATA}")
        print("Place the pl_data folder on your Desktop and try again.")
        sys.exit(1)

    subject_folders = sorted(
        p for p in PL_DATA.iterdir()
        if p.is_dir() and p.name.lower().startswith("pl")
    )
    if not subject_folders:
        print(f"No 'pl data *' folders found in {PL_DATA}")
        sys.exit(1)

    copied  = 0
    skipped = 0

    for sf in subject_folders:
        for f in sorted(sf.rglob("*")):
            if not f.is_file(): continue
            if not (f.suffix.lower() == ".dcm" or
                    (f.suffix == "" and is_dicom(f))):
                continue
            sid = get_subject_id(f)
            if not sid: continue

            out_dir = PIPELINE/"subjects"/sid/"sessions"/"session_01"/"images"
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / (f.stem + ".dcm")

            if dest.exists():
                skipped += 1
                continue

            shutil.copy(f, dest)
            copied += 1
            print(f"copied {sid}/{dest.name}")

    print(f"Done. {copied} DICOMs copied, {skipped} already existed.")


if __name__ == "__main__":
    main()
