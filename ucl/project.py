"""
Project layout for the UCL study — subjects with imaging sessions.

    <root>/
        subjects/
            <SubjectID>/
                subject.json         px/mm, notes
                labels/
                    images/           images staged for labeling
                    masks/            indexed label PNGs
                    points/           landmark coordinate JSONs
                sessions/
                    <SessionName>/
                        images/       ultrasound images for this session
                        results/      inference outputs, measurements CSV
                        session.json  optional per-session overrides (px/mm etc.)
        models/
            ucl_seg_*.pt
            ucl_landmarks_*.pt

Mirrors tendon/project.py participant/trial structure exactly,
substituting subject/session for participant/trial.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
SUBJECTS = ROOT / "subjects"
MODELS   = ROOT / "models"


def ensure_dirs():
    SUBJECTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)


# ---- subject level ----------------------------------------------------------

def subject_dir(sid):         return SUBJECTS / sid
def labels_dir(sid):          return subject_dir(sid) / "labels"
def label_images_dir(sid):    return labels_dir(sid) / "images"
def label_masks_dir(sid):     return labels_dir(sid) / "masks"
def label_points_dir(sid):    return labels_dir(sid) / "points"
def sessions_dir(sid):        return subject_dir(sid) / "sessions"
def smeta_path(sid):          return subject_dir(sid) / "subject.json"


def make_subject(sid: str):
    for d in (label_images_dir(sid), label_masks_dir(sid),
              label_points_dir(sid), sessions_dir(sid)):
        d.mkdir(parents=True, exist_ok=True)
    if not smeta_path(sid).exists():
        save_smeta(sid, {"id": sid, "px_per_mm": None})


def load_smeta(sid: str) -> dict:
    p = smeta_path(sid)
    return json.loads(p.read_text()) if p.exists() else {"id": sid, "px_per_mm": None}


def save_smeta(sid: str, meta: dict):
    smeta_path(sid).write_text(json.dumps(meta, indent=2))


# ---- session level ----------------------------------------------------------

def session_dir(sid, sess):         return sessions_dir(sid) / sess
def session_images_dir(sid, sess):  return session_dir(sid, sess) / "images"
def session_results_dir(sid, sess): return session_dir(sid, sess) / "results"
def sessm_path(sid, sess):          return session_dir(sid, sess) / "session.json"


def make_session(sid: str, sess: str):
    for d in (session_images_dir(sid, sess), session_results_dir(sid, sess)):
        d.mkdir(parents=True, exist_ok=True)


def load_sessm(sid: str, sess: str) -> dict:
    p = sessm_path(sid, sess)
    return json.loads(p.read_text()) if p.exists() else {}


def save_sessm(sid: str, sess: str, meta: dict):
    sessm_path(sid, sess).write_text(json.dumps(meta, indent=2))


def effective_settings(sid: str, sess: str) -> dict:
    """Session overrides merged onto subject defaults (mirrors tendon/project.py)."""
    s = load_smeta(sid)
    s.update({k: v for k, v in load_sessm(sid, sess).items() if v is not None})
    return s


# ---- listing ----------------------------------------------------------------

def list_subjects() -> list[str]:
    ensure_dirs()
    return sorted(p.name for p in SUBJECTS.iterdir() if p.is_dir())


def list_sessions(sid: str) -> list[str]:
    td = sessions_dir(sid)
    if not td.exists(): return []
    return sorted(t.name for t in td.iterdir() if t.is_dir())


def list_models(prefix: str = "") -> list[str]:
    ensure_dirs()
    return sorted(str(p) for p in MODELS.glob(f"{prefix}*.pt"))


def labeled_count(sid: str) -> int:
    masks = list(label_masks_dir(sid).glob("*.png"))
    return len(masks)
