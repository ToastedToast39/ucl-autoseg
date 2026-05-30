"""
UCL measurement engine.

Key functions ported verbatim or closely adapted from the patellar reference:

  mask_quality_ok()    — from tendon_analysis.py (same logic, UCL-tuned thresholds)
  mad_outlier_mask()   — from tendon_analysis.py (identical)
  smooth_series()      — from tendon_analysis.py (identical)
  largest_component()  — from tendon_analysis.py / infer_video.py (identical)
  fit_line_tls()       — from tendon_analysis.py (identical)

New for UCL (no patellar equivalent):
  measure_ucl()        — landmark-to-landmark distances + bone angle + UCL thickness
  UCLMeasurement       — dataclass holding all measurements for one image
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Ported verbatim from tendon_analysis.py
# ---------------------------------------------------------------------------

def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected blob. Verbatim from tendon_analysis.py."""
    from scipy import ndimage
    mask = mask.astype(bool)
    if not mask.any(): return mask
    lbl, n = ndimage.label(mask)
    if n <= 1: return mask
    sizes = ndimage.sum(mask, lbl, index=np.arange(1, n+1))
    return lbl == int(np.argmax(sizes)) + 1


def fit_line_tls(points: np.ndarray):
    """Total-least-squares line via PCA. Returns (origin, unit_direction).
    Verbatim from tendon_analysis.py."""
    p = np.asarray(points, float); c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    d = vt[0]; return c, d / np.linalg.norm(d)


def mad_outlier_mask(vals: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """True where value is an outlier by MAD score.
    Verbatim from tendon_analysis.py."""
    v = np.asarray(vals, float)
    good = ~np.isnan(v)
    if good.sum() < 3: return np.zeros_like(v, bool)
    med = np.median(v[good])
    mad = np.median(np.abs(v[good] - med)) or 1e-9
    score = 0.6745 * (v - med) / mad
    out = np.abs(score) > thresh
    out[~good] = False
    return out


def smooth_series(vals: np.ndarray, win: int) -> np.ndarray:
    """Median then moving-average smoothing. Verbatim from tendon_analysis.py."""
    vals = np.asarray(vals, float); n = len(vals); win = int(win)
    if win < 3 or n < 3: return vals.copy()
    win = min(win, n if n % 2 else n-1)
    if win % 2 == 0: win += 1
    half = win // 2
    v = vals.copy(); nans = np.isnan(v)
    if nans.any() and (~nans).sum() >= 2:
        v[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), v[~nans])
    padded = np.pad(v, half, mode="reflect")
    med = np.array([np.median(padded[i:i+win]) for i in range(n)])
    padded2 = np.pad(med, half, mode="reflect")
    avg = np.convolve(padded2, np.ones(win)/win, mode="valid")[:n]
    avg[nans] = np.nan
    return avg


def mask_quality_ok(mask: np.ndarray, frame_area: float,
                    min_area_frac: float = 0.002,
                    max_area_frac: float = 0.5) -> tuple[bool, str]:
    """Reject implausible masks before measuring.

    Adapted from tendon_analysis.py mask_quality_ok(); thresholds relaxed
    slightly since UCL is smaller and more variable than the patellar tendon.
    Checks: area fraction, minimum column span, aspect ratio (width > height).
    """
    m = np.asarray(mask) > 0
    area = m.sum()
    if area == 0: return False, "empty"
    frac = area / frame_area
    if frac < min_area_frac: return False, f"too small ({frac:.4f})"
    if frac > max_area_frac: return False, f"too large ({frac:.3f})"
    cols = np.where(m.any(axis=0))[0]
    rows = np.where(m.any(axis=1))[0]
    if cols.size < 2: return False, "no column span"
    w = cols.max() - cols.min() + 1
    h = rows.max() - rows.min() + 1
    if w < 0.05 * np.sqrt(frame_area): return False, "too narrow"
    if h > 3 * w: return False, "wrong aspect (too tall)"
    return True, "ok"


# ---------------------------------------------------------------------------
# UCL-specific geometry
# ---------------------------------------------------------------------------

def point_distance(p1, p2) -> float:
    return float(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))


def ucl_thickness_at_midpoint(ucl_mask: np.ndarray) -> Optional[float]:
    """Perpendicular thickness of the UCL mask at its midpoint column."""
    m = np.asarray(ucl_mask) > 0
    cols = np.where(m.any(axis=0))[0]
    if cols.size < 4: return None
    c = cols[len(cols)//2]
    rows = np.where(m[:, c])[0]
    if rows.size == 0: return None
    t = float(rows[-1] - rows[0])
    return t if t > 0 else None


def bone_surface_angle(humerus_mask: np.ndarray,
                        ulna_mask: np.ndarray) -> Optional[float]:
    """Acute angle between humeral and ulnar surface lines (degrees).
    Uses upper-border points + TLS fit — same approach as
    tendon_tuberosity_angle.py."""
    def upper_pts(mask):
        m = largest_component(np.asarray(mask) > 0)
        cols = np.where(m.any(axis=0))[0]
        if cols.size < 4: return None
        ys = np.array([np.where(m[:, c])[0][0] for c in cols], float)
        return np.column_stack([cols.astype(float), ys])
    ph = upper_pts(humerus_mask); pu = upper_pts(ulna_mask)
    if ph is None or pu is None: return None
    _, dh = fit_line_tls(ph); _, du = fit_line_tls(pu)
    ca = min(1.0, max(-1.0, abs(float(np.dot(dh, du)))))
    return float(np.degrees(np.arccos(ca)))


# ---------------------------------------------------------------------------
# Main measurement dataclass + function
# ---------------------------------------------------------------------------

@dataclass
class UCLMeasurement:
    ucl_length_px:         Optional[float] = None
    ucl_length_mm:         Optional[float] = None
    medial_gap_px:         Optional[float] = None
    medial_gap_mm:         Optional[float] = None
    ucl_thickness_mid_px:  Optional[float] = None
    ucl_thickness_mid_mm:  Optional[float] = None
    bone_angle_deg:        Optional[float] = None
    px_per_mm:             Optional[float] = None
    mask_quality_ucl:      str = "not checked"
    notes:                 list = field(default_factory=list)


def _to_mm(px, px_per_mm):
    if px is None or not px_per_mm: return None
    return px / px_per_mm


def measure_ucl(landmarks: dict, seg_mask: Optional[np.ndarray] = None,
                px_per_mm: Optional[float] = None) -> UCLMeasurement:
    """Compute all UCL measurements for one image.

    landmarks : {name: (x,y) or None}  — from HeatmapUNet predictions
    seg_mask  : (H,W) integer label map (1=ucl, 2=bone_humerus, 3=bone_ulna)
    px_per_mm : from calibrate.py; None → px values only
    """
    r = UCLMeasurement(px_per_mm=px_per_mm)

    # landmark distances
    ph = landmarks.get("ucl_humeral"); pu = landmarks.get("ucl_ulnar")
    if ph and pu:
        r.ucl_length_px = point_distance(ph, pu)
        r.ucl_length_mm = _to_mm(r.ucl_length_px, px_per_mm)
    else:
        r.notes.append("ucl_length: attachment landmark(s) missing")

    pg = landmarks.get("gap_humerus"); pgu = landmarks.get("gap_ulna")
    if pg and pgu:
        r.medial_gap_px = point_distance(pg, pgu)
        r.medial_gap_mm = _to_mm(r.medial_gap_px, px_per_mm)
    else:
        r.notes.append("medial_gap: joint-line landmark(s) missing")

    # segmentation-based
    if seg_mask is not None:
        ucl_px = seg_mask == 1
        ok, reason = mask_quality_ok(ucl_px, float(seg_mask.size))
        r.mask_quality_ucl = reason
        if ok:
            t = ucl_thickness_at_midpoint(ucl_px)
            r.ucl_thickness_mid_px = t
            r.ucl_thickness_mid_mm = _to_mm(t, px_per_mm)
        else:
            r.notes.append(f"ucl_thickness: mask rejected ({reason})")
        ang = bone_surface_angle(seg_mask == 2, seg_mask == 3)
        if ang is None:
            r.notes.append("bone_angle: bone mask(s) too small")
        else:
            r.bone_angle_deg = ang
    else:
        r.notes.append("seg_mask not provided")
    return r


def measurement_to_dict(m: UCLMeasurement) -> dict:
    return {
        "ucl_length_px":        m.ucl_length_px,
        "ucl_length_mm":        m.ucl_length_mm,
        "medial_gap_px":        m.medial_gap_px,
        "medial_gap_mm":        m.medial_gap_mm,
        "ucl_thickness_mid_px": m.ucl_thickness_mid_px,
        "ucl_thickness_mid_mm": m.ucl_thickness_mid_mm,
        "bone_angle_deg":       m.bone_angle_deg,
        "px_per_mm":            m.px_per_mm,
        "mask_quality_ucl":     m.mask_quality_ucl,
        "notes":                "; ".join(m.notes),
    }
