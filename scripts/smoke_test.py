"""
Quick self-test — no trained model or GPU required.

Mirrors scripts/smoke_test.py from the patellar reference.
Tests:
  1. pad/unpad round-trip (from data.py)
  2. mask_quality_ok — rejects empty, tiny, wrong-aspect masks
  3. point_distance geometry
  4. make_gaussian_heatmaps + heatmap_to_coords round-trip
  5. UCLMeasurement with synthetic data

Run:  python scripts/smoke_test.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucl.data    import pad_to_multiple, unpad
from ucl.measure import mask_quality_ok, point_distance

try:
    import torch
    from ucl.model import make_gaussian_heatmaps, heatmap_to_coords
    _TORCH = True
except Exception:
    _TORCH = False


def ok(cond, msg):
    status = "OK" if cond else "XX"
    print(f"[{status}] {msg}")
    return cond


def main():
    passed = True

    # 1. pad/unpad
    arr = np.zeros((660, 1032))
    p, pad = pad_to_multiple(arr, 16)
    back = unpad(p, pad)
    passed &= ok(p.shape[0]%16==0 and p.shape[1]%16==0 and back.shape==arr.shape,
                 f"pad/unpad 660x1032 → {p.shape} → {back.shape}")

    # 2. mask_quality_ok
    H, W = 400, 600; area = H*W
    empty = np.zeros((H, W), bool)
    passed &= ok(not mask_quality_ok(empty, area)[0], "mask_quality_ok rejects empty mask")

    tiny = np.zeros((H, W), bool); tiny[200:205, 300:305] = True
    passed &= ok(not mask_quality_ok(tiny, area)[0], "mask_quality_ok rejects tiny mask")

    tall = np.zeros((H, W), bool); tall[100:350, 295:305] = True
    passed &= ok(not mask_quality_ok(tall, area)[0], "mask_quality_ok rejects tall-aspect mask")

    good = np.zeros((H, W), bool); good[180:220, 100:500] = True
    passed &= ok(mask_quality_ok(good, area)[0], "mask_quality_ok accepts ligament-shaped mask")

    # 3. point_distance
    d = point_distance((0, 0), (3, 4))
    passed &= ok(abs(d - 5.0) < 1e-6, f"point_distance (3-4-5 triangle) = {d:.4f}")

    # 4. Gaussian heatmap round-trip
    if _TORCH:
        lms = [(50.0, 30.0), None, (200.0, 150.0), (10.0, 10.0)]
        hm = make_gaussian_heatmaps(lms, 320, 512, sigma=8.0)
        coords = heatmap_to_coords(hm)
        err0 = abs(coords[0][0]-50) + abs(coords[0][1]-30) if coords[0] else 99
        err2 = abs(coords[2][0]-200) + abs(coords[2][1]-150) if coords[2] else 99
        passed &= ok(coords[1] is None, "heatmap_to_coords returns None for absent landmark")
        passed &= ok(err0 < 2 and err2 < 2,
                     f"heatmap round-trip errors: {err0:.1f}px, {err2:.1f}px")
    else:
        print("[--] torch not available; skipping heatmap round-trip test")

    # 5. UCLMeasurement with synthetic landmarks + mask
    from ucl.measure import measure_ucl
    lm_dict = {"ucl_humeral": (100.0, 50.0), "ucl_ulnar": (200.0, 50.0),
               "gap_humerus": (130.0, 60.0), "gap_ulna":  (130.0, 80.0)}
    seg = np.zeros((300, 400), np.int32)
    seg[40:60, 90:210] = 1   # UCL band
    seg[65:80, 90:210] = 2   # humerus
    seg[85:95, 90:210] = 3   # ulna
    m = measure_ucl(lm_dict, seg_mask=seg, px_per_mm=10.0)
    passed &= ok(m.ucl_length_mm is not None and abs(m.ucl_length_mm - 10.0) < 0.5,
                 f"ucl_length_mm = {m.ucl_length_mm}")
    passed &= ok(m.medial_gap_mm is not None and abs(m.medial_gap_mm - 2.0) < 0.5,
                 f"medial_gap_mm = {m.medial_gap_mm}")
    passed &= ok(m.ucl_thickness_mid_mm is not None,
                 f"ucl_thickness_mid_mm = {m.ucl_thickness_mid_mm}")

    print("\nALL PASSED" if passed else "\nSOME FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
