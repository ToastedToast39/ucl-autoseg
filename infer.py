"""
Run both UCL models on a session's images and produce measurements.

Structure mirrors tendon_analysis.py exactly:
  Pass 1: run segmentation model on all images
  Pass 2: run landmark model, apply mask_quality_ok gate,
          compute measurements, MAD-filter outliers, smooth

Outputs (--out):
  measurements.csv      one row per image
  overlays/             QC images: mask tint + landmark dots + measurement text
  <stem>_seg.png        integer label mask per image
  landmarks.json        all predicted landmark coordinates

Usage:
    python scripts/infer.py \
        --seg_model  models/ucl_seg.pt \
        --lm_model   models/ucl_landmarks.pt \
        --images     subjects/UCL_001/sessions/valgus_01/images/ \
        --out        subjects/UCL_001/sessions/valgus_01/results/ \
        --px_per_mm  12.5 --save_overlays
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucl.model   import UNet, HeatmapUNet, heatmap_to_coords
from ucl.data    import pad_to_multiple, unpad, LANDMARK_NAMES, NUM_SEG_CLASSES
from ucl.measure import (measure_ucl, measurement_to_dict,
                          mad_outlier_mask, smooth_series, mask_quality_ok)

try:
    import cv2; _CV2 = True
except Exception: _CV2 = False

PALETTE   = {1: (255,0,255),    # UCL — magenta
             2: (0,165,255),    # humerus — orange
             3: (0,255,0),      # ulna — green
             4: (0,255,255)}    # flexor_pronator — cyan
LM_COLORS = {"ucl_humeral": (255,255,0), "ucl_ulnar": (255,128,0)}


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_seg(path, device):
    ck = torch.load(path, map_location=device)
    nc = ck.get("num_classes", NUM_SEG_CLASSES)
    m  = UNet(in_ch=1, out_ch=nc, base=ck.get("base",32)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("resize"), nc


def load_lm(path, device):
    ck    = torch.load(path, map_location=device)
    n_lm  = ck.get("num_landmarks", len(LANDMARK_NAMES))
    names = ck.get("landmark_names", LANDMARK_NAMES)
    m     = HeatmapUNet(in_ch=1, num_landmarks=n_lm, base=ck.get("base",32)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("resize"), names


# ---------------------------------------------------------------------------
# Inference helpers — mirrors segment_probs / labels_from_probs pattern
# from tendon_analysis.py
# ---------------------------------------------------------------------------

@torch.no_grad()
def seg_probs(model, gray, device, resize):
    """Returns (C,h,w) softmax probs at working res + original (H0,W0)."""
    H0, W0 = gray.shape
    img = gray.astype(np.float32) / 255.0
    if resize:
        rh, rw = resize
        img = np.asarray(Image.fromarray((img*255).astype(np.uint8))
                         .resize((rw,rh), Image.BILINEAR), np.float32)/255.0
    img_p, pad = pad_to_multiple(img, 16)
    t = torch.from_numpy(img_p)[None,None].to(device)
    probs = torch.softmax(model(t), 1)[0].cpu().numpy().astype(np.float32)
    probs = np.stack([unpad(probs[c], pad) for c in range(probs.shape[0])], 0)
    return probs, (H0, W0)


def labels_from_probs(probs, orig_size):
    labels = probs.argmax(0).astype(np.int32)
    H0, W0 = orig_size
    if labels.shape != (H0, W0):
        labels = np.asarray(Image.fromarray(labels.astype(np.uint8))
                            .resize((W0,H0), Image.NEAREST), np.int32)
    return labels


@torch.no_grad()
def run_lm(model, gray, device, resize, lm_names):
    """Returns {name: (x,y) or None} at original image coords."""
    H0, W0 = gray.shape
    img = gray.astype(np.float32)/255.0
    sx = sy = 1.0
    if resize:
        rh, rw = resize; sx, sy = W0/rw, H0/rh
        img = np.asarray(Image.fromarray((img*255).astype(np.uint8))
                         .resize((rw,rh), Image.BILINEAR), np.float32)/255.0
    img_p, pad = pad_to_multiple(img, 16)
    t  = torch.from_numpy(img_p)[None,None].to(device)
    hm = model(t)[0].cpu()
    hm_up = torch.stack([torch.from_numpy(unpad(hm[i].numpy(), pad))
                         for i in range(hm.shape[0])])
    coords = heatmap_to_coords(hm_up)
    return {nm: (coords[i][0]*sx, coords[i][1]*sy) if coords[i] else None
            for i, nm in enumerate(lm_names)}


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def save_overlay(img_path, labels, landmarks, mdict, out_path):
    if not _CV2: return
    bgr  = cv2.cvtColor(np.asarray(Image.open(img_path).convert("RGB")),
                        cv2.COLOR_RGB2BGR)
    tint = np.zeros_like(bgr)
    for cid, col in PALETTE.items():
        tint[labels == cid] = col
    bgr = cv2.addWeighted(bgr, 1.0, tint, 0.3, 0)
    for name, pt in landmarks.items():
        if pt is None: continue
        col = LM_COLORS.get(name, (255,255,255))
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(bgr, (x,y), 6, col, -1)
        cv2.circle(bgr, (x,y), 7, (0,0,0), 1)
        cv2.putText(bgr, name.split("_")[1], (x+8, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    y_txt = 30
    for key in ("ucl_length_mm","medial_gap_mm","ucl_thickness_mid_mm","bone_angle_deg"):
        v = mdict.get(key)
        if v is None: continue
        unit = "deg" if "angle" in key else "mm"
        cv2.putText(bgr, f"{key.replace('_mm','').replace('_deg','')}: {v:.2f}{unit}",
                    (10, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        y_txt += 18
    cv2.imwrite(str(out_path), bgr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_model",    required=True)
    ap.add_argument("--lm_model",     required=True)
    ap.add_argument("--images",       required=True)
    ap.add_argument("--out",          default="results")
    ap.add_argument("--px_per_mm",    type=float, default=None)
    ap.add_argument("--mad",          type=float, default=4.0,
                    help="MAD outlier threshold (0=off). From tendon_analysis.py.")
    ap.add_argument("--smooth",       type=int,   default=5,
                    help="smoothing window for landmark-based measurements (0=off)")
    ap.add_argument("--save_overlays", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.save_overlays: (out/"overlays").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    seg_model, seg_resize, nc   = load_seg(args.seg_model, device)
    lm_model,  lm_resize, names = load_lm(args.lm_model,  device)
    print(f"seg classes: {nc}   landmarks: {names}")

    img_dir = Path(args.images)
    files   = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    if not files: raise SystemExit(f"No images in {img_dir}")
    print(f"Processing {len(files)} images…")

    # ---- Pass 1: segmentation probs for all images --------------------------
    print("pass 1/2: segmenting...")
    all_probs, all_sizes, all_grays = [], [], []
    for fp in files:
        gray = np.asarray(Image.open(fp).convert("L"))
        probs, orig = seg_probs(seg_model, gray, device, seg_resize)
        all_probs.append(probs); all_sizes.append(orig); all_grays.append(gray)

    # ---- Pass 2: landmarks + measurements -----------------------------------
    print("pass 2/2: measuring...")
    rows = []; all_lm = {}; n_rejected = 0
    for i, fp in enumerate(files):
        labels = labels_from_probs(all_probs[i], all_sizes[i])
        H0, W0 = all_sizes[i]
        ucl_mask = labels == 1

        # mask quality gate — ported from tendon_analysis.py
        ok, reason = mask_quality_ok(ucl_mask, float(H0*W0))
        if not ok:
            n_rejected += 1
            rows.append({"image": fp.name, **{k: None for k in
                          ("ucl_length_px","ucl_length_mm","medial_gap_px",
                           "medial_gap_mm","ucl_thickness_mid_px",
                           "ucl_thickness_mid_mm","bone_angle_deg",
                           "px_per_mm","mask_quality_ucl","notes")},
                         "mask_quality_ucl": reason})
            all_lm[fp.stem] = {"landmarks": {}, "measurements": {}}
            continue

        landmarks = run_lm(lm_model, all_grays[i], device, lm_resize, names)
        m         = measure_ucl(landmarks, seg_mask=labels, px_per_mm=args.px_per_mm)
        mdict     = measurement_to_dict(m)

        rows.append({"image": fp.name, **mdict})
        all_lm[fp.stem] = {
            "landmarks":    {k: list(v) if v else None for k,v in landmarks.items()},
            "measurements": mdict,
        }
        # save seg mask
        Image.fromarray(labels.astype(np.uint8)).save(out / f"{fp.stem}_seg.png")
        if args.save_overlays:
            save_overlay(fp, labels, landmarks, mdict,
                         out / "overlays" / f"{fp.stem}_overlay.png")

    # ---- MAD outlier removal on landmark measurements ----------------------
    # mirrors mad_outlier_mask usage in tendon_analysis.py
    MEAS_COLS = ["ucl_length_mm","medial_gap_mm","ucl_thickness_mid_mm","bone_angle_deg"]
    if args.mad and args.mad > 0:
        for col in MEAS_COLS:
            vals = np.array([r.get(col) if r.get(col) is not None else np.nan
                             for r in rows], float)
            bad  = mad_outlier_mask(vals, args.mad)
            for j, r in enumerate(rows):
                if bad[j]: r[col] = None; r.setdefault("notes", ""); r["notes"] += f"{col}_outlier "

    # ---- smooth numeric columns --------------------------------------------
    if args.smooth and args.smooth > 1:
        for col in MEAS_COLS:
            vals = np.array([r.get(col) if r.get(col) is not None else np.nan
                             for r in rows], float)
            sm = smooth_series(vals, args.smooth)
            for j, r in enumerate(rows):
                r[col+"_smooth"] = None if np.isnan(sm[j]) else round(float(sm[j]),4)

    # ---- write outputs ------------------------------------------------------
    if rows:
        keys = list(rows[0].keys())
        with open(out/"measurements.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    with open(out/"landmarks.json","w") as f:
        json.dump(all_lm, f, indent=2)

    # ---- summary (mirrors tendon_analysis.py summary block) ----------------
    valid_rows = [r for r in rows if r.get("mask_quality_ucl") not in (None,"empty","too small","too large","too narrow","wrong aspect (too tall)")]
    print(f"\n=== Summary ({len(valid_rows)} images measured, {n_rejected} rejected by mask gate) ===")
    for col in MEAS_COLS:
        vals = np.array([r.get(col) for r in valid_rows
                         if r.get(col) is not None], float)
        if vals.size:
            unit = "deg" if "angle" in col else ("mm" if args.px_per_mm else "px")
            print(f"  {col:30s}: {np.mean(vals):.3f} +/- {np.std(vals):.3f} {unit}")
    print(f"Results → {out}/")


if __name__ == "__main__":
    main()
