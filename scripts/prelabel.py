"""
Pre-label new UCL images with trained models.

Mirrors scripts/prelabel.py from the patellar reference exactly:
  - predict_labels() → integer label map
  - mask_to_polygon() → approxPolyDP contour
  - write_labelme_json() → .json next to each image

Extended for UCL: also predicts landmark points and writes them as
Labelme "point" shapes in the same JSON, so Labelme shows both
polygon outlines AND point proposals for correction.

Usage:
    python scripts/prelabel.py \
        --seg_model  models/ucl_seg.pt \
        --lm_model   models/ucl_landmarks.pt \
        --images     subjects/UCL_001/sessions/s1/images/ \
        --max_points 40
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucl.model import UNet, HeatmapUNet, heatmap_to_coords
from ucl.data  import pad_to_multiple, unpad, LANDMARK_NAMES, NUM_SEG_CLASSES, SEG_CLASS_MAP

try:
    import cv2; _CV2 = True
except Exception: _CV2 = False

# reverse map: class_id → label name
ID_TO_LABEL = {v: k for k, v in SEG_CLASS_MAP.items()}


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


@torch.no_grad()
def predict_seg(model, gray, device, resize):
    """Verbatim from prelabel.py predict_labels()."""
    H0, W0 = gray.shape
    img = gray.astype(np.float32)/255.0
    if resize:
        rh, rw = resize
        img = np.asarray(Image.fromarray((img*255).astype(np.uint8))
                         .resize((rw,rh), Image.BILINEAR), np.float32)/255.0
    img_p, pad = pad_to_multiple(img, 16)
    t = torch.from_numpy(img_p)[None,None].to(device)
    labels = torch.softmax(model(t), 1)[0].argmax(0).cpu().numpy().astype(np.int32)
    labels = unpad(labels, pad)
    if labels.shape != (H0, W0):
        labels = np.asarray(Image.fromarray(labels.astype(np.uint8))
                            .resize((W0,H0), Image.NEAREST), np.int32)
    return labels


@torch.no_grad()
def predict_lm(model, gray, device, resize, lm_names):
    H0, W0 = gray.shape
    img = gray.astype(np.float32)/255.0; sx = sy = 1.0
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


def largest_component(mask):
    from scipy import ndimage
    mask = mask.astype(bool)
    if not mask.any(): return mask
    lbl, n = ndimage.label(mask)
    if n <= 1: return mask
    sizes = ndimage.sum(mask, lbl, index=np.arange(1, n+1))
    return lbl == int(np.argmax(sizes)) + 1


def mask_to_polygon(mask, max_points=40):
    """Verbatim from prelabel.py mask_to_polygon()."""
    if not _CV2: return None
    mask = largest_component(mask).astype(np.uint8)
    if mask.sum() == 0: return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt  = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    pts  = cv2.approxPolyDP(cnt, 0.003*peri, True).reshape(-1,2)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts)-1, max_points).astype(int)
        pts = pts[idx]
    return pts.astype(float)


def write_labelme_json(img_path, H, W, polygon_shapes, point_shapes):
    """Extended version of write_labelme_json from prelabel.py.
    polygon_shapes: [(label, pts_array)]
    point_shapes:   [(label, (x,y))]
    """
    shapes = []
    for label, pts in polygon_shapes:
        if pts is not None and len(pts) >= 3:
            shapes.append({"label": label, "shape_type": "polygon", "flags": {},
                           "group_id": None,
                           "points": [[float(x),float(y)] for x,y in pts]})
    for label, pt in point_shapes:
        if pt is not None:
            shapes.append({"label": label, "shape_type": "point", "flags": {},
                           "group_id": None,
                           "points": [[float(pt[0]),float(pt[1])]]})
    data = {"version":"5.0.1","flags":{},
            "shapes": shapes,
            "imagePath": Path(img_path).name,
            "imageData": None,
            "imageHeight": int(H), "imageWidth": int(W)}
    out = Path(img_path).with_suffix(".json")
    with open(out,"w") as f: json.dump(data, f, indent=2)
    return out


def main():
    if not _CV2: raise SystemExit("opencv-python required")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_model",  required=True)
    ap.add_argument("--lm_model",   default=None)
    ap.add_argument("--images",     required=True)
    ap.add_argument("--max_points", type=int, default=40)
    ap.add_argument("--overwrite",  action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, seg_resize, nc = load_seg(args.seg_model, device)

    # Only load landmark model if explicitly provided and different from seg model
    lm_model = lm_resize = names = None
    if args.lm_model and args.lm_model != args.seg_model:
        try:
            lm_model, lm_resize, names = load_lm(args.lm_model, device)
        except Exception as e:
            print(f"Warning: could not load landmark model ({e}) — skipping landmarks")
            lm_model = lm_resize = names = None

    print(f"device: {device}  seg_classes: {nc}  landmarks: {names if names else 'none'}")

    img_dir = Path(args.images)
    files   = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.dcm")))
    if not files: raise SystemExit(f"No images in {img_dir}")

    made = skipped = 0
    for fp in files:
        if fp.with_suffix(".json").exists() and not args.overwrite:
            skipped += 1; continue
        if fp.suffix.lower() == ".dcm":
            import pydicom
            ds = pydicom.dcmread(str(fp))
            arr = ds.pixel_array
            if arr.ndim == 3:
                arr = arr.mean(axis=2).astype(np.uint8)
            gray = arr.astype(np.uint8)
        else:
            gray = np.asarray(Image.open(fp).convert("L"))
        H, W = gray.shape
        labels    = predict_seg(seg_model, gray, device, seg_resize)
        landmarks = predict_lm(lm_model, gray, device, lm_resize, names) if lm_model else {}
        polygon_shapes = [(ID_TO_LABEL[cid],
                           mask_to_polygon(labels==cid, args.max_points))
                          for cid in sorted(ID_TO_LABEL)]
        point_shapes   = [(nm, pt) for nm, pt in landmarks.items()]
        write_labelme_json(fp, H, W, polygon_shapes, point_shapes)
        made += 1

    print(f"\nPre-labeled {made} images ({skipped} skipped).")
    print(f"Open '{img_dir}' in Labelme — correct polygons AND landmark points.")


if __name__ == "__main__":
    main()
