"""
Per-class validation Dice for a trained UCL segmentation model.

Near-verbatim port of perclass.py from the patellar reference.
Class names updated for UCL (bg/ucl/bone_humerus/bone_ulna).

Usage:
    python scripts/perclass.py --data _train_seg/ --model models/ucl_seg.pt
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucl.model import UNet
from ucl.data  import UCLSegDataset
from torch.utils.data import DataLoader, random_split

ap = argparse.ArgumentParser()
ap.add_argument("--data",     required=True)
ap.add_argument("--model",    required=True)
ap.add_argument("--resize",   type=int, nargs=2, default=None)
ap.add_argument("--seed",     type=int, default=0)
ap.add_argument("--val_frac", type=float, default=0.2)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.model, map_location=device)
nc = ck.get("num_classes", 4)
model = UNet(in_ch=1, out_ch=nc, base=ck.get("base",32)).to(device)
model.load_state_dict(ck["model"]); model.eval()

resize = tuple(args.resize) if args.resize else ck.get("resize")
ds = UCLSegDataset(args.data, augment=False, resize=resize, num_classes=nc)
n = len(ds); n_val = max(1, int(n*args.val_frac)); n_tr = n - n_val
g = torch.Generator().manual_seed(args.seed)
_, va_idx = random_split(range(n), [n_tr, n_val], generator=g)
val = torch.utils.data.Subset(ds, list(va_idx))
loader = DataLoader(val, batch_size=1)

# UCL class names
NAMES = {1: "humerus", 2: "ulna"}
sums = {c: [] for c in range(1, nc)}

with torch.no_grad():
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        pred = model(img).argmax(1)
        for c in range(1, nc):
            p = (pred==c).float(); t = (msk==c).float()
            if t.sum() > 0:
                d = (2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)
                sums[c].append(d.item())

print(f"\nPer-class validation Dice ({len(val)} val images):")
for c in range(1, nc):
    nm = NAMES.get(c, f"class{c}"); vals = sums[c]
    if vals:
        print(f"  {nm:18s}: {np.mean(vals):.4f}  ({len(vals)} images)")
    else:
        print(f"  {nm:18s}: not present in val set")
