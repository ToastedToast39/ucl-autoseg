"""
Train the UCL landmark localization model (heatmap regression).

Same structure as train_seg.py / scripts/train.py from patellar reference.
Validation metric: mean pixel error (lower = better) instead of Dice.

Usage:
    python scripts/train_landmarks.py --data _train_lm/ --epochs 80 \
        --resize 320 512 --out models/ucl_landmarks.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucl.model import HeatmapUNet, heatmap_mse_loss, heatmap_to_coords
from ucl.data  import UCLLandmarkDataset, LANDMARK_NAMES


@torch.no_grad()
def val_mean_error(model, loader, device):
    model.eval(); errors = []
    for img, target_hm, visible in loader:
        img = img.to(device)
        pred_hm   = model(img)[0].cpu()
        target_hm = target_hm[0]; vis = visible[0]
        pp = heatmap_to_coords(pred_hm)
        tp = heatmap_to_coords(target_hm)
        for i in range(len(pp)):
            if not vis[i].item(): continue
            p, t = pp[i], tp[i]
            if p is None or t is None:
                errors.append(50.0); continue
            errors.append(float(np.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2)))
    return float(np.mean(errors)) if errors else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     required=True)
    ap.add_argument("--epochs",   type=int,   default=80)
    ap.add_argument("--batch",    type=int,   default=4)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--base",     type=int,   default=32)
    ap.add_argument("--sigma",    type=float, default=8.0,
                    help="Gaussian blob std dev in pixels")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--resize",   type=int,   nargs=2, default=None,
                    metavar=("H", "W"))
    ap.add_argument("--out",      default="models/ucl_landmarks.pt")
    ap.add_argument("--seed",     type=int,   default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  landmarks: {LANDMARK_NAMES}")

    resize = tuple(args.resize) if args.resize else None
    n_lm   = len(LANDMARK_NAMES)
    full_tr = UCLLandmarkDataset(args.data, augment=True,  resize=resize, sigma=args.sigma)
    full_va = UCLLandmarkDataset(args.data, augment=False, resize=resize, sigma=args.sigma)

    n = len(full_tr); n_val = max(1, int(n*args.val_frac)); n_tr = n - n_val
    g = torch.Generator().manual_seed(args.seed)
    ti, vi = random_split(range(n), [n_tr, n_val], generator=g)
    tr = torch.utils.data.Subset(full_tr, list(ti))
    va = torch.utils.data.Subset(full_va, list(vi))
    print(f"train {len(tr)}  val {len(va)}")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,  num_workers=2)
    vl = DataLoader(va, batch_size=1,          shuffle=False, num_workers=2)

    model = HeatmapUNet(in_ch=1, num_landmarks=n_lm, base=args.base).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(1, args.epochs+1):
        model.train(); losses = []
        for img, hm, vis in tl:
            img, hm, vis = img.to(device), hm.to(device), vis.to(device)
            opt.zero_grad()
            loss = heatmap_mse_loss(model(img), hm, vis)
            loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        err = val_mean_error(model, vl, device)
        print(f"epoch {ep:3d}  loss {np.mean(losses):.5f}  val_err {err:.2f} px")
        if err < best:
            best = err
            torch.save({"model": model.state_dict(), "base": args.base,
                        "num_landmarks": n_lm, "landmark_names": LANDMARK_NAMES,
                        "resize": resize, "sigma": args.sigma,
                        "val_mean_err_px": best, "task": "landmarks"}, args.out)
            print(f"  saved → {args.out}  (err {best:.2f} px)")
    print(f"\ndone. best val mean error {best:.2f} px")


if __name__ == "__main__":
    main()
