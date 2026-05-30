"""
Train the UCL segmentation U-Net.

Near-verbatim copy of scripts/train.py from the patellar reference.
Only differences:
  - uses UCLSegDataset instead of TendonSegDataset
  - default --num_classes 4 (bg + ucl + bone_humerus + bone_ulna)
  - checkpoint tagged with task="segmentation"

Usage:
    python scripts/train_seg.py --data _train_seg/ --epochs 80 \
        --resize 320 512 --out models/ucl_seg.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucl.model import UNet, dice_ce_loss
from ucl.data  import UCLSegDataset, NUM_SEG_CLASSES


@torch.no_grad()
def val_dice(model, loader, device, nc):
    """Mean foreground Dice — identical to train.py val_dice."""
    model.eval(); per_class = []
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        pred = model(img).argmax(1)
        dices = []
        for c in range(1, nc):
            p = (pred == c).float(); t = (msk == c).float()
            if t.sum() > 0:
                dices.append(((2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)).item())
        if dices: per_class.append(np.mean(dices))
    return float(np.mean(per_class)) if per_class else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        required=True)
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch",       type=int,   default=4)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--base",        type=int,   default=32)
    ap.add_argument("--num_classes", type=int,   default=NUM_SEG_CLASSES)
    ap.add_argument("--val_frac",    type=float, default=0.2)
    ap.add_argument("--resize",      type=int,   nargs=2, default=None,
                    metavar=("H", "W"))
    ap.add_argument("--out",         default="models/ucl_seg.pt")
    ap.add_argument("--seed",        type=int,   default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    resize = tuple(args.resize) if args.resize else None
    nc     = args.num_classes
    full_tr = UCLSegDataset(args.data, augment=True,  resize=resize, num_classes=nc)
    full_va = UCLSegDataset(args.data, augment=False, resize=resize, num_classes=nc)

    n = len(full_tr); n_val = max(1, int(n * args.val_frac)); n_tr = n - n_val
    g = torch.Generator().manual_seed(args.seed)
    ti, vi = random_split(range(n), [n_tr, n_val], generator=g)
    tr = torch.utils.data.Subset(full_tr, list(ti))
    va = torch.utils.data.Subset(full_va, list(vi))
    print(f"train {len(tr)}  val {len(va)}  classes {nc}")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,  num_workers=2)
    vl = DataLoader(va, batch_size=1,          shuffle=False, num_workers=2)

    model = UNet(in_ch=1, out_ch=nc, base=args.base).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs+1):
        model.train(); losses = []
        for img, msk in tl:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            loss = dice_ce_loss(model(img), msk, num_classes=nc)
            loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        d = val_dice(model, vl, device, nc)
        print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  val_dice {d:.4f}")
        if d > best:
            best = d
            torch.save({"model": model.state_dict(), "base": args.base,
                        "num_classes": nc, "resize": resize,
                        "val_dice": best, "task": "segmentation"}, args.out)
            print(f"  saved → {args.out}  (dice {best:.4f})")
    print(f"\ndone. best val dice {best:.4f}")


if __name__ == "__main__":
    main()
