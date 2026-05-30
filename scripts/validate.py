"""
Validate automated UCL measurements against manual ground truth.

Produces Bland-Altman plots and agreement statistics, similar to the
validation step described in UCL_Project_Guide.md Milestone 4.

Manual CSV format (one row per image):
    image, ucl_length_mm, medial_gap_mm, ucl_thickness_mid_mm, bone_angle_deg

Usage:
    python scripts/validate.py \
        --results  subjects/UCL_001/sessions/s1/results/measurements.csv \
        --manual   manual_measurements.csv \
        --out      validation_report/
"""
import argparse, csv
from pathlib import Path
import numpy as np


def read_csv(path):
    return {r["image"]: r for r in csv.DictReader(open(path))}


def bland_altman(auto, manual, label, out_dir, units):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    a, m  = np.array(auto, float), np.array(manual, float)
    diff  = a - m; mean = (a+m)/2
    bias  = diff.mean(); sd = diff.std()
    lo, hi = bias-1.96*sd, bias+1.96*sd
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(mean, diff, alpha=0.7, s=40)
    ax.axhline(bias, c="red",  lw=2, label=f"Bias {bias:+.3f}")
    ax.axhline(hi,   c="gray", lw=1, ls="--", label=f"+1.96SD {hi:+.3f}")
    ax.axhline(lo,   c="gray", lw=1, ls="--", label=f"-1.96SD {lo:+.3f}")
    ax.set_xlabel(f"Mean ({units})"); ax.set_ylabel(f"Auto−Manual ({units})")
    ax.set_title(f"Bland-Altman: {label}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir/f"ba_{label}.png", dpi=110); plt.close()
    return bias, sd, lo, hi, len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--manual",  required=True)
    ap.add_argument("--out",     default="validation_report")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    auto   = read_csv(args.results)
    manual = read_csv(args.manual)
    common = sorted(set(auto) & set(manual))
    if not common:
        raise SystemExit("No images in common. Check 'image' column values match.")
    print(f"{len(common)} images in common")

    METRICS = [("ucl_length_mm","UCL length","mm"),
               ("medial_gap_mm","Medial gap","mm"),
               ("ucl_thickness_mid_mm","UCL thickness","mm"),
               ("bone_angle_deg","Bone angle","deg")]
    summary = []
    for col, label, units in METRICS:
        pairs = []
        for img in common:
            try:
                pairs.append((float(auto[img].get(col,"")),
                               float(manual[img].get(col,""))))
            except (ValueError, TypeError): pass
        if len(pairs) < 3:
            print(f"  {label}: only {len(pairs)} pairs — skipping"); continue
        a_vals, m_vals = [p[0] for p in pairs], [p[1] for p in pairs]
        r = float(np.corrcoef(a_vals, m_vals)[0,1])
        bias, sd, lo, hi, n = bland_altman(a_vals, m_vals, col, out, units)
        summary.append({"metric":label,"n":n,"bias":round(bias,4),
                         "sd":round(sd,4),"lo_95":round(lo,4),"hi_95":round(hi,4),
                         "pearson_r":round(r,4),"units":units})
        print(f"  {label:26s}: bias={bias:+.3f}  SD={sd:.3f}  "
              f"LoA=[{lo:+.3f},{hi:+.3f}]  r={r:.3f}  n={n}")

    if summary:
        with open(out/"validation_summary.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
        print(f"\nSummary → {out}/validation_summary.csv")
        print(f"BA plots → {out}/ba_*.png")


if __name__ == "__main__":
    main()
