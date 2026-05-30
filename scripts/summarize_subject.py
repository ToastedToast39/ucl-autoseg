"""
Cross-session summary for one UCL subject.

Mirrors summarize_participant.py from the patellar reference exactly.
Reads each session's results/measurements.csv, computes per-session
summary stats, and produces a subject-level comparison:

    subjects/<SubjectID>/summary/
        subject_summary.csv       one row per session
        measurements_by_session.png  bar chart of key metrics

Usage:
    python scripts/summarize_subject.py --subject_dir subjects/UCL_001
"""
import argparse, csv
from pathlib import Path
import numpy as np


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    if not rows: return None
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try: cols[k].append(float(v))
            except (ValueError, TypeError): cols[k].append(np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def col(d, *names):
    """First matching column — mirrors summarize_participant.py col()."""
    for n in names:
        for k in d:
            if k == n or k.startswith(n): return d[k]
    return None


def ms(a):
    a = a[~np.isnan(a)] if a is not None else np.array([])
    return (float(np.mean(a)), float(np.std(a))) if a.size else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dir", required=True)
    args = ap.parse_args()
    sdir = Path(args.subject_dir)
    sess_dir = sdir / "sessions"
    if not sess_dir.exists():
        raise SystemExit("No sessions/ folder.")

    summary_dir = sdir / "summary"
    summary_dir.mkdir(exist_ok=True)

    METRICS = [
        ("ucl_length_mm",        "ucl_length"),
        ("medial_gap_mm",        "medial_gap"),
        ("ucl_thickness_mid_mm", "ucl_thickness_mid"),
        ("bone_angle_deg",       "bone_angle"),
    ]

    rows = []
    for tdir in sorted(p for p in sess_dir.iterdir() if p.is_dir()):
        csvp = tdir / "results" / "measurements.csv"
        if not csvp.exists(): continue
        d = read_csv(csvp)
        if d is None: continue
        row = {"session": tdir.name,
               "n_images": int(sum(1 for v in d.get("ucl_length_mm", np.array([]))
                                   if not np.isnan(v)))}
        for colkey, shortname in METRICS:
            arr = col(d, colkey)
            m, s = ms(arr)
            row[f"{shortname}_mean"] = m
            row[f"{shortname}_sd"]   = s
        rows.append(row)

    if not rows:
        print("No analyzed sessions found (run infer.py first)."); return

    keys = list(rows[0].keys())
    with open(summary_dir / "subject_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # bar charts — one per metric, mirroring summarize_participant.py
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sessions = [r["session"] for r in rows]
    x = np.arange(len(sessions))

    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(sessions)*2.5), 8))
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:green"]
    for ax, (colkey, shortname), color in zip(axes.flat, METRICS, colors):
        means = [r.get(f"{shortname}_mean", np.nan) for r in rows]
        sds   = [r.get(f"{shortname}_sd",   np.nan) for r in rows]
        ax.bar(x, means, yerr=sds, capsize=4, color=color, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(sessions, rotation=30, ha="right")
        unit = "deg" if "angle" in colkey else "mm"
        ax.set_ylabel(f"{shortname} ({unit})"); ax.set_title(f"{sdir.name}: {shortname}")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Subject {sdir.name} — measurements by session")
    fig.tight_layout()
    fig.savefig(summary_dir / "measurements_by_session.png", dpi=110)
    plt.close()

    print(f"\n=== {sdir.name}: {len(rows)} sessions ===")
    for r in rows:
        print(f"  {r['session']:20s}  ucl_len={r.get('ucl_length_mean',np.nan):.2f}mm  "
              f"gap={r.get('medial_gap_mean',np.nan):.2f}mm  "
              f"angle={r.get('bone_angle_mean',np.nan):.1f}deg  "
              f"n={r['n_images']}")
    print(f"Summary → {summary_dir}/")


if __name__ == "__main__":
    main()
