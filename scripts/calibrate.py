"""
Interactive px/mm calibration for UCL ultrasound images.

Verbatim copy of scripts/calibrate.py from the patellar reference.
The ruler-based calibration approach applies regardless of anatomy.

Usage:
    python scripts/calibrate.py --image subjects/UCL_001/sessions/s1/images/img001.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    from PIL import Image
    img = np.array(Image.open(args.image).convert("RGB"))

    print("\nINSTRUCTIONS:")
    print("  1. A window opens with your ultrasound image.")
    print("  2. Click TWO points on the depth ruler (e.g. 0 mm and 40 mm ticks).")
    print("     Use the zoom tool first if needed, then click.")
    print("  3. Close the window once both points are clicked.\n")

    clicks = []
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title("Click two ruler ticks (e.g. 0 mm then 40 mm), then close window")

    def onclick(event):
        if event.inaxes != ax or event.xdata is None: return
        clicks.append((event.xdata, event.ydata))
        n = len(clicks)
        ax.plot(event.xdata, event.ydata, "r+", markersize=15, markeredgewidth=2)
        ax.annotate(f"P{n}", (event.xdata, event.ydata), color="red",
                    fontsize=12, xytext=(8,8), textcoords="offset points")
        fig.canvas.draw()
        print(f"  point {n}: x={event.xdata:.1f}  y={event.ydata:.1f}")
        if n >= 2: print("  Two points captured — close the window to continue.")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(clicks) < 2:
        raise SystemExit("Need two clicks. Run again.")

    (x1,y1),(x2,y2) = clicks[0], clicks[1]
    px_dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    print(f"\nPixel distance: {px_dist:.1f} px")

    mm = input("How many mm apart are those two ticks? (e.g. 40): ").strip()
    try: mm = float(mm)
    except ValueError: raise SystemExit("Not a number.")
    if mm <= 0: raise SystemExit("mm must be positive.")

    px_per_mm = px_dist / mm
    print("\n=========================================")
    print(f"  px_per_mm = {px_per_mm:.3f}")
    print("=========================================")
    print(f"\nSave in subject.json or pass as:")
    print(f"  python scripts/infer.py ... --px_per_mm {px_per_mm:.3f}\n")


if __name__ == "__main__":
    main()
