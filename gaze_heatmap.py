"""
Create gaze heatmaps from gaze_recorder.py CSV files.

Usage examples:
    python gaze_heatmap.py recordings/p001_20261001_135646_gaze.csv
    python gaze_heatmap.py recordings/*_gaze.csv                     (one combined heatmap)
    python gaze_heatmap.py recordings/*_gaze.csv --separate          (one heatmap per file)
    python gaze_heatmap.py recordings/p001_..._gaze.csv --background screenshot.png
    python gaze_heatmap.py recordings/p001_..._gaze.csv --start 60 --end 300   (seconds 60-300 only)

Options:
    --background IMG   image to draw the heatmap on (e.g. a screenshot of what was shown)
    --width / --height screen size in pixels (default: background size, else your screen size)
    --sigma PX         blur radius in pixels (default 40). Bigger = smoother
    --start / --end    only use samples within this time range (seconds since recording start)
    --separate         one image per input file instead of one combined image
    --out FILE         output file name (combined mode only)
    --show             also open the result in a window
"""

import argparse
import glob
import os

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle


def screen_size():
    try:
        from screeninfo import get_monitors
        m = get_monitors()[0]
        return m.width, m.height
    except Exception:
        return 1920, 1080


def load_points(path, start=None, end=None):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8",
                         usecols=("t_s", "x", "y"), missing_values="", filling_values=np.nan)
    data = np.atleast_1d(data)
    t, x, y = (data[c].astype(float) for c in ("t_s", "x", "y"))
    keep = ~np.isnan(x) & ~np.isnan(y)
    if start is not None:
        keep &= t >= start
    if end is not None:
        keep &= t <= end
    return x[keep], y[keep]


def heatmap_bounds(x, y, width, height, pad):
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return 0, width, 0, height

    xmin = int(np.floor(min(0, np.min(x[valid]) - pad)))
    xmax = int(np.ceil(max(width, np.max(x[valid]) + pad)))
    ymin = int(np.floor(min(0, np.min(y[valid]) - pad)))
    ymax = int(np.ceil(max(height, np.max(y[valid]) + pad)))
    return xmin, xmax, ymin, ymax


def make_heatmap(x, y, width, height, sigma, bounds):
    xmin, xmax, ymin, ymax = bounds
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    heat, _, _ = np.histogram2d(
        y, x,
        bins=(ymax - ymin, xmax - xmin),
        range=((ymin, ymax), (xmin, xmax)),
    )
    heat = gaussian_filter(heat, sigma)
    if heat.max() > 0:
        heat /= heat.max()

    on_screen = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return heat, int(on_screen.sum()), len(x)


def save_heatmap(heat, out, title, background, width, height, show, bounds):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xmin, xmax, ymin, ymax = bounds
    extent = (xmin, xmax, ymax, ymin)
    dpi = 100

    fig = plt.figure(
        figsize=((xmax - xmin) / dpi, (ymax - ymin) / dpi),
        dpi=dpi
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    if background is not None:
        ax.imshow(background, extent=(0, width, height, 0))
        masked = np.ma.masked_less(heat, 0.02)
        ax.imshow(masked, cmap="jet", alpha=0.55, extent=extent, vmin=0, vmax=1)
    else:
        ax.imshow(heat, cmap="inferno", extent=extent, vmin=0, vmax=1)

    # Outline the original screen, not the expanded heatmap area.
    ax.add_patch(Rectangle(
        (0, 0), width, height,
        linewidth=1, edgecolor="gray", facecolor="none", zorder=5
    ))

    ax.text(10, 30, title, color="white", fontsize=14,
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
            zorder=6)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)

    fig.savefig(out, dpi=dpi)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Gaze heatmaps from gaze_recorder CSV files")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--background")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--sigma", type=float, default=40)
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--separate", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    files = sorted({f for pattern in args.files for f in glob.glob(pattern)})
    if not files:
        ap.error("no matching CSV files found")
    not_csv = [f for f in files if not f.lower().endswith(".csv")]
    if not_csv:
        ap.error(f"not a gaze CSV file: {not_csv[0]}  (images go after --background)")

    background = None
    if args.background:
        from PIL import Image
        background = np.asarray(Image.open(args.background).convert("RGB"))
    if args.width and args.height:
        width, height = args.width, args.height
    elif background is not None:
        height, width = background.shape[:2]
    elif os.path.exists(files[0].replace("_gaze.csv", "_meta.json")):
        import json
        with open(files[0].replace("_gaze.csv", "_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        width, height = meta["screen_width"], meta["screen_height"]
    else:
        width, height = screen_size()

    if background is not None and background.shape[:2] != (height, width):
        from PIL import Image
        background = np.asarray(Image.fromarray(background).resize((width, height)))

    groups = [[f] for f in files] if args.separate else [files]
    for group in groups:
        xs, ys = zip(*(load_points(f, args.start, args.end) for f in group))
        x, y = np.concatenate(xs), np.concatenate(ys)

        bounds = heatmap_bounds(x, y, width, height, pad=3*args.sigma)
        heat, used, total = make_heatmap(x, y, width, height, args.sigma, bounds)

        if len(group) == 1:
            name = os.path.basename(group[0]).replace("_gaze.csv", "")
            out = args.out if (args.out and not args.separate) else os.path.splitext(group[0])[0].replace("_gaze", "") + "_heatmap.png"
        else:
            name = f"{len(group)} recordings combined"
            out = args.out or os.path.join(os.path.dirname(group[0]), "combined_heatmap.png")
            
        print(f"{name}: {used} of {total} samples on screen")
        save_heatmap(heat, out, f"{name}  ({used} samples)", background, width, height, args.show, bounds)


if __name__ == "__main__":
    main()
