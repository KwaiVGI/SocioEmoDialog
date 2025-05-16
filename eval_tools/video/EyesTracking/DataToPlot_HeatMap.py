#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')    # Force non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# Configuration
CSV_FOLDER   = r".\csv"            # Directory containing input CSV files
VIDEO_FOLDER = r".\input_videos"  # Directory containing videos
OUTPUT_DIR   = r".\heatmaps"       # Where to save heatmap images

BINS   = 100     # Number of grid cells per axis for heatmap
SCALE  = 500     # Multiplier for gaze vector length

LE_IDXS = range(0, 28)   # Left-eye landmark indices
RE_IDXS = range(28, 56)  # Right-eye landmark indices

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')  # Supported video extensions

def find_video_path(basename):
    """Locate a video file by basename and supported extensions."""
    for ext in VIDEO_EXTS:
        p = os.path.join(VIDEO_FOLDER, basename + ext)
        if os.path.isfile(p):
            return p
    return None

def compute_endpoints(df):
    """
    Compute gaze endpoint coordinates for each frame.
    Returns:
        numpy array of shape (num_frames*2, 2) with left and right gaze points.
    """
    pts = []
    for _, r in df.iterrows():
        # Left eye center + scaled gaze vector
        lx = np.mean([r[f"eye_lmk_x_{i}"] for i in LE_IDXS])
        ly = np.mean([r[f"eye_lmk_y_{i}"] for i in LE_IDXS])
        pts.append((lx + r["gaze_0_x"] * SCALE,
                    ly + r["gaze_0_y"] * SCALE))
        # Right eye center + scaled gaze vector
        rx = np.mean([r[f"eye_lmk_x_{i}"] for i in RE_IDXS])
        ry = np.mean([r[f"eye_lmk_y_{i}"] for i in RE_IDXS])
        pts.append((rx + r["gaze_1_x"] * SCALE,
                    ry + r["gaze_1_y"] * SCALE))
    return np.array(pts)

def draw_heatmap(H_norm, xedges, yedges, w_res, h_res, out_path, title, vmax):
    """
    Draw and save a normalized heatmap.
    - H_norm: normalized 2D histogram counts
    - xedges, yedges: histogram bin edges
    - w_res, h_res: original video resolution
    - title: plot title
    - vmax: max value for color scaling
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    pcm = ax.pcolormesh(
        xedges, yedges, H_norm.T,
        cmap='hot', norm=norm, shading='auto'
    )
    fig.colorbar(pcm, ax=ax, label='gaze point ratio')

    # Invert Y axis so origin is top-left
    ax.invert_yaxis()

    # Draw cyan border for video resolution
    bx = [0, w_res, w_res, 0, 0]
    by = [0, 0, h_res, h_res, 0]
    ax.plot(bx, by, color='cyan', linewidth=2, zorder=2)

    ax.set_title(title, fontproperties='SimHei')
    ax.set_xlabel('Screen X pixels', fontproperties='SimHei')
    ax.set_ylabel('Screen Y pixels', fontproperties='SimHei')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    # Prepare output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_paths = glob.glob(os.path.join(CSV_FOLDER, '*.csv'))
    if not csv_paths:
        print("⚠ No CSV files found")
        return

    # First pass: gather all gaze points and find global ranges
    video_data = []  # [(basename, endpoints, width, height), ...]
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')

    for csv_path in csv_paths:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        video_p = find_video_path(base)
        if video_p is None:
            print(f"❌ Video not found for: {base}")
            continue

        # Get video resolution
        cap = cv2.VideoCapture(video_p)
        w_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_res = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        df = pd.read_csv(csv_path).rename(columns=str.strip)
        required = {"gaze_0_x", "gaze_0_y", "gaze_1_x", "gaze_1_y"}
        if not required.issubset(df.columns):
            print(f"❌ Missing columns, skipping: {base}")
            continue

        pts = compute_endpoints(df)
        if pts.size == 0:
            continue

        xs, ys = pts[:,0], pts[:,1]
        xmin = min(xmin, xs.min())
        xmax = max(xmax, xs.max())
        ymin = min(ymin, ys.min())
        ymax = max(ymax, ys.max())

        video_data.append((base, pts, w_res, h_res))
        print(f"{base} collected {len(xs)} gaze points")

    if not video_data:
        print("⚠ No valid gaze data")
        return

    print(f"Global X range: [{xmin:.1f}, {xmax:.1f}], Y range: [{ymin:.1f}, {ymax:.1f}]")

    # Second pass: build individual and global histograms
    H_total = None
    records = []
    range_xy = [[xmin, xmax], [ymin, ymax]]

    for base, pts, w_res, h_res in video_data:
        xs, ys = pts[:,0], pts[:,1]
        H, xedges, yedges = np.histogram2d(xs, ys, bins=BINS, range=range_xy)
        H_total = H.copy() if H_total is None else H_total + H
        total = H.sum()
        H_norm = H / total if total > 0 else H
        records.append((base, H_norm, xedges, yedges, w_res, h_res))

    # Normalize global histogram
    total_all = H_total.sum()
    H_global = H_total / total_all if total_all > 0 else H_total
    print(f"Global total points: {int(total_all)}")

    # Determine unified colorbar maximum
    vmax = max(*(h.max() for (_, h, *_) in records), H_global.max())
    print(f"Unified colorbar vmax = {vmax:.4f}")

    # Generate and save per-video heatmaps
    for base, H_norm, xedges, yedges, w_res, h_res in records:
        out_png = os.path.join(OUTPUT_DIR, f"{base}_heatmap.png")
        print(f"→ Saving: {out_png}")
        title = f"{base} gaze heatmap"
        draw_heatmap(H_norm, xedges, yedges, w_res, h_res, out_png, title, vmax)

    # Generate and save the global heatmap
    out_all = os.path.join(OUTPUT_DIR, "global_heatmap.png")
    print(f"→ Saving global heatmap: {out_all}")
    draw_heatmap(H_global, records[0][2], records[0][3],
                 video_data[0][2], video_data[0][3],
                 out_all, "Global gaze heatmap", vmax)

    print("✅ Done.")

if __name__ == "__main__":
    main()
