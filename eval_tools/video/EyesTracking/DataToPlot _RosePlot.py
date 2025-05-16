import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')    # Force non-interactive backend
import matplotlib.pyplot as plt

# Configuration
CSV_FOLDER = r".\csv"       # Directory containing input CSV files
OUTPUT_DIR = r".\roseplots" # Directory to save output rose plot images
BIN_COUNT  = 36                                                       # Number of angular bins (36 sectors of 10° each)

def compute_theta(df):
    """
    Compute the deflection angle θ for each frame using gaze_angle_x and gaze_angle_y.
    θ = atan2(y, x), mapped to [0, 2π).
    Returns:
        numpy array of θ values in radians.
    """
    ax = df["gaze_angle_x"].to_numpy()
    ay = df["gaze_angle_y"].to_numpy()
    theta = np.arctan2(ay, ax)
    # Map negative angles to [0, 2π)
    theta = np.mod(theta, 2 * np.pi)
    return theta

def draw_rose(theta, ax, title):
    """
    Draw a rose (polar histogram) of angular distribution on the given polar axis.
    """
    counts, bins = np.histogram(theta, bins=BIN_COUNT, range=(0, 2 * np.pi))
    width = 2 * np.pi / BIN_COUNT
    ax.bar(bins[:-1], counts, width=width, bottom=0.0, alpha=0.7, edgecolor='k')
    ax.set_title(title, fontproperties="SimHei")
    ax.set_yticklabels([])  # Hide radial grid labels

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_paths = glob.glob(os.path.join(CSV_FOLDER, '*.csv'))
    if not csv_paths:
        print("⚠ No CSV files found")
        return

    all_theta = []  # Accumulate θ values from all files

    # Single-file rose plots
    for csv_path in csv_paths:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path).rename(columns=str.strip)

        # Check for required columns
        if not {"gaze_angle_x", "gaze_angle_y"}.issubset(df.columns):
            print(f"❌ Missing columns, skipping: {base}")
            continue

        theta = compute_theta(df)
        if theta.size == 0:
            print(f"⚠ No valid data: {base}")
            continue

        all_theta.extend(theta.tolist())

        # Plot and save the rose diagram for this file
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        draw_rose(theta, ax, f"{base} deflection angle distribution")
        plt.tight_layout()

        out_png = os.path.join(OUTPUT_DIR, f"{base}_rose.png")
        plt.savefig(out_png)
        plt.close(fig)
        print(f"→ Saved: {out_png}")

    # Global rose plot for all data
    if all_theta:
        all_theta = np.array(all_theta)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        draw_rose(all_theta, ax, "Global deflection angle distribution")
        plt.tight_layout()

        out_all = os.path.join(OUTPUT_DIR, "global_rose.png")
        plt.savefig(out_all)
        plt.close(fig)
        print(f"→ Saved global rose plot: {out_all}")

    print("✅ All done.")

if __name__ == "__main__":
    main()
