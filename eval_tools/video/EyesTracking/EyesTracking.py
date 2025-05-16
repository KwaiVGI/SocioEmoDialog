#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import tempfile
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Configuration
MULTI_EXE         = r".\OpenFace\FaceLandmarkVidMulti.exe"  # OpenFace multi-face tracking executable
INPUT_DIR         = r".\input_videos"                        # Directory of input videos
OUTPUT_DIR        = r".\output_videos"                       # Directory for processed output videos
TEMP_BASE         = r".\temp"                                # Base directory for temporary files
VIDEO_EXTS        = ("*.mp4", "*.avi", "*.mov", "*.mkv")                                             # Supported video extensions

ARROW_SCALE       = 150    # Scale factor for gaze vector arrows
JPEG_QUALITY      = 90     # JPEG quality for extracted frames
EYE_CLOSED_RATIO  = 0.25   # Threshold ratio to determine eye closure based on median eye height

# Landmark index ranges for left/right eye (0–27: left, 28–55: right)
LE_IDXS = range(0, 28)
RE_IDXS = range(28, 56)

# Ensure necessary directories exist
os.makedirs(TEMP_BASE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify that the multi-face tracking executable exists
if not os.path.isfile(MULTI_EXE):
    print("Executable not found:", MULTI_EXE)
    sys.exit(1)

def process_video(video_path):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n=== Processing: {basename} ===")

    # 1) Extract video frames to temporary folder
    tmpdir = tempfile.mkdtemp(prefix=basename + "_", dir=TEMP_BASE)
    frames_dir = os.path.join(tmpdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save each frame as JPEG with specified quality
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(frames_dir, f"f{frame_count:06d}.jpg"), "JPEG", quality=JPEG_QUALITY)
        frame_count += 1
    cap.release()

    if frame_count == 0:
        print("Empty video, skipping")
        shutil.rmtree(tmpdir)
        return

    # 2) Run OpenFace multi-face tracking on extracted frames
    try:
        subprocess.run(
            [MULTI_EXE, "-fdir", frames_dir, "-gaze", "-out_dir", tmpdir],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print("Multi-face tracking failed:", e)
        shutil.rmtree(tmpdir)
        return

    # 3) Read and clean up the generated CSV
    csv_files = glob.glob(os.path.join(tmpdir, "*.csv"))
    if not csv_files:
        print("No CSV output found")
        shutil.rmtree(tmpdir)
        return
    df = pd.read_csv(csv_files[0]).rename(columns=str.strip)

    # Verify required columns exist
    required_cols = {"frame", "face_id", "gaze_0_x", "gaze_0_y", "gaze_1_x", "gaze_1_y"}
    if not required_cols.issubset(df.columns):
        print("CSV missing required columns:", df.columns.tolist())
        shutil.rmtree(tmpdir)
        return

    # Convert types
    df["frame"] = df["frame"].astype(int)
    df["face_id"] = df["face_id"].astype(int)

    # Save the cleaned CSV for this video
    out_csv = os.path.join(OUTPUT_DIR, f"{basename}.csv")
    df.to_csv(out_csv, index=False)
    print("CSV saved:", out_csv)

    # 4) Compute median eye heights and determine closed-eye threshold
    left_heights  = df.apply(lambda r: max(r[f"eye_lmk_y_{i}"] for i in LE_IDXS) -
                                      min(r[f"eye_lmk_y_{i}"] for i in LE_IDXS), axis=1)
    right_heights = df.apply(lambda r: max(r[f"eye_lmk_y_{i}"] for i in RE_IDXS) -
                                      min(r[f"eye_lmk_y_{i}"] for i in RE_IDXS), axis=1)
    df["eye_h_L"] = left_heights
    df["eye_h_R"] = right_heights

    # Compute per-face median eye heights and thresholds
    med_L = df.groupby("face_id")["eye_h_L"].median()
    med_R = df.groupby("face_id")["eye_h_R"].median()
    thr_L = med_L * EYE_CLOSED_RATIO
    thr_R = med_R * EYE_CLOSED_RATIO

    # 5) Calibrate gaze vectors by removing per-face mean bias
    for eye in (0, 1):
        colx, coly = f"gaze_{eye}_x", f"gaze_{eye}_y"
        df[colx] -= df.groupby("face_id")[colx].transform("mean")
        df[coly] -= df.groupby("face_id")[coly].transform("mean")

    # 6) Prepare video writer for overlaying gaze arrows
    cap2 = cv2.VideoCapture(video_path)
    fps = cap2.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap2.release()

    out_video = os.path.join(OUTPUT_DIR, basename + ".mp4")
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Render each frame with gaze arrows
    for f in range(frame_count):
        if f % 100 == 0:
            print(f"  Rendering frame {f+1}/{frame_count}")
        frame_path = os.path.join(frames_dir, f"f{f:06d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Get all detections for this frame
        detections = df[df["frame"] == f]
        for _, r in detections.iterrows():
            fid = r["face_id"]

            # Draw left-eye gaze arrow if eyes are open
            lx = np.mean([r[f"eye_lmk_x_{i}"] for i in LE_IDXS])
            ly = np.mean([r[f"eye_lmk_y_{i}"] for i in LE_IDXS])
            if r["eye_h_L"] >= thr_L.loc[fid]:
                ex0 = int(lx + r["gaze_0_x"] * ARROW_SCALE)
                ey0 = int(ly - r["gaze_0_y"] * ARROW_SCALE)
                cv2.arrowedLine(frame, (int(lx), int(ly)), (ex0, ey0), (0, 0, 255), 2, tipLength=0.2)

            # Draw right-eye gaze arrow if eyes are open
            rx = np.mean([r[f"eye_lmk_x_{i}"] for i in RE_IDXS])
            ry = np.mean([r[f"eye_lmk_y_{i}"] for i in RE_IDXS])
            if r["eye_h_R"] >= thr_R.loc[fid]:
                ex1 = int(rx + r["gaze_1_x"] * ARROW_SCALE)
                ey1 = int(ry - r["gaze_1_y"] * ARROW_SCALE)
                cv2.arrowedLine(frame, (int(rx), int(ry)), (ex1, ey1), (0, 255, 0), 2, tipLength=0.2)

        writer.write(frame)

    writer.release()
    print("Video saved:", out_video)

    # 7) Clean up temporary files
    shutil.rmtree(tmpdir)

def main():
    # Gather all video files from input directory
    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    if not video_files:
        print("⚠ No videos found")
        return

    print(f"Processing {len(video_files)} videos → output at {OUTPUT_DIR}")
    for video in video_files:
        process_video(video)

if __name__ == "__main__":
    main()
