import os
import cv2
import csv
import subprocess
import shutil
import sys

# Configuration
VIDEO_FOLDER      = r".\videos_folder"   # Directory containing input videos
OUTPUT_CSV        = r".\results.csv"      # Path for aggregated results CSV
FEATURE_EXTRACTOR = r".\OpenFace\FeatureExtraction.exe"  # OpenFace CLI executable
TMP_DIR           = r".\openface_tmp"     # Temporary directory for OpenFace outputs
FRAME_INTERVAL    = 20                                                             # Process every Nth frame
MAX_LANDMARKS     = 68                                                             # Number of facial landmarks per frame

def clear_tmp():
    """Remove and recreate the temporary directory for each video."""
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

def process_video(video_path, writer):
    """
    Run OpenFace on a single video, sample its landmark outputs,
    compute coverage, visible area, occlusion confidence, and crop flag,
    then write the per-frame metrics to the CSV writer.
    """
    name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[INFO] Processing video: {name}")

    # 1) Read video resolution
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {name}")
        return
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 2) Clear temporary outputs
    clear_tmp()

    # 3) Call OpenFace FeatureExtraction
    #    - Output 2D landmarks
    #    - Disable MTCNN (use CLNF instead)
    #    - Resize frames by a factor of 2 for detection
    cmd = [
        FEATURE_EXTRACTOR,
        "-f",       video_path,
        "-out_dir", TMP_DIR,
        "-2Dfp",
        "-no_mtcnn",
        "-resize", str(width * 2), str(height * 2)
    ]
    print("  ┗─ Running command:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] OpenFace returned code {proc.returncode}")
        print(proc.stderr, file=sys.stderr)
        return

    # 4) Locate the output CSV (prefer <video_name>.csv)
    candidates = [f for f in os.listdir(TMP_DIR) if f.lower().endswith(".csv")]
    if not candidates:
        print(f"[WARN] No .csv files found in {TMP_DIR}: {os.listdir(TMP_DIR)}")
        return
    feat_csv = os.path.join(TMP_DIR, name + ".csv")
    if not os.path.isfile(feat_csv):
        feat_csv = os.path.join(TMP_DIR, candidates[0])
    print(f"  ┗─ Reading CSV: {feat_csv}")

    # 5) Parse and sample every FRAME_INTERVAL-th frame
    with open(feat_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        any_nonzero = False
        rows = []
        for row in reader:
            frame_no = int(row.get("frame", -1))
            if frame_no < 0 or frame_no % FRAME_INTERVAL != 0:
                continue

            # 5.1) Landmark coverage ratio
            detected = sum(
                1 for i in range(MAX_LANDMARKS)
                if float(row.get(f"x_{i}", 0.0)) > 0 and float(row.get(f"y_{i}", 0.0)) > 0
            )
            coverage = detected / MAX_LANDMARKS

            # 5.2) Visible facial area ratio and crop boundary flag
            xs = [float(row.get(f"x_{i}", 0.0)) for i in range(MAX_LANDMARKS) if float(row.get(f"x_{i}", 0.0)) > 0]
            ys = [float(row.get(f"y_{i}", 0.0)) for i in range(MAX_LANDMARKS) if float(row.get(f"y_{i}", 0.0)) > 0]
            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                area = (x_max - x_min) * (y_max - y_min)
                visible_area = area / (width * height)
                crop_flag = int(x_min <= 1 or y_min <= 1 or x_max >= width - 1 or y_max >= height - 1)
            else:
                visible_area, crop_flag = 0.0, 1

            # 5.3) Occlusion confidence from OpenFace
            occlusion_conf = float(row.get("confidence", 0.0))

            rows.append((frame_no, coverage, visible_area, occlusion_conf, crop_flag))
            if coverage > 0 or visible_area > 0 or occlusion_conf > 0:
                any_nonzero = True

        if not any_nonzero:
            print(f"[WARN] All-zero metrics for {name}; no face detected. Check video quality/lighting.")
        # Write sampled metrics to output CSV
        for frame_no, coverage, visible_area, occlusion_conf, crop_flag in rows:
            writer.writerow([
                name,
                frame_no,
                f"{coverage:.4f}",
                f"{visible_area:.4f}",
                f"{occlusion_conf:.4f}",
                crop_flag
            ])

    # 6) Clean up temporary directory
    clear_tmp()

if __name__ == "__main__":
    # Verify that the OpenFace executable exists
    if not os.path.isfile(FEATURE_EXTRACTOR):
        raise FileNotFoundError(f"FeatureExtraction executable not found: {FEATURE_EXTRACTOR}")

    # Ensure output directory exists, then open CSV for writing
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write CSV header
        writer.writerow([
            "video", "frame",
            "landmark_coverage",
            "visible_area_ratio",
            "occlusion_confidence",
            "crop_flag"
        ])
        # Process each video file in the input directory
        for fn in sorted(os.listdir(VIDEO_FOLDER)):
            if fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                process_video(os.path.join(VIDEO_FOLDER, fn), writer)

    print(f">> All done. Results written to {OUTPUT_CSV}")
