import os
import glob
import pandas as pd
import numpy as np

# Configuration
INPUT_CSV_DIR = r".\csv_outputs"  # Directory with input CSV files
OUTPUT_DIR    = r".\csv_by_person"  # Directory for per-person CSV outputs
LE_IDXS = range(0, 28)   # Indices for left-eye landmarks (0–27)
RE_IDXS = range(28, 56)  # Indices for right-eye landmarks (28–55)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each CSV in the input directory
for csv_path in glob.glob(os.path.join(INPUT_CSV_DIR, "*.csv")):
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Compute x-coordinate of left-eye center for each frame
    df['eye_left_cx']  = df[[f'eye_lmk_x_{i}' for i in LE_IDXS]].mean(axis=1)
    # Compute x-coordinate of right-eye center for each frame
    df['eye_right_cx'] = df[[f'eye_lmk_x_{i}' for i in RE_IDXS]].mean(axis=1)
    # Compute the midpoint between the two eye centers
    df['eye_mid_x']    = (df['eye_left_cx'] + df['eye_right_cx']) / 2

    # Calculate median midpoint x for each face_id
    medians = df.groupby('face_id')['eye_mid_x'].median()
    # Sort face IDs by horizontal position: leftmost as person1, next as person2
    sorted_ids = medians.sort_values().index.tolist()

    # Skip files with fewer than two tracked faces
    if len(sorted_ids) < 2:
        print(f"⚠ Only {len(sorted_ids)} track(s) detected in {basename}.csv – skipping")
        continue

    # Assign face IDs to person1 and person2 based on left-right order
    left_id, right_id = sorted_ids[0], sorted_ids[1]

    # Split the DataFrame into two per-person DataFrames
    df_left  = df[df['face_id'] == left_id].drop(columns=['eye_left_cx', 'eye_right_cx', 'eye_mid_x'])
    df_right = df[df['face_id'] == right_id].drop(columns=['eye_left_cx', 'eye_right_cx', 'eye_mid_x'])

    # Define output paths
    out1 = os.path.join(OUTPUT_DIR, f"{basename}_person1.csv")
    out2 = os.path.join(OUTPUT_DIR, f"{basename}_person2.csv")

    # Save per-person CSVs
    df_left.to_csv(out1, index=False)
    df_right.to_csv(out2, index=False)

    print(f"✔ Generated: {os.path.basename(out1)}, {os.path.basename(out2)}")
