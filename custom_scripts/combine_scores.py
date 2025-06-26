import numpy as np
import os
import csv
import glob

# 🔧 Set your path here
video_name = "Test18"
root_path = f"demo/{video_name}/pywork/"
output_csv = os.path.join(root_path, "combined_scores.csv")

# 📦 Find all score files
score_files = sorted(glob.glob(os.path.join(root_path, "scores_track*.npy")))

# 🧠 Load all score arrays
all_scores = [np.load(f) for f in score_files]
max_len = max(len(s) for s in all_scores)

# 🧱 Pad shorter tracks with empty scores (e.g., NaN)
padded_scores = []
for scores in all_scores:
    if len(scores) < max_len:
        scores = np.pad(scores, (0, max_len - len(scores)), mode='constant', constant_values=np.nan)
    padded_scores.append(scores)

# 📊 Stack into matrix [frame x track]
score_matrix = np.stack(padded_scores, axis=1)

# 📝 Write to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Frame Index"] + [f"Track {i}" for i in range(score_matrix.shape[1])]
    writer.writerow(header)

    for frame_idx in range(score_matrix.shape[0]):
        row = [frame_idx] + [round(score_matrix[frame_idx, j], 4) if not np.isnan(score_matrix[frame_idx, j]) else "" for j in range(score_matrix.shape[1])]
        writer.writerow(row)

print(f"✅ Combined scores saved to: {output_csv}")
