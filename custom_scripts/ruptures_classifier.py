import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- CONFIG ---
csv_path = "demo/businessman_with_headset/pywork/combined_scores.csv"
score_col = "Track 0"
frame_col = "Frame Index"
prominence = 0.1  # Controls how "prominent" a gradient spike must be; adjust if too few/many

# --- Load Data ---
df = pd.read_csv(csv_path)
signal = df[score_col].values
frames = df[frame_col].values

# --- Compute absolute gradient ---
gradient = np.abs(np.diff(signal, prepend=signal[0]))

# --- Find local peaks in gradient (sharp changes) ---
peaks, _ = find_peaks(gradient, prominence=prominence)

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(frames, signal, color="black", label="Score")
plt.plot(frames, gradient, color="skyblue", alpha=0.4, label="|ΔScore|")

# Mark detected gradient jumps
for idx in peaks:
    plt.axvline(frames[idx], color="crimson", linestyle="--", alpha=0.6)

plt.title("Detected Relative Gradient Changes (No Threshold)")
plt.xlabel("Frame Index")
plt.ylabel("Score / Gradient")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print frame indices where big changes occurred ---
print("Detected change points at frames:", frames[peaks])
