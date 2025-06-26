import pandas as pd
import matplotlib.pyplot as plt

# Load your combined CSV
df = pd.read_csv("demo/businessman_with_headset/pywork/combined_scores.csv")  # Update the path as needed

# Plot the speaking scores for Track 0
plt.figure(figsize=(12, 5))
plt.plot(df["Frame Index"], df["Track 0"], label="Track 0", color="royalblue")

plt.xlabel("Frame Index")
plt.ylabel("Speaking Score")
plt.title("Speaking Score Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
