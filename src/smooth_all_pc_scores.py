"""
smooth_all_pc_scores.py
------------------------
Author: Danny Watkins
Date: 2025
Description:
Applies Fourier Transform smoothing to PC1_Score, PC2_Score, and PC3_Score,
and saves the smoothed versions for behavioral analysis and trading.
"""

import pandas as pd
import numpy as np
import scipy.fftpack
import os
import matplotlib.pyplot as plt
import json

# ------------------------ Load Config ------------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
data_dir = config["data_dir"]
visuals_dir = os.path.join(config["visuals_dir"], "fourier")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)

# ------------------------ Load Data ------------------------
rolling_pca = pd.read_csv(f"{data_dir}/rolling_pca_results.csv", parse_dates=["date"])
rolling_pca.set_index("date", inplace=True)
pc_scores = rolling_pca[["PC1_Score", "PC2_Score", "PC3_Score"]]

# ------------------------ Smoothing Function ------------------------
def fourier_smooth(series, label="PC", energy_cutoff=0.85):
    raw_signal = series.to_numpy(dtype=float)
    fft_values = scipy.fftpack.fft(raw_signal)
    frequencies = scipy.fftpack.fftfreq(len(raw_signal))

    power_spectrum = np.abs(fft_values) ** 2
    cumulative_energy = np.cumsum(power_spectrum) / np.sum(power_spectrum)

    # Determine cutoffs
    optimal_cutoff_index = np.argmax(cumulative_energy >= energy_cutoff)
    optimal_cutoff = abs(frequencies[optimal_cutoff_index])

    elbow_cutoff_index = np.argmax(np.gradient(cumulative_energy) < 0.01)
    elbow_cutoff = abs(frequencies[elbow_cutoff_index])

    weighted_cutoff = 0.8 * optimal_cutoff + 0.2 * elbow_cutoff

    # Apply smoothing filter
    filtered_fft = fft_values.copy()
    filtered_fft[np.abs(frequencies) > weighted_cutoff] = 0
    smoothed_signal = np.real(scipy.fftpack.ifft(filtered_fft))

    # Plot raw vs smoothed
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, raw_signal, label=f"Raw {label}", color="blue", alpha=0.6)
    plt.plot(series.index, smoothed_signal, label=f"Smoothed {label}", color="green", linewidth=2)
    plt.title(f"Fourier Smoothing: {label}")
    plt.xlabel("Date")
    plt.ylabel("Score Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{visuals_dir}/fourier_smoothed_{label}.png")
    plt.close()
    print(f"[INFO] Saved plot: {visuals_dir}/fourier_smoothed_{label}.png")

    return smoothed_signal

# ------------------------ Apply to PC Scores ------------------------
print("[INFO] Applying Fourier smoothing to PC scores...")

smoothed_pc1 = fourier_smooth(pc_scores["PC1_Score"], label="PC1_Score")
smoothed_pc2 = fourier_smooth(pc_scores["PC2_Score"], label="PC2_Score")
smoothed_pc3 = fourier_smooth(pc_scores["PC3_Score"], label="PC3_Score")

# ------------------------ Save Output ------------------------
smoothed_df = pd.DataFrame({
    "Date": pc_scores.index,
    "Smoothed_PC1_Score": smoothed_pc1,
    "Smoothed_PC2_Score": smoothed_pc2,
    "Smoothed_PC3_Score": smoothed_pc3
})
smoothed_df.to_csv(f"{data_dir}/fourier_smoothed_pc_scores.csv", index=False)

print(f"[INFO] Fourier-smoothed PC scores saved to '{data_dir}/fourier_smoothed_pc_scores.csv'")
print("[INFO] Done.")
