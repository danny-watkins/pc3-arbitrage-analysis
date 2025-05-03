# smooth_all_pc_scores.py
# ------------------------
# Author: Danny Watkins
# Date: 2025
# Description:
# Applies Fourier Transform smoothing to PC1_Score, PC2_Score, PC3_Score
# and saves the clean smoothed versions for behavioral analysis.

import pandas as pd
import numpy as np
import scipy.fftpack
import os
import matplotlib.pyplot as plt

# Ensure output directories
os.makedirs("data", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

# Load Rolling PCA Results
rolling_pca = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
rolling_pca.set_index("date", inplace=True)

# Focus only on PC scores
pc_scores = rolling_pca[["PC1_Score", "PC2_Score", "PC3_Score"]]

# Function to apply Fourier smoothing
def fourier_smooth(series, label="PC"):
    raw_signal = series.to_numpy(dtype=float)
    fft_values = scipy.fftpack.fft(raw_signal)
    frequencies = scipy.fftpack.fftfreq(len(raw_signal))

    # Compute power spectrum
    power_spectrum = np.abs(fft_values) ** 2
    cumulative_energy = np.cumsum(power_spectrum) / np.sum(power_spectrum)

    # 85% Variance Cutoff
    optimal_cutoff_index = np.argmax(cumulative_energy >= 0.85)
    optimal_cutoff = abs(frequencies[optimal_cutoff_index])

    # Elbow Cutoff
    elbow_cutoff_index = np.argmax(np.gradient(cumulative_energy) < 0.01)
    elbow_cutoff = abs(frequencies[elbow_cutoff_index])

    # Weighted Cutoff (80% Optimal + 20% Elbow)
    weighted_cutoff = 0.8 * optimal_cutoff + 0.2 * elbow_cutoff

    # Apply cutoff
    filtered_fft = fft_values.copy()
    filtered_fft[np.abs(frequencies) > weighted_cutoff] = 0
    smoothed_signal = np.real(scipy.fftpack.ifft(filtered_fft))

    # Plot original vs smoothed
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, raw_signal, label=f"Raw {label}", color="blue", alpha=0.6)
    plt.plot(series.index, smoothed_signal, label=f"Smoothed {label}", color="green", linewidth=2)
    plt.title(f"Fourier Smoothing: {label}")
    plt.xlabel("Date")
    plt.ylabel("Score Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visuals/fourier_smoothed_{label}.png")
    plt.close()

    return smoothed_signal

# Apply smoothing
smoothed_pc1 = fourier_smooth(pc_scores["PC1_Score"], label="PC1_Score")
smoothed_pc2 = fourier_smooth(pc_scores["PC2_Score"], label="PC2_Score")
smoothed_pc3 = fourier_smooth(pc_scores["PC3_Score"], label="PC3_Score")

# Save smoothed scores to new file
smoothed_df = pd.DataFrame({
    "Date": pc_scores.index,
    "Smoothed_PC1_Score": smoothed_pc1,
    "Smoothed_PC2_Score": smoothed_pc2,
    "Smoothed_PC3_Score": smoothed_pc3
})
smoothed_df.to_csv("data/fourier_smoothed_pc_scores.csv", index=False)

print("✅ Fourier smoothed PC1, PC2, PC3 saved to 'data/fourier_smoothed_pc_scores.csv'")
print("✅ Individual plots saved under 'visuals/' directory")
