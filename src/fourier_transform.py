"""
fourier_transform.py
--------------------
Author: Danny Watkins
Updated: 2025
Description:
Applies Fourier Transform smoothing to the full PC3_Score (curvature factor)
to prepare for trade entry and exit detection.
Also saves raw PC3 signal and a standalone plot for reference.
"""

import numpy as np
import pandas as pd
import scipy.fftpack
import matplotlib.pyplot as plt
import os
import json

# -------------------- Load Configuration --------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
data_dir = config.get("data_dir", "data")
visuals_dir = os.path.join(config.get("visuals_dir", "visuals"), "smoothing")
signals_dir = os.path.join(config.get("visuals_dir", "visuals"), "signals")

# -------------------- Ensure output directories --------------------
os.makedirs(data_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)
os.makedirs(signals_dir, exist_ok=True)

# -------------------- Load Rolling PCA Results --------------------
rolling_pca = pd.read_csv(os.path.join(data_dir, "rolling_pca_results.csv"), parse_dates=["date"])

# -------------------- Focus only on PC3_Score --------------------
pc3_series = rolling_pca["PC3_Score"].to_numpy(dtype=float)

# -------------------- Apply FFT --------------------
fft_values = scipy.fftpack.fft(pc3_series)
frequencies = scipy.fftpack.fftfreq(len(pc3_series))

# -------------------- Power Spectrum --------------------
power_spectrum = np.abs(fft_values) ** 2
cumulative_energy = np.cumsum(power_spectrum) / np.sum(power_spectrum)

# Step 1: 85% Variance Retention
optimal_cutoff_index = np.argmax(cumulative_energy >= 0.85)
optimal_cutoff = abs(frequencies[optimal_cutoff_index])

# Step 2: Elbow Point
elbow_cutoff_index = np.argmax(np.gradient(cumulative_energy) < 0.01)
elbow_cutoff = abs(frequencies[elbow_cutoff_index])

# Step 3: Weighted Cutoff
weighted_cutoff = 0.8 * optimal_cutoff + 0.2 * elbow_cutoff
print(f"85% Variance Cutoff: {optimal_cutoff:.5f}")
print(f"Elbow Point Cutoff: {elbow_cutoff:.5f}")
print(f"Weighted Cutoff (80/20): {weighted_cutoff:.5f}")

# -------------------- Apply Cutoff --------------------
filtered_fft = fft_values.copy()
filtered_fft[np.abs(frequencies) > weighted_cutoff] = 0
smoothed_signal = np.real(scipy.fftpack.ifft(filtered_fft))

# -------------------- Save Smoothed Output --------------------
filtered_df = pd.DataFrame({
    "Date": rolling_pca["date"],
    "PC3_Score": pc3_series,
    "Smoothed_PC3_Score": smoothed_signal
})
filtered_df.to_csv(os.path.join(data_dir, "fourier_filtered_signals.csv"), index=False)
print(f"[OK] Smoothed PC3 signal saved to '{data_dir}/fourier_filtered_signals.csv'")

# -------------------- Save Raw PC3 Signal for Visuals --------------------
raw_signal_df = pd.DataFrame({
    "Date": rolling_pca["date"],
    "PC3_Score": pc3_series
})
raw_signal_df.to_csv(os.path.join(signals_dir, "raw_pc3_signal.csv"), index=False)
print(f"[OK] Raw PC3 signal saved to '{signals_dir}/raw_pc3_signal.csv'")

# -------------------- Comparison Visualization --------------------
plt.figure(figsize=(12, 6))
plt.plot(rolling_pca["date"], pc3_series, label="Raw PC3_Score", color="blue", alpha=0.6)
plt.plot(rolling_pca["date"], smoothed_signal, label="Smoothed PC3_Score (Fourier)", color="green", linewidth=2)
plt.title("PC3 Score vs Fourier Smoothed PC3 Score")
plt.xlabel("Date")
plt.ylabel("PC3 Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(visuals_dir, "pc3_vs_smoothed_pc3.png")
plt.savefig(plot_path)
plt.close()
print(f"[OK] Fourier smoothing visualization saved to '{plot_path}'")

# -------------------- Raw PC3 Signal Only Plot --------------------
plt.figure(figsize=(12, 5))
plt.plot(rolling_pca["date"], pc3_series, color="blue", linewidth=1.5)
plt.title("Raw PC3 Score Over Time")
plt.xlabel("Date")
plt.ylabel("PC3 Score")
plt.grid(True)
plt.tight_layout()
raw_plot_path = os.path.join(signals_dir, "raw_pc3_signal_plot.png")
plt.savefig(raw_plot_path)
plt.close()
print(f"[OK] Raw PC3 signal plot saved to '{raw_plot_path}'")

# -------------------- Energy Spectrum Zoomed-In Visualization --------------------
plt.figure(figsize=(10, 5))

# Sort frequency and power arrays by absolute frequency
abs_freq = np.abs(frequencies)
sorted_indices = np.argsort(abs_freq)
abs_freq_sorted = abs_freq[sorted_indices]
power_sorted = power_spectrum[sorted_indices]

# Plot the full power spectrum
plt.plot(abs_freq_sorted, power_sorted, label="Power Spectrum", color="black", alpha=0.6)

# Fill area under retained frequencies (green)
plt.fill_between(abs_freq_sorted, power_sorted, where=(abs_freq_sorted <= weighted_cutoff),
                 color="green", alpha=0.3, label="Retained Frequencies")

# Fill area under filtered frequencies (red)
plt.fill_between(abs_freq_sorted, power_sorted, where=(abs_freq_sorted > weighted_cutoff),
                 color="red", alpha=0.2, label="Filtered Out Frequencies")

# Add cutoff line
plt.axvline(x=weighted_cutoff, color="black", linestyle="--", linewidth=2, label="Weighted Cutoff (80/20)")

plt.title("Energy Spectrum with Weighted Cutoff (Zoomed) for PC3 Score")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.xlim(0, 0.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
spectrum_path = os.path.join(visuals_dir, "energy_spectrum_zoomed.png")
plt.savefig(spectrum_path)
plt.close()

print(f"[OK] Energy spectrum visualization saved to '{spectrum_path}'")
