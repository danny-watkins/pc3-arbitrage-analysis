# fourier_transform.py
# --------------------
# Author: Danny Watkins
# Date: 2025

# Description:
# Applies a Fourier Transform (FFT) to Rolling PCA dislocations 
# to detect periodic cycles and filter out noise.

# Generates separate visualizations for each cutoff method:
# 1. Elbow Point Method
# 2. 85% Variance Retention Method
# 3. Combined 80/20 Weighted Signal
# 4. Overlay of raw signal with combined smoothed signal
# 5. Energy Spectrum Visualization with Cutoff Lines (Zoomed)

import numpy as np
import pandas as pd
import scipy.fftpack
import matplotlib.pyplot as plt
import json
import os

# Ensure visuals directory exists
os.makedirs("visuals", exist_ok=True)

# Load the best maturity from config.json
with open("config.json", "r") as f:
    config = json.load(f)

best_maturity = config["belly_maturity"]  # e.g., "PC3_DGS5"
print(f"Using Belly Maturity for Fourier Transform: {best_maturity}")

# Load Rolling PCA Results
rolling_pca = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])

# Select the PC3 value series for the dynamically selected maturity
pc3_series = rolling_pca[best_maturity].to_numpy(dtype=float)

# Apply FFT (Fast Fourier Transform) to the PC3 value series
fft_values = scipy.fftpack.fft(pc3_series)
frequencies = scipy.fftpack.fftfreq(len(pc3_series))

# Compute the power spectrum (magnitude squared of FFT coefficients)
power_spectrum = np.abs(fft_values)**2
cumulative_energy = np.cumsum(power_spectrum) / np.sum(power_spectrum)  # Normalize

# Step 1️⃣: Energy-Based Cutoff (85% Variance Retention)
optimal_cutoff_index = np.argmax(cumulative_energy >= 0.85)
optimal_cutoff = abs(frequencies[optimal_cutoff_index])

# Step 2️⃣: Elbow Point Method (Diminishing Returns)
elbow_cutoff_index = np.argmax(np.gradient(cumulative_energy) < 0.01)
elbow_cutoff = abs(frequencies[elbow_cutoff_index])

print(f"🔍 85% Variance Cutoff: {optimal_cutoff:.5f}")
print(f"🔍 Elbow Point Cutoff: {elbow_cutoff:.5f}")

# Step 3️⃣: Combined Weighted Cutoff (80/20)
weighted_cutoff = 0.8 * optimal_cutoff + 0.2 * elbow_cutoff
weighted_cutoff_index = np.argmax(frequencies >= weighted_cutoff)
print(f"🔍 Weighted Cutoff (80/20): {weighted_cutoff:.5f}")

# Visualization: Energy Spectrum with Weighted Cutoff Line (Zoomed)
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], power_spectrum[:len(frequencies)//2], label="Power Spectrum", color="black")
plt.axvline(x=weighted_cutoff, color="purple", linestyle="--", label="Weighted Cutoff (80/20)")
plt.fill_between(frequencies[:weighted_cutoff_index], power_spectrum[:weighted_cutoff_index], color="lightgreen", alpha=0.5)
plt.fill_between(frequencies[weighted_cutoff_index:], power_spectrum[weighted_cutoff_index:], color="red", alpha=0.5)
plt.title("Energy Spectrum with Weighted Cutoff (Zoomed)")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.xlim(0, 0.05)
plt.legend()
plt.grid(True)
plt.savefig("visuals/energy_spectrum_zoomed.png")
plt.close()

# Apply the weighted cutoff to smooth the signal
filtered_fft = fft_values.copy()
filtered_fft[np.abs(frequencies) > weighted_cutoff] = 0
smoothed_signal = np.real(scipy.fftpack.ifft(filtered_fft))

# Save the smoothed signal for trading signals
filtered_df = pd.DataFrame({"Date": rolling_pca["date"], best_maturity: smoothed_signal})
filtered_df.to_csv("data/fourier_filtered_signals.csv", index=False)

# Visualization: Overlay Raw Signal with Smoothed Signal
plt.figure(figsize=(12, 6))
plt.plot(rolling_pca["date"], pc3_series, label="Raw Signal", color="gray", alpha=0.6)
plt.plot(rolling_pca["date"], smoothed_signal, label="Weighted Smoothed Signal (80/20)", color="purple", linestyle="--")
plt.title("Overlay of Raw Signal with Smoothed Signal (80/20)")
plt.xlabel("Date")
plt.ylabel("PC3 Value")
plt.legend()
plt.grid(True)
plt.savefig("visuals/overlay_smoothed_signal.png")
plt.close()

print("✅ Fourier Transform applied and visuals saved to 'visuals/' folder.")

