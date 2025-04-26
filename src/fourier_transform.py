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
# 6. Smoothed PC3 signal vs. DGS5 yield
# 7. Overlay of DGS5 yield, raw PC3, and smoothed PC3 signal
# 8. Trading Signals Overlayed on All Three Lines
# 9. Trading Signals Overlayed on DGS5 Yield Alone (all normalized)
# 10. Correlation of Slopes: Fourier vs DGS5

import numpy as np
import pandas as pd
import scipy.fftpack
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import zscore, pearsonr

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
filtered_df = pd.DataFrame({"Date": rolling_pca["date"], "Fourier_Signal": smoothed_signal})
filtered_df.to_csv("data/fourier_filtered_signals.csv", index=False)

# Load actual DGS5 yield curve data
yield_curve = pd.read_csv("data/yield_curve_data.csv", parse_dates=["Date"])
yield_curve.set_index("Date", inplace=True)
dgs5_rates = yield_curve["DGS5"].reindex(rolling_pca["date"]).reset_index(drop=True)



