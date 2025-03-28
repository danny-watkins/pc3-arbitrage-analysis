"""
fourier_transform.py
--------------------
Author: Danny Watkins
Date: 2025

Description:
Applies a Fourier Transform (FFT) to Rolling PCA dislocations 
to detect periodic cycles and filter out noise.

The cutoff frequency is selected **automatically** based on:
1. The Energy Threshold Method (85% variance retention).
2. The Elbow Point Method (where adding more frequencies gives diminishing returns).
"""

import numpy as np
import pandas as pd
import scipy.fftpack
import matplotlib.pyplot as plt
import json

# Load the best maturity from config.json
with open("config.json", "r") as f:
    config = json.load(f)

best_maturity = config["belly_maturity"]  # e.g., "PC3_DGS5"
print(f"🔍 Using Belly Maturity for Fourier Transform: {best_maturity}")

# Load Rolling PCA Results (ensure this is the 10 years of rolling PCA data)
rolling_pca = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])

# Select the PC3 value series for the dynamically selected maturity
pc3_series = rolling_pca[best_maturity].to_numpy(dtype=float)

# Apply FFT (Fast Fourier Transform) to the PC3 value series
fft_values = scipy.fftpack.fft(pc3_series)
frequencies = scipy.fftpack.fftfreq(len(pc3_series))

# Compute the power spectrum (magnitude squared of FFT coefficients)
power_spectrum = np.abs(fft_values)**2
cumulative_energy = np.cumsum(power_spectrum) / np.sum(power_spectrum)  # Normalize

# Step 1️⃣: Find the Energy-Based Cutoff Frequency (85% variance retained)
optimal_cutoff_index = np.argmax(cumulative_energy >= 0.85)  # Find first index reaching 85%
optimal_cutoff = abs(frequencies[optimal_cutoff_index])

# Step 2️⃣: Find the Elbow Point (Where Variance Gains Drop Off)
elbow_cutoff_index = np.argmax(np.gradient(cumulative_energy) < 0.01)  # First drop below 1% gain
elbow_cutoff = abs(frequencies[elbow_cutoff_index])

# Step 3️⃣: Choose the More Conservative Cutoff
final_cutoff = min(optimal_cutoff, elbow_cutoff)
print(f"🔍 Selected Fourier Cutoff: {final_cutoff:.5f} (Automatically Determined)")

# Apply the cutoff: Zero out high-frequency components (filtering out noise)
filtered_fft = fft_values.copy()
filtered_fft[np.abs(frequencies) > final_cutoff] = 0

# Reconstruct the filtered (smoothed) signal using Inverse FFT
filtered_signal = np.real(scipy.fftpack.ifft(filtered_fft))

# Store the filtered data in a DataFrame for clarity
filtered_df = pd.DataFrame({"Date": rolling_pca["date"], best_maturity: filtered_signal})
filtered_df.to_csv("data/fourier_filtered_signals.csv", index=False)

# Visualization of the Fourier Smoothed Signal
plt.figure(figsize=(12, 6))
plt.plot(rolling_pca["date"], pc3_series, label="Original PC3 Series", alpha=0.5, color="gray")
plt.plot(rolling_pca["date"], filtered_signal, label="Fourier-Smoothed Signal", color="blue")
plt.title(f"PC3 Smoothed Signal - Original vs. Fourier Filtered ({best_maturity})")
plt.xlabel("Date")
plt.ylabel("PC3 Value")
plt.legend()
plt.grid()
plt.savefig("visuals/pc3_smoothed_signal.png")
plt.show()

print("✅ Fourier Transform applied and visuals saved to 'visuals/' folder.")
