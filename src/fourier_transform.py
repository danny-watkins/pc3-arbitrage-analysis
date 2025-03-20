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

# Select the dislocation series for the dynamically selected maturity
dislocation_series = rolling_pca[best_maturity].to_numpy(dtype=float)

# Apply FFT (Fast Fourier Transform) to the dislocation series
fft_values = scipy.fftpack.fft(dislocation_series)
frequencies = scipy.fftpack.fftfreq(len(dislocation_series))

# Compute the power spectrum (magnitude squared of FFT coefficients)
power_spectrum = np.abs(fft_values)**2
cumulative_energy = np.cumsum(power_spectrum) / np.sum(power_spectrum)  # Normalize

# Step 1️⃣: **Find the Energy-Based Cutoff Frequency (85% variance retained)**
optimal_cutoff_index = np.argmax(cumulative_energy >= 0.85)  # Find first index reaching 85%
optimal_cutoff = abs(frequencies[optimal_cutoff_index])

# Step 2️⃣: **Find the Elbow Point (Where Variance Gains Drop Off)**
elbow_cutoff_index = np.argmax(np.gradient(cumulative_energy) < 0.01)  # First drop below 1% gain
elbow_cutoff = abs(frequencies[elbow_cutoff_index])

# Step 3️⃣: **Choose the More Conservative Cutoff**
final_cutoff = min(optimal_cutoff, elbow_cutoff)  # Use the **lower** of the two

print(f"🔍 Selected Fourier Cutoff: {final_cutoff:.5f} (Automatically Determined)")

# Apply the cutoff: Zero out high-frequency components (filtering out noise)
filtered_fft = fft_values.copy()
filtered_fft[np.abs(frequencies) > final_cutoff] = 0  # Zero out high frequencies

# Reconstruct the filtered (smoothed) signal using Inverse FFT
filtered_signal = np.real(scipy.fftpack.ifft(filtered_fft))

# Store the filtered data in a DataFrame for clarity
filtered_df = pd.DataFrame({"Date": rolling_pca["date"], best_maturity: filtered_signal})

# Save the filtered output to a new CSV file
filtered_df.to_csv("data/fourier_filtered_signals.csv", index=False)

# Energy Analysis: Compare Low vs. High Frequency Energy
total_energy = np.sum(power_spectrum)
low_freq_energy = np.sum(power_spectrum[np.abs(frequencies) <= final_cutoff])
high_freq_energy = total_energy - low_freq_energy
energy_ratio = low_freq_energy / high_freq_energy if high_freq_energy != 0 else np.inf

print("\n🔍 Fourier Energy Analysis:")
print(f"Total Energy: {total_energy:.2f}")
print(f"Low Frequency Energy: {low_freq_energy:.2f} (Retained)")
print(f"High Frequency Energy: {high_freq_energy:.2f} (Filtered Out)")
print(f"Energy Ratio (Low to High Frequencies): {energy_ratio:.2f}")

# ------------------ VISUALIZATIONS ------------------

# Frequency Spectrum (Full)
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(fft_values), label="Original Spectrum", color="blue")
plt.axvline(final_cutoff, color="red", linestyle="--", label=f"Cutoff @ {final_cutoff:.5f}")
plt.title(f"Fourier Transform - Frequency Spectrum ({best_maturity})")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.savefig("visuals/fourier_full_spectrum.png")
plt.show()

# Frequency Spectrum (Zoomed-in on Low Frequencies)
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(fft_values), label="Original Spectrum", color="blue")
plt.axvline(final_cutoff, color="red", linestyle="--", label=f"Cutoff @ {final_cutoff:.5f}")
plt.xlim(-0.05, 0.05)  # Focus on low frequencies
plt.title(f"Fourier Transform - Low Frequency Focus ({best_maturity})")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.savefig("visuals/fourier_zoomed_spectrum.png")
plt.show()

# Filtered PC3 Dislocation Signal Over Time
plt.figure(figsize=(12, 6))
plt.plot(rolling_pca["date"], dislocation_series, label="Original PC3 Dislocation", alpha=0.5, color="gray")
plt.plot(rolling_pca["date"], filtered_signal, label="Filtered Signal", color="blue")
plt.title(f"PC3 Dislocation - Filtered vs. Original ({best_maturity})")
plt.xlabel("Date")
plt.ylabel("PC3 Dislocation")
plt.legend()
plt.grid()
plt.savefig("visuals/fourier_filtered_signal.png")
plt.show()

print(f"✅ Fourier Transform applied and visuals saved to 'visuals/' folder (Cutoff @ {final_cutoff:.5f}).")
