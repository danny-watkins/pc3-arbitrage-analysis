"""
trading_signals.py
------------------
Author: Danny Watkins
Date: 2025
Description:
Generates trading signals based on the dynamically selected best PCA maturity 
with **adaptive thresholds** based on historical dislocation volatility.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Load best maturity from config.json
with open("config.json", "r") as f:
    config = json.load(f)

best_maturity = config["best_maturity"]  # E.g., "PC1_DGS30"
print(f"Using Best Maturity for Trading: {best_maturity}")

# Load Data
pca_dislocations = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
fourier_signals = pd.read_csv("data/fourier_filtered_signals.csv", parse_dates=["Date"])

# Ensure column alignment (the column with the filtered Fourier signal is dynamically renamed)
fourier_signals.rename(columns={best_maturity: "Filtered_Signal"}, inplace=True)

# Merge on Date (making sure the Date format in both datasets matches)
merged = pd.merge(fourier_signals, pca_dislocations, left_on="Date", right_on="date", how="inner")

# Compute Dynamic Thresholds based on the selected best maturity
k = .35  # Adjust sensitivity factor (can be changed based on your preference)
mean_val = merged[best_maturity].mean()  # Mean of the selected maturity's dislocation
std_dev = merged[best_maturity].std()    # Standard deviation of the selected maturity's dislocation

# Compute the dynamic thresholds based on mean and standard deviation
short_threshold = mean_val + k * std_dev   # Short signal threshold (above this value, generate short signal)
long_threshold = mean_val - k * std_dev    # Long signal threshold (below this value, generate long signal)

# Print the thresholds for reference
print(f"Dynamic Thresholds for {best_maturity}:")
print(f"  - Long  (+1) Signal at: {long_threshold:.4f}")
print(f"  - Short (-1) Signal at: {short_threshold:.4f}")

# Generate Trading Signals based on the filtered signal and thresholds
signals = np.where(merged["Filtered_Signal"] < long_threshold, 1,  # Long Signal (buy)
                   np.where(merged["Filtered_Signal"] > short_threshold, -1,  # Short Signal (sell)
                            0))  # Hold (no action)

# Track the number of trades performed and trades per year
trade_count = 0
trades_per_year = {}
previous_signal = 0  # Initially, no position

for i, signal in enumerate(signals):
    year = merged["Date"].iloc[i].year  # Extract the year from the date
    if signal != 0 and signal != previous_signal:  # A trade (change in position)
        trade_count += 1
        if year not in trades_per_year:
            trades_per_year[year] = 1
        else:
            trades_per_year[year] += 1
    previous_signal = signal

# Save the generated trading signals to a CSV file
trading_signals = pd.DataFrame({"Date": merged["Date"], "Trading_Signal": signals})
trading_signals.to_csv("data/trading_signals.csv", index=False)

print(f"✅ Trading signals generated for {best_maturity}")
print(f"Total Trades Performed: {trade_count}")
print(f"Trades Per Year: {trades_per_year}")

# Visualization of the trading signals with dynamic thresholds
plt.figure(figsize=(12, 6))

# Plot the unfiltered PCA dislocations (red, dotted, opaque)
plt.plot(merged["Date"], merged[best_maturity], color="red", label="Unfiltered Signal", linestyle=":", alpha=0.5)

# Plot the filtered PCA dislocations (green, solid)
plt.plot(merged["Date"], merged["Filtered_Signal"], color="green", label="Filtered Signal (Fourier)")

# Plot the dynamic thresholds for Long and Short signals (orange and blue)
plt.axhline(short_threshold, color="blue", linestyle="--", label=f"Short Threshold ({short_threshold:.4f})")  # Short signal threshold
plt.axhline(long_threshold, color="orange", linestyle="--", label=f"Long Threshold ({long_threshold:.4f})")  # Long signal threshold
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)  # Zero line for reference

# Adding labels and title
plt.xlabel("Date", fontsize=14)
plt.ylabel(f"{best_maturity} Value", fontsize=14)
plt.title(f"{best_maturity} Over Time with Dynamic Thresholds", fontsize=16)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate dates for better readability

# Show the plot
plt.show()
