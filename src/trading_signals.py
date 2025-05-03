import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/fourier_filtered_signals.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Calculate dislocation and 3-month Z-score (approx. 63 trading days)
df["Dislocation"] = df["PC3_Score"] - df["Smoothed_PC3_Score"]
df["Z"] = (df["Dislocation"] - df["Dislocation"].rolling(63).mean()) / df["Dislocation"].rolling(63).std()

# Slope and prior slope
df["Slope"] = df["Smoothed_PC3_Score"].diff()
df["Prev_Slope"] = df["Slope"].shift(1)

# Signal logic
df["Position"] = 0
signals = []
current_position = 0

for i in range(2, len(df)):
    date = df.index[i]
    z = df["Z"].iloc[i]
    slope = df["Slope"].iloc[i]
    prev_slope = df["Prev_Slope"].iloc[i]
    smoothed = df["Smoothed_PC3_Score"].iloc[i]

    # Only act when slope is clearly moving (avoid flat transitions)
    if abs(slope) < 0.005:
        df.at[date, "Position"] = current_position
        continue

    if current_position == 0:
        if z < -1.5 and slope > 0:
            current_position = 1
            signals.append((date, smoothed, "BUY"))
        elif z > 1.5 and slope < 0:
            current_position = -1
            signals.append((date, smoothed, "SELL"))
    elif current_position == 1:
        if slope < 0:
            signals.append((date, smoothed, "EXIT"))
            current_position = 0
    elif current_position == -1:
        if slope > 0:
            signals.append((date, smoothed, "EXIT"))
            current_position = 0

    df.at[date, "Position"] = current_position

# Convert signals to DataFrame
signal_df = pd.DataFrame(signals, columns=["Date", "Price", "Signal"])

# === PLOT ===
plt.figure(figsize=(16, 8))
plt.plot(df.index, df["PC3_Score"], label="Raw PC3", color="gray", alpha=0.4)
plt.plot(df.index, df["Smoothed_PC3_Score"], label="Smoothed PC3", color="blue", linewidth=2)

# Signal markers
for sig, marker, color in [("BUY", "^", "green"), ("SELL", "v", "red"), ("EXIT", "o", "black")]:
    subset = signal_df[signal_df["Signal"] == sig]
    plt.scatter(subset["Date"], subset["Price"], marker=marker, color=color, s=100, label=sig, edgecolors='k')

plt.title("PC3 Signal Trades (3-Month Z-Score + Slope Logic)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
