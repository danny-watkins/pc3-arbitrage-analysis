"""
pc3_signal_trading.py
---------------------
Author: Danny Watkins
Date: 2025
Description:
Generates trading signals from PC3 dislocations using Z-score and slope logic.
Logic: Enter trades when Z-score exceeds threshold AND slope reverses direction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# -------------------- Load Configuration --------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
data_dir = config.get("data_dir", "data")
visuals_dir = os.path.join(config.get("visuals_dir", "visuals"), "signals")
os.makedirs(visuals_dir, exist_ok=True)

# -------------------- Load Data --------------------
df = pd.read_csv(os.path.join(data_dir, "fourier_filtered_signals.csv"), parse_dates=["Date"])
df.set_index("Date", inplace=True)

# -------------------- Compute Indicators --------------------
df["Dislocation"] = df["PC3_Score"] - df["Smoothed_PC3_Score"]
df["Z"] = (df["Dislocation"] - df["Dislocation"].rolling(63).mean()) / df["Dislocation"].rolling(63).std()
df["Slope"] = df["Smoothed_PC3_Score"].diff()
df["Prev_Slope"] = df["Slope"].shift(1)

# -------------------- Signal Logic --------------------
df["Position"] = 0
signals = []
current_position = 0

for i in range(2, len(df)):
    date = df.index[i]
    z = df["Z"].iloc[i]
    slope = df["Slope"].iloc[i]
    prev_slope = df["Prev_Slope"].iloc[i]
    smoothed = df["Smoothed_PC3_Score"].iloc[i]

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
    elif current_position == 1 and slope < 0:
        current_position = 0
        signals.append((date, smoothed, "EXIT"))
    elif current_position == -1 and slope > 0:
        current_position = 0
        signals.append((date, smoothed, "EXIT"))

    df.at[date, "Position"] = current_position

# -------------------- Save Signal Data --------------------
signal_df = pd.DataFrame(signals, columns=["Date", "Price", "Signal"])
signal_df.to_csv(os.path.join(data_dir, "pc3_signal_trades.csv"), index=False)

# -------------------- Plot --------------------
plt.figure(figsize=(16, 8))
plt.plot(df.index, df["PC3_Score"], label="Raw PC3", color="gray", alpha=0.4)
plt.plot(df.index, df["Smoothed_PC3_Score"], label="Smoothed PC3", color="blue", linewidth=2)

for sig, marker, color in [("BUY", "^", "green"), ("SELL", "v", "red"), ("EXIT", "o", "black")]:
    subset = signal_df[signal_df["Signal"] == sig]
    plt.scatter(subset["Date"], subset["Price"], marker=marker, color=color, s=100, label=sig, edgecolors='k')

plt.title("PC3 Signal Trades (3-Month Z-Score + Slope Logic)")
plt.xlabel("Date")
plt.ylabel("PC3 Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "pc3_signal_trades.png"))
plt.close()

print(f"[INFO] PC3 signal plot saved to '{visuals_dir}/pc3_signal_trades.png'")
print(f"[INFO] Signal data saved to '{data_dir}/pc3_signal_trades.csv'")
