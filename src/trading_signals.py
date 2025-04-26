"""
trading_signals.py
------------------
Author: Danny Watkins
Date: 2025
Description:
Generates trading signals based on the dynamically selected best PCA maturity 
with adaptive thresholds based on historical dislocation volatility.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Load best maturity from config.json
with open("config.json", "r") as f:
    config = json.load(f)

best_maturity = config['belly_maturity']  # E.g., "PC1_DGS30"
print(f"Using Best Maturity for Trading: {best_maturity}")

# Load Data
pca_dislocations = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
fourier_signals = pd.read_csv("data/fourier_filtered_signals.csv", parse_dates=["Date"])

# Ensure column alignment (the column with the filtered Fourier signal is dynamically renamed)
fourier_signals.rename(columns={best_maturity: "Fourier_Signal"}, inplace=True)

# Merge on Date (making sure the Date format in both datasets matches)
merged = pd.merge(
    fourier_signals, pca_dislocations[["date", best_maturity]], 
    left_on="Date", right_on="date", how="inner"
)

# Calculate Difference
print("✅ Merged Data Columns:", merged.columns)
print("✅ Fourier Signals Data Columns:", fourier_signals.columns)
print("✅ PCA Dislocations Data Columns:", pca_dislocations.columns)
merged["Difference"] = merged[best_maturity] - merged["Fourier_Signal"]

# Calculate Rolling Mean and Std Dev for Dislocation
window_size = 250
merged["Rolling_Mean"] = merged["Difference"].rolling(window=window_size, min_periods=1).mean()
merged["Rolling_Std"] = merged["Difference"].rolling(window=window_size, min_periods=1).std()

# Determine Dislocation Flag
merged["Dislocation_Flag"] = (abs(merged["Difference"] - merged["Rolling_Mean"]) > 1.5 * merged["Rolling_Std"]).astype(int)

# Calculate Fourier Slope using simple differences
merged["Fourier_Slope"] = merged["Fourier_Signal"].diff()

# Generate Trading Signals
merged["Trading_Signal"] = 0
buy_condition = (merged["Dislocation_Flag"] == 1) & (merged["Difference"] < 0) & (merged["Fourier_Slope"] > 0)
sell_condition = (merged["Dislocation_Flag"] == 1) & (merged["Difference"] > 0) & (merged["Fourier_Slope"] < 0)
merged.loc[buy_condition, "Trading_Signal"] = 1
merged.loc[sell_condition, "Trading_Signal"] = -1

# Backtesting
initial_balance = 100000
balance = initial_balance
position = 0
returns = []
equity_curve = [initial_balance]  # Initialize with starting balance

# Tracking variables
total_trades = 0
for i in range(1, len(merged)):
    signal = merged.loc[i, "Trading_Signal"]
    price_change = merged.loc[i, best_maturity] - merged.loc[i - 1, best_maturity]

    # Calculate daily return based on current position
    daily_return = position * price_change
    balance += daily_return
    returns.append(daily_return)
    equity_curve.append(balance)

    # If trading signal differs from current position, make a trade
    if signal != 0 and signal != position:
        position = signal
        total_trades += 1
        print(f"Trade at {merged.loc[i, 'Date']} with signal {signal}")

print(f"Total Trades Made: {total_trades}")

# Calculate Performance Metrics
total_return = (balance / initial_balance - 1) * 100
annualized_return = (1 + total_return / 100) ** (252 / len(merged)) - 1
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
drawdown = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
max_drawdown = drawdown.max() * 100

print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")


# Visualization of Trades on Raw and Fourier Signal
plt.figure(figsize=(16, 8))
plt.plot(merged["Date"], merged[best_maturity], label="Raw Signal", color="blue", alpha=0.7)
plt.plot(merged["Date"], merged["Fourier_Signal"], label="Fourier Signal", color="green")

# Plotting Buy and Sell Signals
buy_signals = merged[merged["Trading_Signal"] == 1]
sell_signals = merged[merged["Trading_Signal"] == -1]
plt.scatter(buy_signals["Date"], buy_signals[best_maturity], color="green", marker="^", label="Buy Signal")
plt.scatter(sell_signals["Date"], sell_signals[best_maturity], color="red", marker="v", label="Sell Signal")

plt.title("Trading Signals on Raw and Fourier Signal")
plt.xlabel("Date")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)
plt.show()
