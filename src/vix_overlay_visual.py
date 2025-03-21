import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

# Load Data
start_date = "2013-01-01"
end_date = "2023-12-31"

# Load rolling PCA results (contains PC3 dislocations)
rolling_pca_results = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
rolling_pca_results.set_index("date", inplace=True)

# Load VIX data
vix_data = pd.read_csv("data/vix_data.csv", index_col=0, parse_dates=True)

# Load monetary policy data
monetary_policy_data = pd.read_csv("data/monetary_policy.csv", parse_dates=["Date"])

# Ensure all data is aligned to the same timeline (2013-2023)
rolling_pca_results = rolling_pca_results.loc[start_date:end_date]
vix_data = vix_data.loc[start_date:end_date]
monetary_policy_data = monetary_policy_data[(monetary_policy_data["Date"] >= start_date) & (monetary_policy_data["Date"] <= end_date)]

# Merge data
merged_data = pd.merge(rolling_pca_results[["PC3_Dislocation"]], vix_data, left_index=True, right_index=True, how='inner')
merged_data.columns = ['PC3', 'VIX']

# Merge with monetary policy data
monetary_policy_data['Monetary Policy Change'] = 1  # Mark all dates as policy changes
monetary_policy_data.set_index("Date", inplace=True)
merged_data = pd.merge(merged_data, monetary_policy_data, left_index=True, right_index=True, how='left')
merged_data['Monetary Policy Change'] = merged_data['Significant Change'].fillna(0)

# Classify Market Regimes
volatility_threshold = merged_data['VIX'].median()
merged_data['Market Regime'] = np.where(
    merged_data['VIX'] < volatility_threshold, 'Low Volatility', 'High Volatility'
)
merged_data['Market Regime'] = np.where(
    merged_data['Monetary Policy Change'] != 0, 'Monetary Policy Change', merged_data['Market Regime']
)

# Check if data is loaded correctly
if merged_data.empty:
    print("❌ Merged data frame is empty. Please check data loading.")
else:
    print("✅ Merged data frame is loaded correctly.")
    print(f"Merged data sample:\n{merged_data.head()}")

# Mean Reversion Calculation (Half-Life)
def calculate_half_life(series):
    series = series - series.mean()
    lagged = series.shift(1).fillna(0)
    delta = series - lagged
    beta = np.polyfit(lagged, delta, 1)[0]
    half_life = -np.log(2) / beta
    return half_life

# Calculate Half-Life for each regime
regimes = merged_data['Market Regime'].unique()
half_lives = {}
for regime in regimes:
    regime_data = merged_data[merged_data['Market Regime'] == regime]['PC3']
    half_life = calculate_half_life(regime_data)
    half_lives[regime] = half_life

print("\nPC3 Mean-Reversion Half-Lives by Regime:")
for regime, half_life in half_lives.items():
    print(f"{regime}: {half_life:.2f} days")

# Correlation Analysis between PC3 and VIX by Regime
correlations = {}
for regime in regimes:
    regime_data = merged_data[merged_data['Market Regime'] == regime]
    correlation = regime_data['PC3'].corr(regime_data['VIX'])
    correlations[regime] = correlation
    print(f"Correlation between PC3 and VIX ({regime}): {correlation:.2f}")

# ------------------ VISUALIZATIONS ------------------

# Time Series Plot: PC3 Dislocations with Market Regime Shading
plt.figure(figsize=(14, 8))

# Plot PC3 Dislocations
ax1 = plt.gca()  # Primary y-axis
sns.lineplot(data=merged_data, x=merged_data.index, y='PC3', label="PC3 Dislocations", color="blue", linewidth=1.5, ax=ax1)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label="Mean Reversion Line")

# Shade regions for market regimes
for regime, color in zip(['Low Volatility', 'High Volatility', 'Monetary Policy Change'], ['green', 'red', 'orange']):
    regime_dates = merged_data[merged_data['Market Regime'] == regime].index
    if not regime_dates.empty:
        plt.fill_between(regime_dates, merged_data.loc[regime_dates, 'PC3'].min(), merged_data.loc[regime_dates, 'PC3'].max(), 
                         color=color, alpha=0.3, label=regime)

# Highlight Policy Changes
for date in monetary_policy_data.index:
    plt.axvline(x=date, color='red', linestyle=':', linewidth=1, alpha=0.7, label="Policy Change" if date == monetary_policy_data.index[0] else "")

# Add VIX on secondary y-axis
ax2 = ax1.twinx()  # Secondary y-axis
sns.lineplot(data=merged_data, x=merged_data.index, y='VIX', label="VIX Index", color="purple", linestyle="--", linewidth=1.2, ax=ax2)

# Labels and title
ax1.set_xlabel("Date")
ax1.set_ylabel("PC3 Dislocation")
ax2.set_ylabel("VIX Level")
plt.title("PC3 Dislocations with Market Regime Shading and VIX Overlay")
plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig("visuals/pc3_vix_overlay_improved.png")
plt.show()

# Half-Life Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(half_lives.keys(), half_lives.values(), color=['green', 'red', 'orange'])

# Add annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.2f} days', ha='center', va='bottom')

# Highlight key insight
plt.text(2, 10, 'Fastest Reversion', ha='center', va='bottom', color='blue', fontsize=10)

# Labels and title
plt.title('Half-Life of PC3 Dislocations by Market Regime')
plt.xlabel('Market Regime')
plt.ylabel('Half-Life (Days)')
plt.ylim(0, 90)
plt.savefig("visuals/pc3_half_life_improved.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(pd.DataFrame(correlations, index=[0]), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation between PC3 and VIX by Market Regime')
plt.savefig("visuals/pc3_vix_correlation_improved.png")
plt.show()

# ------------------ VALIDATION ------------------

# Plot Distribution of PC3 Dislocations
plt.figure(figsize=(8, 6))
sns.histplot(merged_data['PC3'], kde=True, color='blue')
plt.axvline(merged_data['PC3'].mean(), color='red', linestyle='--', label='Mean')
plt.title("Distribution of PC3 Dislocations")
plt.xlabel("PC3 Dislocation")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("visuals/pc3_distribution.png")
plt.show()

# Augmented Dickey-Fuller Test for Mean Reversion
adf_result = adfuller(merged_data['PC3'])
print("\nAugmented Dickey-Fuller Test Results:")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print(f"Critical Values: {adf_result[4]}")