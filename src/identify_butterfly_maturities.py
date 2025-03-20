"""
identify_butterfly_maturities.py
--------------------------------
Author: Danny Watkins
Date: 2025
Description:
Identifies the optimal butterfly trade structure by selecting:
- The "belly" (the maturity with the highest absolute PC3 dislocation)
- The "wings" (one short-term maturity due to Fed influence, one longer maturity that is valid)
Saves these selections for later Fourier analysis and trading.
"""

import pandas as pd
import json
import numpy as np

# Load PCA dislocations dataset
pca_dislocations = pd.read_csv("data/rolling_pca_results.csv")

# Identify all PC3 maturities dynamically (excluding "PC3_Dislocation" column)
pc3_columns = [col for col in pca_dislocations.columns if col.startswith("PC3_") and col != "PC3_Dislocation"]

# Step 1️⃣: Find the Belly Maturity (Highest Absolute PC3 Dislocation)
pc3_dislocation_means = {maturity: pca_dislocations[maturity].abs().mean() for maturity in pc3_columns}
belly_maturity = max(pc3_dislocation_means, key=pc3_dislocation_means.get)
belly_suffix = belly_maturity.replace("PC3_DGS", "")  # Remove "PC3_DGS" prefix to get the maturity name

print(f"✅ Selected Belly Maturity: {belly_maturity}")

# Step 2️⃣: Find Wing Maturities

# Function to extract numeric maturity values and differentiate MO vs Y
def extract_maturity_number(maturity_label):
    maturity = maturity_label.replace("PC3_DGS", "").replace("DGS", "").replace("MO", "").replace("Y", "")
    is_month = "MO" in maturity_label  # True for MO maturities
    try:
        return (is_month, int(maturity))  # Return (is_month flag, maturity value)
    except ValueError:
        return (False, None)  # Return invalid flag

# Custom sorting function to ensure MO maturities come first, followed by years in numerical order
def custom_maturity_sort(maturity_label):
    is_month, num = extract_maturity_number(maturity_label)
    return (not is_month, num)  # Sort by (is_month flag, numeric value)

# Sort PC3 maturities correctly
sorted_pc3_maturities = sorted(pc3_columns, key=custom_maturity_sort)

# ✅ Select the shortest available PC3 maturity as the left wing (MO maturities allowed)
left_wing = sorted_pc3_maturities[0]  # Shortest-term PC3 maturity

# Get belly maturity numeric value
_, belly_value = extract_maturity_number(belly_suffix)

# ✅ Exclude "MO" maturities from the right-wing selection
valid_right_wings = [
    m for m in sorted_pc3_maturities
    if extract_maturity_number(m.replace("PC3_DGS", ""))[1] is not None
    and extract_maturity_number(m.replace("PC3_DGS", ""))[1] > belly_value
    and not extract_maturity_number(m.replace("PC3_DGS", ""))[0]  # 🚀 **Ensure only "Y" maturities are selected**
]

# Ensure we found at least one valid right-wing maturity
if not valid_right_wings:
    raise ValueError("❌ No valid right-wing maturities found that are longer than the belly.")

# Pick the **first valid "Y" maturity after the belly**
right_wing = valid_right_wings[0]

print(f"✅ Selected Left Wing: {left_wing} (Fed-driven, MO allowed)")
print(f"✅ Selected Right Wing: {right_wing} (Valid right-wing maturity)")

# Step 3️⃣: Save Belly and Wings to Config
config = {
    "belly_maturity": belly_maturity,  # This is PC3
    "left_wing_maturity": left_wing,  # Shortest PC3 maturity (Fed-driven, MO allowed)
    "right_wing_maturity": right_wing  # First valid "Y" maturity after the belly
}

with open("config.json", "w") as f:
    json.dump(config, f)

print("✅ Butterfly maturities saved to config.json")


