"""
compute_butterfly_hedge_ratios.py
---------------------------------
Author: Danny Watkins
Date: 2025
Description:
Computes hedge ratios for butterfly trading by neutralizing PC1 and PC2 exposure,
ensuring that the final position is only sensitive to PC3.
"""

import pandas as pd
import json
import numpy as np

# Load butterfly maturities from config.json
with open("config.json", "r") as f:
    config = json.load(f)

belly_maturity = config["belly_maturity"][4:]  # Remove "PC3_" prefix
left_wing_maturity = config["left_wing_maturity"][4:]
right_wing_maturity = config["right_wing_maturity"][4:]

# Load PCA results
pca_dislocations = pd.read_csv("data/rolling_pca_results.csv")

# Extract PC1 and PC2 loadings for the selected maturities
pc1_belly = pca_dislocations[f"PC1_{belly_maturity}"].mean()
pc2_belly = pca_dislocations[f"PC2_{belly_maturity}"].mean()

pc1_left_wing = pca_dislocations[f"PC1_{left_wing_maturity}"].mean()
pc2_left_wing = pca_dislocations[f"PC2_{left_wing_maturity}"].mean()

pc1_right_wing = pca_dislocations[f"PC1_{right_wing_maturity}"].mean()
pc2_right_wing = pca_dislocations[f"PC2_{right_wing_maturity}"].mean()

# Step 1️: Construct the 2x2 matrix A (left and right wings only)
A = np.array([
    [pc1_left_wing, pc1_right_wing],  # PC1 equation
    [pc2_left_wing, pc2_right_wing]   # PC2 equation
])

# Step 2️: Construct the vector b (negative belly PC1 and PC2)
b = np.array([-pc1_belly, -pc2_belly])  # Move belly terms to the right-hand side

# Debugging: Print the system to check correctness
print("Solving for Hedge Ratios:")
print(f"A Matrix:\n{A}")
print(f"b Vector:\n{b}")

# Step 3️: Check if the system is solvable (Avoid Singular Matrix Issues)
if np.linalg.cond(A) > 1e10:
    raise ValueError("❌ Hedge ratio matrix A is ill-conditioned. Check wing selection.")

# Solve for wing weights that neutralize PC1 and PC2
wings_weights = np.linalg.solve(A, b)  # Directly solving the system

# Normalize the hedge ratios so that the belly weight is **fixed at -1**
belly_weight = -1.0

# Ensure wings are proportionate
left_wing_weight = wings_weights[0]
right_wing_weight = wings_weights[1]

# Normalize so that left + right sum to 1, keeping relative proportions
total_weight = abs(left_wing_weight) + abs(right_wing_weight)

# Add stability check before normalizing
if total_weight > 0:
    left_wing_weight = abs(left_wing_weight) / total_weight
    right_wing_weight = abs(right_wing_weight) / total_weight
else:
    raise ValueError("❌ Computed hedge ratios are invalid (division by zero). Check PC1/PC2 values.")

# Print computed hedge ratios for verification
print("\nComputed Hedge Ratios:")
print(f"Belly Weight: {belly_weight}")
print(f"Left Wing Weight: {left_wing_weight:.4f}")
print(f"Right Wing Weight: {right_wing_weight:.4f}")

# Save hedge ratios
hedge_ratios = {
    "belly_weight": belly_weight,
    "left_wing_weight": left_wing_weight,
    "right_wing_weight": right_wing_weight
}

# Save back to config.json
config["hedge_ratios"] = hedge_ratios
with open("config.json", "w") as f:
    json.dump(config, f)

print("✅ Butterfly hedge ratios computed and saved to config.json")
