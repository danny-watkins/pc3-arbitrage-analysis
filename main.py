"""
main.py
--------------------
Author: Danny Watkins
Date: 2025
Description:
Master controller script to run the full PCA-Fourier yield curve pipeline in order.
"""

import subprocess

scripts = [
    "data_loader.py",
    "rolling_pca_analysis.py",
    "fourier_transform.py",
    "smooth_all_pc_scores.py",
    "pc_macro_sensitivity_analysis.py",
    "pc3_mean_reversion_analysis.py",
    "pc3_signal_trading.py",
    "verify_pc3_vs_butterfly.py",
    "pca_geometry_visuals.py",
]

for script in scripts:
    print(f"\n[RUNNING] {script}")
    try:
        subprocess.run(["python", f"src/{script}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script} failed:\n{e}")
