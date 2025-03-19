"""
main.py
-------
Author: Danny Watkins
Date: 2025
Description: 
Runs the full pipeline: data collection, Rolling PCA, Fourier 
filtering, trading signals, RL training, and backtesting.
"""

import os

print("Starting pipeline...")

# Run all scripts
os.system("python src/data_loader.py")
os.system("python src/rolling_pca.py")
os.system("python src/fourier_transform.py")
os.system("python src/trading_signals.py")
os.system("python src/rl_trading_agent.py")

print("Pipeline completed successfully!")
