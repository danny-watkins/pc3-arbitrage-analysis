# PC3 Spectral Analysis: A PCA and Fourier  Modeling Approach for Yield Curve Curvature

Author: Danny Watkins  
Date: 2025

---

## Overview

The PC3 Spectral Arbitrage project analyzes U.S. Treasury yield curve dynamics by applying Principal Component Analysis (PCA) and Fourier Transform smoothing to identify dislocations in the third principal component (PC3), associated with curvature. This dislocation signal is used to generate long/short trade signals based on Z-score deviations and slope reversals.

The pipeline includes statistical diagnostics (e.g., mean reversion metrics), macro context overlays (e.g., VIX sensitivity), and signal validation against butterfly curvature and synthetic portfolios.

---
### Project Structure

```text
pc3-spectral-arbitrage/
├── config.json                      # Central config file for parameters
├── requirements.txt                # Python dependencies
├── scripts/
│   ├── main.py                     # Master script to run entire pipeline
│   ├── data_loader.py              # Fetches and saves FRED data (yield, VIX, Fed Funds)
│   ├── rolling_pca_analysis.py     # Rolling PCA analysis and score generation
│   ├── fourier_transform.py        # Fourier smoothing of PC3
│   ├── smooth_all_pc_scores.py     # Fourier smoothing for PC1–PC3
│   ├── pc3_signal_trading.py       # Trade signal generation from PC3 dislocations
│   ├── pc3_mean_reversion_analysis.py  # Mean reversion metrics + plots
│   ├── pc_macro_sensitivity_analysis.py # VIX sensitivity by volatility regime
│   ├── verify_pc3_vs_butterfly.py  # Validate PC3 vs butterfly & synthetic curvature
│   └── pca_geometry_visuals.py     # Visual PCA interpretation (2D, 3D)
├── data/                           # All intermediate and final datasets (CSV)
├── visuals/                        # All figures, grouped by subfolder
└── README.md                       # This file


---

## Key Features

- Rolling PCA to identify structural shifts in yield curve shape
- Fourier-based smoothing to isolate cyclical behavior in PC3
- Z-score + slope-based logic to generate trade signals
- Mean reversion analysis using half-life, ADF, MAD, and zero crossings
- VIX correlation analysis by volatility regime


---

## Installation

1. Clone the repository
   git clone https://github.com/dannywatkins/pc3-spectral-arbitrage.git
   cd pc3-spectral-arbitrage

2. Create a virtual environment (optional)
   python3 -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Set your FRED API key
   export FRED_API_KEY=your_fred_key_here

---

## Usage

Run the full pipeline:
   python scripts/main.py

Run individual modules:
   python scripts/pc3_signal_trading.py
   python scripts/rolling_pca_analysis.py
   python scripts/fourier_transform.py
   ...

---

## Output

- CSVs: All data stored in /data (e.g., PC scores, smoothed signals)
- PNGs: All figures saved in /visuals with subfolders for each module
- Logs: Print statements confirm successful saves and pipeline status

### Additional Materials

- **Presentation:**  
  [PC3 Spectral Arbitrage – Canva Slides](https://www.canva.com/design/DAGh8Y-SZw8/n1P25jCb5vHezFNze0_n6A/edit?utm_content=DAGh8Y-SZw8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

- **Research Paper (LaTeX):**  
  The full academic write-up including mathematical derivations, model explanation, and results is located in:  
  [`reports/pc3_spectral_arbitrage.tex`](reports/pc3_spectral_arbitrage.tex)


---

## Contact

Danny Watkins  
maximus@arizona.edu
