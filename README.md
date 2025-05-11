# PC3 Spectral Analysis: A PCA and Fourier Modeling Approach for Yield Curve Curvature

Author: Danny Watkins  
University of Arizona — MATH485: Mathematical Modeling  
Date: 2025

---

## Overview

This project analyzes structural shape shifts in the U.S. Treasury yield curve by applying Principal Component Analysis (PCA) and Fourier Transform smoothing to the third principal component (PC3) — the factor most associated with curvature.

We construct a real-time curvature signal using PCA eigenvector loadings and implement a trade signal framework based on statistical dislocation thresholds and slope-confirmation logic. Results show that traditional proxies like 2s5s10s butterflies fail to track curvature reliably, and we propose a PCA-weighted portfolio as a cleaner alternative.

---

## Project Structure

```
pc3-spectral-arbitrage/
├── config.json                      # Central config file
├── requirements.txt                 # Python dependencies
├── main.py                          # Runs full pipeline (calls all modules)
├── reports/
│   ├── PC3_Spectral_Analysis.pdf    # Final research paper (compiled)
├── data/                            # Processed yield data, PC scores, signals
├── visuals/                         # Output figures (organized by folder)
│   ├── comparisons/
│   ├── macro_sensitivity/
│   ├── smoothing/
│   ├── signals/
│   └── pca_geometry/
├── scripts/
│   ├── data_loader.py
│   ├── rolling_pca_analysis.py
│   ├── fourier_transform.py
│   ├── smooth_all_pc_scores.py
│   ├── pc3_signal_trading.py
│   ├── pc3_mean_reversion_analysis.py
│   ├── pc_macro_sensitivity_analysis.py
│   ├── verify_pc3_vs_butterfly.py
│   └── pca_geometry_visuals.py
└── README.md
```

---

## Key Features

- Rolling PCA to isolate level (PC1), slope (PC2), and curvature (PC3)
- Fourier-based smoothing of PC3 to filter high-frequency noise
- Trade signal framework using Z-score dislocation and slope reversal
- Statistical validation of curvature mean-reversion using ADF, half-life, MAD
- Macro context overlays with VIX-based regime segmentation
- Proxy analysis of traditional 2s5s10s butterfly trades
- PCA-weighted portfolio construction for tracking curvature directly

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dannywatkins/pc3-spectral-arbitrage.git
cd pc3-spectral-arbitrage
```

2. (Optional) Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your FRED API key as an environment variable:
```bash
export FRED_API_KEY=your_fred_key_here
```

---

## Usage

To run the full pipeline:

```bash
python main.py
```

To run individual modules (example):

```bash
python scripts/fourier_transform.py
python scripts/pc3_signal_trading.py
```

---

## Output

- Processed data saved to `data/`
- Visualizations stored in `visuals/` (organized by module)
- All final output files used in the paper are reproducible via code

---

## Additional Materials

- **Presentation:**  
  [PC3 Spectral Arbitrage – Canva Slides](https://www.canva.com/design/DAGh8Y-SZw8/n1P25jCb5vHezFNze0_n6A/edit?utm_content=DAGh8Y-SZw8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

- **Research Paper (PDF):**  
  [`reports/PC3_Spectral_Analysis.pdf`](reports/PC3_Spectral_Analysis.pdf)

---

## Contact

Danny Watkins  
maximus@arizona.edu
