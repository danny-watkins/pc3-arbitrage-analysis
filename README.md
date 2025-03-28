# PC3 Spectral Arbitrage: A PCA-Fourier Learning Model

## Overview
The PC3 Spectral Arbitrage project implements a PCA-Fourier learning model focused on yield curve arbitrage. The project is designed to identify and exploit dislocations in the yield curve through a combination of Principal Component Analysis (PCA) and Fourier Transform techniques. The model leverages spectral analysis to detect cyclical patterns and anomalies, enhancing trading decisions and risk management strategies.

### Project Structure
The project is organized as follows:

```
project_root/
├── data/                 # Datasets used for analysis
├── scripts/              # Core analysis scripts
├── models/               # Machine learning models and utilities
├── visuals/              # Scripts and images for visualizations
├── tests/                # Unit tests for each module
├── reports/              # Final results and analysis reports
└── README.md             # Project documentation (this file)
```

### Key Features
- PCA for dimensionality reduction and principal component extraction
- Fourier Transform for detecting cyclical patterns in yield curve dislocations
- Machine learning models for yield curve prediction and arbitrage opportunity detection
- Backtesting framework for strategy evaluation
- Visualization tools for data analysis and result presentation

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/dannywatkins/pc3-spectral-arbitrage.git
   cd pc3-spectral-arbitrage
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: .\env\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the main analysis script:
```bash
python scripts/main.py
```

To visualize results:
```bash
python scripts/visuals_generator.py
```

### Example Output
The analysis output includes:
- PCA visualization and interpretation
- Fourier spectral analysis plots
- Trading signals based on identified dislocations
- Performance metrics from backtesting

## Contributions
Contributions are welcome. Please fork the repository and submit a pull request with detailed changes.

## Contact
For questions or collaborations, feel free to reach out to Danny Watkins at maximus@arizona.edu.

