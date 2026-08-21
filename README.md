# House Price Prediction with Gradient Descent from Scratch

A multi-feature linear regression project comparing three gradient descent variants (Batch, Stochastic, Mini-Batch) implemented from scratch in NumPy, validated against scikit-learn.

## Overview

This project predicts house prices using 5 numeric features (bedrooms, bathrooms, sqft_living, sqft_lot, floors). Rather than just calling `sklearn.LinearRegression()`, all three gradient descent optimizers are implemented from scratch using NumPy matrix operations, then benchmarked against each other and against scikit-learn's exact solution.

## Dataset

- **Source:** Kaggle House Price dataset (4,600 listings, 18 columns)
- **Cleaning:** Removed 289 rows — invalid `$0` prices and statistical outliers (IQR method, 1.5×IQR beyond Q1/Q3)
- **Result:** Price standard deviation dropped from ~$564K to ~$216K (61% reduction), confirming outliers were meaningfully distorting the raw data

## Methodology

1. **Data cleaning & exploration** — Pandas, outlier detection via IQR
2. **Feature scaling** — standardization (mean=0, std=1) on all 5 features
3. **Gradient descent from scratch** — Batch, Stochastic (SGD), and Mini-Batch (batch_size=32), each implemented as independent NumPy functions using matrix multiplication (`X @ weights`)
4. **Validation** — compared final weights/bias against `sklearn.linear_model.LinearRegression`

## Key Findings

- **Strongest predictor:** `sqft_living` — confirmed both visually (scatter plots) and numerically (consistently largest-magnitude weight, ~142K-146K, across all three methods)
- **Weakest predictor:** `sqft_lot` — most scattered relationship, smallest-magnitude weight
- **Convergence:** Batch GD and Mini-Batch GD converged smoothly within ~200 epochs and matched each other closely. SGD remained visibly noisy throughout training, including a sharp cost spike around epoch 800
- **Accuracy vs scikit-learn:** Batch GD matched scikit-learn's bias term exactly ($487,456.90) and all feature weights within a few hundred dollars. SGD showed the largest deviation, even flipping sign on one feature weight

| Method | Bias | Matches sklearn |
|---|---|---|
| Batch GD | 487,456.90 | Almost exact |
| Mini-Batch GD | 494,522.23 | Close |
| SGD | 490,944.89 | Least precise |
| Scikit-learn | 487,456.90 | — |

## Tech Stack

- Python, NumPy, Pandas, Matplotlib, scikit-learn
- Jupyter Notebook (VS Code)

## Project Structure

```
house-price-gradient-descent/
├── data/
│   └── house_prices.csv
├── notebooks/
│   └── house_price_analysis.ipynb
├── requirements.txt
└── README.md
```

## Running This Project

```bash
pip install -r requirements.txt
jupyter notebook notebooks/house_price_analysis.ipynb
```

## What I'd Explore Next

- Run SGD for more epochs / with a decaying learning rate to test if it converges closer to the optimal solution
- Tune Mini-Batch's `batch_size` (16 vs 64 vs 128) to study the speed/stability tradeoff
- Add categorical features (city, waterfront) via one-hot encoding
- Benchmark actual wall-clock training time across all three methods
