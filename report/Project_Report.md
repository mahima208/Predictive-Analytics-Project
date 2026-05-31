# Project Report — Predictive Analytics Using Historical Data

## Overview
This project builds a predictive analytics workflow for sales forecasting using historical sales data. It includes data loading, preprocessing, baseline modeling, ARIMA time-series forecasting, evaluation, and visualization.

## Data
Source: `data/sales_data.csv` — expects a `date` column and a numeric sales column.
Preprocessing: daily resampling, forward-fill missing values, and sorting by date.

## Methods
- Linear trend regression: uses integer time index as regressor to establish a simple baseline.
- ARIMA(1,1,1): fits an autoregressive integrated moving average model to the training series.
- Evaluation: MAE, RMSE, MAPE, and AIC.
- Visualization: actual vs predicted, prediction intervals, residuals distribution, rolling MAE.

## Results (sample dataset)
See `report/Results_Summary.md` for numeric metrics and interpretation.
Visual outputs are in `notebooks/sales_forecasting.ipynb` and use `src/visualization.py`.

## How to reproduce
1. Create a Python environment and install requirements:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
2. Open `notebooks/sales_forecasting.ipynb` and run all cells.
3. Replace `data/sales_data.csv` with your dataset (ensure `date` and `sales` columns).

## Next steps
- Tune ARIMA hyperparameters or use automatic order selection.
- Add seasonal and external regressors.
- Use saved model artifacts from `models/` and add a retraining pipeline.
