# Results Summary

Dataset: sample `data/sales_data.csv` (daily sales)

Models evaluated:
- Linear trend regression (baseline)
- ARIMA(1,1,1)

Test window: last 30 days (train/test split)

Key metrics (on sample dataset):
- Linear trend: MAE=3.224, RMSE=3.329, MAPE=0.658%
- ARIMA(1,1,1): MAE=1.393, RMSE=1.595, MAPE=0.276%
- ARIMA AIC: 365.387

Interpretation:
- ARIMA outperforms the simple linear-trend baseline on MAE/RMSE/MAPE for this dataset.
- Residual analysis and rolling MAE plots (in the notebook) show forecast stability over the test window.

Recommended next steps:
1. Grid-search ARIMA orders or use `pmdarima.auto_arima` to find better hyperparameters.
2. Add seasonality/holiday regressors if domain events affect sales.
3. Implement model persistence (joblib) and a scheduled retraining job.