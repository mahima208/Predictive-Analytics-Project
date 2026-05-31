# Predictive Analytics Using Historical Data

This project demonstrates a small end-to-end predictive analytics pipeline for sales forecasting using historical sales data.

Project structure:
- `data/` - contains `sales_data.csv`
- `notebooks/` - contains `sales_forecasting.ipynb`
- `src/` - helper modules for data processing, modeling, visualization, and retraining
- `models/` - saved model artifacts and versioned metadata
- `report/` - project report and results summary

Getting started:
1. Create and activate a Python environment:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
2. Open `notebooks/sales_forecasting.ipynb` in Jupyter or VS Code and run all cells.

Notes:
- The loader detects common date and sales column names and renames them to `date` and `sales`.
- The notebook trains a linear trend baseline and an ARIMA(1,1,1) model, evaluates both, and plots forecasts, residuals, and rolling error.
- `src/retrain.py` saves trained models to `models/` and includes timestamped metadata.

Next improvements:
- Add ARIMA hyperparameter tuning or automatic order selection
- Add seasonal and external regressors
- Add a small deployment endpoint or scheduled retraining pipeline
