"""Paths relative to the forecasting component, and the notebook model settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/18100245.csv"
MODEL_PATH = PROJECT_ROOT / "models/forecast_model.pkl"
PREDICTIONS_PATH = PROJECT_ROOT / "data/predictions/forecasts.json"

HISTORY_FEATURES = ["LAG_1", "LAG_2", "LAG_3", "ROLLING_MEAN_3", "PRICE_CHANGE_3"]
FEATURE_COLUMNS = ["PRODUCTS", "MONTH", "YEAR", *HISTORY_FEATURES]
MODEL_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
}
