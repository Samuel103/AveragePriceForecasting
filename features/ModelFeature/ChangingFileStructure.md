# Feature — Refactor AverageForecasting notebook into a Python project

## Goal
Convert the existing `forecasting(1).ipynb` notebook into a clean, modular Python project using `uv`, Polars, and LightGBM.

**Use the notebook as the source of truth.** Move and reuse its working logic instead of redesigning the forecasting approach. Keep the implementation simple and readable.

## Target structure

```text
src
|-forecasting/
| ├── data/
| │   ├── raw/
| │   ├── processed/
| │   └── predictions/
| ├── models/
| ├── notebooks/
| │   ├── exploration.ipynb
| │   └── training.ipynb
| ├── src/
| │   └── average_forecasting/
| │       ├── __init__.py
| │       ├── config.py
| │       ├── data_loader.py
| │       ├── features.py
| │       ├── train.py
| │       ├── evaluate.py
| │       └── predict.py
| ├── tests/
| ├── main.py
| ├── pyproject.toml
| └── uv.lock
```

## Implementation

### `config.py`
Centralize project paths and model parameters. Resolve paths relative to the project root, not the current working directory.

### `data_loader.py`
Load the source CSV with **Polars** and separator `;`. Preserve the current notebook transformations:
- Filter geography to `Québec`.
- Keep and rename the columns to `DATE`, `PRODUCTS`, and `VALUE`.
- Exclude records before `2024-01`.
- Normalize dates and numeric prices for downstream steps.

Move the source CSV currently referenced as `src/forecasting/data/18100245.csv` to `data/raw/`, or adapt the loader to the actual input path.

### `features.py`
Extract the existing feature-engineering functions from the notebook:
- `prepare_training_data()`
- `prepare_next_month_data()`
- `lightgbm_features()`
- Shared monthly feature-building logic

Keep the existing features: `PRODUCTS`, `MONTH`, `YEAR`, `LAG_1`, `LAG_2`, `LAG_3`, `ROLLING_MEAN_3`, and `PRICE_CHANGE_3`.

Each row represents a **target month**. Lags must use earlier calendar months only. Keep missing months explicit; do not forward-fill missing prices or leak future values. Require three complete preceding months for an eligible prediction. Maintain consistent product categorical encoding between training and inference.

### `train.py`
Use the current `lightgbm.LGBMRegressor` settings and feature order. Train **one model for all products**, with `PRODUCTS` as a categorical feature. Save the final trained model in `models/` and support loading it for predictions.

### `evaluate.py`
Keep the chronological holdout approach from the notebook:
- Latest observed eligible month = validation.
- Earlier eligible months = training.
- Calculate **MAE** (dollars) and **WAPE** (percent) across the validation products.
- Do not use the holdout for fitting or tuning the evaluated model.
- Handle a zero WAPE denominator safely.

Keep this evaluation simple; do not add complex cross-validation frameworks.

### `predict.py`
After evaluating, train the final model on all eligible observed data and forecast the **next two consecutive months** using the notebook's recursive approach:
1. Predict the first future month.
2. Append those predictions to the historical data.
3. Recalculate the features and predict the second month.

Do not round intermediate predictions. If any product lacks the required recent history, or the model produces nonfinite prices, raise a clear error instead of exporting partial results.

Export JSON to `data/predictions/forecasts.json` (replacing the notebook's old `data/forecasts.json` location). Each record must contain:

```json
{
  "PRODUCTS": "Example product",
  "DATE": "2026-11",
  "VALUE": 12.34
}
```

The example above illustrates the schema, not an expected prediction.

### `main.py`
Provide a straightforward entry point that loads data, builds features, evaluates the model, trains the final model, saves it, predicts two months, and exports JSON. Run it with `uv run python main.py`.

### `notebooks/`
Keep notebooks for exploration and experimentation only. Import reusable functions from `src/average_forecasting/` instead of duplicating production logic inside notebook cells.

### `tests/`
Add a few focused tests covering feature lags (including missing calendar months), chronological split, MAE/WAPE calculations, and the two-month prediction output schema.

## Acceptance criteria
- [ ] Project uses `uv` and a `src/` package layout.
- [ ] The existing notebook logic is separated into the modules above.
- [ ] Data loading and feature engineering use Polars.
- [ ] Model behavior and LightGBM configuration remain consistent with the notebook.
- [ ] Validation respects chronological order and reports MAE and WAPE.
- [ ] Final model can be saved and loaded.
- [ ] Forecasting produces two months for every product using recursive predictions.
- [ ] Predictions are exported to `data/predictions/forecasts.json`.
- [ ] `uv run python main.py` runs the end-to-end pipeline.
- [ ] Basic unit tests pass.

## Out of scope
- API or user interface.
- Agents or LLM integration.
- Database or cloud services.
- Hyperparameter tuning and new forecasting algorithms.
- Complex orchestration or unnecessary abstractions.

## Note for Codex
Inspect `forecasting(1).ipynb` before implementing anything. Prefer extracting and adapting the existing code. Preserve its business rules and pay special attention to preventing time-series data leakage. Keep functions small and interfaces explicit.
