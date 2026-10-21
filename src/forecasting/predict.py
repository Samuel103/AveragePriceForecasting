"""Recursive two-month forecasts and JSON export."""

import json
from pathlib import Path

import polars as pl

from .config import PREDICTIONS_PATH
from .features import _monthly_features, lightgbm_features, prepare_next_month_data
from .train import TrainedModel


def forecast_two_months(prices: pl.DataFrame, trained_model: TrainedModel) -> pl.DataFrame:
    """Forecast every product twice, feeding predictions back into price history."""
    # Reuse preprocessing to normalize dates, prices, and the product vocabulary.
    # Remove only its extra, empty next month; retain the original historical grid.
    prepared = _monthly_features(prices)
    history = prepared.filter(pl.col("DATE") < pl.col("DATE").max()).select(
        "DATE", pl.col("PRODUCTS").cast(pl.String), "VALUE"
    )
    products = set(history.get_column("PRODUCTS").to_list())
    vocabulary = tuple(prepared.schema["PRODUCTS"].categories.to_list())
    if vocabulary != trained_model.products:
        raise ValueError("Prediction products must match the saved model product vocabulary.")
    predictions = []

    for _ in range(2):
        next_month_data = prepare_next_month_data(history)
        missing_products = products - set(next_month_data.get_column("PRODUCTS").to_list())
        if missing_products:
            raise ValueError(
                "Cannot forecast every product: three complete recent monthly prices "
                f"are required for {sorted(missing_products)}."
            )

        values = pl.Series(
            "VALUE",
            trained_model.predict(lightgbm_features(next_month_data).to_numpy()),
            dtype=pl.Float64,
        )
        if values.null_count() or not values.is_finite().all():
            raise ValueError("The model returned nonfinite forecast prices.")
        month_predictions = next_month_data.select(
            pl.col("PRODUCTS").cast(pl.String), "DATE"
        ).with_columns(values)
        predictions.append(month_predictions)
        history = pl.concat([history, month_predictions.select(history.columns)])

    return pl.concat(predictions).sort("PRODUCTS", "DATE")


def export_forecasts(
    predictions: pl.DataFrame, path: str | Path = PREDICTIONS_PATH
) -> Path:
    records = predictions.select(
        "PRODUCTS", pl.col("DATE").dt.strftime("%Y-%m"), "VALUE"
    ).to_dicts()
    # Serialize before touching the output, rejecting nonfinite JSON numbers.
    payload = json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path
