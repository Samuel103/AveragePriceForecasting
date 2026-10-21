"""Chronological holdout evaluation with the notebook's MAE and WAPE."""

import numpy as np
import polars as pl

from .features import lightgbm_features
from .train import train_model


def chronological_split(dataset: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    if dataset.get_column("DATE").n_unique() < 2:
        raise ValueError(
            "Training requires at least two eligible target months for a chronological holdout."
        )
    validation_month = dataset.get_column("DATE").max()
    return (
        dataset.filter(pl.col("DATE") < validation_month),
        dataset.filter(pl.col("DATE") == validation_month),
    )


def calculate_metrics(actual, predicted) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Metrics require equally sized, nonempty vectors.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite prices and predictions.")
    errors = np.abs(actual - predicted)
    total_actual = actual.sum()
    return {
        "MAE_DOLLARS": float(errors.mean()),
        "WAPE_PERCENT": float(100 * errors.sum() / total_actual) if total_actual != 0 else None,
    }


def evaluate_model(dataset: pl.DataFrame) -> pl.DataFrame:
    """Fit on earlier months only; evaluate once on the latest eligible month."""
    training, validation = chronological_split(dataset)
    model = train_model(training)
    metrics = calculate_metrics(
        validation.get_column("VALUE").to_numpy(),
        model.predict(lightgbm_features(validation).to_numpy()),
    )
    return pl.DataFrame({
        "VALIDATION_MONTH": [validation.get_column("DATE").max()],
        "PRODUCTS_EVALUATED": [validation.height],
        "MAE_DOLLARS": [metrics["MAE_DOLLARS"]],
        "WAPE_PERCENT": pl.Series([metrics["WAPE_PERCENT"]], dtype=pl.Float64),
    })
