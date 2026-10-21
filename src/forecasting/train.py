"""Fit one LightGBM regressor and persist its product encoding with it."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import FEATURE_COLUMNS, MODEL_PARAMS, MODEL_PATH
from .features import lightgbm_features


@dataclass
class TrainedModel:
    regressor: lgb.LGBMRegressor
    products: tuple[str, ...]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.regressor.predict(inputs)


def train_model(dataset: pl.DataFrame) -> TrainedModel:
    """Fit only the supplied rows, preserving their complete Enum vocabulary."""
    if dataset.is_empty():
        raise ValueError("Training requires eligible observed monthly prices.")
    inputs = lightgbm_features(dataset).to_numpy()
    products = tuple(dataset.schema["PRODUCTS"].categories.to_list())
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(
        inputs,
        dataset.get_column("VALUE").to_numpy(),
        feature_name=FEATURE_COLUMNS,
        categorical_feature=["PRODUCTS"],
    )
    return TrainedModel(model, products)


def save_model(model: TrainedModel, path: str | Path = MODEL_PATH) -> Path:
    """Save the estimator and vocabulary as a single local artifact."""
    path = Path(path)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def load_model(path: str | Path = MODEL_PATH) -> TrainedModel:
    """Load a trusted local model artifact produced by save_model."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)
