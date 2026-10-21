"""Focused checks for temporal correctness and the exported forecast contract."""

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from forecasting.config import FEATURE_COLUMNS, MODEL_PARAMS
from forecasting.data_loader import load_prices
from forecasting.evaluate import calculate_metrics, chronological_split, evaluate_model
from forecasting.features import (
    _monthly_features,
    lightgbm_features,
    prepare_next_month_data,
    prepare_training_data,
)
from forecasting.predict import export_forecasts, forecast_two_months
from forecasting.train import load_model, save_model, train_model


def price_history():
    # Reverse order deliberately: feature construction must sort calendar months.
    return pl.DataFrame([
        {"PRODUCTS": product, "DATE": f"2024-{month:02}", "VALUE": float(month + offset)}
        for product, offset in [("B", 10), ("A", 0)]
        for month in range(6, 0, -1)
    ])


class RecursiveModel:
    products = ("A", "B")

    def __init__(self):
        self.inputs = []

    def predict(self, inputs):
        self.inputs.append(inputs.copy())
        return inputs[:, FEATURE_COLUMNS.index("LAG_1")] + 0.123456789


class FeatureTests(unittest.TestCase):
    def test_target_month_uses_only_prior_calendar_prices(self):
        prices = price_history()
        dataset = prepare_training_data(prices)
        april = dataset.filter(
            (pl.col("PRODUCTS") == "A") & (pl.col("DATE") == date(2024, 4, 1))
        ).row(0, named=True)
        self.assertEqual(
            [april[k] for k in ["MONTH", "YEAR", "LAG_1", "LAG_2", "LAG_3",
                                "ROLLING_MEAN_3", "PRICE_CHANGE_3", "VALUE"]],
            [4, 2024, 3.0, 2.0, 1.0, 2.0, 2.0, 4.0],
        )
        changed = prices.with_columns(
            pl.when(pl.col("DATE") >= "2024-04").then(999.0)
            .otherwise(pl.col("VALUE")).alias("VALUE")
        )
        april_changed = prepare_training_data(changed).filter(pl.col("DATE") == date(2024, 4, 1))
        np.testing.assert_array_equal(
            lightgbm_features(dataset.filter(pl.col("DATE") == date(2024, 4, 1))).to_numpy(),
            lightgbm_features(april_changed).to_numpy(),
        )

    def test_missing_calendar_month_is_explicit_and_never_filled(self):
        prices = price_history().filter(
            ~((pl.col("PRODUCTS") == "A") & (pl.col("DATE") == "2024-05"))
        )
        grid = _monthly_features(prices)
        june = grid.filter(
            (pl.col("PRODUCTS") == "A") & (pl.col("DATE") == date(2024, 6, 1))
        ).row(0, named=True)
        self.assertIsNone(june["LAG_1"])
        self.assertEqual(june["LAG_2"], 4)
        self.assertEqual(june["LAG_3"], 3)
        self.assertEqual(prepare_next_month_data(prices)["PRODUCTS"].to_list(), ["B"])
        self.assertEqual(
            prepare_training_data(prices).filter(pl.col("PRODUCTS") == "A")["DATE"].to_list(),
            [date(2024, 4, 1)],
        )

    def test_nonfinite_prices_are_missing_and_duplicates_rejected(self):
        for value in [None, float("nan"), float("inf"), -float("inf")]:
            prices = price_history().with_columns(
                pl.when((pl.col("PRODUCTS") == "A") & (pl.col("DATE") == "2024-05"))
                .then(pl.lit(value)).otherwise(pl.col("VALUE")).alias("VALUE")
            )
            self.assertEqual(prepare_next_month_data(prices)["PRODUCTS"].to_list(), ["B"])
        prices = price_history()
        with self.assertRaisesRegex(ValueError, "one price per product and month"):
            prepare_training_data(pl.concat([prices, prices.head(1)]))

    def test_encoding_retains_full_product_vocabulary_in_subsets(self):
        training = prepare_training_data(price_history())
        only_b = training.filter(pl.col("PRODUCTS") == "B")
        self.assertEqual(lightgbm_features(only_b)["PRODUCTS"].unique().to_list(), [1])
        self.assertEqual(lightgbm_features(training).columns, FEATURE_COLUMNS)


class EvaluationTests(unittest.TestCase):
    def test_holdout_is_latest_eligible_month_and_never_used_for_fit(self):
        dataset = prepare_training_data(price_history())
        training, validation = chronological_split(dataset)
        self.assertLess(training["DATE"].max(), validation["DATE"].min())
        self.assertEqual(validation["DATE"].unique().to_list(), [date(2024, 6, 1)])
        with patch("forecasting.evaluate.train_model", return_value=RecursiveModel()) as fit:
            result = evaluate_model(dataset)
        self.assertTrue(fit.call_args.args[0].equals(training))
        self.assertEqual(result["PRODUCTS_EVALUATED"].item(), 2)
        with self.assertRaisesRegex(ValueError, "two eligible target months"):
            chronological_split(validation)

    def test_mae_wape_and_zero_denominator(self):
        metrics = calculate_metrics([10, 30], [12, 24])
        self.assertEqual(metrics, {"MAE_DOLLARS": 4.0, "WAPE_PERCENT": 20.0})
        self.assertEqual(calculate_metrics([0, 0], [1, 3]),
                         {"MAE_DOLLARS": 2.0, "WAPE_PERCENT": None})
        with self.assertRaisesRegex(ValueError, "finite"):
            calculate_metrics([1], [float("inf")])


class PredictionTests(unittest.TestCase):
    def test_two_consecutive_months_schema_and_unrounded_recursion(self):
        model = RecursiveModel()
        forecasts = forecast_two_months(price_history(), model)
        self.assertEqual(forecasts.height, 4)
        self.assertEqual(forecasts["DATE"].unique().sort().to_list(),
                         [date(2024, 7, 1), date(2024, 8, 1)])
        self.assertEqual(forecasts.group_by("PRODUCTS").len()["len"].to_list(), [2, 2])
        np.testing.assert_allclose(
            model.inputs[1][:, FEATURE_COLUMNS.index("LAG_1")],
            [6.123456789, 16.123456789], rtol=0, atol=1e-12,
        )
        self.assertAlmostEqual(forecasts.filter(pl.col("PRODUCTS") == "A")["VALUE"][1],
                               6.246913578, places=12)
        with TemporaryDirectory() as tmp:
            path = export_forecasts(forecasts, Path(tmp) / "predictions/forecasts.json")
            records = json.loads(path.read_text())
            for record in records:
                self.assertEqual(set(record), {"PRODUCTS", "DATE", "VALUE"})
                self.assertRegex(record["DATE"], r"^2024-0[78]$")
                self.assertIsInstance(record["PRODUCTS"], str)
                self.assertIsInstance(record["VALUE"], float)

    def test_forecast_crosses_year_boundary(self):
        prices = price_history().with_columns(
            pl.col("DATE").str.to_date("%Y-%m").dt.offset_by("6mo")
        )
        result = forecast_two_months(prices, RecursiveModel())
        self.assertEqual(result["DATE"].unique().sort().to_list(),
                         [date(2025, 1, 1), date(2025, 2, 1)])

    def test_incomplete_recent_history_and_changed_vocabulary_fail(self):
        prices = price_history().filter(
            ~((pl.col("PRODUCTS") == "A") & (pl.col("DATE") == "2024-05"))
        )
        model = RecursiveModel()
        with self.assertRaisesRegex(ValueError, "recent monthly prices.*A"):
            forecast_two_months(prices, model)
        self.assertEqual(model.inputs, [])
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            forecast_two_months(price_history().filter(pl.col("PRODUCTS") == "B"), model)

    def test_nonfinite_predictions_fail_in_either_month(self):
        for invalid in [None, float("nan"), float("inf")]:
            for failed_month in [1, 2]:
                model = RecursiveModel()
                calls = 0

                def predict(inputs):
                    nonlocal calls
                    calls += 1
                    return [invalid, invalid] if calls == failed_month else [1.0, 2.0]

                model.predict = predict
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    forecast_two_months(price_history(), model)

    def test_real_model_save_load_preserves_predictions_and_settings(self):
        prices = price_history()
        model = train_model(prepare_training_data(prices))
        for name, expected_value in MODEL_PARAMS.items():
            self.assertEqual(model.regressor.get_params()[name], expected_value)
        expected = forecast_two_months(prices, model)
        with TemporaryDirectory() as tmp:
            restored = load_model(save_model(model, Path(tmp) / "models/model.pkl"))
            self.assertEqual(restored.products, ("A", "B"))
            self.assertTrue(forecast_two_months(prices, restored).equals(expected))


class LoaderTests(unittest.TestCase):
    def test_business_filters_and_normalization(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            path.write_text(
                'PÉRIODE DE RÉFÉRENCE;GÉO;Produits;VALEUR\n'
                '2023-12;Québec;A;10\n'
                '2024-01;Canada;A;20\n'
                '2024-01;Québec;A;12.34\n'
                '2024-02;Québec;A;NaN\n'
                '2024-03;Québec;A;..\n', encoding="utf-8",
            )
            prices = load_prices(path)
        self.assertEqual(prices.columns, ["DATE", "PRODUCTS", "VALUE"])
        self.assertEqual(prices.schema["DATE"], pl.Date)
        self.assertEqual(prices["VALUE"].to_list(), [12.34, None, None])
        self.assertEqual(prices["DATE"].min(), date(2024, 1, 1))


if __name__ == "__main__":
    unittest.main()
