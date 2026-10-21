"""Run with `uv run forecast` from the repository root."""

from forecasting.data_loader import load_prices
from forecasting.evaluate import evaluate_model
from forecasting.features import prepare_training_data
from forecasting.predict import export_forecasts, forecast_two_months
from forecasting.train import save_model, train_model


def main() -> None:
    prices = load_prices()
    training_data = prepare_training_data(prices)
    results = evaluate_model(training_data).row(0, named=True)
    print(f"Holdout month: {results['VALIDATION_MONTH']:%Y-%m}; "
          f"products: {results['PRODUCTS_EVALUATED']}")
    print(f"MAE: ${results['MAE_DOLLARS']:.2f}")
    wape = results["WAPE_PERCENT"]
    print(f"WAPE: {wape:.2f}%" if wape is not None else "WAPE: N/A (total actual price is zero)")

    model = train_model(training_data)
    model_path = save_model(model)
    predictions = forecast_two_months(prices, model)
    output_path = export_forecasts(predictions)
    print(f"Saved model to {model_path}")
    print(f"Exported {predictions.height} predictions to {output_path}")


if __name__ == "__main__":
    main()
