# Goal

As a consumer, I want to forecast product prices for the next two months using the trained LightGBM model and export the predictions to a JSON file.

# Acceptance Criteria

- Generate price forecasts for **two consecutive months** following the last available month in the dataset.
- Generate predictions for each product in the dataset.
- Use a recursive forecasting approach, where the first month's predicted prices are used to calculate the features required for the second month's predictions.
- Perform postprocessing to format the predictions into a structured JSON file.
- Each prediction must include:
  - `PRODUCTS`: Product identifier.
  - `DATE`: Forecasted month in `YYYY-MM` format.
  - `VALUE`: Predicted product price.
- Export the JSON file to the `data` directory at the root of the `AverageForecasting` project.
- Create the `data` directory if it does not exist.
- Keep forecasting and postprocessing logic within the existing Jupyter Notebook.

# Expected JSON Output

```json
[
  {
    "PRODUCTS": "Product_A",
    "DATE": "2026-11",
    "VALUE": 4.52
  },
  {
    "PRODUCTS": "Product_A",
    "DATE": "2026-12",
    "VALUE": 4.61
  }
]
```