# Description

As a consumer, I want to train a machine learning model using historical product price data to predict prices one month ahead.

The training process should be implemented entirely within the existing Jupyter Notebook.

# Acceptance Criteria

- Use **LightGBM** to train the model.
- Use the prepared dataset containing historical prices and engineered features.
- Evaluate the model using the following metrics:
  - **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual prices, expressed in dollars (e.g., $0.45).
  - **WAPE (Weighted Absolute Percentage Error):** Measures the total absolute prediction error divided by the sum of actual prices, expressed as a percentage. Useful for evaluating overall prediction accuracy across products with different prices.
- Display the evaluation results in the notebook.
- Keep all training and evaluation code **inside the existing Jupyter Notebook**.
- Do not create additional Python files or modules.