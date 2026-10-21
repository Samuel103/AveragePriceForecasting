# Goal

As a user, I want to ask the AI agent questions about forecasting results and receive accurate answers based on the generated predictions.

# Acceptance Criteria

- Create a tool to read and query the forecasting data stored in `/data/forecasts.json`.
- The agent must automatically use this tool whenever a question relates to forecasting results.
- The tool must support filtering forecasts by product and date.
- The agent must be able to answer questions about predicted prices, price trends, and price changes.
- The agent must base its answers on the actual forecasting data and must not invent values.
- Handle missing files, unknown products, and unavailable forecasts gracefully.

# Out of Scope

- Modifying or generating forecasts.
- Training or retraining the forecasting model.
- External web searches to validate predictions.