# Description

I want to prepare my dataset for training a machine learning model that predicts product prices for the next month.

The current dataset does not contain all the features required for training. We need to generate additional features based on historical price data.

Each row should represent a product for a specific month.

# Example of the Target Dataset

| Feature | Description |
|---|---|
| `PRODUCTS` | Unique product identifier |
| `MONTH` | Month of the year (1–12) to capture seasonality |
| `YEAR` | Year to capture long-term price trends |
| `LAG_1` | Product price from the previous month |
| `LAG_2` | Product price from two months ago |
| `LAG_3` | Product price from three months ago |
| `ROLLING_MEAN_3` | Average product price over the previous three months |
| `PRICE_CHANGE_3` | Price change between the previous month and three months ago |
| `VALUE` | Target variable: actual product price for the month |

# Requirements

- Use **Polars** for all data transformations.
- Calculate lag and rolling features independently for each product.
- Sort the data chronologically before generating features.
- Prevent data leakage by using only historical prices to calculate features.
- Handle missing values caused by insufficient historical data.
- Prepare the final dataset for training with **LightGBM**.
- The model should predict product prices **one month ahead**.