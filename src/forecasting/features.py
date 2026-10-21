"""Calendar-aligned features shared by training and recursive inference."""

import polars as pl

from .config import FEATURE_COLUMNS, HISTORY_FEATURES


def _monthly_features(prices: pl.DataFrame) -> pl.DataFrame:
    """Include an empty next month so training and inference share feature logic."""
    missing = {"DATE", "PRODUCTS", "VALUE"} - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if prices.is_empty():
        raise ValueError("At least one monthly price is required.")

    date = pl.col("DATE")
    if prices.schema["DATE"] == pl.String:
        date = date.str.to_date("%Y-%m")
    elif prices.schema["DATE"] != pl.Date and not isinstance(
        prices.schema["DATE"], pl.Datetime
    ):
        raise ValueError("DATE must contain YYYY-MM strings, dates, or datetimes.")

    monthly = prices.select(
        date.cast(pl.Date).dt.month_start().alias("DATE"),
        pl.col("PRODUCTS").cast(pl.String),
        pl.col("VALUE").cast(pl.Float64).fill_nan(None),
    ).with_columns(
        pl.when(pl.col("VALUE").is_finite())
        .then(pl.col("VALUE"))
        .otherwise(None)
        .alias("VALUE")
    )
    if monthly.select(pl.any_horizontal(pl.col("DATE", "PRODUCTS").is_null()).any()).item():
        raise ValueError("DATE and PRODUCTS cannot contain nulls.")
    if monthly.select(pl.struct("PRODUCTS", "DATE").is_duplicated().any()).item():
        raise ValueError("Expected one price per product and month; filter geography first.")

    # A fixed Enum preserves the same categorical codes across time-based splits.
    products = sorted(monthly.get_column("PRODUCTS").unique().to_list())
    monthly = monthly.with_columns(pl.col("PRODUCTS").cast(pl.Enum(products)))
    next_month = monthly.select(pl.col("DATE").max().dt.offset_by("1mo")).item()
    calendar = (
        monthly.group_by("PRODUCTS")
        .agg(pl.col("DATE").min().alias("START"))
        .with_columns(
            pl.date_ranges("START", pl.lit(next_month), interval="1mo").alias("DATE")
        )
        .explode("DATE")
        .select("PRODUCTS", "DATE")
    )
    features = (
        calendar.join(monthly, on=["PRODUCTS", "DATE"], how="left")
        .sort("PRODUCTS", "DATE")
        .with_columns(
            pl.col("DATE").dt.month().alias("MONTH"),
            pl.col("DATE").dt.year().alias("YEAR"),
            *[
                pl.col("VALUE").shift(lag).over("PRODUCTS").alias(f"LAG_{lag}")
                for lag in (1, 2, 3)
            ],
        )
        .with_columns(
            ((pl.col("LAG_1") + pl.col("LAG_2") + pl.col("LAG_3")) / 3)
            .alias("ROLLING_MEAN_3"),
            (pl.col("LAG_1") - pl.col("LAG_3")).alias("PRICE_CHANGE_3"),
        )
    )
    return features.select("DATE", *FEATURE_COLUMNS, "VALUE").sort("DATE", "PRODUCTS")


def prepare_training_data(prices: pl.DataFrame) -> pl.DataFrame:
    """Return labeled rows with three complete historical calendar months.

    DATE, MONTH and YEAR describe the target month. VALUE is that month's
    observed price; every price feature uses only earlier months. Missing target
    prices and incomplete histories are dropped without filling from the future.
    PRODUCTS is an Enum suitable for categorical encoding with Polars.
    """
    return _monthly_features(prices).drop_nulls(["VALUE", *HISTORY_FEATURES])


def prepare_next_month_data(prices: pl.DataFrame) -> pl.DataFrame:
    """Return eligible products for the month after the latest input month.

    VALUE is intentionally null. Products missing any of the last three prices
    are omitted. Pass the same product universe as training to preserve codes.
    """
    features = _monthly_features(prices)
    return features.filter(pl.col("DATE") == pl.col("DATE").max()).drop_nulls(
        HISTORY_FEATURES
    )


def lightgbm_features(dataset: pl.DataFrame) -> pl.DataFrame:
    """Select numeric model inputs, encoding PRODUCTS with its fixed Enum codes.

    Pass the result's .to_numpy() to LightGBM with feature_name=FEATURE_COLUMNS
    and categorical_feature=["PRODUCTS"]. Keep DATE for chronological splitting
    and VALUE for labels; neither belongs in the feature matrix.
    """
    if not isinstance(dataset.schema["PRODUCTS"], pl.Enum):
        raise ValueError("PRODUCTS must retain the Enum from preprocessing.")
    return dataset.select(
        pl.col("PRODUCTS").to_physical().cast(pl.Int32),
        *[pl.col(name) for name in FEATURE_COLUMNS if name != "PRODUCTS"],
    )
