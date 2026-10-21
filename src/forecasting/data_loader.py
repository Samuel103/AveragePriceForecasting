"""Load the Statistics Canada source using the notebook's business filters."""

from pathlib import Path

import polars as pl

from .config import RAW_DATA_PATH


def load_prices(path: str | Path = RAW_DATA_PATH) -> pl.DataFrame:
    """Return Québec prices from January 2024, with month-start dates."""
    prices = (
        pl.read_csv(path, separator=";", schema_overrides={"VALEUR": pl.String})
        .filter(pl.col("GÉO") == "Québec")
        .select(
            pl.col("PÉRIODE DE RÉFÉRENCE").str.to_date("%Y-%m").alias("DATE"),
            pl.col("Produits").cast(pl.String).alias("PRODUCTS"),
            pl.col("VALEUR").cast(pl.Float64, strict=False).alias("VALUE"),
        )
        .filter(pl.col("DATE") >= pl.date(2024, 1, 1))
        .with_columns(
            pl.when(pl.col("VALUE").is_finite())
            .then(pl.col("VALUE"))
            .otherwise(None)
            .alias("VALUE")
        )
    )
    return prices.sort("DATE", "PRODUCTS")
