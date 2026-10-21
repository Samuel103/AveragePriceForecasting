"""Forecast lookup exposed to the Agents SDK."""

import json

from agents import function_tool

from agent.services.forecasts import query_forecasts


@function_tool
def read_forecasts(
    product: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Read actual exported predictions, available products/months, and price changes.

    Args:
        product: Full product name or case/accent-insensitive substring. Omit for all
            products or to discover French product names before querying a translation.
        date: Exact month in YYYY-MM. Do not combine with start_date or end_date.
        start_date: Inclusive first month in YYYY-MM; omit for all earlier months.
        end_date: Inclusive last month in YYYY-MM; omit for all later months.

    Changes compare the first and last returned month per product. A single month
    cannot establish a trend. Missing months are never interpolated. This tool
    only reads forecasts; it cannot provide observed historical prices.
    """
    return json.dumps(
        query_forecasts(product, date, start_date, end_date),
        ensure_ascii=False,
        allow_nan=False,
    )
