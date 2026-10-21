"""Read-only queries against the forecasting pipeline's JSON export."""

import json
import math
import re
import unicodedata
from pathlib import Path


FORECASTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "forecasting/data/predictions/forecasts.json"
)


def _normalize(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(char)
    ).strip()


def _valid_month(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"[0-9]{4}-(0[1-9]|1[0-2])", value
    ) is not None and value[:4] != "0000"


def query_forecasts(
    product: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    path: Path | None = None,
) -> dict:
    """Filter forecasts and compute changes only across returned observations."""
    source = path if path is not None else FORECASTS_PATH
    result = {"source": str(source), "forecasts": [], "changes": []}

    def error(status: str, message: str) -> dict:
        return {**result, "status": status, "message": message}

    for value in (date, start_date, end_date):
        if value is not None and not _valid_month(value):
            return error("invalid_query", "Dates must use YYYY-MM with a valid month.")
    if date is not None and (start_date is not None or end_date is not None):
        return error("invalid_query", "Use either date or a date range, not both.")
    if start_date and end_date and start_date > end_date:
        return error("invalid_query", "start_date must be on or before end_date.")
    if product is not None and not _normalize(product):
        return error("invalid_query", "Product must not be blank; omit it to list all products.")

    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return error("missing_file", "The forecast export is missing. No forecasts are available.")
    except (OSError, UnicodeError):
        return error("unreadable_file", "The forecast export could not be read.")
    except ValueError:
        return error("invalid_data", "The forecast export is not valid JSON.")

    if not isinstance(records, list):
        return error("invalid_data", "The forecast export must contain a list of records.")
    seen = set()
    clean = []
    for row in records:
        if not isinstance(row, dict):
            return error("invalid_data", "A forecast record is invalid.")
        name, month, value = (row.get(key) for key in ("PRODUCTS", "DATE", "VALUE"))
        if (
            not isinstance(name, str) or not name.strip() or not _valid_month(month)
            or type(value) not in (int, float)
        ):
            return error("invalid_data", "A forecast record has an invalid product, date, or price.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (name, month) in seen:
            return error("invalid_data", "Forecasts contain nonfinite prices or duplicate product/month records.")
        seen.add((name, month))
        clean.append({"PRODUCTS": name, "DATE": month, "VALUE": value})

    products = sorted({row["PRODUCTS"] for row in clean})
    matches = products
    if product is not None:
        needle = _normalize(product)
        exact = [name for name in products if _normalize(name) == needle]
        matches = exact or [name for name in products if needle in _normalize(name)]
    result["available_products"] = products
    result["matched_products"] = matches
    result["available_dates"] = sorted({row["DATE"] for row in clean if row["PRODUCTS"] in matches})
    if product is not None and not matches:
        return error("unknown_product", "No matching product. Use the available product names; do not substitute another product.")

    lower, upper = (date, date) if date is not None else (start_date, end_date)
    selected = sorted(
        (row for row in clean if row["PRODUCTS"] in matches
         and (lower is None or row["DATE"] >= lower)
         and (upper is None or row["DATE"] <= upper)),
        key=lambda row: (row["PRODUCTS"], row["DATE"]),
    )
    result["forecasts"] = selected
    if not selected:
        return error("no_forecasts", "No forecasts are available for the requested filters.")

    for name in matches:
        rows = [row for row in selected if row["PRODUCTS"] == name]
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        delta = last["VALUE"] - first["VALUE"]
        steps = [right["VALUE"] - left["VALUE"] for left, right in zip(rows, rows[1:])]
        trend = (
            "stable" if all(step == 0 for step in steps)
            else "increasing" if all(step >= 0 for step in steps)
            else "decreasing" if all(step <= 0 for step in steps)
            else "mixed"
        )
        result["changes"].append({
            "PRODUCTS": name, "start_date": first["DATE"], "end_date": last["DATE"],
            "start_value": first["VALUE"], "end_value": last["VALUE"],
            "absolute_change": delta,
            "percent_change": delta / first["VALUE"] * 100 if first["VALUE"] != 0 else None,
            "trend": trend,
        })
    return {**result, "status": "ok"}
