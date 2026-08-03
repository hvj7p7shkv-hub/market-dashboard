#!/usr/bin/env python3
"""
Fetch Nifty 500 constituents from NSE and identify fresh/new 52-week highs
and stocks within 3% of their 52-week highs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
import warnings
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(os.environ.get("MARKET_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "nse_52w_highs"
BASE_URL = "https://www.nseindia.com"
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/market-data/live-equity-market",
        }
    )
    session.get(BASE_URL, timeout=20)
    return session


def to_float(value) -> float | None:
    if value in (None, "", "-", "—"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def first_value(row: dict, names: list[str]):
    lowered = {str(key).lower(): key for key in row}
    for name in names:
        if "." in name:
            current = row
            found = True
            for part in name.split("."):
                if not isinstance(current, dict):
                    found = False
                    break
                part_key = {str(key).lower(): key for key in current}.get(part.lower())
                if part_key is None:
                    found = False
                    break
                current = current.get(part_key)
            if found and current not in (None, "", "-", "—"):
                return current
            continue
        key = lowered.get(name.lower())
        if key is not None and row.get(key) not in (None, "", "-", "—"):
            return row.get(key)
    return None


def fetch_index(session: requests.Session, index: str) -> list[dict]:
    url = f"{BASE_URL}/api/equity-stockIndices?index={quote(index)}"
    response = session.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def normalize_rows(rows: list[dict], near_pct: float) -> pd.DataFrame:
    records = []
    for row in rows:
        symbol = first_value(row, ["symbol", "identifier", "meta.symbol"])
        if not symbol or str(symbol).upper() in {"NIFTY 500", "NIFTY500"}:
            continue
        company = first_value(row, ["meta.companyName", "companyName", "name", "identifier"]) or symbol
        last_price = to_float(first_value(row, ["lastPrice", "ltp", "last"]))
        open_price = to_float(first_value(row, ["open", "openPrice"]))
        high = to_float(first_value(row, ["dayHigh", "high"]))
        low = to_float(first_value(row, ["dayLow", "low"]))
        previous_close = to_float(first_value(row, ["previousClose", "prevClose"]))
        change_pct = to_float(first_value(row, ["pChange", "perChange", "changePercent"]))
        year_high = to_float(first_value(row, ["yearHigh", "wkhi", "high52", "high52Week", "52WeekHigh"]))
        year_low = to_float(first_value(row, ["yearLow", "wklo", "low52", "low52Week", "52WeekLow"]))
        volume = to_float(first_value(row, ["totalTradedVolume", "totalTradedQty", "volume"]))
        value = to_float(first_value(row, ["totalTradedValue", "totalTradedVal", "value"]))
        if last_price is None or year_high is None or year_high <= 0:
            continue
        distance = (last_price / year_high - 1) * 100
        fresh_high = last_price >= year_high or (high is not None and high >= year_high)
        records.append(
            {
                "symbol": str(symbol).strip(),
                "company": str(company).strip(),
                "last_price": round(last_price, 2),
                "open": open_price,
                "day_high": high,
                "day_low": low,
                "previous_close": previous_close,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "year_high": round(year_high, 2),
                "year_low": round(year_low, 2) if year_low is not None else None,
                "distance_from_52w_high_pct": round(distance, 2),
                "fresh_52w_high": bool(fresh_high),
                "within_3pct_52w_high": bool(distance >= -near_pct),
                "volume": int(volume) if volume is not None else None,
                "traded_value": value,
            }
        )
    data = pd.DataFrame(records)
    if data.empty:
        return data
    return data.sort_values(["fresh_52w_high", "distance_from_52w_high_pct", "change_pct"], ascending=[False, False, False])


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Nifty 500 fresh and near 52-week highs from NSE.")
    parser.add_argument("--index", default="NIFTY 500", help="NSE index name. Default: NIFTY 500.")
    parser.add_argument("--near-pct", type=float, default=3.0, help="Near-high threshold in percent. Default: 3.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw", action="store_true", help="Also write raw NSE JSON.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        session = make_session()
        rows = fetch_index(session, args.index)
    except requests.RequestException as exc:
        raise SystemExit(f"NSE request failed: {exc}") from exc
    time.sleep(0.2)
    data = normalize_rows(rows, args.near_pct)
    stamp = dt.date.today().isoformat()
    slug = args.index.lower().replace(" ", "-")
    csv_path = args.output_dir / f"{stamp}-{slug}-52w-highs.csv"
    data.to_csv(csv_path, index=False)
    if args.raw:
        raw_path = args.output_dir / f"{stamp}-{slug}-raw.json"
        raw_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fresh_count = int(data["fresh_52w_high"].sum()) if not data.empty else 0
    near_count = int(data["within_3pct_52w_high"].sum()) if not data.empty else 0
    print(f"Rows: {len(data)}")
    print(f"Fresh 52-week highs: {fresh_count}")
    print(f"Within {args.near_pct:.1f}% of 52-week high: {near_count}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
