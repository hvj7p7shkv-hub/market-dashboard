#!/usr/bin/env python3
"""
Fetch or compute Nifty 500 fresh/new 52-week highs and stocks within 3% of
their 52-week highs.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
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
DEFAULT_INDEX = "NIFTY 500"
BASE_URL = "https://www.nseindia.com"
NIFTY500_CONSTITUENT_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

try:
    import yfinance as yf
except Exception:  # pragma: no cover - runtime dependency
    yf = None


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


def clean_symbol(symbol: object) -> str:
    return re.sub(r"\s+", "", str(symbol).strip().upper())


def yahoo_ticker(symbol: object) -> str:
    symbol = clean_symbol(symbol)
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def fetch_index(session: requests.Session, index: str) -> list[dict]:
    url = f"{BASE_URL}/api/equity-stockIndices?index={quote(index)}"
    response = session.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def normalize_rows(rows: list[dict], near_pct: float, source: str) -> pd.DataFrame:
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
                "symbol": clean_symbol(symbol),
                "company": str(company).strip(),
                "ticker": yahoo_ticker(symbol),
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
                "source": source,
            }
        )
    return sort_highs(pd.DataFrame(records))


def fetch_niftyindices_constituents() -> pd.DataFrame:
    response = requests.get(
        NIFTY500_CONSTITUENT_URL,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Accept": "text/csv,*/*",
        },
    )
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def normalize_constituents(data: pd.DataFrame) -> pd.DataFrame:
    lowered = {str(column).strip().lower(): column for column in data.columns}
    symbol_column = lowered.get("symbol")
    name_column = lowered.get("company name") or lowered.get("company") or lowered.get("name")
    if symbol_column is None:
        raise ValueError("Constituent CSV does not contain a Symbol column.")
    records = []
    seen: set[str] = set()
    for _, row in data.iterrows():
        symbol = clean_symbol(row.get(symbol_column, ""))
        if not symbol or symbol in seen:
            continue
        company = str(row.get(name_column, symbol)).strip() if name_column else symbol
        records.append({"symbol": symbol, "company": company, "ticker": yahoo_ticker(symbol)})
        seen.add(symbol)
    if not records:
        raise ValueError("Constituent CSV did not contain usable symbols.")
    return pd.DataFrame(records).sort_values("symbol").reset_index(drop=True)


def get_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce").dropna()


def clean_close(frame: pd.DataFrame) -> pd.Series:
    close = get_series(frame, "Close")
    if close.empty:
        close = get_series(frame, "Adj Close")
    return close.astype(float)


def extract_ticker_frame(downloaded: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()
    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy() if single_ticker else pd.DataFrame()
    level0 = [str(value) for value in downloaded.columns.get_level_values(0)]
    level1 = [str(value) for value in downloaded.columns.get_level_values(1)]
    if ticker in level0:
        return downloaded.xs(ticker, axis=1, level=0, drop_level=True).dropna(how="all")
    if ticker in level1:
        return downloaded.xs(ticker, axis=1, level=1, drop_level=True).dropna(how="all")
    return pd.DataFrame()


def yfinance_download(tickers: list[str], period: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return yf.download(
            tickers=tickers if len(tickers) > 1 else tickers[0],
            period=period,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=40,
        )


def download_history(tickers: list[str], period: str, chunk_size: int) -> dict[str, pd.DataFrame]:
    history: dict[str, pd.DataFrame] = {}
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        downloaded = yfinance_download(chunk, period)
        for ticker in chunk:
            history[ticker] = extract_ticker_frame(downloaded, ticker, single_ticker=len(chunk) == 1)
        time.sleep(0.2)
    return history


def latest_on_close_index(series: pd.Series, close_index: pd.Index) -> float | None:
    aligned = series.reindex(close_index).dropna()
    if aligned.empty:
        return None
    return float(aligned.iloc[-1])


def compute_yahoo_rows(universe: pd.DataFrame, near_pct: float, period: str, chunk_size: int) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed, so Yahoo fallback cannot run.")
    tickers = universe["ticker"].dropna().astype(str).tolist()
    history = download_history(tickers, period, chunk_size)
    records = []
    lookup = universe.set_index("ticker").to_dict(orient="index")
    for ticker in tickers:
        meta = lookup.get(ticker, {})
        frame = history.get(ticker, pd.DataFrame())
        close = clean_close(frame)
        high = get_series(frame, "High")
        low = get_series(frame, "Low")
        volume = get_series(frame, "Volume")
        if len(close) < 20:
            continue
        high = high.reindex(close.index).dropna()
        low = low.reindex(close.index).dropna()
        volume = volume.reindex(close.index).fillna(0)
        if high.empty:
            high = close
        last_price = float(close.iloc[-1])
        previous_close = float(close.iloc[-2]) if len(close) >= 2 else None
        day_high = latest_on_close_index(high, close.index)
        day_low = latest_on_close_index(low, close.index)
        latest_volume = latest_on_close_index(volume, close.index)
        year_high = float(high.max())
        year_low = float(low.min()) if not low.empty else float(close.min())
        if year_high <= 0:
            continue
        distance = (last_price / year_high - 1) * 100
        change_pct = (last_price / previous_close - 1) * 100 if previous_close else None
        fresh_high = last_price >= year_high or (day_high is not None and day_high >= year_high)
        records.append(
            {
                "symbol": meta.get("symbol", ticker.replace(".NS", "")),
                "company": meta.get("company", meta.get("symbol", ticker.replace(".NS", ""))),
                "ticker": ticker,
                "last_date": close.index[-1].date().isoformat(),
                "last_price": round(last_price, 2),
                "open": None,
                "day_high": round(day_high, 2) if day_high is not None else None,
                "day_low": round(day_low, 2) if day_low is not None else None,
                "previous_close": round(previous_close, 2) if previous_close is not None else None,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "year_high": round(year_high, 2),
                "year_low": round(year_low, 2),
                "distance_from_52w_high_pct": round(distance, 2),
                "fresh_52w_high": bool(fresh_high),
                "within_3pct_52w_high": bool(distance >= -near_pct),
                "volume": int(latest_volume) if latest_volume is not None else None,
                "traded_value": None,
                "source": "Nifty Indices constituents + Yahoo history",
            }
        )
    return sort_highs(pd.DataFrame(records))


def sort_highs(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    return data.sort_values(
        ["fresh_52w_high", "distance_from_52w_high_pct", "change_pct"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch or compute Nifty 500 fresh and near 52-week highs.")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="NSE index name. Default: NIFTY 500.")
    parser.add_argument("--near-pct", type=float, default=3.0, help="Near-high threshold in percent. Default: 3.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw", action="store_true", help="Also write raw NSE JSON when NSE live data works.")
    parser.add_argument("--period", default="1y", help="Yahoo fallback lookback period. Default: 1y.")
    parser.add_argument("--chunk-size", type=int, default=80, help="Yahoo fallback download batch size.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.60,
        help="Minimum Yahoo fallback constituent coverage required before writing the snapshot.",
    )
    parser.add_argument(
        "--universe-csv",
        type=Path,
        default=None,
        help="Optional local constituent CSV with Symbol and Company Name columns.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.universe_csv:
        source = "Nifty Indices constituents + Yahoo history"
        constituent_frame = pd.read_csv(args.universe_csv)
        universe = normalize_constituents(constituent_frame)
        data = compute_yahoo_rows(universe, args.near_pct, args.period, args.chunk_size)
        coverage = len(data) / len(universe) if len(universe) else 0.0
        if coverage < args.min_coverage:
            raise SystemExit(
                f"Yahoo coverage was only {len(data)}/{len(universe)} ({coverage:.0%}); "
                "52-week-high snapshot was not overwritten."
            )
        print(f"Yahoo coverage: {len(data)}/{len(universe)}")
    else:
        source = "NSE live"
        try:
            session = make_session()
            rows = fetch_index(session, args.index)
            data = normalize_rows(rows, args.near_pct, source=source)
            if data.empty:
                raise RuntimeError("NSE live data returned no usable Nifty 500 52-week-high rows.")
        except Exception as exc:
            print(f"NSE live request failed: {exc}")
            print("Falling back to the official Nifty Indices constituent list and Yahoo history...")
            source = "Nifty Indices constituents + Yahoo history"
            try:
                constituent_frame = fetch_niftyindices_constituents()
                universe = normalize_constituents(constituent_frame)
                data = compute_yahoo_rows(universe, args.near_pct, args.period, args.chunk_size)
                coverage = len(data) / len(universe) if len(universe) else 0.0
                if coverage < args.min_coverage:
                    raise RuntimeError(
                        f"Yahoo fallback coverage was only {len(data)}/{len(universe)} ({coverage:.0%}); "
                        "52-week-high snapshot was not overwritten."
                    )
                print(f"Yahoo fallback coverage: {len(data)}/{len(universe)}")
            except Exception as fallback_exc:
                raise SystemExit(f"Nifty 500 highs could not be refreshed: {fallback_exc}") from fallback_exc
    time.sleep(0.2)
    stamp = dt.date.today().isoformat()
    slug = args.index.lower().replace(" ", "-")
    csv_path = args.output_dir / f"{stamp}-{slug}-52w-highs.csv"
    data.to_csv(csv_path, index=False)
    if args.raw and rows:
        raw_path = args.output_dir / f"{stamp}-{slug}-raw.json"
        raw_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fresh_count = int(data["fresh_52w_high"].sum()) if not data.empty else 0
    near_count = int(data["within_3pct_52w_high"].sum()) if not data.empty else 0
    print(f"Source: {source}")
    print(f"Rows: {len(data)}")
    print(f"Fresh 52-week highs: {fresh_count}")
    print(f"Within {args.near_pct:.1f}% of 52-week high: {near_count}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
