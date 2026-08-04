#!/usr/bin/env python3
"""
Build a Nifty 500 rotation table using price momentum, volume expansion,
relative strength, and RSI state.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import math
import os
import re
import time
import warnings
from pathlib import Path
from urllib.parse import quote

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import pandas as pd
import requests

ROOT = Path(os.environ.get("MARKET_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "nifty500_rotation"
DEFAULT_INDEX = "NIFTY 500"
DEFAULT_BENCHMARK = "^CRSLDX"
DEFAULT_BENCHMARK_FALLBACKS = ["^CNX500", "^NSEI", "^NSEMDCP50"]
BASE_URL = "https://www.nseindia.com"
NIFTY500_CONSTITUENT_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

try:
    import yfinance as yf
except Exception:  # pragma: no cover - runtime dependency
    yf = None

try:
    from setup_pattern_detection import scan_setups
except Exception:  # pragma: no cover - optional local helper
    scan_setups = None


def make_nse_session() -> requests.Session:
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


def fetch_index_rows(index: str) -> list[dict]:
    session = make_nse_session()
    url = f"{BASE_URL}/api/equity-stockIndices?index={quote(index)}"
    response = session.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


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


def normalize_constituent_csv(data: pd.DataFrame, index: str) -> pd.DataFrame:
    lowered = {str(column).strip().lower(): column for column in data.columns}
    symbol_column = lowered.get("symbol")
    name_column = lowered.get("company name") or lowered.get("company") or lowered.get("name")
    sector_column = lowered.get("industry") or lowered.get("sector")
    if symbol_column is None:
        raise ValueError("Constituent CSV does not contain a Symbol column.")
    records = []
    seen: set[str] = set()
    for _, row in data.iterrows():
        symbol = clean_symbol(row.get(symbol_column, ""))
        if not symbol or symbol in seen:
            continue
        name = str(row.get(name_column, symbol)).strip() if name_column else symbol
        sector = str(row.get(sector_column, "Unknown")).strip() if sector_column else "Unknown"
        records.append(
            {
                "symbol": symbol,
                "name": name,
                "ticker": yahoo_ticker(symbol),
                "sector": sector or "Unknown",
                "source_index": index,
            }
        )
        seen.add(symbol)
    if not records:
        raise ValueError("Constituent CSV did not contain usable symbols.")
    return pd.DataFrame(records).sort_values("symbol").reset_index(drop=True)


def clean_symbol(symbol: str) -> str:
    return re.sub(r"\s+", "", str(symbol).strip().upper())


def yahoo_ticker(symbol: str) -> str:
    symbol = clean_symbol(symbol)
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def load_universe_from_nse_live(index: str) -> pd.DataFrame:
    rows = fetch_index_rows(index)
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        symbol = clean_symbol(first_value(row, ["symbol", "identifier", "meta.symbol"]) or "")
        if not symbol or symbol in {"NIFTY500", "NIFTY 500"} or symbol in seen:
            continue
        company = first_value(row, ["meta.companyName", "companyName", "name", "identifier"]) or symbol
        sector = first_value(row, ["meta.industry", "industry", "sector", "meta.sector"]) or "Unknown"
        records.append(
            {
                "symbol": symbol,
                "name": str(company).strip(),
                "ticker": yahoo_ticker(symbol),
                "sector": str(sector).strip() or "Unknown",
                "source_index": index,
            }
        )
        seen.add(symbol)
    data = pd.DataFrame(records)
    if data.empty:
        raise RuntimeError(f"No constituents returned by NSE for {index}.")
    return data.sort_values("symbol").reset_index(drop=True)


def load_universe(index: str, universe_csv: Path | None = None) -> pd.DataFrame:
    if universe_csv is not None:
        return normalize_constituent_csv(pd.read_csv(universe_csv), index)
    try:
        return normalize_constituent_csv(fetch_niftyindices_constituents(), index)
    except Exception as first_error:
        try:
            return load_universe_from_nse_live(index)
        except Exception as second_error:
            raise RuntimeError(
                "Could not fetch Nifty 500 constituents from the official Nifty Indices CSV "
                f"or NSE live API. CSV error: {first_error}. NSE error: {second_error}"
            ) from second_error


def pct_return(series: pd.Series, days: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= days:
        return None
    return float((clean.iloc[-1] / clean.iloc[-days - 1] - 1) * 100)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def latest_rolling(series: pd.Series, window: int) -> float | None:
    if len(series.dropna()) < window:
        return None
    value = series.rolling(window).mean().iloc[-1]
    return float(value) if pd.notna(value) else None


def get_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = [str(value) for value in frame.columns.get_level_values(0)]
        level1 = [str(value) for value in frame.columns.get_level_values(1)]
        if column in level0:
            series = frame.xs(column, axis=1, level=0, drop_level=True)
        elif column in level1:
            series = frame.xs(column, axis=1, level=1, drop_level=True)
        else:
            return pd.Series(dtype=float)
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return pd.to_numeric(series, errors="coerce").dropna()
    if column not in frame:
        return pd.Series(dtype=float)
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce").dropna()


def clean_close(frame: pd.DataFrame) -> pd.Series:
    close = get_series(frame, "Adj Close")
    if close.empty:
        close = get_series(frame, "Close")
    return close.astype(float)


def clean_volume(frame: pd.DataFrame, close_index: pd.Index) -> pd.Series:
    volume = get_series(frame, "Volume")
    if volume.empty:
        return pd.Series(0.0, index=close_index)
    return volume.reindex(close_index).fillna(0).astype(float)


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


def download_benchmark(preferred: str, fallbacks: list[str], period: str) -> tuple[str, pd.Series]:
    candidates = [preferred, *fallbacks]
    seen: set[str] = set()
    for ticker in candidates:
        ticker = ticker.strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        frame = yfinance_download([ticker], period)
        frame = extract_ticker_frame(frame, ticker, single_ticker=True)
        close = clean_close(frame)
        if len(close) >= 70:
            return ticker, close
    print(
        "Benchmark data was not available for any of "
        f"{', '.join(seen)}. Continuing with market-breadth metrics only."
    )
    return "", pd.Series(dtype=float)


def signal_for(row: dict[str, object]) -> str:
    setup_quality = str(row.get("setup_quality") or "")
    primary_setup = str(row.get("primary_setup") or "")
    rs_above = bool(row.get("rs_above_50dma"))
    rs_20 = row.get("rs_return_20d_pct")
    vol = row.get("volume_ratio_20d")
    rsi_now = row.get("rsi_14")
    rsi_prev = row.get("rsi_5d_ago")
    above_50 = bool(row.get("above_sma_50"))
    above_200 = bool(row.get("above_sma_200"))
    if "risk" in setup_quality.lower() or "head-and-shoulders risk" in primary_setup.lower():
        return "Risk review"
    if isinstance(rsi_now, (int, float)) and isinstance(rsi_prev, (int, float)) and rsi_prev < 35 <= rsi_now:
        return "Mean reversion bounce"
    if rs_above and isinstance(rs_20, (int, float)) and rs_20 > 0 and above_50:
        if isinstance(vol, (int, float)) and vol >= 1.5:
            return "Strengthening with volume"
        return "Strengthening"
    if above_50 and above_200 and isinstance(rsi_now, (int, float)) and rsi_now > 72:
        return "Leader but extended"
    if setup_quality in {"Actionable watch", "Watch"} and primary_setup:
        return f"Setup watch: {primary_setup}"
    if not rs_above and isinstance(rs_20, (int, float)) and rs_20 < 0:
        return "Weakening"
    return "Neutral / watch"


def score_row(row: dict[str, object]) -> float:
    score = 0.0
    if row.get("above_sma_20"):
        score += 0.7
    if row.get("above_sma_50"):
        score += 1.0
    if row.get("above_sma_200"):
        score += 0.8
    if row.get("rs_above_50dma"):
        score += 1.2
    rs_20 = row.get("rs_return_20d_pct")
    if isinstance(rs_20, (int, float)):
        score += max(min(rs_20 / 5, 1.2), -1.2)
    vol = row.get("volume_ratio_20d")
    if isinstance(vol, (int, float)) and vol >= 1.5:
        score += 0.8
    rsi_now = row.get("rsi_14")
    rsi_prev = row.get("rsi_5d_ago")
    if isinstance(rsi_now, (int, float)):
        if 45 <= rsi_now <= 68:
            score += 0.8
        elif rsi_now > 75:
            score -= 0.7
    if isinstance(rsi_now, (int, float)) and isinstance(rsi_prev, (int, float)) and rsi_prev < 35 <= rsi_now:
        score += 0.9
    setup_score = row.get("setup_score")
    if isinstance(setup_score, (int, float)):
        score += max(min(setup_score, 4), -3) * 0.35
    return round(score, 2)


def round_value(value, digits: int = 2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def analyse(
    universe: pd.DataFrame,
    period: str,
    benchmark: str,
    benchmark_fallbacks: list[str],
    chunk_size: int,
) -> tuple[pd.DataFrame, str]:
    if yf is None:
        raise SystemExit("yfinance is not installed.")
    benchmark_ticker, benchmark_close = download_benchmark(benchmark, benchmark_fallbacks, period)
    has_benchmark = not benchmark_close.empty
    tickers = universe["ticker"].dropna().astype(str).tolist()
    history = download_history(tickers, period, chunk_size)
    rows: list[dict[str, object]] = []
    lookup = universe.set_index("ticker").to_dict(orient="index")
    for ticker in tickers:
        meta = lookup.get(ticker, {})
        symbol = meta.get("symbol", ticker.replace(".NS", ""))
        name = meta.get("name", symbol)
        frame = history.get(ticker, pd.DataFrame())
        close = clean_close(frame)
        if len(close) < 70:
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "ticker": ticker,
                    "sector": meta.get("sector", "Unknown"),
                    "downloaded": False,
                    "download_note": "Yahoo returned no usable price history",
                    "source_index": meta.get("source_index", DEFAULT_INDEX),
                    "benchmark_ticker": benchmark_ticker,
                }
            )
            continue
        volume = clean_volume(frame, close.index)
        aligned = (
            pd.concat([close.rename("stock"), benchmark_close.rename("benchmark")], axis=1).dropna()
            if has_benchmark
            else pd.DataFrame()
        )
        if has_benchmark and len(aligned) < 50:
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "ticker": ticker,
                    "sector": meta.get("sector", "Unknown"),
                    "downloaded": False,
                    "download_note": "Insufficient overlap with benchmark history",
                    "source_index": meta.get("source_index", DEFAULT_INDEX),
                    "benchmark_ticker": benchmark_ticker,
                }
            )
            continue
        ratio = aligned["stock"] / aligned["benchmark"] if has_benchmark else pd.Series(dtype=float)
        latest = float(close.iloc[-1])
        rsi_series = rsi(close).dropna()
        latest_volume = float(volume.iloc[-1])
        avg_volume_20 = latest_rolling(volume, 20)
        turnover_proxy = latest * latest_volume
        sma_20 = latest_rolling(close, 20)
        sma_50 = latest_rolling(close, 50)
        sma_200 = latest_rolling(close, 200)
        rs_sma_50 = latest_rolling(ratio, 50) if has_benchmark else None
        rsi_now = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
        rsi_5d_ago = float(rsi_series.iloc[-6]) if len(rsi_series) >= 6 else None
        row = {
            "symbol": symbol,
            "name": name,
            "ticker": ticker,
            "sector": meta.get("sector", "Unknown"),
            "downloaded": True,
            "download_note": "",
            "source_index": meta.get("source_index", DEFAULT_INDEX),
            "benchmark_ticker": benchmark_ticker,
            "last_date": close.index[-1].date().isoformat(),
            "last": round(latest, 2),
            "return_1d_pct": pct_return(close, 1),
            "return_5d_pct": pct_return(close, 5),
            "return_20d_pct": pct_return(close, 20),
            "volume": round_value(latest_volume, 0),
            "avg_volume_20": round_value(avg_volume_20, 0),
            "volume_ratio_20d": round(latest_volume / avg_volume_20, 2) if avg_volume_20 else None,
            "turnover_proxy": round_value(turnover_proxy, 2),
            "rsi_14": round_value(rsi_now),
            "rsi_5d_ago": round_value(rsi_5d_ago),
            "above_sma_20": bool(sma_20 and latest > sma_20),
            "above_sma_50": bool(sma_50 and latest > sma_50),
            "above_sma_200": bool(sma_200 and latest > sma_200) if sma_200 else None,
            "distance_sma_20_pct": round_value((latest / sma_20 - 1) * 100) if sma_20 else None,
            "distance_sma_50_pct": round_value((latest / sma_50 - 1) * 100) if sma_50 else None,
            "rs_ratio": round_value(float(ratio.iloc[-1]), 8) if has_benchmark and len(ratio) else None,
            "rs_ratio_sma_50": round_value(rs_sma_50, 8),
            "rs_distance_sma_50_pct": round_value((float(ratio.iloc[-1]) / rs_sma_50 - 1) * 100)
            if has_benchmark and rs_sma_50
            else None,
            "rs_return_20d_pct": pct_return(ratio, 20) if has_benchmark else None,
            "rs_above_50dma": bool(rs_sma_50 and float(ratio.iloc[-1]) > rs_sma_50)
            if has_benchmark and len(ratio)
            else None,
        }
        if scan_setups is not None:
            row.update(scan_setups(frame))
        row["rotation_signal"] = signal_for(row)
        row["rotation_score"] = score_row(row)
        rows.append(row)
    data = pd.DataFrame(rows)
    for column in ["return_1d_pct", "return_5d_pct", "return_20d_pct", "rs_return_20d_pct"]:
        if column in data:
            data[column] = data[column].map(lambda value: round(float(value), 2) if pd.notna(value) else None)
    if "rotation_score" in data:
        data = data.sort_values(
            ["downloaded", "rotation_score", "rs_return_20d_pct", "volume_ratio_20d"],
            ascending=[False, False, False, False],
            na_position="last",
        )
    return data.reset_index(drop=True), benchmark_ticker


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse Nifty 500 stock rotation.")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="NSE index name. Default: NIFTY 500.")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--benchmark-fallbacks",
        default=",".join(DEFAULT_BENCHMARK_FALLBACKS),
        help="Comma-separated Yahoo benchmark fallbacks.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--universe-csv",
        type=Path,
        default=None,
        help="Optional local constituent CSV with Symbol and Company Name columns.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional constituent limit for smoke tests.")
    parser.add_argument("--chunk-size", type=int, default=80, help="Yahoo download batch size.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.65,
        help="Minimum constituent download coverage required before replacing the rotation snapshot.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        universe = load_universe(args.index, args.universe_csv)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit and args.limit > 0:
        universe = universe.head(args.limit).copy()
    stamp = dt.date.today().isoformat()
    universe_path = args.output_dir / f"{stamp}-nifty500-universe.csv"
    universe.to_csv(universe_path, index=False)
    fallbacks = [item.strip() for item in args.benchmark_fallbacks.split(",") if item.strip()]
    data, benchmark_ticker = analyse(universe, args.period, args.benchmark, fallbacks, args.chunk_size)
    downloaded_count = int(data.get("downloaded", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    expected_count = len(universe)
    coverage = downloaded_count / expected_count if expected_count else 0.0
    output = args.output_dir / f"{stamp}-nifty500-rotation.csv"
    if coverage < args.min_coverage:
        existing_snapshots = list(args.output_dir.glob("*-nifty500-rotation.csv"))
        if downloaded_count > 0 and not existing_snapshots:
            data.to_csv(output, index=False)
            print(
                f"Warning: Nifty 500 download coverage was only {downloaded_count}/{expected_count} "
                f"({coverage:.0%}). Wrote the partial first snapshot so the dashboard can show coverage."
            )
            print(f"Nifty 500 constituents: {expected_count}")
            print(f"Nifty 500 rotation coverage: {downloaded_count}/{expected_count}")
            print(f"Benchmark used: {benchmark_ticker or 'unavailable'}")
            print(f"Universe CSV: {universe_path}")
            print(f"Rotation CSV: {output}")
            return 0
        raise SystemExit(
            f"Nifty 500 download coverage was {downloaded_count}/{expected_count} "
            f"({coverage:.0%}); rotation snapshot was not overwritten. Universe CSV: {universe_path}"
        )
    data.to_csv(output, index=False)
    print(f"Nifty 500 constituents: {expected_count}")
    print(f"Nifty 500 rotation coverage: {downloaded_count}/{expected_count}")
    print(f"Benchmark used: {benchmark_ticker or 'unavailable'}")
    print(f"Universe CSV: {universe_path}")
    print(f"Rotation CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
