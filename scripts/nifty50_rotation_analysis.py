#!/usr/bin/env python3
"""
Build a Nifty 50 rotation table using price momentum, volume expansion,
relative strength versus Nifty 50, and RSI state.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import math
import os
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("MARKET_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "nifty50_rotation"
DEFAULT_BENCHMARK = "^NSEI"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

try:
    import yfinance as yf
except Exception:  # pragma: no cover - runtime dependency
    yf = None


NIFTY50 = [
    ("ADANIENT", "Adani Enterprises", "ADANIENT.NS"),
    ("ADANIPORTS", "Adani Ports", "ADANIPORTS.NS"),
    ("APOLLOHOSP", "Apollo Hospitals", "APOLLOHOSP.NS"),
    ("ASIANPAINT", "Asian Paints", "ASIANPAINT.NS"),
    ("AXISBANK", "Axis Bank", "AXISBANK.NS"),
    ("BAJAJ-AUTO", "Bajaj Auto", "BAJAJ-AUTO.NS"),
    ("BAJFINANCE", "Bajaj Finance", "BAJFINANCE.NS"),
    ("BAJAJFINSV", "Bajaj Finserv", "BAJAJFINSV.NS"),
    ("BEL", "Bharat Electronics", "BEL.NS"),
    ("BHARTIARTL", "Bharti Airtel", "BHARTIARTL.NS"),
    ("CIPLA", "Cipla", "CIPLA.NS"),
    ("COALINDIA", "Coal India", "COALINDIA.NS"),
    ("DRREDDY", "Dr Reddy's Labs", "DRREDDY.NS"),
    ("EICHERMOT", "Eicher Motors", "EICHERMOT.NS"),
    ("ETERNAL", "Eternal", "ETERNAL.NS"),
    ("GRASIM", "Grasim", "GRASIM.NS"),
    ("HCLTECH", "HCL Technologies", "HCLTECH.NS"),
    ("HDFCBANK", "HDFC Bank", "HDFCBANK.NS"),
    ("HDFCLIFE", "HDFC Life", "HDFCLIFE.NS"),
    ("HEROMOTOCO", "Hero MotoCorp", "HEROMOTOCO.NS"),
    ("HINDALCO", "Hindalco", "HINDALCO.NS"),
    ("HINDUNILVR", "Hindustan Unilever", "HINDUNILVR.NS"),
    ("ICICIBANK", "ICICI Bank", "ICICIBANK.NS"),
    ("INDUSINDBK", "IndusInd Bank", "INDUSINDBK.NS"),
    ("INFY", "Infosys", "INFY.NS"),
    ("ITC", "ITC", "ITC.NS"),
    ("JIOFIN", "Jio Financial Services", "JIOFIN.NS"),
    ("JSWSTEEL", "JSW Steel", "JSWSTEEL.NS"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "KOTAKBANK.NS"),
    ("LT", "Larsen & Toubro", "LT.NS"),
    ("M&M", "Mahindra & Mahindra", "M&M.NS"),
    ("MARUTI", "Maruti Suzuki", "MARUTI.NS"),
    ("NESTLEIND", "Nestle India", "NESTLEIND.NS"),
    ("NTPC", "NTPC", "NTPC.NS"),
    ("ONGC", "ONGC", "ONGC.NS"),
    ("POWERGRID", "Power Grid", "POWERGRID.NS"),
    ("RELIANCE", "Reliance Industries", "RELIANCE.NS"),
    ("SBILIFE", "SBI Life", "SBILIFE.NS"),
    ("SBIN", "State Bank of India", "SBIN.NS"),
    ("SHRIRAMFIN", "Shriram Finance", "SHRIRAMFIN.NS"),
    ("SUNPHARMA", "Sun Pharma", "SUNPHARMA.NS"),
    ("TATACONSUM", "Tata Consumer", "TATACONSUM.NS"),
    ("TATAMOTORS", "Tata Motors", "TATAMOTORS.NS"),
    ("TATASTEEL", "Tata Steel", "TATASTEEL.NS"),
    ("TCS", "TCS", "TCS.NS"),
    ("TECHM", "Tech Mahindra", "TECHM.NS"),
    ("TITAN", "Titan", "TITAN.NS"),
    ("TRENT", "Trent", "TRENT.NS"),
    ("ULTRACEMCO", "UltraTech Cement", "ULTRACEMCO.NS"),
    ("WIPRO", "Wipro", "WIPRO.NS"),
]


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


def clean_close(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    close = data["Adj Close"] if "Adj Close" in data else data["Close"]
    return close.dropna().astype(float)


def download_one(ticker: str, period: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return yf.download(ticker, period=period, auto_adjust=False, progress=False, timeout=25)


def signal_for(row: dict[str, object]) -> str:
    rs_above = bool(row.get("rs_above_50dma"))
    rs_20 = row.get("rs_return_20d_pct")
    vol = row.get("volume_ratio_20d")
    rsi_now = row.get("rsi_14")
    rsi_prev = row.get("rsi_5d_ago")
    above_50 = bool(row.get("above_sma_50"))
    above_200 = bool(row.get("above_sma_200"))
    if isinstance(rsi_now, (int, float)) and isinstance(rsi_prev, (int, float)) and rsi_prev < 35 <= rsi_now:
        return "Mean reversion bounce"
    if rs_above and isinstance(rs_20, (int, float)) and rs_20 > 0 and above_50:
        if isinstance(vol, (int, float)) and vol >= 1.5:
            return "Strengthening with volume"
        return "Strengthening"
    if above_50 and above_200 and isinstance(rsi_now, (int, float)) and rsi_now > 72:
        return "Leader but extended"
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
    return round(score, 2)


def analyse(period: str, benchmark: str) -> pd.DataFrame:
    benchmark_data = download_one(benchmark, period)
    benchmark_close = clean_close(benchmark_data)
    if benchmark_close.empty:
        raise SystemExit(f"No benchmark data downloaded for {benchmark}.")
    rows = []
    for symbol, name, ticker in NIFTY50:
        data = download_one(ticker, period)
        if data.empty:
            rows.append({"symbol": symbol, "name": name, "ticker": ticker, "downloaded": False})
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close = clean_close(data)
        if len(close) < 70:
            rows.append({"symbol": symbol, "name": name, "ticker": ticker, "downloaded": False})
            continue
        volume = data["Volume"].reindex(close.index).fillna(0).astype(float) if "Volume" in data else pd.Series(0, index=close.index)
        aligned = pd.concat([close.rename("stock"), benchmark_close.rename("benchmark")], axis=1).dropna()
        ratio = aligned["stock"] / aligned["benchmark"]
        latest = float(close.iloc[-1])
        rsi_series = rsi(close)
        latest_volume = float(volume.iloc[-1])
        avg_volume_20 = float(volume.rolling(20).mean().iloc[-1])
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else math.nan
        rs_sma_50 = float(ratio.rolling(50).mean().iloc[-1]) if len(ratio) >= 50 else math.nan
        row = {
            "symbol": symbol,
            "name": name,
            "ticker": ticker,
            "downloaded": True,
            "last_date": close.index[-1].date().isoformat(),
            "last": round(latest, 2),
            "return_1d_pct": pct_return(close, 1),
            "return_5d_pct": pct_return(close, 5),
            "return_20d_pct": pct_return(close, 20),
            "volume_ratio_20d": round(latest_volume / avg_volume_20, 2) if avg_volume_20 else None,
            "rsi_14": round(float(rsi_series.iloc[-1]), 2),
            "rsi_5d_ago": round(float(rsi_series.iloc[-6]), 2) if len(rsi_series.dropna()) >= 6 else None,
            "above_sma_20": bool(latest > sma_20),
            "above_sma_50": bool(latest > sma_50),
            "above_sma_200": bool(latest > sma_200) if not math.isnan(sma_200) else None,
            "distance_sma_20_pct": round((latest / sma_20 - 1) * 100, 2) if sma_20 else None,
            "distance_sma_50_pct": round((latest / sma_50 - 1) * 100, 2) if sma_50 else None,
            "rs_ratio": round(float(ratio.iloc[-1]), 8) if not ratio.empty else None,
            "rs_ratio_sma_50": round(rs_sma_50, 8) if not math.isnan(rs_sma_50) else None,
            "rs_distance_sma_50_pct": round((float(ratio.iloc[-1]) / rs_sma_50 - 1) * 100, 2)
            if not ratio.empty and not math.isnan(rs_sma_50) and rs_sma_50
            else None,
            "rs_return_20d_pct": pct_return(ratio, 20),
            "rs_above_50dma": bool(float(ratio.iloc[-1]) > rs_sma_50) if not ratio.empty and not math.isnan(rs_sma_50) else None,
        }
        row["rotation_signal"] = signal_for(row)
        row["rotation_score"] = score_row(row)
        rows.append(row)
    data = pd.DataFrame(rows)
    for column in ["return_1d_pct", "return_5d_pct", "return_20d_pct", "rs_return_20d_pct"]:
        if column in data:
            data[column] = data[column].map(lambda value: round(float(value), 2) if pd.notna(value) else None)
    return data.sort_values(["rotation_score", "rs_return_20d_pct", "volume_ratio_20d"], ascending=[False, False, False], na_position="last")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse Nifty 50 sector/stock rotation.")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80,
        help="Minimum share of Nifty 50 constituents that must download before replacing the snapshot.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = analyse(args.period, args.benchmark)
    downloaded_count = int(data.get("downloaded", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    expected_count = len(NIFTY50)
    coverage = downloaded_count / expected_count if expected_count else 0.0
    if coverage < args.min_coverage:
        raise SystemExit(
            f"Nifty 50 download coverage was {downloaded_count}/{expected_count} "
            f"({coverage:.0%}); rotation snapshot was not overwritten."
        )
    output = args.output_dir / f"{dt.date.today().isoformat()}-nifty50-rotation.csv"
    data.to_csv(output, index=False)
    print(f"Nifty 50 rotation coverage: {downloaded_count}/{expected_count}")
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
