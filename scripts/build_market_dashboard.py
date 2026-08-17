#!/usr/bin/env python3
"""
Build a local market dashboard from news wires, index technicals, and the
52-week-high technical ranking output.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import math
import os
import re
import shutil
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("MARKET_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("MARKET_DASHBOARD_DATA_ROOT", ROOT / "outputs"))
DEFAULT_OUTPUT_DIR = DATA_ROOT / "market_dashboard"
WIRES_DIR = DATA_ROOT / "market_news_wires"
TECHNICAL_SUMMARY = DATA_ROOT / "batch_52w_technical" / "technical_ranked_summary.csv"
NSE_HIGH_DIR = DATA_ROOT / "nse_52w_highs"
MUTUAL_FUND_NAV_DIR = DATA_ROOT / "mutual_fund_navs"
NIFTY50_ROTATION_DIR = DATA_ROOT / "nifty50_rotation"
NIFTY500_ROTATION_DIR = DATA_ROOT / "nifty500_rotation"
BEAR_DASHBOARD_NOTES_DIR = DATA_ROOT / "bear_dashboard_notes"
LIVE_MARKET_SNAPSHOT = DATA_ROOT / "live_market" / "latest-live-market-snapshot.json"
BREADTH_LEADERSHIP_SNAPSHOT = DEFAULT_OUTPUT_DIR / "breadth_leadership_snapshot.json"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

try:
    import yfinance as yf
except Exception:  # pragma: no cover - runtime dependency
    yf = None


INDEX_TICKERS = [
    ("India", "Nifty 50", "^NSEI"),
    ("India", "Nifty Midcap 50", "^NSEMDCP50"),
    ("India", "Nifty Bank", "^NSEBANK"),
    ("India", "India VIX", "^INDIAVIX"),
    ("India Sectors", "Nifty Auto", "^CNXAUTO"),
    ("India Sectors", "Nifty IT", "^CNXIT"),
    ("India Sectors", "Nifty FMCG", "^CNXFMCG"),
    ("India Sectors", "Nifty Pharma", "^CNXPHARMA"),
    ("India Sectors", "Nifty Metal", "^CNXMETAL"),
    ("India Sectors", "Nifty Realty", "^CNXREALTY"),
    ("India Sectors", "Nifty Energy", "^CNXENERGY"),
    ("India Sectors", "Nifty PSU Bank", "^CNXPSUBANK"),
    ("India Sectors", "Nifty Private Bank", "^NIFPVTBANK"),
    ("Asia", "Nikkei 225", "^N225"),
    ("Asia", "KOSPI", "^KS11"),
    ("Asia", "Hang Seng", "^HSI"),
    ("Asia", "Shanghai Composite", "000001.SS"),
    ("Asia", "ASX 200", "^AXJO"),
    ("US", "S&P 500", "^GSPC"),
    ("US", "Nasdaq 100", "^NDX"),
    ("US", "Dow Jones", "^DJI"),
    ("US", "Russell 2000", "^RUT"),
    ("Europe", "DAX", "^GDAXI"),
    ("Europe", "FTSE 100", "^FTSE"),
    ("Europe", "CAC 40", "^FCHI"),
    ("Europe", "Euro Stoxx 50", "^STOXX50E"),
]


def clean_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_live_market_snapshot(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def merge_breadth_layers(
    eod_layer: dict[str, object],
    live_layer: dict[str, object] | None,
) -> dict[str, object]:
    """Keep EOD structure metrics while replacing participation with live data."""
    merged = dict(eod_layer or {})
    eod_summary = dict(merged.get("summary") or {})
    eod_as_of = str(eod_summary.get("as_of_date") or "")
    live_layer = live_layer or {}
    live_summary = dict(live_layer.get("summary") or {})
    live_as_of = str(live_summary.get("as_of_date") or "")
    if live_as_of and eod_as_of and live_as_of < eod_as_of:
        eod_summary["technical_as_of_date"] = eod_as_of
        eod_summary["breadth_layer"] = "EOD cache (newer than live snapshot)"
        merged["summary"] = eod_summary
        merged["alerts"] = []
        return merged
    if not live_summary:
        eod_summary["technical_as_of_date"] = eod_as_of
        eod_summary["breadth_layer"] = "EOD fallback"
        merged["summary"] = eod_summary
        merged["alerts"] = []
        return merged

    eod_summary.update(live_summary)
    eod_summary["advance_decline_ratio"] = live_summary.get("ad_ratio")
    eod_summary["technical_as_of_date"] = eod_as_of
    eod_summary["breadth_as_of_timestamp"] = live_summary.get("as_of_timestamp", "")
    eod_summary["breadth_layer"] = "Live/delayed"
    merged["summary"] = eod_summary
    merged["sector_breadth"] = live_layer.get("sector_breadth") or merged.get("sector_breadth") or []
    merged["alerts"] = live_layer.get("alerts") or []
    merged["live_leaders"] = live_layer.get("leaders") or []
    merged["live_laggards"] = live_layer.get("laggards") or []
    return merged


def pct_return(series: pd.Series, days: int):
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


def yf_series(frame: pd.DataFrame, column: str) -> pd.Series:
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


def usable_index_rows(indices: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in indices
        if not row.get("is_stale")
    ]


def align_index_dates_with_region(indices: list[dict[str, object]]) -> dict[str, str]:
    """Exclude an older quote when peers from the same region have a newer session."""
    reference_dates: dict[str, str] = {}
    for row in indices:
        region = str(row.get("region") or "Other")
        last_date = str(row.get("last_date") or "")
        if last_date and last_date > reference_dates.get(region, ""):
            reference_dates[region] = last_date
    for row in indices:
        region = str(row.get("region") or "Other")
        last_date = str(row.get("last_date") or "")
        reference_date = reference_dates.get(region, "")
        row["region_reference_date"] = reference_date
        if last_date and reference_date and last_date < reference_date:
            row["is_stale"] = True
            note = f"older than the {region} session ({reference_date})"
            existing = str(row.get("data_note") or "").strip()
            row["data_note"] = "; ".join(part for part in [existing, note] if part)
    return reference_dates


def india_reference_date(indices: list[dict[str, object]]) -> str:
    for row in indices:
        if row.get("ticker") == "^NSEI" and row.get("last_date"):
            return str(row["last_date"])
    india_dates = [
        str(row.get("last_date"))
        for row in indices
        if row.get("region") in {"India", "India Sectors"} and row.get("last_date")
    ]
    return max(india_dates) if india_dates else ""


def find_latest_wires() -> Path | None:
    candidates = sorted(WIRES_DIR.glob("*ai-market-wires*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_wires(path: Path | None, today_only: bool) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    data = pd.read_csv(path)
    if today_only and "published" in data:
        today = dt.date.today().isoformat()
        data = data[data["published"].astype(str).str.startswith(today)]
    records: list[dict[str, object]] = []
    for _, row in data.fillna("").iterrows():
        records.append(
            {
                "published": str(row.get("published", "")),
                "score": clean_number(row.get("score")) or 0,
                "themes": str(row.get("themes", "")),
                "source": str(row.get("source", "")),
                "title": str(row.get("title", "")),
                "link": str(row.get("link", "")),
                "description": str(row.get("description", "")),
            }
        )
    records.sort(key=lambda item: (item["score"], item["published"]), reverse=True)
    return records


def market_theme_counts(wires: list[dict[str, object]]) -> dict[str, int]:
    counts = {
        "AI / tech risk": 0,
        "market selling": 0,
        "India market": 0,
        "macro / flows": 0,
        "mutual funds": 0,
    }
    for row in wires:
        text = f"{row.get('themes', '')} {row.get('title', '')}".lower()
        for key in list(counts):
            if key.lower() in text:
                counts[key] += 1
        if re.search(r"\b(mutual fund|sip|amfi|fii|fpi)\b", text):
            counts["mutual funds"] += 1
    return counts


def wire_date(row: dict[str, object]) -> str:
    published = str(row.get("published", ""))
    match = re.match(r"\d{4}-\d{2}-\d{2}", published)
    return match.group(0) if match else ""


def latest_wire_slice(wires: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    dated_rows = [(wire_date(row), row) for row in wires]
    dates = [date for date, _ in dated_rows if date]
    if not dates:
        return "", wires
    latest = max(dates)
    return latest, [row for date, row in dated_rows if date == latest]


def title_contains(row: dict[str, object], terms: list[str]) -> bool:
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    return any(re.search(rf"\b{re.escape(term.lower())}\b", text) for term in terms)


def unique_wires(wires: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the newest copy of syndicated headlines without hiding the raw wire list."""
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for row in sorted(wires, key=lambda item: str(item.get("published", "")), reverse=True):
        key = re.sub(r"[^a-z0-9]+", " ", str(row.get("title", "")).lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def wire_stamp(row: dict[str, object]) -> str:
    published = str(row.get("published", "")).strip()
    match = re.match(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})", published)
    return f"{match.group(1)} {match.group(2)}" if match else published[:16]


def sample_title(wires: list[dict[str, object]], terms: list[str]) -> str:
    for row in wires:
        if title_contains(row, terms):
            return str(row.get("title", "")).strip()
    return str(wires[0].get("title", "")).strip() if wires else ""


def narrative_from_wires(wires: list[dict[str, object]]) -> list[dict[str, str]]:
    latest_date, latest_wires = latest_wire_slice(wires)
    scoped_wires = unique_wires(latest_wires or wires)
    counts = market_theme_counts(scoped_wires)
    top_titles = " ".join(str(row.get("title", "")).lower() for row in scoped_wires[:40])
    asia_ai = any(term in top_titles for term in ["kospi", "nikkei", "chip", "semiconductor", "ai"])
    crude = any(term in top_titles for term in ["crude", "oil", "brent"])
    flows = any(term in top_titles for term in ["fii", "fpi", "flows", "rupee"])
    india_relief = any(
        term in top_titles
        for term in ["surge", "jump", "rebound", "gains", "up for", "above", "relief", "crude falls", "oil falls"]
    )
    items = []
    date_text = f" for {latest_date}" if latest_date else ""
    if scoped_wires:
        newest = scoped_wires[0]
        source = str(newest.get("source", "")).strip()
        source_text = f" · {source}" if source else ""
        items.append(
            {
                "label": f"Newest Wire · {wire_stamp(newest)}",
                "title": str(newest.get("title", "")).strip(),
                "text": (
                    f"Latest dated headline{source_text}. Today's unique wire set contains "
                    f"{len(scoped_wires)} headlines; the cards below are recalculated from this same set."
                ),
            }
        )
    if india_relief and asia_ai:
        items.append(
            {
                "label": f"Wire Bias · {latest_date}" if latest_date else "Wire Bias",
                "title": "India relief, global AI overhang",
                "text": (
                    f"The latest wire slice{date_text} has {counts['India market']} India-market wires, "
                    f"{counts['market selling']} selling/risk wires, and {counts['AI / tech risk']} AI/tech-risk wires. "
                    "The read is mixed: domestic relief is visible, but global tech pressure remains the overhang."
                ),
            }
        )
    elif india_relief:
        items.append(
            {
                "label": f"Wire Bias · {latest_date}" if latest_date else "Wire Bias",
                "title": "India relief is leading the tape",
                "text": (
                    f"The latest wire slice{date_text} is being led by India recovery language, with "
                    f"{counts['India market']} India-market wires versus {counts['market selling']} selling/risk wires."
                ),
            }
        )
    elif counts["market selling"] or counts["AI / tech risk"]:
        items.append(
            {
                "label": f"Wire Bias · {latest_date}" if latest_date else "Wire Bias",
                "title": "Risk-off still dominates the wires",
                "text": (
                    f"The latest wire slice{date_text} has {counts['market selling']} selling/risk wires and "
                    f"{counts['AI / tech risk']} AI/tech-risk wires, so the bias remains defensive."
                ),
            }
        )
    if asia_ai:
        title = sample_title(scoped_wires, ["kospi", "nikkei", "chip", "semiconductor", "ai"])
        items.append(
            {
                "label": f"AI / Chips · {latest_date}" if latest_date else "AI / Chips",
                "title": "Asian AI and chip risk-off",
                "text": f"AI/chip-linked selling is still the key external risk cluster. Representative wire: {title}",
            }
        )
    if crude:
        title = sample_title(scoped_wires, ["crude", "oil", "brent"])
        items.append(
            {
                "label": f"India Trigger · {latest_date}" if latest_date else "India Trigger",
                "title": "Crude is the local swing factor",
                "text": f"Oil is active in the India read-through today, so local moves should be read with crude and rupee context. Representative wire: {title}",
            }
        )
    if flows:
        title = sample_title(scoped_wires, ["fii", "fpi", "flows", "rupee"])
        items.append(
            {
                "label": f"Flows · {latest_date}" if latest_date else "Flows",
                "title": "Foreign-flow and risk appetite lens",
                "text": f"Flow and currency language is present in today's wires, so breadth confirmation matters more than an index-only read. Representative wire: {title}",
            }
        )
    if counts["mutual funds"]:
        items.append(
            {
                "label": "Funds",
                "title": "Mutual fund participation is still visible",
                "text": "Fund-related wires are active, but this dashboard should be read for category dispersion rather than a single fund-flow signal.",
            }
        )
    return items[:4]


def download_index_history(period: str, max_stale_days: int, outlier_pct: float) -> list[dict[str, object]]:
    if yf is None:
        return []
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date()
    rows: list[dict[str, object]] = []
    for region, name, ticker in INDEX_TICKERS:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                daily = yf.download(ticker, period=period, auto_adjust=False, progress=False, timeout=25)
                intraday = yf.download(
                    ticker,
                    period="5d",
                    interval="5m",
                    auto_adjust=False,
                    progress=False,
                    timeout=25,
                )
            if daily.empty:
                continue
            raw_close = yf_series(daily, "Close")
            adjusted_close = yf_series(daily, "Adj Close")
            if adjusted_close.empty:
                adjusted_close = raw_close.copy()
            intraday_close = yf_series(intraday, "Close")
            if raw_close.empty or adjusted_close.empty:
                continue

            quote_type = "Yahoo daily-close fallback"
            if not intraday_close.empty:
                quote_timestamp = pd.Timestamp(intraday_close.index[-1])
                last_date = quote_timestamp.date()
                last = float(intraday_close.iloc[-1])
                previous_candidates = [
                    float(value)
                    for index, value in raw_close.items()
                    if pd.Timestamp(index).date() < last_date
                ]
                previous = previous_candidates[-1] if previous_candidates else math.nan
                quote_type = "Yahoo 5-minute quote"
            else:
                quote_timestamp = pd.Timestamp(raw_close.index[-1])
                last_date = quote_timestamp.date()
                last = float(raw_close.iloc[-1])
                previous = float(raw_close.iloc[-2]) if len(raw_close) >= 2 else math.nan

            # EOD indicators use adjusted closes and exclude the quote session.
            # That keeps corporate actions from looking like technical breaks
            # and prevents partial intraday bars from rewriting rankings.
            technical_close = adjusted_close[
                [pd.Timestamp(index).date() < last_date for index in adjusted_close.index]
            ]
            if technical_close.empty:
                technical_close = adjusted_close.iloc[:-1] if len(adjusted_close) > 1 else adjusted_close
            if technical_close.empty:
                continue
            technical_last = float(technical_close.iloc[-1])
            technical_date = pd.Timestamp(technical_close.index[-1]).date()
            stale_days = max((today - last_date).days, 0)
            return_1d = (last / previous - 1) * 100 if math.isfinite(previous) and previous > 0 else None
            is_stale = stale_days > max_stale_days
            is_outlier = return_1d is not None and abs(return_1d) >= outlier_pct
            notes = []
            if is_stale:
                notes.append(f"stale quote: {stale_days} calendar days old")
            if is_outlier:
                notes.append(f"outlier 1D move: {return_1d:.2f}%")
            row: dict[str, object] = {
                "region": region,
                "name": name,
                "ticker": ticker,
                "last_date": last_date.isoformat(),
                "quote_timestamp": str(quote_timestamp),
                "quote_type": quote_type,
                "technical_as_of_date": technical_date.isoformat(),
                "previous_close": round(previous, 2) if math.isfinite(previous) else None,
                "last": round(last, 2),
                "return_1d_pct": return_1d,
                "return_5d_pct": pct_return(technical_close, 5),
                "return_20d_pct": pct_return(technical_close, 20),
                "rsi_14": round(float(rsi(technical_close).iloc[-1]), 2),
                "stale_days": stale_days,
                "is_stale": is_stale,
                "is_outlier": is_outlier,
                "data_note": "; ".join(notes),
            }
            for window in (20, 50, 200):
                ma = technical_close.rolling(window).mean().iloc[-1]
                row[f"sma_{window}"] = None if math.isnan(ma) else round(float(ma), 2)
                row[f"distance_sma_{window}_pct"] = (
                    None if math.isnan(ma) else round((technical_last / float(ma) - 1) * 100, 2)
                )
                row[f"above_sma_{window}"] = None if math.isnan(ma) else bool(technical_last > float(ma))
            rows.append(row)
        except Exception:
            continue
    align_index_dates_with_region(rows)
    return rows


def regional_breadth(indices: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_region: dict[str, list[dict[str, object]]] = {}
    for row in usable_index_rows(indices):
        by_region.setdefault(str(row.get("region", "Other")), []).append(row)
    for region, items in sorted(by_region.items()):
        returns_1d = [clean_number(item.get("return_1d_pct")) for item in items]
        returns_1d = [value for value in returns_1d if value is not None]
        returns_5d = [clean_number(item.get("return_5d_pct")) for item in items]
        returns_5d = [value for value in returns_5d if value is not None]
        advancers = sum(1 for value in returns_1d if value > 0)
        decliners = sum(1 for value in returns_1d if value < 0)
        above_20 = sum(1 for item in items if item.get("above_sma_20") is True)
        above_50 = sum(1 for item in items if item.get("above_sma_50") is True)
        rows.append(
            {
                "region": region,
                "count": len(items),
                "advancers": advancers,
                "decliners": decliners,
                "advance_decline_ratio": round(advancers / decliners, 2) if decliners else None,
                "average_1d_pct": round(sum(returns_1d) / len(returns_1d), 2) if returns_1d else None,
                "average_5d_pct": round(sum(returns_5d) / len(returns_5d), 2) if returns_5d else None,
                "above_20dma_pct": round(above_20 / len(items) * 100, 1) if items else None,
                "above_50dma_pct": round(above_50 / len(items) * 100, 1) if items else None,
            }
        )
    return rows


def global_leaders_laggards(indices: list[dict[str, object]], limit: int = 5) -> dict[str, list[dict[str, object]]]:
    valid = [
        row
        for row in usable_index_rows(indices)
        if clean_number(row.get("return_1d_pct")) is not None and str(row.get("region", "")) != "India Sectors"
    ]
    ranked = sorted(valid, key=lambda row: clean_number(row.get("return_1d_pct")) or 0, reverse=True)
    return {
        "leaders": ranked[:limit],
        "laggards": list(reversed(ranked[-limit:])),
    }


def load_india_breadth_universe(path: Path, limit: int) -> list[tuple[str, str, str]]:
    if not path.exists():
        return []
    data = pd.read_csv(path)
    if "YahooTicker" not in data.columns:
        return []
    rows: list[tuple[str, str, str]] = []
    for _, row in data.head(limit).iterrows():
        ticker = str(row.get("YahooTicker") or "").strip()
        if not ticker:
            continue
        symbol = str(row.get("Symbol") or ticker).strip()
        sector = str(row.get("Sector") or "Unknown").strip()
        rows.append((symbol, ticker, sector))
    return rows


def download_india_breadth(
    path: Path,
    limit: int,
    period: str = "3mo",
    expected_date: str = "",
) -> dict[str, object]:
    universe = load_india_breadth_universe(path, limit)
    if yf is None or not universe:
        return {"summary": {}, "leaders": [], "laggards": [], "sector_breadth": [], "source_count": len(universe)}
    rows: list[dict[str, object]] = []
    for symbol, ticker, sector in universe:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                data = yf.download(ticker, period=period, auto_adjust=False, progress=False, timeout=20)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close = data["Close"] if "Close" in data else data["Adj Close"]
            close = close.dropna().astype(float)
            volume = data["Volume"].reindex(close.index).fillna(0).astype(float) if "Volume" in data else pd.Series(0, index=close.index)
            if len(close) < 25:
                continue
            last = float(close.iloc[-1])
            previous = float(close.iloc[-2])
            ret_1d = (last / previous - 1) * 100
            latest_volume = float(volume.iloc[-1])
            avg_volume_20 = float(volume.rolling(20).mean().iloc[-1])
            turnover_proxy = latest_volume * last
            rows.append(
                {
                    "symbol": symbol,
                    "ticker": ticker,
                    "sector": sector,
                    "last_date": close.index[-1].date().isoformat(),
                    "last": round(last, 2),
                    "return_1d_pct": round(ret_1d, 2),
                    "return_5d_pct": pct_return(close, 5),
                    "volume": latest_volume,
                    "avg_volume_20": avg_volume_20,
                    "volume_ratio_20d": round(latest_volume / avg_volume_20, 2) if avg_volume_20 else None,
                    "turnover_proxy": turnover_proxy,
                    "above_20dma": bool(last > float(close.rolling(20).mean().iloc[-1])),
                    "above_50dma": bool(last > float(close.rolling(50).mean().iloc[-1])) if len(close) >= 50 else None,
                }
            )
        except Exception:
            continue

    raw_downloaded_count = len(rows)
    date_counts = pd.Series([row["last_date"] for row in rows], dtype=str).value_counts().to_dict()
    as_of_date = expected_date or (str(max(date_counts, key=date_counts.get)) if date_counts else "")
    mismatched_date_count = sum(1 for row in rows if as_of_date and row.get("last_date") != as_of_date)
    if as_of_date:
        rows = [row for row in rows if row.get("last_date") == as_of_date]

    total = len(rows)
    advancers = [row for row in rows if (row.get("return_1d_pct") or 0) > 0]
    decliners = [row for row in rows if (row.get("return_1d_pct") or 0) < 0]
    up_turnover = sum(float(row.get("turnover_proxy") or 0) for row in advancers)
    down_turnover = sum(float(row.get("turnover_proxy") or 0) for row in decliners)
    total_turnover = up_turnover + down_turnover
    high_volume_decliners = [
        row for row in decliners if (clean_number(row.get("volume_ratio_20d")) or 0) >= 1.5
    ]
    sector_rows = []
    if rows:
        frame = pd.DataFrame(rows)
        for sector, group in frame.groupby("sector"):
            sector_rows.append(
                {
                    "sector": sector,
                    "count": int(len(group)),
                    "advancers": int((group["return_1d_pct"] > 0).sum()),
                    "decliners": int((group["return_1d_pct"] < 0).sum()),
                    "average_1d_pct": round(float(group["return_1d_pct"].mean()), 2),
                    "average_volume_ratio_20d": round(float(group["volume_ratio_20d"].dropna().mean()), 2)
                    if not group["volume_ratio_20d"].dropna().empty
                    else None,
                }
            )
        sector_rows.sort(key=lambda row: (row["average_1d_pct"], row["advancers"]), reverse=True)

    return {
        "summary": {
            "source_count": len(universe),
            "downloaded_count": total,
            "raw_downloaded_count": raw_downloaded_count,
            "as_of_date": as_of_date,
            "expected_date": expected_date,
            "date_aligned": bool(as_of_date and (not expected_date or as_of_date == expected_date)),
            "date_counts": date_counts,
            "excluded_date_mismatch": mismatched_date_count,
            "advancers": len(advancers),
            "decliners": len(decliners),
            "unchanged": total - len(advancers) - len(decliners),
            "advance_decline_ratio": round(len(advancers) / len(decliners), 2) if decliners else None,
            "up_turnover_share_pct": round(up_turnover / total_turnover * 100, 1) if total_turnover else None,
            "down_turnover_share_pct": round(down_turnover / total_turnover * 100, 1) if total_turnover else None,
            "high_volume_decliners": len(high_volume_decliners),
            "above_20dma_pct": round(sum(1 for row in rows if row.get("above_20dma")) / total * 100, 1) if total else None,
            "above_50dma_pct": round(sum(1 for row in rows if row.get("above_50dma")) / total * 100, 1) if total else None,
        },
        "leaders": sorted(rows, key=lambda row: row.get("return_1d_pct") or 0, reverse=True)[:8],
        "laggards": sorted(rows, key=lambda row: row.get("return_1d_pct") or 0)[:8],
        "sector_breadth": sector_rows[:12],
        "source_count": len(universe),
    }


def load_leaders(path: Path, limit: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = pd.read_csv(path)
    if data.empty:
        return []
    fields = [
        "Symbol",
        "Description",
        "Sector",
        "last_close",
        "rank_score",
        "setup_label",
        "chart_pattern_view",
        "rsi_14",
        "return_1m_pct",
        "relative_strength_ratio_distance_sma_50_pct",
        "relative_strength_leader",
        "distance_from_52w_high_pct",
        "pnf_signal",
    ]
    data = data[[field for field in fields if field in data.columns]].head(limit)
    return json.loads(data.where(pd.notna(data), None).to_json(orient="records"))


def find_latest_nse_highs() -> Path | None:
    candidates = sorted(NSE_HIGH_DIR.glob("*nifty-500-52w-highs.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def symbol_key(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper().replace(".NS", ""))


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def summarize_nifty500_breadth(data: pd.DataFrame, source_count: int) -> dict[str, object]:
    if data.empty:
        return {
            "source_count": source_count,
            "downloaded_count": 0,
            "advancers": 0,
            "decliners": 0,
            "unchanged": 0,
            "advance_decline_ratio": None,
            "up_turnover_share_pct": None,
            "down_turnover_share_pct": None,
            "high_volume_decliners": 0,
            "above_20dma_pct": None,
            "above_50dma_pct": None,
            "above_200dma_pct": None,
            "sector_breadth": [],
            "as_of_date": "",
        }

    rows = data.to_dict(orient="records")
    total = len(rows)
    eligible_200 = [row for row in rows if clean_number(row.get("above_sma_200")) is not None]
    advancers = [row for row in rows if (clean_number(row.get("return_1d_pct")) or 0) > 0]
    decliners = [row for row in rows if (clean_number(row.get("return_1d_pct")) or 0) < 0]
    high_volume_decliners = [
        row for row in decliners if (clean_number(row.get("volume_ratio_20d")) or 0) >= 1.5
    ]

    def row_turnover(row: dict[str, object]) -> float:
        turnover = clean_number(row.get("turnover_proxy"))
        if turnover is not None:
            return turnover
        volume = clean_number(row.get("volume")) or 0
        last = clean_number(row.get("last")) or 0
        return volume * last

    up_turnover = sum(row_turnover(row) for row in advancers)
    down_turnover = sum(row_turnover(row) for row in decliners)
    total_turnover = up_turnover + down_turnover
    sector_rows: list[dict[str, object]] = []
    if "sector" in data.columns:
        for sector, group in data.groupby("sector", dropna=False):
            returns = [clean_number(value) for value in group.get("return_1d_pct", pd.Series(dtype=float))]
            returns = [value for value in returns if value is not None]
            sector_advancers = sum(1 for value in returns if value > 0)
            sector_decliners = sum(1 for value in returns if value < 0)
            volume_ratios = [
                clean_number(value)
                for value in group.get("volume_ratio_20d", pd.Series(dtype=float))
            ]
            volume_ratios = [value for value in volume_ratios if value is not None]
            sector_rows.append(
                {
                    "sector": str(sector or "Unknown"),
                    "count": int(len(group)),
                    "advancers": int(sector_advancers),
                    "decliners": int(sector_decliners),
                    "advance_decline_ratio": round(sector_advancers / sector_decliners, 2)
                    if sector_decliners
                    else None,
                    "average_1d_pct": round(sum(returns) / len(returns), 2) if returns else None,
                    "above_50dma_pct": round(
                        sum(1 for _, item in group.iterrows() if boolish(item.get("above_sma_50")))
                        / len(group)
                        * 100,
                        1,
                    )
                    if len(group)
                    else None,
                    "average_volume_ratio_20d": round(sum(volume_ratios) / len(volume_ratios), 2)
                    if volume_ratios
                    else None,
                }
            )
        sector_rows.sort(
            key=lambda row: (
                clean_number(row.get("average_1d_pct")) or -999,
                row.get("advancers") or 0,
            ),
            reverse=True,
        )

    dates = [str(value) for value in data.get("last_date", pd.Series(dtype=str)).dropna() if str(value)]
    as_of_date = max(set(dates), key=dates.count) if dates else ""
    return {
        "source_count": source_count,
        "downloaded_count": total,
        "advancers": len(advancers),
        "decliners": len(decliners),
        "unchanged": total - len(advancers) - len(decliners),
        "advance_decline_ratio": round(len(advancers) / len(decliners), 2) if decliners else None,
        "up_turnover_share_pct": round(up_turnover / total_turnover * 100, 1) if total_turnover else None,
        "down_turnover_share_pct": round(down_turnover / total_turnover * 100, 1) if total_turnover else None,
        "high_volume_decliners": len(high_volume_decliners),
        "above_20dma_pct": round(
            sum(1 for row in rows if boolish(row.get("above_sma_20"))) / total * 100,
            1,
        )
        if total
        else None,
        "above_50dma_pct": round(
            sum(1 for row in rows if boolish(row.get("above_sma_50"))) / total * 100,
            1,
        )
        if total
        else None,
        "above_200dma_pct": round(
            sum(1 for row in eligible_200 if boolish(row.get("above_sma_200"))) / len(eligible_200) * 100,
            1,
        )
        if eligible_200
        else None,
        "above_200dma_eligible": len(eligible_200),
        "sector_breadth": sector_rows[:15],
        "as_of_date": as_of_date,
    }


def build_nifty500_technical_leaders(data: pd.DataFrame, technical_path: Path, limit: int) -> list[dict[str, object]]:
    if data.empty:
        return []
    pool = data.copy()
    if "within_3pct_52w_high" in pool:
        pool = pool[pool["within_3pct_52w_high"].astype(bool)]
    if pool.empty:
        return []

    technical_lookup: dict[str, dict[str, object]] = {}
    if technical_path.exists():
        technical = pd.read_csv(technical_path)
        technical = technical.where(pd.notna(technical), None)
        if "Symbol" in technical:
            for _, row in technical.iterrows():
                key = symbol_key(row.get("Symbol"))
                if key and key not in technical_lookup:
                    technical_lookup[key] = row.to_dict()

    records: list[dict[str, object]] = []
    for _, row in pool.iterrows():
        key = symbol_key(row.get("symbol"))
        technical_row = technical_lookup.get(key, {})
        rank_score = clean_number(technical_row.get("rank_score"))
        rs_distance = clean_number(technical_row.get("relative_strength_ratio_distance_sma_50_pct"))
        rsi_14 = clean_number(technical_row.get("rsi_14"))
        one_month = clean_number(technical_row.get("return_1m_pct"))
        distance = clean_number(row.get("distance_from_52w_high_pct"))
        change = clean_number(row.get("change_pct"))
        fresh = boolish(row.get("fresh_52w_high"))
        pnf_signal = str(technical_row.get("pnf_signal") or "").strip()
        rs_leader = boolish(technical_row.get("relative_strength_leader"))
        setup_label = str(
            technical_row.get("setup_label")
            or ("Fresh 52-week high" if fresh else "Near 52-week high")
        ).strip()
        chart_view = str(
            technical_row.get("chart_pattern_view")
            or "Price is close to or at a 52-week high. Technical overlay pending; check RS, RSI, and P&F before upgrading the reading."
        ).strip()
        sort_score = (
            (rank_score or 0) * 10
            + (8 if rs_leader else 0)
            + (4 if "bullish" in pnf_signal.lower() else 0)
            + (3 if fresh else 0)
            + (change or 0)
            + max(distance or -20, -20) / 2
        )
        records.append(
            {
                "symbol": row.get("symbol"),
                "company": row.get("company"),
                "last_price": row.get("last_price"),
                "change_pct": change,
                "distance_from_52w_high_pct": distance,
                "rank_score": rank_score,
                "relative_strength_leader": rs_leader if technical_row else None,
                "relative_strength_ratio_distance_sma_50_pct": rs_distance,
                "rsi_14": round(rsi_14, 2) if rsi_14 is not None else None,
                "return_1m_pct": one_month,
                "pnf_signal": pnf_signal or None,
                "setup_label": setup_label,
                "chart_pattern_view": chart_view,
                "has_technical_overlay": bool(technical_row),
                "_sort_score": sort_score,
            }
        )
    records = sorted(records, key=lambda item: item.get("_sort_score") or 0, reverse=True)
    for item in records:
        item.pop("_sort_score", None)
    return records[:limit]


def load_nse_highs(path: Path | None, limit: int) -> dict[str, object]:
    if path is None or not path.exists():
        return {"fresh_highs": [], "near_highs": [], "technical_leaders": [], "source_file": ""}
    data = pd.read_csv(path)
    if data.empty:
        return {"fresh_highs": [], "near_highs": [], "technical_leaders": [], "source_file": str(path)}
    data = data.where(pd.notna(data), None)
    fresh = data[data["fresh_52w_high"].astype(bool)] if "fresh_52w_high" in data.columns else data.head(0)
    near = data[data["within_3pct_52w_high"].astype(bool)] if "within_3pct_52w_high" in data.columns else data.head(0)
    technical_leaders = build_nifty500_technical_leaders(data, TECHNICAL_SUMMARY, limit)
    laggards = data.copy()
    worst_day = data.copy()
    if "change_pct" in fresh:
        fresh = fresh.sort_values(["change_pct", "distance_from_52w_high_pct"], ascending=[False, False], na_position="last")
    if "distance_from_52w_high_pct" in near:
        near = near.sort_values(["distance_from_52w_high_pct", "change_pct"], ascending=[False, False], na_position="last")
    if "distance_from_52w_high_pct" in laggards:
        laggards = laggards.sort_values(["distance_from_52w_high_pct", "change_pct"], ascending=[True, True], na_position="last")
    if "change_pct" in worst_day:
        worst_day = worst_day.sort_values(["change_pct", "distance_from_52w_high_pct"], ascending=[True, True], na_position="last")
    return {
        "fresh_highs": json.loads(fresh.head(limit).to_json(orient="records")),
        "near_highs": json.loads(near.head(limit).to_json(orient="records")),
        "technical_leaders": technical_leaders,
        "structural_laggards": json.loads(laggards.head(limit).to_json(orient="records")),
        "worst_day_moves": json.loads(worst_day.head(limit).to_json(orient="records")),
        "source_file": str(path),
        "source_name": str(data["source"].dropna().iloc[0]) if "source" in data and len(data["source"].dropna()) else "NSE live",
        "fresh_count": int(len(fresh)),
        "near_count": int(len(near)),
        "technical_leaders_count": int(len(technical_leaders)),
        "technical_overlay_count": int(sum(1 for row in technical_leaders if row.get("has_technical_overlay"))),
        "total_count": int(len(data)),
    }


def find_latest_mutual_fund_navs() -> Path | None:
    candidates = sorted(MUTUAL_FUND_NAV_DIR.glob("*mutual-fund-navs.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_mutual_fund_navs(path: Path | None, limit: int) -> dict[str, object]:
    if path is None or not path.exists():
        return {"funds": [], "source_file": ""}
    data = pd.read_csv(path)
    if data.empty:
        return {"funds": [], "source_file": str(path), "count": 0}
    data = data.where(pd.notna(data), None)
    sort_columns = [column for column in ["return_5d_pct", "return_20d_pct"] if column in data.columns]
    if sort_columns:
        data = data.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
    return {
        "funds": json.loads(data.head(limit).to_json(orient="records")),
        "source_file": str(path),
        "count": int(len(data)),
    }


def find_latest_nifty50_rotation() -> Path | None:
    candidates = sorted(NIFTY50_ROTATION_DIR.glob("*nifty50-rotation.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_nifty50_rotation(path: Path | None, limit: int) -> dict[str, object]:
    if path is None or not path.exists():
        return {"leaders": [], "weakening": [], "mean_reversion": [], "source_file": ""}
    data = pd.read_csv(path)
    if data.empty:
        return {"leaders": [], "weakening": [], "mean_reversion": [], "source_file": str(path), "count": 0}
    data = data.where(pd.notna(data), None)
    if "rotation_score" in data:
        data = data.sort_values(["rotation_score", "rs_return_20d_pct"], ascending=[False, False], na_position="last")
    weakening = data[data["rotation_signal"].astype(str).str.contains("Weakening", case=False, na=False)] if "rotation_signal" in data else data.head(0)
    mean_reversion = data[
        data["rotation_signal"].astype(str).str.contains("Mean reversion", case=False, na=False)
    ] if "rotation_signal" in data else data.head(0)
    return {
        "leaders": json.loads(data.head(limit).to_json(orient="records")),
        "weakening": json.loads(weakening.head(limit).to_json(orient="records")),
        "mean_reversion": json.loads(mean_reversion.head(limit).to_json(orient="records")),
        "source_file": str(path),
        "count": int(len(data)),
    }


def nifty500_file_market_date(path: Path) -> str:
    try:
        dates = pd.read_csv(path, usecols=["last_date"])["last_date"].dropna().astype(str)
    except Exception:
        return ""
    return str(dates.value_counts().index[0]) if not dates.empty else ""


def find_latest_nifty500_rotation() -> Path | None:
    candidates = list(NIFTY500_ROTATION_DIR.glob("*nifty500-rotation.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (nifty500_file_market_date(path), path.stat().st_mtime))


def load_nifty500_rotation(path: Path | None, limit: int, expected_date: str = "") -> dict[str, object]:
    empty = {
        "leaders": [],
        "composite_leaders": [],
        "strengthening": [],
        "extended": [],
        "setup_candidates": [],
        "weakening": [],
        "laggards": [],
        "mean_reversion": [],
        "risk_review": [],
        "source_file": "",
        "count": 0,
        "downloaded_count": 0,
        "coverage_pct": None,
        "benchmark_ticker": "",
        "summary": {},
        "as_of_date": "",
        "expected_date": expected_date,
        "date_aligned": False,
    }
    if path is None or not path.exists():
        return empty
    data = pd.read_csv(path)
    if data.empty:
        empty["source_file"] = str(path)
        return empty
    data = data.where(pd.notna(data), None)
    raw_downloaded = (
        data[data["downloaded"].astype(str).str.lower().isin(["true", "1"])]
        if "downloaded" in data
        else data
    )
    date_counts = (
        raw_downloaded["last_date"].dropna().astype(str).value_counts().to_dict()
        if "last_date" in raw_downloaded
        else {}
    )
    as_of_date = expected_date or (str(max(date_counts, key=date_counts.get)) if date_counts else "")
    downloaded = (
        raw_downloaded[raw_downloaded["last_date"].astype(str) == as_of_date]
        if as_of_date and "last_date" in raw_downloaded
        else raw_downloaded
    )
    mismatched_date_count = int(len(raw_downloaded) - len(downloaded))
    downloaded_count = int(len(downloaded))
    total_count = int(len(data))
    coverage_pct = round(downloaded_count / total_count * 100, 1) if total_count else None
    summary = summarize_nifty500_breadth(downloaded, total_count)
    summary["coverage_pct"] = coverage_pct
    summary["raw_downloaded_count"] = int(len(raw_downloaded))
    summary["date_counts"] = date_counts
    summary["excluded_date_mismatch"] = mismatched_date_count
    summary["expected_date"] = expected_date
    summary["date_aligned"] = bool(as_of_date and (not expected_date or as_of_date == expected_date))

    def sorted_frame(frame: pd.DataFrame, columns: list[str], ascending: list[bool]) -> pd.DataFrame:
        available = [column for column in columns if column in frame.columns]
        if not available:
            return frame
        available_ascending = [ascending[columns.index(column)] for column in available]
        return frame.sort_values(available, ascending=available_ascending, na_position="last")

    signal = (
        downloaded["rotation_signal"].astype(str)
        if "rotation_signal" in downloaded
        else pd.Series("", index=downloaded.index)
    )
    setup_quality = (
        downloaded["setup_quality"].astype(str)
        if "setup_quality" in downloaded
        else pd.Series("", index=downloaded.index)
    )
    setup_score = (
        pd.to_numeric(downloaded["setup_score"], errors="coerce")
        if "setup_score" in downloaded
        else pd.Series(math.nan, index=downloaded.index)
    )
    has_setup_text = (
        downloaded["primary_setup"].astype(str).str.len() > 0
        if "primary_setup" in downloaded
        else pd.Series(False, index=downloaded.index)
    )
    leaders = sorted_frame(downloaded.copy(), ["rotation_score", "rs_return_20d_pct"], [False, False])
    strengthening = sorted_frame(
        downloaded[signal.str.contains("Strengthening", case=False, na=False)].copy(),
        ["rotation_score", "rs_return_20d_pct"],
        [False, False],
    )
    extended = sorted_frame(
        downloaded[signal.str.contains("Leader but extended", case=False, na=False)].copy(),
        ["rotation_score", "rs_return_20d_pct"],
        [False, False],
    )
    risk_review = sorted_frame(
        downloaded[
            signal.str.contains("Risk", case=False, na=False)
            | setup_quality.str.contains("Risk", case=False, na=False)
        ].copy(),
        ["rotation_score", "rs_return_20d_pct"],
        [True, True],
    )
    setup_candidates = sorted_frame(
        downloaded[
            ((setup_score >= 2.3) | setup_quality.isin(["Actionable watch", "Watch"]) | signal.str.contains("Setup watch", case=False, na=False))
            & ~setup_quality.str.contains("Risk", case=False, na=False)
            & has_setup_text
        ].copy(),
        ["setup_score", "rotation_score", "rs_return_20d_pct"],
        [False, False, False],
    )
    laggards = sorted_frame(downloaded.copy(), ["rotation_score", "rs_return_20d_pct"], [True, True])
    weakening = (
        downloaded[downloaded["rotation_signal"].astype(str).str.contains("Weakening", case=False, na=False)]
        if "rotation_signal" in downloaded
        else downloaded.head(0)
    )
    if weakening.empty:
        weakening = laggards.head(limit)
    mean_reversion = (
        downloaded[downloaded["rotation_signal"].astype(str).str.contains("Mean reversion", case=False, na=False)]
        if "rotation_signal" in downloaded
        else downloaded.head(0)
    )
    benchmark_ticker = ""
    if "benchmark_ticker" in data and len(data["benchmark_ticker"].dropna()):
        benchmark_ticker = str(data["benchmark_ticker"].dropna().iloc[0])
    return {
        "leaders": json.loads(leaders.head(limit).to_json(orient="records")),
        "composite_leaders": json.loads(leaders.head(limit).to_json(orient="records")),
        "strengthening": json.loads(strengthening.head(limit).to_json(orient="records")),
        "extended": json.loads(extended.head(limit).to_json(orient="records")),
        "setup_candidates": json.loads(setup_candidates.head(limit).to_json(orient="records")),
        "weakening": json.loads(weakening.head(limit).to_json(orient="records")),
        "laggards": json.loads(laggards.head(limit).to_json(orient="records")),
        "mean_reversion": json.loads(mean_reversion.head(limit).to_json(orient="records")),
        "risk_review": json.loads(risk_review.head(limit).to_json(orient="records")),
        "source_file": str(path),
        "count": total_count,
        "downloaded_count": downloaded_count,
        "coverage_pct": coverage_pct,
        "benchmark_ticker": benchmark_ticker,
        "summary": summary,
        "as_of_date": as_of_date,
        "expected_date": expected_date,
        "date_aligned": bool(as_of_date and (not expected_date or as_of_date == expected_date)),
    }


def find_latest_bear_dashboard_notes() -> Path | None:
    candidates = sorted(
        BEAR_DASHBOARD_NOTES_DIR.glob("*bear-watchlist-notes.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_bear_dashboard_notes(path: Path | None, limit: int, include_note_text: bool) -> dict[str, object]:
    if path is None or not path.exists():
        return {"notes": [], "source_file": "", "count": 0, "includes_full_text": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        notes = []
    cleaned = []
    for note in notes[:limit]:
        if not isinstance(note, dict):
            continue
        item = {
            "id": note.get("id", ""),
            "title": note.get("title", "Untitled"),
            "modified": note.get("modified", ""),
            "bear_link": note.get("bear_link", ""),
            "excerpt": note.get("excerpt", ""),
            "images": note.get("images", []),
        }
        if include_note_text:
            item["note"] = note.get("note", "")
        cleaned.append(item)
    cleaned.sort(key=lambda item: str(item.get("modified", "")), reverse=True)
    return {
        "tag": payload.get("tag", ""),
        "generated_at": payload.get("generated_at", ""),
        "notes": cleaned,
        "source_file": str(path),
        "count": int(payload.get("count") or len(notes)),
        "shown_count": len(cleaned),
        "includes_full_text": include_note_text,
    }


def copy_bear_dashboard_assets(bear_notes_path: Path | None, output_dir: Path) -> None:
    if bear_notes_path is None:
        return
    source_assets = bear_notes_path.parent / "assets"
    if not source_assets.exists():
        return
    destination_assets = output_dir / "assets"
    destination_assets.mkdir(parents=True, exist_ok=True)
    for source in source_assets.iterdir():
        if source.is_file():
            shutil.copy2(source, destination_assets / source.name)


def market_posture(indices: list[dict[str, object]], wires: list[dict[str, object]]) -> dict[str, str]:
    valid_indices = usable_index_rows(indices)
    latest_date, latest_wires = latest_wire_slice(wires)
    scoped_wires = latest_wires or wires
    wire_sample = scoped_wires
    selling_wires = sum(1 for row in wire_sample if "market selling" in str(row.get("themes", "")).lower())
    ai_wires = sum(1 for row in wire_sample if "ai / tech risk" in str(row.get("themes", "")).lower())
    india_wires = sum(1 for row in wire_sample if "india market" in str(row.get("themes", "")).lower())
    title_blob = " ".join(str(row.get("title", "")).lower() for row in wire_sample)
    india_relief = any(
        term in title_blob
        for term in ["surge", "jump", "rebound", "gains", "up for", "above", "relief", "crude falls", "oil falls"]
    )
    weak_indices = sum(1 for row in valid_indices if (row.get("distance_sma_20_pct") or 0) < 0)
    above_50 = sum(1 for row in valid_indices if row.get("above_sma_50") is True)
    if india_relief and ai_wires >= 5:
        stance = "India relief, AI overhang"
    elif ai_wires >= 5 and weak_indices >= 3:
        stance = "Risk-off, AI-led"
    elif selling_wires >= 8:
        stance = "Risk-off, broad"
    elif above_50 >= max(2, len(valid_indices) // 2):
        stance = "Constructive but selective"
    else:
        stance = "Mixed"
    return {
        "stance": stance,
        "wire_count": str(len(scoped_wires)),
        "wire_total_count": str(len(wires)),
        "wire_scope_date": latest_date,
        "ai_wire_count": str(ai_wires),
        "selling_wire_count": str(selling_wires),
        "india_wire_count": str(india_wires),
        "weak_index_count": str(weak_indices),
        "valid_index_count": str(len(valid_indices)),
    }


def html_page(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Wire and Technical Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8f5;
      --ink: #151715;
      --muted: #646b65;
      --panel: #ffffff;
      --line: #d9ded8;
      --green: #16785f;
      --red: #bb3d3d;
      --amber: #a06b19;
      --blue: #276f9f;
      --shadow: 0 12px 32px rgba(28, 35, 30, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 22px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    .topbar {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
      max-width: 1480px;
      margin: 0 auto;
    }}
    h1 {{ margin: 0; font-size: 24px; font-weight: 750; letter-spacing: 0; }}
    .stamp {{ color: var(--muted); margin-top: 4px; }}
    .header-actions {{
      display: flex;
      align-items: stretch;
      justify-content: flex-end;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .refresh-panel {{
      min-width: 220px;
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 12px;
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .refresh-panel b {{ display: block; }}
    .refresh-panel button {{ margin-top: 8px; width: 100%; }}
    .stance {{
      min-width: 220px;
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 12px;
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .stance b {{ display: block; font-size: 18px; }}
    main {{
      width: 100%;
      box-sizing: border-box;
      max-width: 1480px;
      margin: 0 auto;
      padding: 18px 28px 40px;
      display: grid;
      gap: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .panel {{
      min-width: 0;
      max-width: 100%;
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .note {{ padding: 16px; min-height: 142px; }}
    .eyebrow {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .note h2 {{ margin: 7px 0 8px; font-size: 17px; line-height: 1.25; }}
    .note p {{ margin: 0; color: #313832; }}
    .section-title {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 16px 0;
      gap: 12px;
    }}
    .section-title h2 {{ margin: 0; font-size: 18px; }}
    .section-title span {{ color: var(--muted); font-size: 12px; }}
    .title-with-info {{ display: inline-flex; align-items: center; gap: 7px; }}
    .info-button {{
      width: 19px; height: 19px; min-height: 19px; flex: 0 0 19px;
      display: inline-grid; place-items: center; padding: 0;
      border: 1px solid #b9c2b6; border-radius: 50%; background: #fff;
      color: var(--muted); font-size: 12px; font-weight: 800; line-height: 1;
    }}
    .info-button:hover, .info-button:focus-visible, .info-button[aria-expanded="true"] {{
      border-color: var(--blue); color: var(--blue); background: #f4f8fc; outline: none;
    }}
    .info-popover {{
      position: fixed; z-index: 100; width: min(330px, calc(100vw - 24px));
      padding: 12px 13px; border: 1px solid #cfd7cc; border-radius: 8px;
      background: #fff; color: var(--ink); box-shadow: 0 8px 24px rgba(30, 45, 35, .18);
      font-size: 13px; font-weight: 500; line-height: 1.45;
    }}
    .info-popover[hidden] {{ display: none; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #edf0ec; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 650; background: #fbfcfa; position: sticky; top: 86px; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .pos {{ color: var(--green); font-weight: 650; }}
    .neg {{ color: var(--red); font-weight: 650; }}
    .flat {{ color: var(--muted); font-weight: 650; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f8faf7;
      white-space: nowrap;
      font-size: 12px;
    }}
    .leader {{ border-color: #add7c8; background: #eef8f4; color: var(--green); }}
    .risk {{ border-color: #e4c1c1; background: #fff4f4; color: var(--red); }}
    .watch {{ border-color: #d9c38e; background: #fff9ea; color: var(--amber); }}
    .tabs {{ display: flex; gap: 8px; padding: 12px 16px 0; flex-wrap: wrap; }}
    .wire-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px 0;
      flex-wrap: wrap;
    }}
    .wire-controls .tabs {{ padding: 0; }}
    .date-filter {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    select {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      min-height: 34px;
      padding: 0 10px;
      color: var(--ink);
      font: inherit;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      min-height: 34px;
      padding: 0 11px;
      cursor: pointer;
      color: var(--ink);
    }}
    button.active {{ border-color: #222; background: #202420; color: #fff; }}
    .region-row {{ cursor: pointer; }}
    .region-row:hover {{ background: #fbfcfa; }}
    .region-toggle {{
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      margin-right: 8px;
      border-radius: 50%;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      vertical-align: 1px;
    }}
    .region-row.open .region-toggle {{ background: #202420; color: #fff; border-color: #202420; }}
    .detail-row {{ display: none; background: #fbfcfa; }}
    .detail-row.open {{ display: table-row; }}
    .detail-cell {{ padding: 0; }}
    .detail-wrap {{ padding: 12px 16px 16px 48px; }}
    .detail-wrap table {{ background: #fff; border: 1px solid #e7ebe6; border-radius: 8px; overflow: hidden; }}
    .detail-wrap table th {{ position: static; }}
    .detail-heading {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .wire-list {{ display: grid; gap: 10px; padding: 12px 16px 16px; }}
    .wire {{
      display: grid;
      grid-template-columns: 86px 1fr auto;
      gap: 12px;
      padding: 12px;
      border: 1px solid #e7ebe6;
      border-radius: 8px;
      background: #fff;
    }}
    .wire a {{ color: #174f7a; text-decoration: none; font-weight: 650; }}
    .wire a:hover {{ text-decoration: underline; }}
    .source {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .data-warning {{ color: var(--red); font-weight: 700; }}
    .score {{ font-variant-numeric: tabular-nums; color: var(--muted); }}
    .note-browser {{ display: grid; grid-template-columns: 320px 1fr; gap: 0; }}
    .note-list {{ border-right: 1px solid #edf0ec; max-height: 560px; overflow: auto; }}
    .note-item {{
      width: 100%;
      min-height: 0;
      display: block;
      border: 0;
      border-bottom: 1px solid #edf0ec;
      border-radius: 0;
      padding: 12px 14px;
      background: #fff;
      text-align: left;
    }}
    .note-item.active {{ background: #eef8f4; color: var(--ink); }}
    .note-item.warning {{ box-shadow: inset 4px 0 0 #c43f49; }}
    .note-warning-label {{
      display: inline-flex;
      margin-top: 7px;
      padding: 3px 7px;
      border: 1px solid #efb5ba;
      border-radius: 999px;
      background: #fff1f2;
      color: #a52833;
      font-size: 11px;
      font-weight: 750;
    }}
    .note-reader {{ min-height: 560px; padding: 16px; overflow: auto; }}
    .note-reader pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 12px 0 0;
      font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .note-content {{ display: grid; gap: 10px; margin-top: 12px; }}
    .note-content p {{ margin: 0; white-space: pre-wrap; }}
    .note-content h3 {{ margin: 8px 0 0; font-size: 15px; }}
    .note-warning {{
      display: grid;
      gap: 4px;
      margin: 12px 0;
      padding: 12px 14px;
      border: 1px solid #efb5ba;
      border-left: 4px solid #c43f49;
      border-radius: 6px;
      background: #fff1f2;
      color: #7d2028;
    }}
    .note-warning strong {{ color: #9f2631; }}
    .note-image {{
      width: min(100%, 980px);
      max-height: 520px;
      object-fit: contain;
      border: 1px solid #e7ebe6;
      border-radius: 8px;
      background: #fff;
      cursor: zoom-in;
    }}
    .note-actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
    .note-actions a {{
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: #174f7a;
      text-decoration: none;
      background: #fff;
    }}
    .image-modal {{
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      background: rgba(12, 14, 13, 0.88);
      z-index: 10;
      padding: 24px;
    }}
    .image-modal.open {{ display: grid; }}
    .image-modal img {{
      max-width: 96vw;
      max-height: 92vh;
      object-fit: contain;
      background: #fff;
      border-radius: 8px;
    }}
    .image-modal button {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: #fff;
      color: #111;
    }}
    .two-col {{ display: grid; grid-template-columns: 1.05fr 1fr; gap: 18px; }}
    .three-col {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 14px 16px 4px;
    }}
    .metric {{
      border: 1px solid #e7ebe6;
      border-radius: 8px;
      background: #fbfcfa;
      padding: 10px;
      min-height: 76px;
    }}
    .metric b {{ display: block; font-size: 20px; margin-top: 4px; font-variant-numeric: tabular-nums; }}
    .empty {{ padding: 16px; color: var(--muted); }}
    @media (max-width: 980px) {{
      header {{ position: static; padding: 14px 16px 12px; }}
      h1 {{ font-size: 20px; line-height: 1.2; }}
      main {{ padding: 12px 12px 28px; gap: 12px; }}
      .grid, .two-col, .three-col, .metric-grid {{ grid-template-columns: 1fr; }}
      #marketHealthPanel {{ order: -2; }}
      #narrative {{ order: -1; }}
      .topbar {{ grid-template-columns: 1fr; }}
      .header-actions {{ display: grid; grid-template-columns: 1fr 1fr; justify-content: stretch; }}
      .stance, .refresh-panel {{ min-width: 0; padding: 8px 10px; box-shadow: none; }}
      .stance b {{ font-size: 15px; }}
      .refresh-panel b {{ font-size: 13px; }}
      .refresh-panel button {{ min-height: 30px; margin-top: 6px; }}
      th {{ position: static; }}
      .wire {{ grid-template-columns: 1fr; }}
      .note-browser {{ grid-template-columns: 1fr; }}
      .note-list {{ border-right: 0; border-bottom: 1px solid #edf0ec; max-height: 320px; }}
    }}
    @media (max-width: 560px) {{
      .header-actions {{ grid-template-columns: 1fr; }}
      .refresh-panel, .stance {{ width: 100%; }}
      .section-title {{ align-items: flex-start; padding: 12px 12px 0; }}
      .section-title h2 {{ font-size: 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>Market Wire and Technical Dashboard</h1>
        <div class="stamp" id="generated"></div>
      </div>
      <div class="header-actions">
        <div class="refresh-panel">
          <span class="eyebrow">Data refresh</span>
          <b>Every 10 minutes during market hours</b>
          <span id="refreshStatus" class="stamp">Checking the latest snapshot</span>
          <button id="refreshNow" type="button">Refresh now</button>
        </div>
        <div class="stance">
          <span class="eyebrow">Current posture</span>
          <b id="stance"></b>
          <span id="stanceDetail" class="stamp"></span>
        </div>
      </div>
    </div>
  </header>
  <main>
    <section class="grid" id="narrative"></section>
    <section class="panel" id="marketHealthPanel">
      <div class="section-title">
        <h2 class="title-with-info">Market Health + Leadership Deployment <button type="button" class="info-button" data-help="Combines broad Nifty 500 health with the number and quality of genuine leaders to guide how much capital may be deployed." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span id="researchAsOf">Fixed current Nifty 500</span>
      </div>
      <div id="breadthLeadership"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2 class="title-with-info">Global Equity Breadth <button type="button" class="info-button" data-help="Groups valid global indices by region. Returns, advancing versus declining indices, and moving-average participation show whether strength is broad or narrow." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span>Region-wise index map</span>
      </div>
      <div id="regionalBreadth"></div>
    </section>
    <section class="three-col">
      <div class="panel">
        <div class="section-title">
          <h2 class="title-with-info">Global Leaders <button type="button" class="info-button" data-help="The strongest valid global indices ranked by their latest one-day percentage move." aria-label="Explain this section" aria-expanded="false">i</button></h2>
          <span>1-day move</span>
        </div>
        <div id="globalLeaders"></div>
      </div>
      <div class="panel">
        <div class="section-title">
          <h2 class="title-with-info">Global Laggards <button type="button" class="info-button" data-help="The weakest valid global indices ranked by their latest one-day percentage move." aria-label="Explain this section" aria-expanded="false">i</button></h2>
          <span>1-day move</span>
        </div>
        <div id="globalLaggards"></div>
      </div>
      <div class="panel">
        <div class="section-title">
          <h2 class="title-with-info">India Breadth + Volume <button type="button" class="info-button" data-help="Shows how widely stocks participate in the Indian market move and whether unusual volume confirms that move." aria-label="Explain this section" aria-expanded="false">i</button></h2>
          <span>52W technical screen universe</span>
        </div>
        <div id="indiaBreadth"></div>
      </div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2 class="title-with-info">Mutual Fund NAV Monitor <button type="button" class="info-button" data-help="Tracks selected AMFI mutual-fund NAVs across short and medium horizons. These are end-of-day fund returns, not intraday prices." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span>AMFI NAV movement</span>
      </div>
      <div id="mutualFundNavs"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2 class="title-with-info">Nifty 50 Rotation <button type="button" class="info-button" data-help="Ranks Nifty 50 stocks using relative strength, trend, RSI and volume to identify strengthening, extended, weakening and mean-reversion groups." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span>Relative strength, volume, RSI</span>
      </div>
      <div id="nifty50Rotation"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2 class="title-with-info">Nifty 500 Breadth + Rotation <button type="button" class="info-button" data-help="Measures participation across the fixed Nifty 500 universe and identifies stocks and sectors gaining or losing relative strength." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span>Full universe participation</span>
      </div>
      <div id="nifty500Rotation"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2 class="title-with-info">Nifty 500 52-Week Highs <button type="button" class="info-button" data-help="Separates fresh highs, near-high continuation candidates and laggards. More high-quality new highs generally indicate expanding leadership." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span>Technical leaders, highs, and laggards</span>
      </div>
      <div id="nseHighs"></div>
    </section>
    <section class="two-col">
      <div class="panel">
        <div class="section-title">
          <h2 class="title-with-info">Market Technicals <button type="button" class="info-button" data-help="Shows each index's recent return, placement versus its 20-day and 50-day averages, and 14-day RSI." aria-label="Explain this section" aria-expanded="false">i</button></h2>
          <span>Index placement</span>
        </div>
        <div id="indices"></div>
      </div>
      <div class="panel">
        <div class="section-title">
          <h2 class="title-with-info">Leadership Watchlist <button type="button" class="info-button" data-help="A focused list of stocks showing strong relative strength and proximity to 52-week highs. It is a research watchlist, not an automatic buy list." aria-label="Explain this section" aria-expanded="false">i</button></h2>
          <span>52-week-high screen</span>
        </div>
        <div id="leaders"></div>
      </div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2 class="title-with-info">Bear Watchlist Notes <button type="button" class="info-button" data-help="Your locally stored Bear research notes and attached charts, ordered by most recently modified." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span id="bearNoteCount"></span>
      </div>
      <div id="bearNotes"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2 class="title-with-info">News Wires <button type="button" class="info-button" data-help="Current market headlines grouped by theme. They provide context for price and breadth signals but do not independently determine exposure." aria-label="Explain this section" aria-expanded="false">i</button></h2>
        <span id="wireCount"></span>
      </div>
      <div class="wire-controls">
        <div class="tabs" id="tabs"></div>
        <label class="date-filter">
          Date
          <select id="dateFilter" aria-label="Filter wires by date"></select>
        </label>
      </div>
      <div class="wire-list" id="wires"></div>
    </section>
  </main>
  <div class="image-modal" id="imageModal" aria-hidden="true">
    <button id="imageModalClose">Close</button>
    <img id="imageModalImg" alt="Expanded chart image">
  </div>
  <div class="info-popover" id="infoPopover" role="tooltip" hidden></div>
  <script>
    /* DASHBOARD_PAYLOAD_START */
    const payload = {data_json};
    /* DASHBOARD_PAYLOAD_END */
    const AUTO_REFRESH_MS = 10 * 60 * 1000;
    const VIEW_STATE_KEY = 'market-dashboard-view-state-v1';
    let savedViewState = {{}};
    try {{
      savedViewState = JSON.parse(sessionStorage.getItem(VIEW_STATE_KEY) || '{{}}');
    }} catch (error) {{
      savedViewState = {{}};
    }}
    let activeBearNote = Number(savedViewState.activeBearNote || 0);
    function saveViewState() {{
      const noteList = document.querySelector('#bearNotes .note-list');
      sessionStorage.setItem(VIEW_STATE_KEY, JSON.stringify({{
        activeBearNote,
        pageScrollY: window.scrollY,
        noteListScrollTop: noteList ? noteList.scrollTop : Number(savedViewState.noteListScrollTop || 0),
      }}));
    }}
    function reloadLatestSnapshot() {{
      saveViewState();
      const nextUrl = new URL(window.location.href);
      nextUrl.searchParams.set('refresh', String(Date.now()));
      window.location.replace(nextUrl.toString());
    }}
    async function checkForDashboardUpdate(forceReload = false) {{
      const status = document.getElementById('refreshStatus');
      if (window.location.protocol === 'file:') {{
        status.textContent = 'Local file: rebuild on this Mac for new data';
        if (forceReload) reloadLatestSnapshot();
        return;
      }}
      status.textContent = 'Checking for newer data...';
      try {{
        if (forceReload) {{
          status.textContent = 'Refreshing wires and Bear notes...';
          try {{
            await fetch('/api/refresh', {{ method: 'POST', cache: 'no-store' }});
          }} catch (refreshError) {{
            // Static/GitHub hosting has no local refresh endpoint; reload the latest published payload.
          }}
        }}
        const response = await fetch(`market_dashboard_data.json?refresh=${{Date.now()}}`, {{ cache: 'no-store' }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const latest = await response.json();
        if (forceReload || (latest.generated_at && latest.generated_at !== payload.generated_at)) {{
          reloadLatestSnapshot();
          return;
        }}
        status.textContent = `Current as of ${{payload.generated_at}}`;
      }} catch (error) {{
        status.textContent = 'Could not check just now; will retry automatically';
      }}
    }}
    document.getElementById('refreshNow').addEventListener('click', () => checkForDashboardUpdate(true));
    if (window.location.protocol !== 'file:') {{
      window.setInterval(() => checkForDashboardUpdate(false), AUTO_REFRESH_MS);
    }} else {{
      document.getElementById('refreshStatus').textContent = 'Local file: rebuild on this Mac for new data';
    }}
    const fmt = new Intl.NumberFormat('en-IN', {{ maximumFractionDigits: 2 }});
    const pct = value => value === null || value === undefined || Number.isNaN(Number(value)) ? '' : `${{Number(value).toFixed(2)}}%`;
    const cls = value => value === null || value === undefined || Number.isNaN(Number(value)) ? 'flat' : Number(value) > 0 ? 'pos' : Number(value) < 0 ? 'neg' : 'flat';
    const shortDate = value => (value || '').slice(0, 10);
    const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}}[char]));
    const infoPopover = document.getElementById('infoPopover');
    let activeInfoButton = null;
    function closeInfo() {{
      infoPopover.hidden = true;
      if (activeInfoButton) activeInfoButton.setAttribute('aria-expanded', 'false');
      activeInfoButton = null;
    }}
    function openInfo(button) {{
      if (activeInfoButton === button && !infoPopover.hidden) {{ closeInfo(); return; }}
      closeInfo();
      activeInfoButton = button;
      button.setAttribute('aria-expanded', 'true');
      infoPopover.textContent = button.dataset.help || '';
      infoPopover.hidden = false;
      const rect = button.getBoundingClientRect();
      const width = Math.min(330, window.innerWidth - 24);
      const left = Math.min(Math.max(12, rect.left), window.innerWidth - width - 12);
      infoPopover.style.left = `${{left}}px`;
      infoPopover.style.top = `${{Math.min(rect.bottom + 8, window.innerHeight - infoPopover.offsetHeight - 12)}}px`;
    }}
    document.addEventListener('click', event => {{
      const button = event.target.closest('.info-button');
      if (button) {{ event.stopPropagation(); openInfo(button); }} else if (!event.target.closest('#infoPopover')) closeInfo();
    }});
    document.addEventListener('keydown', event => {{ if (event.key === 'Escape') closeInfo(); }});
    window.addEventListener('scroll', closeInfo, {{ passive: true }});
    const metricHelp = {{
      'Market WOE constructive': 'Percentage of stocks scoring at least 3 of 4: close above SMA10, close above SMA20, bullish Supertrend (20, 2.5), and RSI14 at or above 50.',
      'Above SMA50': 'Percentage of the fixed Nifty 500 universe closing above its 50-day simple moving average. Around 50% is breadth equilibrium.',
      'Above SMA100': 'Percentage of the fixed universe above its 100-day simple moving average; a medium-term participation measure.',
      'Above SMA200': 'Percentage above the 200-day simple moving average; a long-term measure of market health.',
      'MA200 rising': 'Percentage of stocks whose 200-day moving average is rising, showing whether long-term trends are improving.',
      '20D new highs': 'Percentage of stocks making a new 20-trading-day high, used to detect fresh leadership.',
      'Momentum average': 'Cross-sectional average 20-day stock return. Positive values mean the typical stock has upward momentum.',
      'Momentum dispersion': 'Spread of 20-day returns across stocks. High dispersion means winners and losers are separating more widely.',
      'Exceptional leaders': 'Stocks passing the strongest combined leadership-quality conditions, even if broad market breadth is weak.',
      'Emerging leaders': 'Stocks whose relative strength and setup quality are improving but have not yet reached exceptional-leader status.',
      'Near ATH': 'Number of stocks at or close to their all-time high, a sign of durable price leadership.',
      'Positive RS vs Nifty': 'Number of stocks outperforming the Nifty benchmark over the model’s relative-strength window.'
    }};
    const research = payload.breadth_leadership || {{}};
    const rb = research.breadth || {{}};
    const rh = research.market_health || {{}};
    const rl = research.leadership || {{}};
    document.getElementById('researchAsOf').textContent = research.as_of_date ? `Fixed Nifty 500 · ${{research.as_of_date}}` : 'Fixed current Nifty 500';
    const researchMetrics = [
      ['Market WOE constructive', pct(rh.woe_constructive_pct)],
      ['Above SMA50', pct(rb.above_sma50_pct)], ['Above SMA100', pct(rb.above_sma100_pct)], ['Above SMA200', pct(rb.above_sma200_pct)],
      ['MA200 rising', pct(rb.ma200_rising_pct)], ['20D new highs', pct(rb.new_high20_pct)],
      ['Momentum average', pct(rb.momentum20_avg_pct)], ['Momentum dispersion', pct(rb.momentum20_dispersion_pct)],
      ['Exceptional leaders', rl.exceptional_count ?? 0], ['Emerging leaders', rl.emerging_count ?? 0],
      ['Near ATH', rl.ath_or_near_ath_count ?? 0], ['Positive RS vs Nifty', rl.rs_positive_count ?? 0]
    ].map(([label,value]) => `<div class="metric"><span class="eyebrow title-with-info">${{label}}<button type="button" class="info-button" data-help="${{esc(metricHelp[label] || '')}}" aria-label="Explain ${{esc(label)}}" aria-expanded="false">i</button></span><b>${{value}}</b></div>`).join('');
    const researchLeaderRows = (rl.top40_enter || []).slice(0,15).map(row => `<tr>
      <td><strong>${{esc(row.symbol)}}</strong><div class="source">${{esc(row.sector)}}</div></td><td class="num">${{row.rank}}</td>
      <td class="num">${{Number(row.leader_score || 0).toFixed(1)}}</td><td class="num ${{cls(row.rs20_pct)}}">${{pct(row.rs20_pct)}}</td>
      <td class="num ${{cls(row.rs_acceleration_pct)}}">${{pct(row.rs_acceleration_pct)}}</td><td class="num">${{row.woe}}/4</td>
      <td>${{row.within_3pct_ath ? '<span class="pill leader">Near ATH</span>' : row.new_high20 ? '<span class="pill leader">20D high</span>' : ''}}</td>
      <td><span class="pill ${{row.exceptional ? 'leader' : row.emerging ? 'watch' : 'neutral'}}">${{row.exceptional ? 'Exceptional' : row.emerging ? 'Emerging' : 'Leader'}}</span></td></tr>`).join('');
    document.getElementById('breadthLeadership').innerHTML = research.as_of_date ? `
      <div class="metric-grid">${{researchMetrics}}</div>
      <div class="empty" style="margin:12px 16px;"><strong>Deployment posture: ${{esc(rh.deployment_posture || '')}}</strong><br>
      Breadth equilibrium is 50%; oversold/overbought reference zones are 15%/85%. WOE is constructive at 3 of 4. Leadership uses equal-weight cross-sectional percentile ranks, weekly Top 40 entry / Top 60 retention.</div>
      <div style="padding:0 16px 16px"><div class="detail-heading">RS, ATH and emerging leadership</div>
      <table><thead><tr><th>Stock</th><th class="num">Rank</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS accel</th><th class="num">WOE</th><th>High status</th><th>Quality</th></tr></thead><tbody>${{researchLeaderRows}}</tbody></table></div>
      <div class="two-col" style="padding:0 16px 16px"><div class="empty"><strong>Three drawdowns remain distinct</strong><br>Peak-to-trough NAV drawdown; Entry-Capital Drawdown measured from each deployment point; intra-trade drawdown measured within each open position.</div><div class="empty"><strong>Exposure ladder</strong><br>Weak + no leaders → cash · weak + exceptional leaders → concentrated partial exposure · expanding leadership → progressive exposure · broad health + strong leadership → full exposure eligible.</div></div>`
      : '<div class="empty">Build the fixed-universe breadth and leadership snapshot to populate this section.</div>';
    const normalizeRef = value => decodeURIComponent(String(value || '').trim());
    const imageModal = document.getElementById('imageModal');
    const imageModalImg = document.getElementById('imageModalImg');
    document.getElementById('imageModalClose').addEventListener('click', () => imageModal.classList.remove('open'));
    imageModal.addEventListener('click', event => {{
      if (event.target === imageModal) imageModal.classList.remove('open');
    }});
    function openImageModal(src) {{
      imageModalImg.src = src;
      imageModal.classList.add('open');
    }}
    function versionedBearAsset(path) {{
      const version = encodeURIComponent(String((payload.bear_watchlist_notes || {{}}).generated_at || payload.generated_at || 'current'));
      return `${{path}}${{String(path).includes('?') ? '&' : '?'}}v=${{version}}`;
    }}
    function renderNoteContent(note, fullTextEnabled) {{
      const text = fullTextEnabled ? String(note.note || '') : String(note.excerpt || 'Full note text was not embedded in this dashboard build.');
      const images = note.images || [];
      let imageCursor = 0;
      const blocks = [];
      for (const line of text.split('\\n')) {{
        const imageMatch = line.match(/^\\s*!\\[[^\\]]*\\]\\(([^)]+)\\)\\s*$/);
        if (imageMatch) {{
          const ref = normalizeRef(imageMatch[1]);
          let image = images.find((item, idx) => idx >= imageCursor && normalizeRef(item.original_ref) === ref);
          if (!image) image = images[imageCursor];
          if (image && image.asset_path) {{
            imageCursor = images.indexOf(image) + 1;
            const imagePath = versionedBearAsset(image.asset_path);
            blocks.push(`<img class="note-image" src="${{esc(imagePath)}}" alt="${{esc(note.title || 'Bear note image')}}" data-fullsrc="${{esc(imagePath)}}">`);
          }} else {{
            blocks.push(`<p>${{esc(line)}}</p>`);
          }}
          continue;
        }}
        if (line.startsWith('## ')) {{
          blocks.push(`<h3>${{esc(line.replace(/^##\\s+/, ''))}}</h3>`);
        }} else if (line.startsWith('# ')) {{
          blocks.push(`<h3>${{esc(line.replace(/^#\\s+/, ''))}}</h3>`);
        }} else if (line.trim()) {{
          blocks.push(`<p>${{esc(line)}}</p>`);
        }} else {{
          blocks.push('<p></p>');
        }}
      }}
      if (!blocks.length && images.length) {{
        images.forEach(image => {{
          const imagePath = versionedBearAsset(image.asset_path);
          blocks.push(`<img class="note-image" src="${{esc(imagePath)}}" alt="${{esc(note.title || 'Bear note image')}}" data-fullsrc="${{esc(imagePath)}}">`);
        }});
      }}
      return `<div class="note-content">${{blocks.join('')}}</div>`;
    }}
    const quality = payload.data_quality || {{}};
    const qualityParts = [];
    if (quality.index_expected) qualityParts.push(`indices ${{quality.index_downloaded}}/${{quality.index_expected}}`);
    if (quality.index_valid !== undefined) qualityParts.push(`valid indices ${{quality.index_valid}}/${{quality.index_downloaded}}`);
    if (quality.index_stale) qualityParts.push(`stale indices ${{quality.index_stale}}`);
    if (quality.index_outliers) qualityParts.push(`outlier indices ${{quality.index_outliers}}`);
    if (quality.breadth_expected) qualityParts.push(`India breadth ${{quality.breadth_downloaded}}/${{quality.breadth_expected}}`);
    if (quality.india_market_date) qualityParts.push(`India session ${{quality.india_market_date}}`);
    if (quality.nifty500_expected) qualityParts.push(`Nifty 500 ${{quality.nifty500_downloaded}}/${{quality.nifty500_expected}}`);
    const coverageText = qualityParts.length ? ` | Coverage: ${{qualityParts.join(', ')}}` : '';
    document.getElementById('generated').textContent = `Generated ${{payload.generated_at}} from ${{payload.wire_file || 'wire data'}}${{coverageText}}`;
    document.getElementById('stance').textContent = payload.posture.stance;
    const postureDate = payload.posture.wire_scope_date ? `${{payload.posture.wire_scope_date}} · ` : '';
    document.getElementById('stanceDetail').textContent = `${{postureDate}}${{payload.posture.wire_count}} latest-date wires, ${{payload.posture.india_wire_count || 0}} India, ${{payload.posture.ai_wire_count}} AI/tech, ${{payload.posture.selling_wire_count}} selling`;

    const narrative = document.getElementById('narrative');
    narrative.innerHTML = payload.narrative.map(item => `
      <article class="panel note">
        <div class="eyebrow">${{item.label}}</div>
        <h2>${{item.title}}</h2>
        <p>${{item.text}}</p>
      </article>`).join('') || '<div class="panel empty">No current wire cluster found.</div>';

    const regionId = region => `region-${{String(region).toLowerCase().replace(/[^a-z0-9]+/g, '-')}}`;
    function indexDataNote(row) {{
      return row.data_note ? `<div class="source data-warning">${{esc(row.data_note)}}</div>` : '';
    }}
    function regionDetailRows(region) {{
      const rows = payload.indices
        .filter(item => item.region === region && !item.is_stale && !item.is_outlier)
        .sort((a, b) => (Number(b.return_1d_pct) || 0) - (Number(a.return_1d_pct) || 0));
      return rows.map(item => `
        <tr>
          <td><strong>${{item.name}}</strong><div class="source">${{item.ticker}} · ${{item.last_date || ''}}</div>${{indexDataNote(item)}}</td>
          <td class="num">${{fmt.format(item.last || 0)}}</td>
          <td class="num ${{cls(item.return_1d_pct)}}">${{pct(item.return_1d_pct)}}</td>
          <td class="num ${{cls(item.return_5d_pct)}}">${{pct(item.return_5d_pct)}}</td>
          <td class="num ${{cls(item.distance_sma_20_pct)}}">${{pct(item.distance_sma_20_pct)}}</td>
          <td class="num ${{cls(item.distance_sma_50_pct)}}">${{pct(item.distance_sma_50_pct)}}</td>
          <td class="num">${{item.rsi_14 ?? ''}}</td>
        </tr>`).join('');
    }}
    const regionalRows = payload.regional_breadth.map(row => {{
      const id = regionId(row.region);
      const details = regionDetailRows(row.region);
      return `
        <tr class="region-row" data-region-id="${{id}}">
          <td><span class="region-toggle">+</span><strong>${{row.region}}</strong><div class="source">${{row.count}} valid indices</div></td>
          <td class="num ${{cls(row.average_1d_pct)}}">${{pct(row.average_1d_pct)}}</td>
          <td class="num ${{cls(row.average_5d_pct)}}">${{pct(row.average_5d_pct)}}</td>
          <td class="num">${{row.advancers}} / ${{row.decliners}}</td>
          <td class="num">${{row.advance_decline_ratio ?? ''}}</td>
          <td class="num">${{pct(row.above_20dma_pct)}}</td>
          <td class="num">${{pct(row.above_50dma_pct)}}</td>
        </tr>
        <tr class="detail-row" id="${{id}}">
          <td colspan="7" class="detail-cell">
            <div class="detail-wrap">
              <div class="detail-heading">${{row.region}} index constituents, sorted by 1-day move</div>
              ${{details ? `
                <table>
                  <thead><tr><th>Index</th><th class="num">Last</th><th class="num">1D</th><th class="num">5D</th><th class="num">vs 20DMA</th><th class="num">vs 50DMA</th><th class="num">RSI</th></tr></thead>
                  <tbody>${{details}}</tbody>
                </table>` : '<div class="empty">No index constituents available for this region.</div>'}}
            </div>
          </td>
        </tr>`;
    }}).join('');
    document.getElementById('regionalBreadth').innerHTML = regionalRows ? `
      <table>
        <thead><tr><th>Region</th><th class="num">Avg 1D</th><th class="num">Avg 5D</th><th class="num">A/D</th><th class="num">A/D ratio</th><th class="num">Above 20DMA</th><th class="num">Above 50DMA</th></tr></thead>
        <tbody>${{regionalRows}}</tbody>
      </table>` : '<div class="empty">Global regional breadth will appear after rebuilding from Terminal with network access.</div>';
    document.querySelectorAll('.region-row').forEach(row => {{
      row.addEventListener('click', () => {{
        const detail = document.getElementById(row.dataset.regionId);
        const toggle = row.querySelector('.region-toggle');
        const isOpen = detail.classList.toggle('open');
        row.classList.toggle('open', isOpen);
        if (toggle) toggle.textContent = isOpen ? '-' : '+';
      }});
    }});

    function globalMoveRows(rows) {{
      return rows.map(row => `
        <tr>
          <td><strong>${{row.name}}</strong><div class="source">${{row.region}} · ${{row.ticker}}</div></td>
          <td class="num ${{cls(row.return_1d_pct)}}">${{pct(row.return_1d_pct)}}</td>
          <td class="num ${{cls(row.return_5d_pct)}}">${{pct(row.return_5d_pct)}}</td>
        </tr>`).join('');
    }}
    const globalLeaderRows = globalMoveRows(payload.global_leaders_laggards.leaders || []);
    const globalLaggardRows = globalMoveRows(payload.global_leaders_laggards.laggards || []);
    document.getElementById('globalLeaders').innerHTML = globalLeaderRows ? `
      <table><thead><tr><th>Index</th><th class="num">1D</th><th class="num">5D</th></tr></thead><tbody>${{globalLeaderRows}}</tbody></table>` : '<div class="empty">No global leader data yet.</div>';
    document.getElementById('globalLaggards').innerHTML = globalLaggardRows ? `
      <table><thead><tr><th>Index</th><th class="num">1D</th><th class="num">5D</th></tr></thead><tbody>${{globalLaggardRows}}</tbody></table>` : '<div class="empty">No global laggard data yet.</div>';

    const breadth = payload.india_breadth || {{}};
    const breadthSummary = breadth.summary || {{}};
    const breadthMetrics = [
      ['A/D', `${{breadthSummary.advancers ?? 0}} / ${{breadthSummary.decliners ?? 0}}`],
      ['A/D ratio', breadthSummary.advance_decline_ratio ?? ''],
      ['Up turnover share', pct(breadthSummary.up_turnover_share_pct)],
      ['High-volume decliners', breadthSummary.high_volume_decliners ?? 0],
      ['Above 20DMA', pct(breadthSummary.above_20dma_pct)],
      ['Above 50DMA', pct(breadthSummary.above_50dma_pct)],
      ['Downloaded', `${{breadthSummary.downloaded_count ?? 0}} / ${{breadthSummary.source_count ?? 0}}`],
      ['Down turnover share', pct(breadthSummary.down_turnover_share_pct)],
      ['Live breadth as of', breadthSummary.breadth_as_of_timestamp || breadthSummary.as_of_date || 'Unavailable'],
      ['EOD technicals as of', breadthSummary.technical_as_of_date || 'Unavailable']
    ];
    const breadthMetricHtml = breadthMetrics.map(([label, value]) => `
      <div class="metric"><span class="eyebrow">${{label}}</span><b>${{value}}</b></div>`).join('');
    const sectorRows = (breadth.sector_breadth || []).map(row => `
      <tr>
        <td>${{row.sector}}</td>
        <td class="num">${{row.advancers}} / ${{row.decliners}}</td>
        <td class="num ${{cls(row.average_1d_pct)}}">${{pct(row.average_1d_pct)}}</td>
        <td class="num">${{row.average_volume_ratio_20d ?? ''}}</td>
      </tr>`).join('');
    const breadthAlertRows = (breadth.alerts || []).map(row => `
      <tr>
        <td><strong>${{row.symbol || row.name || ''}}</strong><div class="source">${{row.sector || ''}}</div></td>
        <td class="num">${{row.ltp ?? ''}}</td>
        <td class="num ${{cls(row.day_move_pct)}}">${{pct(row.day_move_pct)}}</td>
        <td class="num">${{row.volume_ratio_20d == null ? '' : Number(row.volume_ratio_20d).toFixed(2) + 'x'}}</td>
        <td>${{row.alert || ''}}</td>
      </tr>`).join('');
    document.getElementById('indiaBreadth').innerHTML = breadthSummary.downloaded_count ? `
      <div class="empty" style="padding:0 0 12px;">This is the ranked technical-screen universe, not the full Nifty 500. Advances/declines, LTP, turnover pressure, and volume are from the latest live/delayed snapshot. Moving-average placement remains fixed to the last completed EOD technical session.</div>
      <div class="metric-grid">${{breadthMetricHtml}}</div>
      <table>
        <thead><tr><th>Sector</th><th class="num">A/D</th><th class="num">Avg 1D</th><th class="num">Vol/20D</th></tr></thead>
        <tbody>${{sectorRows}}</tbody>
      </table>
      ${{breadthAlertRows ? `<div class="detail-heading" style="padding:16px 16px 0;">Intraday exceptions</div><table><thead><tr><th>Stock</th><th class="num">LTP</th><th class="num">Move</th><th class="num">Vol/20D</th><th>Alert</th></tr></thead><tbody>${{breadthAlertRows}}</tbody></table>` : ''}}` : '<div class="empty">India breadth and volume will populate after the live-market refresh succeeds. The EOD technical file is retained separately.</div>';

    const mutualFunds = payload.mutual_fund_navs || {{}};
    const fundRows = (mutualFunds.funds || []).map(row => `
      <tr>
        <td><strong>${{row.label || row.scheme_name || ''}}</strong><div class="source">${{row.scheme_name || (row.matched === false ? 'No AMFI match found' : '')}}</div></td>
        <td class="num">${{row.nav ?? ''}}</td>
        <td>${{row.nav_date || ''}}</td>
        <td class="num ${{cls(row.return_1d_pct)}}">${{pct(row.return_1d_pct)}}</td>
        <td class="num ${{cls(row.return_5d_pct)}}">${{pct(row.return_5d_pct)}}</td>
        <td class="num ${{cls(row.return_20d_pct)}}">${{pct(row.return_20d_pct)}}</td>
        <td class="num ${{cls(row.return_60d_pct)}}">${{pct(row.return_60d_pct)}}</td>
      </tr>`).join('');
    const matchedFunds = (mutualFunds.funds || []).filter(row => row.matched !== false).length;
    const fundReturns5d = (mutualFunds.funds || [])
      .map(row => Number(row.return_5d_pct))
      .filter(value => !Number.isNaN(value));
    const bestFund5d = fundReturns5d.length ? Math.max(...fundReturns5d) : null;
    const worstFund5d = fundReturns5d.length ? Math.min(...fundReturns5d) : null;
    document.getElementById('mutualFundNavs').innerHTML = mutualFunds.source_file ? `
      <div class="metric-grid">
        <div class="metric"><span class="eyebrow">Funds loaded</span><b>${{mutualFunds.count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Matched</span><b>${{matchedFunds}}</b></div>
        <div class="metric"><span class="eyebrow">Best 5D NAV</span><b class="${{cls(bestFund5d)}}">${{pct(bestFund5d)}}</b></div>
        <div class="metric"><span class="eyebrow">Worst 5D NAV</span><b class="${{cls(worstFund5d)}}">${{pct(worstFund5d)}}</b></div>
      </div>
      <table>
        <thead><tr><th>Fund</th><th class="num">NAV</th><th>Date</th><th class="num">1D</th><th class="num">5D</th><th class="num">20D</th><th class="num">60D</th></tr></thead>
        <tbody>${{fundRows}}</tbody>
      </table>` : '<div class="empty">Run the AMFI mutual-fund NAV fetcher to populate HDFC 500 and other fund NAV movement.</div>';

    const rotation = payload.nifty50_rotation || {{}};
    function rotationPillClass(signal) {{
      const text = String(signal || '');
      if (text.includes('Risk') || text.includes('Weakening')) return 'risk';
      if (text.includes('Mean') || text.includes('Setup')) return 'watch';
      return 'leader';
    }}
    function rotationRows(rows) {{
      return (rows || []).map(row => `
        <tr>
          <td><strong>${{row.symbol || ''}}</strong><div class="source">${{row.name || ''}}</div></td>
          <td><span class="pill ${{rotationPillClass(row.rotation_signal)}}">${{row.rotation_signal || ''}}</span></td>
          <td class="num">${{row.rotation_score ?? ''}}</td>
          <td class="num ${{cls(row.rs_return_20d_pct)}}">${{pct(row.rs_return_20d_pct)}}</td>
          <td class="num ${{cls(row.rs_distance_sma_50_pct)}}">${{pct(row.rs_distance_sma_50_pct)}}</td>
          <td class="num">${{row.volume_ratio_20d ?? ''}}</td>
          <td class="num">${{row.rsi_14 ?? ''}}</td>
          <td class="num ${{cls(row.distance_sma_50_pct)}}">${{pct(row.distance_sma_50_pct)}}</td>
        </tr>`).join('');
    }}
    function setupRows(rows) {{
      return (rows || []).map(row => `
        <tr>
          <td><strong>${{row.symbol || ''}}</strong><div class="source">${{row.name || ''}}</div></td>
          <td><strong>${{esc(row.primary_setup || '')}}</strong><div class="source">${{esc(row.setup_tags || '')}}</div></td>
          <td><span class="pill ${{String(row.setup_quality || '').includes('Actionable') ? 'leader' : 'watch'}}">${{esc(row.setup_quality || '')}}</span></td>
          <td class="num">${{row.setup_score ?? ''}}</td>
          <td class="num">${{row.setup_pivot_level ?? ''}}</td>
          <td class="num">${{pct(row.setup_retest_gap_pct)}}</td>
          <td class="num">${{row.setup_volume_ratio ?? ''}}</td>
          <td class="num">${{row.rsi_14 ?? ''}}</td>
          <td>${{esc(row.setup_commentary || '')}}</td>
        </tr>`).join('');
    }}
    const rotationLeaderRows = rotationRows(rotation.leaders);
    const rotationWeakRows = rotationRows(rotation.weakening);
    const rotationMeanRows = rotationRows(rotation.mean_reversion);
    document.getElementById('nifty50Rotation').innerHTML = rotation.source_file ? `
      <div class="metric-grid">
        <div class="metric"><span class="eyebrow">Universe</span><b>${{rotation.count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Top bucket</span><b>${{(rotation.leaders || []).length}}</b></div>
        <div class="metric"><span class="eyebrow">Weakening</span><b>${{(rotation.weakening || []).length}}</b></div>
        <div class="metric"><span class="eyebrow">Mean reversion</span><b>${{(rotation.mean_reversion || []).length}}</b></div>
      </div>
      <div style="padding: 10px 16px 0;">
        <div class="detail-heading">Strongest rotation candidates</div>
      </div>
      <table>
        <thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead>
        <tbody>${{rotationLeaderRows}}</tbody>
      </table>
      <div class="two-col" style="padding: 10px 16px 16px;">
        <div>
          <div class="detail-heading">Weakening names</div>
          ${{rotationWeakRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotationWeakRows}}</tbody></table>` : '<div class="empty">No weakening bucket in latest file.</div>'}}
        </div>
        <div>
          <div class="detail-heading">Mean-reversion bounces</div>
          ${{rotationMeanRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotationMeanRows}}</tbody></table>` : '<div class="empty">No mean-reversion bucket in latest file.</div>'}}
        </div>
      </div>` : '<div class="empty">Run the Nifty 50 rotation analysis to populate relative-strength, volume, and RSI rotation.</div>';

    const rotation500 = payload.nifty500_rotation || {{}};
    const rotation500Summary = rotation500.summary || {{}};
    const rotation500BreadthMetrics = [
      ['A/D', `${{rotation500Summary.advancers ?? 0}} / ${{rotation500Summary.decliners ?? 0}}`],
      ['A/D ratio', rotation500Summary.advance_decline_ratio ?? ''],
      ['Up turnover share', pct(rotation500Summary.up_turnover_share_pct)],
      ['High-volume decliners', rotation500Summary.high_volume_decliners ?? 0],
      ['Above 20DMA', pct(rotation500Summary.above_20dma_pct)],
      ['Above 50DMA', pct(rotation500Summary.above_50dma_pct)],
      ['Above 200DMA', pct(rotation500Summary.above_200dma_pct)],
      ['Downloaded', `${{rotation500Summary.downloaded_count ?? rotation500.downloaded_count ?? 0}} / ${{rotation500Summary.source_count ?? rotation500.count ?? 0}}`],
      ['Down turnover share', pct(rotation500Summary.down_turnover_share_pct)],
      ['Breadth as of', rotation500Summary.breadth_as_of_timestamp || rotation500Summary.as_of_date || 'Unavailable'],
      ['EOD technicals as of', rotation500Summary.technical_as_of_date || rotation500.as_of_date || 'Unavailable']
    ];
    const rotation500BreadthHtml = rotation500BreadthMetrics.map(([label, value]) => `
      <div class="metric"><span class="eyebrow">${{label}}</span><b>${{value}}</b></div>`).join('');
    const rotation500SectorRows = (rotation500Summary.sector_breadth || []).map(row => `
      <tr>
        <td>${{row.sector || 'Unknown'}}<div class="source">${{row.count ?? 0}} stocks</div></td>
        <td class="num">${{row.advancers ?? 0}} / ${{row.decliners ?? 0}}</td>
        <td class="num ${{cls(row.average_1d_pct)}}">${{pct(row.average_1d_pct)}}</td>
        <td class="num">${{pct(row.above_50dma_pct)}}</td>
        <td class="num">${{row.average_volume_ratio_20d ?? ''}}</td>
      </tr>`).join('');
    const rotation500StrengthRows = rotationRows(rotation500.strengthening);
    const rotation500SetupRows = setupRows(rotation500.setup_candidates);
    const rotation500ExtendedRows = rotationRows(rotation500.extended);
    const rotation500WeakRows = rotationRows(rotation500.weakening);
    const rotation500LaggardRows = rotationRows(rotation500.laggards);
    const rotation500MeanRows = rotationRows(rotation500.mean_reversion);
    const rotation500RiskRows = rotationRows(rotation500.risk_review);
    const rotation500AlertRows = (rotation500.alerts || []).map(row => `
      <tr>
        <td><strong>${{row.symbol || row.name || ''}}</strong><div class="source">${{row.sector || ''}}</div></td>
        <td class="num">${{row.ltp ?? ''}}</td>
        <td class="num ${{cls(row.day_move_pct)}}">${{pct(row.day_move_pct)}}</td>
        <td class="num">${{row.volume_ratio_20d == null ? '' : Number(row.volume_ratio_20d).toFixed(2) + 'x'}}</td>
        <td>${{row.alert || ''}}</td>
      </tr>`).join('');
    document.getElementById('nifty500Rotation').innerHTML = rotation500.source_file ? `
      <div class="metric-grid">
        ${{rotation500BreadthHtml}}
      </div>
      <div class="empty" style="margin: 12px 16px;">
        This is the fixed current Nifty 500 universe. Participation, volume, RS, RSI, and moving averages use the cache through ${{rotation500Summary.as_of_date || rotation500.as_of_date || 'the latest completed session'}}. Data layer: ${{rotation500Summary.breadth_layer || 'EOD cache'}}. True RS leaders are strengthening versus ${{rotation500.benchmark_ticker || 'the benchmark'}}.
      </div>
      ${{rotation500AlertRows ? `<div style="padding:0 16px 12px;"><div class="detail-heading">Intraday exceptions</div><table><thead><tr><th>Stock</th><th class="num">LTP</th><th class="num">Move</th><th class="num">Vol/20D</th><th>Alert</th></tr></thead><tbody>${{rotation500AlertRows}}</tbody></table></div>` : ''}}
      <div style="padding: 0 16px 12px;">
        <div class="detail-heading">Nifty 500 sector breadth</div>
        ${{rotation500SectorRows ? `<table><thead><tr><th>Sector</th><th class="num">A/D</th><th class="num">Avg 1D</th><th class="num">Above 50DMA</th><th class="num">Vol/20D</th></tr></thead><tbody>${{rotation500SectorRows}}</tbody></table>` : '<div class="empty">Sector breadth will appear after the next Nifty 500 refresh includes sector data.</div>'}}
      </div>
      <div style="padding: 10px 16px 0;">
        <div class="detail-heading">True RS leaders</div>
      </div>
      ${{rotation500StrengthRows ? `<table>
        <thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead>
        <tbody>${{rotation500StrengthRows}}</tbody>
      </table>` : '<div class="empty">No Nifty 500 names are in the true strengthening bucket in the latest snapshot.</div>'}}
      <div style="padding: 10px 16px 0;">
        <div class="detail-heading">Setup candidates near entry zones</div>
      </div>
      ${{rotation500SetupRows ? `<table>
        <thead><tr><th>Stock</th><th>Setup</th><th>Quality</th><th class="num">Setup score</th><th class="num">Pivot</th><th class="num">Retest gap</th><th class="num">Vol/20D</th><th class="num">RSI</th><th>Read</th></tr></thead>
        <tbody>${{rotation500SetupRows}}</tbody>
      </table>` : '<div class="empty">No current breakout-retest, flag, or inverse head-and-shoulders setup was detected in the latest snapshot.</div>'}}
      <div class="two-col" style="padding: 10px 16px 16px;">
        <div>
          <div class="detail-heading">Extended leaders</div>
          ${{rotation500ExtendedRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotation500ExtendedRows}}</tbody></table>` : '<div class="empty">No extended-leader bucket in latest file.</div>'}}
        </div>
        <div>
          <div class="detail-heading">Mean-reversion bounces</div>
          ${{rotation500MeanRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotation500MeanRows}}</tbody></table>` : '<div class="empty">No mean-reversion bucket in latest file.</div>'}}
        </div>
      </div>
      <div class="two-col" style="padding: 0 16px 16px;">
        <div>
          <div class="detail-heading">Weakening names</div>
          ${{rotation500WeakRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotation500WeakRows}}</tbody></table>` : '<div class="empty">No weakening bucket in latest file.</div>'}}
        </div>
        <div>
          <div class="detail-heading">Risk review / lowest scores</div>
          ${{(rotation500RiskRows || rotation500LaggardRows) ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotation500RiskRows || rotation500LaggardRows}}</tbody></table>` : '<div class="empty">No risk or laggard bucket in latest file.</div>'}}
        </div>
      </div>` : '<div class="empty">No Nifty 500 rotation snapshot was found. Run the market dashboard refresh once; if Yahoo blocks a few symbols, the dashboard will still use the latest usable snapshot and show the download coverage here.</div>';

    const highs = payload.nse_highs || {{}};
    function technicalHighRows(rows) {{
      return (rows || []).map(row => {{
        const hasOverlay = Boolean(row.has_technical_overlay);
        const rawSetupLabel = String(row.setup_label || '');
        const displaySetupLabel = rawSetupLabel === 'High-proximity watch' ? 'Near high, needs RS/RSI check' : rawSetupLabel;
        const statusText = row.relative_strength_leader
          ? 'RS leader'
          : hasOverlay
            ? 'Technical watch'
            : '52W high watch';
        const statusClass = row.relative_strength_leader ? 'leader' : 'watch';
        const pnfText = row.pnf_signal || (hasOverlay ? '' : 'Technical overlay pending');
        return `
          <tr>
            <td><strong>${{esc(row.symbol || '')}}</strong><div class="source">${{esc(row.company || '')}}</div><div class="source">${{esc(displaySetupLabel)}}</div></td>
            <td class="num">${{row.rank_score ?? ''}}</td>
            <td><span class="pill ${{statusClass}}">${{esc(statusText)}}</span></td>
            <td class="num ${{cls(row.relative_strength_ratio_distance_sma_50_pct)}}">${{pct(row.relative_strength_ratio_distance_sma_50_pct)}}</td>
            <td class="num">${{row.rsi_14 ?? ''}}</td>
            <td>${{esc(pnfText)}}</td>
            <td class="num ${{cls(row.distance_from_52w_high_pct)}}">${{pct(row.distance_from_52w_high_pct)}}</td>
            <td>${{esc(row.chart_pattern_view || '')}}</td>
          </tr>`;
      }}).join('');
    }}
    function highRows(rows) {{
      return (rows || []).map(row => `
        <tr>
          <td><strong>${{row.symbol || ''}}</strong><div class="source">${{row.company || row.identifier || ''}}</div></td>
          <td class="num">${{row.last_price ?? ''}}</td>
          <td class="num ${{cls(row.change_pct)}}">${{pct(row.change_pct)}}</td>
          <td class="num">${{row.year_high ?? ''}}</td>
          <td class="num ${{cls(row.distance_from_52w_high_pct)}}">${{pct(row.distance_from_52w_high_pct)}}</td>
          <td class="num">${{row.volume ?? ''}}</td>
        </tr>`).join('');
    }}
    const nifty500TechnicalLeaderRows = technicalHighRows(highs.technical_leaders);
    const freshHighRows = highRows(highs.fresh_highs);
    const nearHighRows = highRows(highs.near_highs);
    const structuralLaggardRows = highRows(highs.structural_laggards);
    const worstDayRows = highRows(highs.worst_day_moves);
    document.getElementById('nseHighs').innerHTML = highs.source_file ? `
      <div class="metric-grid">
        <div class="metric"><span class="eyebrow">Fresh highs</span><b>${{highs.fresh_count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Within 3%</span><b>${{highs.near_count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Technical leaders</span><b>${{highs.technical_leaders_count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Universe</span><b>${{highs.total_count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Source</span><b>${{highs.source_name || 'NSE live'}}</b></div>
      </div>
      <div class="empty" style="padding:0 16px 10px;">
        The technical leaders table uses the same RS, RSI, P&F, and reading fields as the leadership watchlist when that overlay is available. The raw fresh/near-high and laggard lists remain below it for breadth context.
      </div>
      <div style="padding: 10px 16px 16px;">
        <div class="detail-heading">Nifty 500 technical leaders</div>
        ${{nifty500TechnicalLeaderRows ? `<table><thead><tr><th>Stock</th><th class="num">Rank</th><th>Status</th><th class="num">RS vs 50DMA</th><th class="num">RSI</th><th>P&F</th><th class="num">52W gap</th><th>Read</th></tr></thead><tbody>${{nifty500TechnicalLeaderRows}}</tbody></table>` : '<div class="empty">No Nifty 500 technical leaders found in the latest file.</div>'}}
      </div>
      <div class="two-col" style="padding: 10px 16px 16px;">
        <div>
          <div class="detail-heading">Technical leaders: fresh 52-week highs</div>
          ${{freshHighRows ? `<table><thead><tr><th>Stock</th><th class="num">Last</th><th class="num">Chg</th><th class="num">52W high</th><th class="num">Distance</th><th class="num">Volume</th></tr></thead><tbody>${{freshHighRows}}</tbody></table>` : '<div class="empty">No fresh highs found in the latest file.</div>'}}
        </div>
        <div>
          <div class="detail-heading">Continuation leaders: within 3% of 52-week high</div>
          ${{nearHighRows ? `<table><thead><tr><th>Stock</th><th class="num">Last</th><th class="num">Chg</th><th class="num">52W high</th><th class="num">Distance</th><th class="num">Volume</th></tr></thead><tbody>${{nearHighRows}}</tbody></table>` : '<div class="empty">No near-high candidates found in the latest file.</div>'}}
        </div>
      </div>
      <div class="two-col" style="padding: 0 16px 16px;">
        <div>
          <div class="detail-heading">Structural laggards vs 52-week high</div>
          ${{structuralLaggardRows ? `<table><thead><tr><th>Stock</th><th class="num">Last</th><th class="num">Chg</th><th class="num">52W high</th><th class="num">Distance</th><th class="num">Volume</th></tr></thead><tbody>${{structuralLaggardRows}}</tbody></table>` : '<div class="empty">No structural laggards found in the latest file.</div>'}}
        </div>
        <div>
          <div class="detail-heading">Weakest day moves</div>
          ${{worstDayRows ? `<table><thead><tr><th>Stock</th><th class="num">Last</th><th class="num">Chg</th><th class="num">52W high</th><th class="num">Distance</th><th class="num">Volume</th></tr></thead><tbody>${{worstDayRows}}</tbody></table>` : '<div class="empty">No weak day moves found in the latest file.</div>'}}
        </div>
      </div>` : '<div class="empty">The Nifty 500 52-week-high file is not present yet. Run the NSE 52-week-high fetcher first, then rebuild the dashboard. If the fetcher fails, NSE may be blocking the request and the dashboard will keep this section empty.</div>';

    const indexRows = payload.indices.map(row => `
      <tr>
        <td><strong>${{row.name}}</strong><div class="source">${{row.region}} · ${{row.ticker}} · ${{row.last_date || ''}}</div>${{indexDataNote(row)}}</td>
        <td class="num">${{fmt.format(row.last || 0)}}</td>
        <td class="num ${{cls(row.return_1d_pct)}}">${{pct(row.return_1d_pct)}}</td>
        <td class="num ${{cls(row.return_5d_pct)}}">${{pct(row.return_5d_pct)}}</td>
        <td class="num ${{cls(row.distance_sma_20_pct)}}">${{pct(row.distance_sma_20_pct)}}</td>
        <td class="num ${{cls(row.distance_sma_50_pct)}}">${{pct(row.distance_sma_50_pct)}}</td>
        <td class="num">${{row.rsi_14 ?? ''}}</td>
      </tr>`).join('');
    document.getElementById('indices').innerHTML = indexRows ? `
      <table>
        <thead><tr><th>Index</th><th class="num">Last</th><th class="num">1D</th><th class="num">5D</th><th class="num">vs 20DMA</th><th class="num">vs 50DMA</th><th class="num">RSI</th></tr></thead>
        <tbody>${{indexRows}}</tbody>
      </table>` : '<div class="empty">Index data will appear after the dashboard is rebuilt from Terminal with network access.</div>';

    const leaderRows = payload.leaders.map(row => `
      <tr>
        <td><strong>${{row.Symbol || ''}}</strong><div class="source">${{row.Description || ''}}</div></td>
        <td class="num">${{row.rank_score ?? ''}}</td>
        <td><span class="pill ${{row.relative_strength_leader ? 'leader' : 'watch'}}">${{row.relative_strength_leader ? 'RS leader' : 'Watch'}}</span></td>
        <td class="num ${{cls(row.relative_strength_ratio_distance_sma_50_pct)}}">${{pct(row.relative_strength_ratio_distance_sma_50_pct)}}</td>
        <td class="num">${{row.rsi_14 ?? ''}}</td>
        <td>${{row.chart_pattern_view || ''}}</td>
      </tr>`).join('');
    document.getElementById('leaders').innerHTML = leaderRows ? `
      <table>
        <thead><tr><th>Stock</th><th class="num">Rank</th><th>Status</th><th class="num">RS vs 50DMA</th><th class="num">RSI</th><th>Read</th></tr></thead>
        <tbody>${{leaderRows}}</tbody>
      </table>` : '<div class="empty">Run the 52-week-high technical screen to populate this section.</div>';

    const bearNotes = payload.bear_watchlist_notes || {{}};
    const bearNoteRows = bearNotes.notes || [];
    activeBearNote = Math.max(0, Math.min(activeBearNote, Math.max(0, bearNoteRows.length - 1)));
    function renderBearNoteReader() {{
      const note = bearNoteRows[activeBearNote] || {{}};
      const reader = document.querySelector('#bearNotes .note-reader');
      if (!reader) return;
      const needsChartCorrection = /Technical status:\s*Chart correction required/i.test(String(note.note || note.excerpt || ''));
      const chartWarning = needsChartCorrection
        ? `<div class="note-warning"><strong>Wrong chart attached</strong><span>The current image is Suven Life Sciences, not Rashi Peripherals. Replace it before relying on any technical reading.</span></div>`
        : '';
      const helper = bearNotes.includes_full_text
        ? ''
        : '<div class="empty" style="padding-left:0;">This build contains only excerpts. Rebuild with Bear note text enabled to read complete notes here.</div>';
      reader.innerHTML = `
        <div class="eyebrow">Selected Note</div>
        <h2>${{esc(note.title || 'Select a note')}}</h2>
        <div class="source">${{esc(note.modified || '')}}</div>
        ${{chartWarning}}
        <div class="note-actions">
          ${{note.bear_link ? `<a href="${{esc(note.bear_link)}}">Open in Bear</a>` : ''}}
        </div>
        ${{helper}}
        ${{renderNoteContent(note, bearNotes.includes_full_text)}}`;
      reader.querySelectorAll('.note-image').forEach(image => {{
        image.addEventListener('click', () => openImageModal(image.dataset.fullsrc || image.src));
      }});
    }}
    function renderBearNotes() {{
      document.getElementById('bearNoteCount').textContent = bearNotes.source_file
        ? `${{bearNotes.shown_count ?? 0}} shown · ${{bearNotes.includes_full_text ? 'full text embedded' : 'links/excerpts only'}}`
        : '';
      if (!bearNotes.source_file) {{
        document.getElementById('bearNotes').innerHTML = '<div class="empty">Run the Bear watchlist export to populate stock-note links and optional note text.</div>';
        return;
      }}
      const list = bearNoteRows.map((row, index) => {{
        const needsChartCorrection = /Technical status:\s*Chart correction required/i.test(String(row.note || row.excerpt || ''));
        return `
          <button class="note-item ${{index === activeBearNote ? 'active' : ''}} ${{needsChartCorrection ? 'warning' : ''}}" data-index="${{index}}">
            <strong>${{esc(row.title)}}</strong>
            <div class="source">${{esc(row.modified || '')}}</div>
            <div class="source">${{esc(row.excerpt || '')}}</div>
            ${{needsChartCorrection ? '<span class="note-warning-label">Wrong chart attached</span>' : ''}}
          </button>`;
      }}).join('');
      document.getElementById('bearNotes').innerHTML = `
        <div class="note-browser">
          <div class="note-list">${{list || '<div class="empty">No Bear notes exported.</div>'}}</div>
          <div class="note-reader"></div>
        </div>`;
      document.querySelectorAll('.note-item').forEach(button => {{
        button.addEventListener('click', () => {{
          activeBearNote = Number(button.dataset.index || 0);
          saveViewState();
          document.querySelectorAll('#bearNotes .note-item').forEach(item => {{
            item.classList.toggle('active', item === button);
          }});
          renderBearNoteReader();
        }});
      }});
      renderBearNoteReader();
    }}
    renderBearNotes();

    const filters = [
      ['all', 'All'],
      ['AI / tech risk', 'AI/Tech'],
      ['market selling', 'Selling'],
      ['India market', 'India'],
      ['macro / flows', 'Macro/Flows'],
      ['mutual', 'Funds']
    ];
    let active = 'all';
    let activeDate = 'all';
    const tabs = document.getElementById('tabs');
    const dateFilter = document.getElementById('dateFilter');
    const availableDates = [...new Set(payload.wires.map(row => shortDate(row.published)).filter(Boolean))].sort().reverse();
    function renderTabs() {{
      tabs.innerHTML = filters.map(([key, label]) => `<button class="${{active === key ? 'active' : ''}}" data-key="${{key}}">${{label}}</button>`).join('');
      tabs.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {{
        active = button.dataset.key;
        renderTabs();
        renderWires();
      }}));
    }}
    function renderDateFilter() {{
      const options = [['all', 'All dates'], ...availableDates.map(date => [date, date])];
      dateFilter.innerHTML = options.map(([value, label]) => `<option value="${{value}}">${{label}}</option>`).join('');
      const today = new Date().toISOString().slice(0, 10);
      if (availableDates.includes(today)) {{
        activeDate = today;
        dateFilter.value = today;
      }}
      dateFilter.addEventListener('change', () => {{
        activeDate = dateFilter.value;
        renderWires();
      }});
    }}
    function renderWires() {{
      const rows = payload.wires.filter(row => {{
        const text = `${{row.themes}} ${{row.title}}`.toLowerCase();
        const themeMatch = active === 'all' || text.includes(active.toLowerCase());
        const dateMatch = activeDate === 'all' || shortDate(row.published) === activeDate;
        return themeMatch && dateMatch;
      }}).slice(0, 80);
      document.getElementById('wireCount').textContent = `${{rows.length}} shown · ${{activeDate === 'all' ? 'all dates' : activeDate}}`;
      document.getElementById('wires').innerHTML = rows.map(row => `
        <article class="wire">
          <div class="source">${{shortDate(row.published)}}<br>${{row.source}}</div>
          <div>
            <a href="${{row.link}}" target="_blank" rel="noreferrer">${{row.title}}</a>
            <div class="source">${{row.themes}}</div>
          </div>
          <div class="score">score ${{row.score}}</div>
        </article>`).join('') || '<div class="empty">No wires for this filter.</div>';
    }}
    renderTabs();
    renderDateFilter();
    renderWires();
    window.requestAnimationFrame(() => {{
      const noteList = document.querySelector('#bearNotes .note-list');
      if (noteList) noteList.scrollTop = Number(savedViewState.noteListScrollTop || 0);
      if (savedViewState.pageScrollY) window.scrollTo(0, Number(savedViewState.pageScrollY));
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local market dashboard.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wires", type=Path, default=None)
    parser.add_argument("--all-wires", action="store_true", help="Use all available wires instead of only today.")
    parser.add_argument("--period", default="1y", help="History period for index technicals.")
    parser.add_argument("--leaders", type=int, default=20, help="Number of stock leaders to show.")
    parser.add_argument("--india-breadth-limit", type=int, default=120, help="Number of Indian stocks to use for tracked breadth.")
    parser.add_argument("--highs-limit", type=int, default=25, help="Number of fresh/near 52-week highs to show.")
    parser.add_argument("--funds-limit", type=int, default=20, help="Number of mutual fund NAV rows to show.")
    parser.add_argument("--rotation-limit", type=int, default=12, help="Number of Nifty 50 rotation rows to show per bucket.")
    parser.add_argument("--nifty500-rotation-limit", type=int, default=15, help="Number of Nifty 500 rotation rows to show per bucket.")
    parser.add_argument(
        "--min-index-coverage",
        type=float,
        default=0.70,
        help="Minimum share of configured global indices that must download before replacing the dashboard.",
    )
    parser.add_argument(
        "--min-valid-index-coverage",
        type=float,
        default=0.60,
        help="Minimum share of configured indices that must pass date and outlier checks.",
    )
    parser.add_argument(
        "--max-index-stale-days",
        type=int,
        default=3,
        help="Exclude index quotes older than this many calendar days from bias and breadth calculations.",
    )
    parser.add_argument(
        "--index-outlier-pct",
        type=float,
        default=12.0,
        help="Exclude index rows with a one-day move above this threshold from bias and breadth calculations.",
    )
    parser.add_argument(
        "--min-breadth-coverage",
        type=float,
        default=0.70,
        help="Minimum share of the Indian breadth universe that must download before replacing the dashboard.",
    )
    parser.add_argument(
        "--min-nifty500-coverage",
        type=float,
        default=0.70,
        help="Minimum same-session Nifty 500 coverage required before replacing the dashboard.",
    )
    parser.add_argument(
        "--require-aligned-market-date",
        action="store_true",
        help="Refuse to publish if Indian breadth datasets do not match the Nifty 50 market date.",
    )
    parser.add_argument("--bear-notes", type=Path, default=None, help="Bear dashboard-notes JSON to include.")
    parser.add_argument(
        "--live-market-snapshot",
        type=Path,
        default=LIVE_MARKET_SNAPSHOT,
        help="Live/delayed breadth, LTP, turnover, and alert snapshot.",
    )
    parser.add_argument("--bear-notes-limit", type=int, default=80, help="Number of Bear watchlist notes to show.")
    parser.add_argument(
        "--include-bear-note-text",
        action="store_true",
        help="Embed full Bear note text in the dashboard. Be careful if publishing online.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wire_path = args.wires or find_latest_wires()
    wires = load_wires(wire_path, today_only=not args.all_wires)
    indices = download_index_history(args.period, args.max_index_stale_days, args.index_outlier_pct)
    index_expected = len(INDEX_TICKERS)
    index_coverage = len(indices) / index_expected if index_expected else 0.0
    valid_indices = usable_index_rows(indices)
    valid_index_coverage = len(valid_indices) / index_expected if index_expected else 0.0
    stale_indices = [row for row in indices if row.get("is_stale")]
    outlier_indices = [row for row in indices if row.get("is_outlier")]
    india_market_date = india_reference_date(indices)
    if index_coverage < args.min_index_coverage:
        raise SystemExit(
            f"Index download coverage was {len(indices)}/{index_expected} "
            f"({index_coverage:.0%}); dashboard was not overwritten."
        )
    if valid_index_coverage < args.min_valid_index_coverage:
        raise SystemExit(
            f"Valid index coverage was {len(valid_indices)}/{index_expected} "
            f"({valid_index_coverage:.0%}) after date/outlier checks; dashboard was not overwritten."
        )
    if args.require_aligned_market_date and not india_market_date:
        raise SystemExit("Nifty 50 market date was unavailable; dashboard was not overwritten.")
    leaders = load_leaders(TECHNICAL_SUMMARY, args.leaders)
    live_market = load_live_market_snapshot(args.live_market_snapshot)
    india_breadth_eod = download_india_breadth(
        TECHNICAL_SUMMARY,
        args.india_breadth_limit,
        expected_date="",
    )
    india_breadth = merge_breadth_layers(
        india_breadth_eod,
        live_market.get("tracked_screen") if live_market else None,
    )
    breadth_summary = india_breadth.get("summary", {})
    breadth_expected = int(breadth_summary.get("source_count") or 0)
    breadth_downloaded = int(breadth_summary.get("downloaded_count") or 0)
    breadth_coverage = breadth_downloaded / breadth_expected if breadth_expected else 0.0
    if breadth_coverage < args.min_breadth_coverage:
        raise SystemExit(
            f"Indian breadth download coverage was {breadth_downloaded}/{breadth_expected} "
            f"({breadth_coverage:.0%}); dashboard was not overwritten."
        )
    nse_highs = load_nse_highs(find_latest_nse_highs(), args.highs_limit)
    mutual_fund_navs = load_mutual_fund_navs(find_latest_mutual_fund_navs(), args.funds_limit)
    nifty50_rotation = load_nifty50_rotation(find_latest_nifty50_rotation(), args.rotation_limit)
    nifty500_rotation_eod = load_nifty500_rotation(
        find_latest_nifty500_rotation(),
        args.nifty500_rotation_limit,
        expected_date="",
    )
    nifty500_rotation = merge_breadth_layers(
        nifty500_rotation_eod,
        live_market.get("nifty500") if live_market else None,
    )
    nifty500_summary = nifty500_rotation.get("summary", {})
    nifty500_expected = int(nifty500_summary.get("source_count") or nifty500_rotation.get("count") or 0)
    nifty500_downloaded = int(
        nifty500_summary.get("downloaded_count") or nifty500_rotation.get("downloaded_count") or 0
    )
    nifty500_coverage = nifty500_downloaded / nifty500_expected if nifty500_expected else 0.0
    if args.require_aligned_market_date and nifty500_coverage < args.min_nifty500_coverage:
        raise SystemExit(
            f"Same-session Nifty 500 coverage was {nifty500_downloaded}/{nifty500_expected} "
            f"({nifty500_coverage:.0%}) for {india_market_date}; dashboard was not overwritten."
        )
    bear_notes_path = args.bear_notes or find_latest_bear_dashboard_notes()
    bear_watchlist_notes = load_bear_dashboard_notes(
        bear_notes_path,
        args.bear_notes_limit,
        args.include_bear_note_text,
    )
    breadth_leadership = load_live_market_snapshot(BREADTH_LEADERSHIP_SNAPSHOT)
    payload = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M IST"),
        "wire_file": wire_path.name if wire_path else "",
        "data_quality": {
            "index_downloaded": len(indices),
            "index_valid": len(valid_indices),
            "index_stale": len(stale_indices),
            "index_outliers": len(outlier_indices),
            "index_expected": index_expected,
            "index_coverage_pct": round(index_coverage * 100, 1),
            "index_valid_coverage_pct": round(valid_index_coverage * 100, 1),
            "breadth_downloaded": breadth_downloaded,
            "breadth_expected": breadth_expected,
            "breadth_coverage_pct": round(breadth_coverage * 100, 1),
            "india_market_date": india_market_date,
            "breadth_as_of_date": breadth_summary.get("as_of_date", ""),
            "breadth_date_mismatches": breadth_summary.get("excluded_date_mismatch", 0),
            "nifty500_downloaded": nifty500_downloaded,
            "nifty500_expected": nifty500_expected,
            "nifty500_coverage_pct": round(nifty500_coverage * 100, 1),
            "nifty500_as_of_date": nifty500_summary.get("as_of_date", ""),
            "nifty500_date_mismatches": nifty500_summary.get("excluded_date_mismatch", 0),
        },
        "posture": market_posture(indices, wires),
        "narrative": narrative_from_wires(wires),
        "indices": indices,
        "regional_breadth": regional_breadth(indices),
        "global_leaders_laggards": global_leaders_laggards(indices),
        "india_breadth": india_breadth,
        "nse_highs": nse_highs,
        "mutual_fund_navs": mutual_fund_navs,
        "nifty50_rotation": nifty50_rotation,
        "nifty500_rotation": nifty500_rotation,
        "breadth_leadership": breadth_leadership,
        "live_market": live_market,
        "bear_watchlist_notes": bear_watchlist_notes,
        "leaders": leaders,
        "wires": wires,
    }
    data_path = args.output_dir / "market_dashboard_data.json"
    html_path = args.output_dir / "index.html"
    data_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(html_page(payload), encoding="utf-8")
    copy_bear_dashboard_assets(bear_notes_path, args.output_dir)
    print(f"Dashboard: {html_path}")
    print(f"Data: {data_path}")
    print(f"Wires: {len(wires)}")
    print(f"Indices: {len(indices)}")
    print(f"India breadth stocks: {india_breadth.get('summary', {}).get('downloaded_count', 0)}")
    print(f"Nifty 500 fresh highs: {nse_highs.get('fresh_count', 0)}")
    print(f"Mutual fund NAVs: {mutual_fund_navs.get('count', 0)}")
    print(f"Nifty 50 rotation rows: {nifty50_rotation.get('count', 0)}")
    print(f"Nifty 500 rotation rows: {nifty500_rotation.get('downloaded_count', 0)}/{nifty500_rotation.get('count', 0)}")
    print(f"Bear watchlist notes: {bear_watchlist_notes.get('shown_count', 0)}")
    print(f"Leaders: {len(leaders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
