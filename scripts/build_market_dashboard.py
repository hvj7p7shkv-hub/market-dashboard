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


def narrative_from_wires(wires: list[dict[str, object]]) -> list[dict[str, str]]:
    counts = market_theme_counts(wires)
    top_titles = " ".join(row["title"].lower() for row in wires[:30])
    asia_ai = any(term in top_titles for term in ["kospi", "nikkei", "chip", "semiconductor", "ai"])
    crude = any(term in top_titles for term in ["crude", "oil", "brent"])
    flows = any(term in top_titles for term in ["fii", "fpi", "flows", "rupee"])
    items = []
    if asia_ai:
        items.append(
            {
                "label": "Dominant Wire",
                "title": "Asian AI and chip risk-off",
                "text": "Today's strongest wire cluster is around Korea/Japan semiconductor selling and AI-capex anxiety spilling into regional equities.",
            }
        )
    if crude:
        items.append(
            {
                "label": "India Pressure",
                "title": "Crude remains the local macro swing factor",
                "text": "Recent Indian-market weakness still has a crude component; relief rallies are also being linked to lower oil.",
            }
        )
    if flows:
        items.append(
            {
                "label": "Flows",
                "title": "Foreign-flow and risk appetite lens",
                "text": "The wires continue to mention FII/FPI pressure and global risk appetite, so breadth matters more than index-only moves.",
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


def download_index_history(period: str) -> list[dict[str, object]]:
    if yf is None:
        return []
    rows: list[dict[str, object]] = []
    for region, name, ticker in INDEX_TICKERS:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                data = yf.download(ticker, period=period, auto_adjust=False, progress=False, timeout=25)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close = data["Adj Close"] if "Adj Close" in data else data["Close"]
            close = close.dropna().astype(float)
            if close.empty:
                continue
            last = float(close.iloc[-1])
            row: dict[str, object] = {
                "region": region,
                "name": name,
                "ticker": ticker,
                "last_date": close.index[-1].date().isoformat(),
                "last": round(last, 2),
                "return_1d_pct": pct_return(close, 1),
                "return_5d_pct": pct_return(close, 5),
                "return_20d_pct": pct_return(close, 20),
                "rsi_14": round(float(rsi(close).iloc[-1]), 2),
            }
            for window in (20, 50, 200):
                ma = close.rolling(window).mean().iloc[-1]
                row[f"sma_{window}"] = None if math.isnan(ma) else round(float(ma), 2)
                row[f"distance_sma_{window}_pct"] = None if math.isnan(ma) else round((last / float(ma) - 1) * 100, 2)
                row[f"above_sma_{window}"] = None if math.isnan(ma) else bool(last > float(ma))
            rows.append(row)
        except Exception:
            continue
    return rows


def regional_breadth(indices: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_region: dict[str, list[dict[str, object]]] = {}
    for row in indices:
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
        for row in indices
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


def download_india_breadth(path: Path, limit: int, period: str = "3mo") -> dict[str, object]:
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
            close = data["Adj Close"] if "Adj Close" in data else data["Close"]
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


def find_latest_nifty500_rotation() -> Path | None:
    candidates = sorted(NIFTY500_ROTATION_DIR.glob("*nifty500-rotation.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_nifty500_rotation(path: Path | None, limit: int) -> dict[str, object]:
    empty = {
        "leaders": [],
        "weakening": [],
        "laggards": [],
        "mean_reversion": [],
        "source_file": "",
        "count": 0,
        "downloaded_count": 0,
        "coverage_pct": None,
        "benchmark_ticker": "",
    }
    if path is None or not path.exists():
        return empty
    data = pd.read_csv(path)
    if data.empty:
        empty["source_file"] = str(path)
        return empty
    data = data.where(pd.notna(data), None)
    downloaded = (
        data[data["downloaded"].astype(str).str.lower().isin(["true", "1"])]
        if "downloaded" in data
        else data
    )
    downloaded_count = int(len(downloaded))
    total_count = int(len(data))
    leaders = downloaded.copy()
    if "rotation_score" in leaders:
        leaders = leaders.sort_values(["rotation_score", "rs_return_20d_pct"], ascending=[False, False], na_position="last")
    laggards = downloaded.copy()
    if "rotation_score" in laggards:
        laggards = laggards.sort_values(["rotation_score", "rs_return_20d_pct"], ascending=[True, True], na_position="last")
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
        "weakening": json.loads(weakening.head(limit).to_json(orient="records")),
        "laggards": json.loads(laggards.head(limit).to_json(orient="records")),
        "mean_reversion": json.loads(mean_reversion.head(limit).to_json(orient="records")),
        "source_file": str(path),
        "count": total_count,
        "downloaded_count": downloaded_count,
        "coverage_pct": round(downloaded_count / total_count * 100, 1) if total_count else None,
        "benchmark_ticker": benchmark_ticker,
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
    selling_wires = sum(1 for row in wires[:50] if "market selling" in str(row.get("themes", "")).lower())
    ai_wires = sum(1 for row in wires[:50] if "ai / tech risk" in str(row.get("themes", "")).lower())
    weak_indices = sum(1 for row in indices if (row.get("distance_sma_20_pct") or 0) < 0)
    above_50 = sum(1 for row in indices if row.get("above_sma_50") is True)
    if ai_wires >= 5 and weak_indices >= 3:
        stance = "Risk-off, AI-led"
    elif selling_wires >= 8:
        stance = "Risk-off, broad"
    elif above_50 >= max(2, len(indices) // 2):
        stance = "Constructive but selective"
    else:
        stance = "Mixed"
    return {
        "stance": stance,
        "wire_count": str(len(wires)),
        "ai_wire_count": str(ai_wires),
        "selling_wire_count": str(selling_wires),
        "weak_index_count": str(weak_indices),
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
    .source {{ color: var(--muted); font-size: 12px; }}
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
      .grid, .two-col, .three-col, .metric-grid {{ grid-template-columns: 1fr; }}
      .topbar {{ grid-template-columns: 1fr; }}
      .header-actions {{ justify-content: stretch; }}
      .stance, .refresh-panel {{ min-width: 0; flex: 1 1 220px; }}
      th {{ position: static; }}
      .wire {{ grid-template-columns: 1fr; }}
      .note-browser {{ grid-template-columns: 1fr; }}
      .note-list {{ border-right: 0; border-bottom: 1px solid #edf0ec; max-height: 320px; }}
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
    <section class="panel">
      <div class="section-title">
        <h2>Global Equity Breadth</h2>
        <span>Region-wise index map</span>
      </div>
      <div id="regionalBreadth"></div>
    </section>
    <section class="three-col">
      <div class="panel">
        <div class="section-title">
          <h2>Global Leaders</h2>
          <span>1-day move</span>
        </div>
        <div id="globalLeaders"></div>
      </div>
      <div class="panel">
        <div class="section-title">
          <h2>Global Laggards</h2>
          <span>1-day move</span>
        </div>
        <div id="globalLaggards"></div>
      </div>
      <div class="panel">
        <div class="section-title">
          <h2>India Breadth + Volume</h2>
          <span>52W technical screen universe</span>
        </div>
        <div id="indiaBreadth"></div>
      </div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2>Mutual Fund NAV Monitor</h2>
        <span>AMFI NAV movement</span>
      </div>
      <div id="mutualFundNavs"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2>Nifty 50 Rotation</h2>
        <span>Relative strength, volume, RSI</span>
      </div>
      <div id="nifty50Rotation"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2>Nifty 500 Rotation</h2>
        <span>Broad-market leaders and laggards</span>
      </div>
      <div id="nifty500Rotation"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2>Nifty 500 52-Week Highs</h2>
        <span>Technical leaders, highs, and laggards</span>
      </div>
      <div id="nseHighs"></div>
    </section>
    <section class="two-col">
      <div class="panel">
        <div class="section-title">
          <h2>Market Technicals</h2>
          <span>Index placement</span>
        </div>
        <div id="indices"></div>
      </div>
      <div class="panel">
        <div class="section-title">
          <h2>Leadership Watchlist</h2>
          <span>52-week-high screen</span>
        </div>
        <div id="leaders"></div>
      </div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2>Bear Watchlist Notes</h2>
        <span id="bearNoteCount"></span>
      </div>
      <div id="bearNotes"></div>
    </section>
    <section class="panel">
      <div class="section-title">
        <h2>News Wires</h2>
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
            blocks.push(`<img class="note-image" src="${{esc(image.asset_path)}}" alt="${{esc(note.title || 'Bear note image')}}" data-fullsrc="${{esc(image.asset_path)}}">`);
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
        images.forEach(image => blocks.push(`<img class="note-image" src="${{esc(image.asset_path)}}" alt="${{esc(note.title || 'Bear note image')}}" data-fullsrc="${{esc(image.asset_path)}}">`));
      }}
      return `<div class="note-content">${{blocks.join('')}}</div>`;
    }}
    const quality = payload.data_quality || {{}};
    const qualityParts = [];
    if (quality.index_expected) qualityParts.push(`indices ${{quality.index_downloaded}}/${{quality.index_expected}}`);
    if (quality.breadth_expected) qualityParts.push(`India breadth ${{quality.breadth_downloaded}}/${{quality.breadth_expected}}`);
    const coverageText = qualityParts.length ? ` | Coverage: ${{qualityParts.join(', ')}}` : '';
    document.getElementById('generated').textContent = `Generated ${{payload.generated_at}} from ${{payload.wire_file || 'wire data'}}${{coverageText}}`;
    document.getElementById('stance').textContent = payload.posture.stance;
    document.getElementById('stanceDetail').textContent = `${{payload.posture.wire_count}} wires, ${{payload.posture.ai_wire_count}} AI/tech, ${{payload.posture.selling_wire_count}} selling`;

    const narrative = document.getElementById('narrative');
    narrative.innerHTML = payload.narrative.map(item => `
      <article class="panel note">
        <div class="eyebrow">${{item.label}}</div>
        <h2>${{item.title}}</h2>
        <p>${{item.text}}</p>
      </article>`).join('') || '<div class="panel empty">No current wire cluster found.</div>';

    const regionId = region => `region-${{String(region).toLowerCase().replace(/[^a-z0-9]+/g, '-')}}`;
    function regionDetailRows(region) {{
      const rows = payload.indices
        .filter(item => item.region === region)
        .sort((a, b) => (Number(b.return_1d_pct) || 0) - (Number(a.return_1d_pct) || 0));
      return rows.map(item => `
        <tr>
          <td><strong>${{item.name}}</strong><div class="source">${{item.ticker}} · ${{item.last_date || ''}}</div></td>
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
          <td><span class="region-toggle">+</span><strong>${{row.region}}</strong><div class="source">${{row.count}} indices</div></td>
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
      ['Down turnover share', pct(breadthSummary.down_turnover_share_pct)]
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
    document.getElementById('indiaBreadth').innerHTML = breadthSummary.downloaded_count ? `
      <div class="empty" style="padding:0 0 12px;">This is the current ranked technical-screen universe from <code>technical_ranked_summary.csv</code>, not the full Nifty 500. It downloaded ${{breadthSummary.downloaded_count ?? 0}} of ${{breadthSummary.source_count ?? 0}} stocks and measures market participation using advances/declines, turnover proxy, moving-average placement, and volume pressure.</div>
      <div class="metric-grid">${{breadthMetricHtml}}</div>
      <table>
        <thead><tr><th>Sector</th><th class="num">A/D</th><th class="num">Avg 1D</th><th class="num">Vol/20D</th></tr></thead>
        <tbody>${{sectorRows}}</tbody>
      </table>` : '<div class="empty">India breadth and volume will populate after rebuilding from Terminal with network access. Current universe is the technical-ranked stock list.</div>';

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
    function rotationRows(rows) {{
      return (rows || []).map(row => `
        <tr>
          <td><strong>${{row.symbol || ''}}</strong><div class="source">${{row.name || ''}}</div></td>
          <td><span class="pill ${{String(row.rotation_signal || '').includes('Weakening') ? 'risk' : String(row.rotation_signal || '').includes('Mean') ? 'watch' : 'leader'}}">${{row.rotation_signal || ''}}</span></td>
          <td class="num">${{row.rotation_score ?? ''}}</td>
          <td class="num ${{cls(row.rs_return_20d_pct)}}">${{pct(row.rs_return_20d_pct)}}</td>
          <td class="num ${{cls(row.rs_distance_sma_50_pct)}}">${{pct(row.rs_distance_sma_50_pct)}}</td>
          <td class="num">${{row.volume_ratio_20d ?? ''}}</td>
          <td class="num">${{row.rsi_14 ?? ''}}</td>
          <td class="num ${{cls(row.distance_sma_50_pct)}}">${{pct(row.distance_sma_50_pct)}}</td>
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
    const rotation500LeaderRows = rotationRows(rotation500.leaders);
    const rotation500WeakRows = rotationRows(rotation500.weakening);
    const rotation500LaggardRows = rotationRows(rotation500.laggards);
    const rotation500MeanRows = rotationRows(rotation500.mean_reversion);
    document.getElementById('nifty500Rotation').innerHTML = rotation500.source_file ? `
      <div class="metric-grid">
        <div class="metric"><span class="eyebrow">Universe</span><b>${{rotation500.count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Downloaded</span><b>${{rotation500.downloaded_count ?? 0}}</b></div>
        <div class="metric"><span class="eyebrow">Coverage</span><b>${{pct(rotation500.coverage_pct)}}</b></div>
        <div class="metric"><span class="eyebrow">Benchmark</span><b>${{rotation500.benchmark_ticker || 'Nifty'}}</b></div>
      </div>
      <div class="empty" style="margin: 12px 16px;">
        This layer scans the Nifty 500 for stocks gaining or losing relative strength versus the benchmark, then combines it with volume, RSI, and moving-average placement.
      </div>
      <div style="padding: 10px 16px 0;">
        <div class="detail-heading">Broad-market leaders</div>
      </div>
      <table>
        <thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead>
        <tbody>${{rotation500LeaderRows}}</tbody>
      </table>
      <div class="two-col" style="padding: 10px 16px 16px;">
        <div>
          <div class="detail-heading">Weakening names</div>
          ${{rotation500WeakRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotation500WeakRows}}</tbody></table>` : '<div class="empty">No weakening bucket in latest file.</div>'}}
        </div>
        <div>
          <div class="detail-heading">Lowest rotation scores</div>
          ${{rotation500LaggardRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotation500LaggardRows}}</tbody></table>` : '<div class="empty">No laggard bucket in latest file.</div>'}}
        </div>
      </div>
      <div style="padding: 0 16px 16px;">
        <div class="detail-heading">Mean-reversion bounces</div>
        ${{rotation500MeanRows ? `<table><thead><tr><th>Stock</th><th>Signal</th><th class="num">Score</th><th class="num">RS 20D</th><th class="num">RS vs 50DMA</th><th class="num">Vol/20D</th><th class="num">RSI</th><th class="num">vs 50DMA</th></tr></thead><tbody>${{rotation500MeanRows}}</tbody></table>` : '<div class="empty">No mean-reversion bucket in latest file.</div>'}}
      </div>` : '<div class="empty">Run the Nifty 500 rotation analysis to populate broad-market relative-strength, volume, and RSI rotation.</div>';

    const highs = payload.nse_highs || {{}};
    function technicalHighRows(rows) {{
      return (rows || []).map(row => {{
        const hasOverlay = Boolean(row.has_technical_overlay);
        const statusText = row.relative_strength_leader
          ? 'RS leader'
          : hasOverlay
            ? 'Technical watch'
            : '52W high watch';
        const statusClass = row.relative_strength_leader ? 'leader' : 'watch';
        const pnfText = row.pnf_signal || (hasOverlay ? '' : 'Technical overlay pending');
        return `
          <tr>
            <td><strong>${{esc(row.symbol || '')}}</strong><div class="source">${{esc(row.company || '')}}</div><div class="source">${{esc(row.setup_label || '')}}</div></td>
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
        <td><strong>${{row.name}}</strong><div class="source">${{row.region}} · ${{row.ticker}} · ${{row.last_date || ''}}</div></td>
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
        "--min-breadth-coverage",
        type=float,
        default=0.70,
        help="Minimum share of the Indian breadth universe that must download before replacing the dashboard.",
    )
    parser.add_argument("--bear-notes", type=Path, default=None, help="Bear dashboard-notes JSON to include.")
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
    indices = download_index_history(args.period)
    index_expected = len(INDEX_TICKERS)
    index_coverage = len(indices) / index_expected if index_expected else 0.0
    if index_coverage < args.min_index_coverage:
        raise SystemExit(
            f"Index download coverage was {len(indices)}/{index_expected} "
            f"({index_coverage:.0%}); dashboard was not overwritten."
        )
    leaders = load_leaders(TECHNICAL_SUMMARY, args.leaders)
    india_breadth = download_india_breadth(TECHNICAL_SUMMARY, args.india_breadth_limit)
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
    nifty500_rotation = load_nifty500_rotation(find_latest_nifty500_rotation(), args.nifty500_rotation_limit)
    bear_notes_path = args.bear_notes or find_latest_bear_dashboard_notes()
    bear_watchlist_notes = load_bear_dashboard_notes(
        bear_notes_path,
        args.bear_notes_limit,
        args.include_bear_note_text,
    )
    payload = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M IST"),
        "wire_file": wire_path.name if wire_path else "",
        "data_quality": {
            "index_downloaded": len(indices),
            "index_expected": index_expected,
            "index_coverage_pct": round(index_coverage * 100, 1),
            "breadth_downloaded": breadth_downloaded,
            "breadth_expected": breadth_expected,
            "breadth_coverage_pct": round(breadth_coverage * 100, 1),
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
