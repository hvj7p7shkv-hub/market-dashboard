#!/usr/bin/env python3
"""Build the intraday/delayed market layer used by the market dashboard."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - runtime dependency
    yf = None


ROOT = Path(os.environ.get("MARKET_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
DEFAULT_SCREEN = ROOT / "outputs" / "batch_52w_technical" / "technical_ranked_summary.csv"
DEFAULT_NIFTY500 = ROOT / "outputs" / "nifty500_universe" / "ind_nifty500list.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "live_market" / "latest-live-market-snapshot.json"


def number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def market_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(MARKET_TIMEZONE)
    return timestamp.tz_convert(MARKET_TIMEZONE)


def numeric_series(frame: pd.DataFrame | None, column: str) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    values: pd.Series | pd.DataFrame | None = None
    if isinstance(frame.columns, pd.MultiIndex):
        if column in frame.columns.get_level_values(0):
            values = frame.xs(column, axis=1, level=0)
        elif column in frame.columns.get_level_values(1):
            values = frame.xs(column, axis=1, level=1)
    elif column in frame.columns:
        values = frame[column]
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.DataFrame):
        for subcolumn in values.columns:
            series = pd.to_numeric(values[subcolumn], errors="coerce").dropna()
            if not series.empty:
                return series.astype(float)
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce").dropna().astype(float)


def extract_frame(downloaded: pd.DataFrame, ticker: str, batch_size: int) -> pd.DataFrame | None:
    if downloaded is None or downloaded.empty:
        return None
    if isinstance(downloaded.columns, pd.MultiIndex):
        level_0 = set(downloaded.columns.get_level_values(0))
        level_1 = set(downloaded.columns.get_level_values(1))
        if ticker in level_0:
            return downloaded[ticker].dropna(how="all")
        if ticker in level_1:
            return downloaded.xs(ticker, axis=1, level=1).dropna(how="all")
        return None
    return downloaded.dropna(how="all") if batch_size == 1 else None


def download(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=30,
            )
    except Exception:
        return pd.DataFrame()


def load_universe(path: Path, universe_name: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if universe_name == "nifty500":
        if "YahooTicker" not in data.columns:
            data["YahooTicker"] = data["Symbol"].astype(str).str.strip().str.upper() + ".NS"
        name_column = "Company Name" if "Company Name" in data.columns else "Symbol"
        sector_column = "Industry" if "Industry" in data.columns else "Sector"
    else:
        name_column = "Description" if "Description" in data.columns else "Symbol"
        sector_column = "Sector" if "Sector" in data.columns else "Industry"
    result = pd.DataFrame(
        {
            "symbol": data["Symbol"].astype(str).str.strip().str.upper(),
            "ticker": data["YahooTicker"].astype(str).str.strip(),
            "name": data[name_column].astype(str).str.strip(),
            "sector": data[sector_column].astype(str).str.strip() if sector_column in data else "",
        }
    )
    result = result[result["ticker"].ne("") & result["ticker"].ne("nan")]
    return result.drop_duplicates("ticker").reset_index(drop=True)


def prior_close(daily_close: pd.Series, quote_date: dt.date) -> float:
    candidates = [
        float(value)
        for index, value in daily_close.items()
        if market_timestamp(index).date() < quote_date
    ]
    return candidates[-1] if candidates else math.nan


def quote_row(
    meta: pd.Series,
    intraday_frame: pd.DataFrame | None,
    daily_frame: pd.DataFrame | None,
) -> dict[str, object] | None:
    intraday_close = numeric_series(intraday_frame, "Close")
    intraday_volume = numeric_series(intraday_frame, "Volume")
    daily_close = numeric_series(daily_frame, "Close")
    daily_volume = numeric_series(daily_frame, "Volume")

    source = "Yahoo 5-minute quote"
    if not intraday_close.empty:
        timestamp = market_timestamp(intraday_close.index[-1])
        last = float(intraday_close.iloc[-1])
        quote_date = timestamp.date()
        session_volume = sum(
            float(value)
            for index, value in intraday_volume.items()
            if market_timestamp(index).date() == quote_date and float(value) >= 0
        )
    elif not daily_close.empty:
        timestamp = market_timestamp(daily_close.index[-1])
        last = float(daily_close.iloc[-1])
        quote_date = timestamp.date()
        same_day_volume = [
            float(value)
            for index, value in daily_volume.items()
            if market_timestamp(index).date() == quote_date and float(value) >= 0
        ]
        session_volume = same_day_volume[-1] if same_day_volume else math.nan
        source = "Yahoo daily-close fallback"
    else:
        return None

    previous = prior_close(daily_close, quote_date)
    if not math.isfinite(previous) or previous <= 0:
        return None
    completed_volume = [
        float(value)
        for index, value in daily_volume.items()
        if market_timestamp(index).date() < quote_date and float(value) >= 0
    ]
    average_20d = float(pd.Series(completed_volume[-20:]).mean()) if completed_volume else math.nan
    volume_ratio = session_volume / average_20d if average_20d and math.isfinite(session_volume) else math.nan
    move_pct = (last / previous - 1) * 100
    turnover = last * session_volume if math.isfinite(session_volume) else 0.0
    alert = ""
    if abs(move_pct) >= 5 and math.isfinite(volume_ratio) and volume_ratio >= 2:
        alert = f"Exceptional move ({move_pct:+.2f}%) on {volume_ratio:.1f}x 20-day volume"
    elif abs(move_pct) >= 5:
        alert = f"Exceptional price move ({move_pct:+.2f}%)"
    elif math.isfinite(volume_ratio) and volume_ratio >= 2:
        alert = f"Unusual volume ({volume_ratio:.1f}x 20-day average)"

    return {
        "symbol": str(meta["symbol"]),
        "ticker": str(meta["ticker"]),
        "name": str(meta["name"]),
        "sector": str(meta["sector"]),
        "ltp": round(last, 4),
        "previous_close": round(previous, 4),
        "day_move_pct": round(move_pct, 4),
        "session_volume": round(session_volume, 0) if math.isfinite(session_volume) else None,
        "average_volume_20d": round(average_20d, 0) if math.isfinite(average_20d) else None,
        "volume_ratio_20d": round(volume_ratio, 3) if math.isfinite(volume_ratio) else None,
        "turnover_proxy": round(turnover, 0),
        "quote_date": quote_date.isoformat(),
        "quote_timestamp": timestamp.strftime("%Y-%m-%d %H:%M %Z"),
        "quote_type": source,
        "alert": alert,
    }


def summarize(rows: list[dict[str, object]], source_count: int) -> dict[str, object]:
    advancers = sum(number(row.get("day_move_pct")) > 0 for row in rows)
    decliners = sum(number(row.get("day_move_pct")) < 0 for row in rows)
    unchanged = len(rows) - advancers - decliners
    up_turnover = sum(number(row.get("turnover_proxy")) for row in rows if number(row.get("day_move_pct")) > 0)
    down_turnover = sum(number(row.get("turnover_proxy")) for row in rows if number(row.get("day_move_pct")) < 0)
    total_turnover = up_turnover + down_turnover
    dates = [str(row.get("quote_date") or "") for row in rows if row.get("quote_date")]
    timestamps = [str(row.get("quote_timestamp") or "") for row in rows if row.get("quote_timestamp")]
    high_volume_decliners = sum(
        number(row.get("day_move_pct")) < 0 and number(row.get("volume_ratio_20d")) >= 1.5
        for row in rows
    )
    return {
        "source_count": source_count,
        "downloaded_count": len(rows),
        "coverage_pct": round(len(rows) / source_count * 100, 1) if source_count else 0,
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "ad_ratio": round(advancers / decliners, 2) if decliners else None,
        "up_turnover_share_pct": round(up_turnover / total_turnover * 100, 1) if total_turnover else None,
        "down_turnover_share_pct": round(down_turnover / total_turnover * 100, 1) if total_turnover else None,
        "high_volume_decliners": high_volume_decliners,
        "intraday_quotes": sum(row.get("quote_type") == "Yahoo 5-minute quote" for row in rows),
        "daily_fallbacks": sum(row.get("quote_type") != "Yahoo 5-minute quote" for row in rows),
        "as_of_date": max(dates) if dates else "",
        "as_of_timestamp": max(timestamps) if timestamps else "",
    }


def build_universe_snapshot(universe: pd.DataFrame, batch_size: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for start in range(0, len(universe), batch_size):
        batch = universe.iloc[start : start + batch_size]
        tickers = batch["ticker"].tolist()
        intraday = download(tickers, "5d", "5m")
        daily = download(tickers, "3mo", "1d")
        for _, meta in batch.iterrows():
            ticker = str(meta["ticker"])
            row = quote_row(
                meta,
                extract_frame(intraday, ticker, len(tickers)),
                extract_frame(daily, ticker, len(tickers)),
            )
            if row:
                rows.append(row)

    # Breadth must compare one market session with itself. A ticker that fell
    # back to an older daily bar is excluded instead of contaminating today's
    # advances/declines with a stale move.
    all_rows = rows
    date_counts = pd.Series(
        [str(row.get("quote_date") or "") for row in rows if row.get("quote_date")],
        dtype=str,
    ).value_counts()
    reference_date = str(date_counts.index[0]) if not date_counts.empty else ""
    rows = [row for row in rows if str(row.get("quote_date") or "") == reference_date]
    stale_rows_excluded = len(all_rows) - len(rows)

    leaders = sorted(rows, key=lambda row: number(row.get("day_move_pct")), reverse=True)[:12]
    laggards = sorted(rows, key=lambda row: number(row.get("day_move_pct")))[:12]
    alerts = sorted(
        [row for row in rows if row.get("alert")],
        key=lambda row: (number(row.get("volume_ratio_20d")), abs(number(row.get("day_move_pct")))),
        reverse=True,
    )[:20]
    sector_breadth: list[dict[str, object]] = []
    row_frame = pd.DataFrame(rows)
    if not row_frame.empty and "sector" in row_frame.columns:
        for sector, group in row_frame.groupby("sector", dropna=False):
            moves = pd.to_numeric(group["day_move_pct"], errors="coerce")
            volumes = pd.to_numeric(group["volume_ratio_20d"], errors="coerce")
            sector_breadth.append(
                {
                    "sector": str(sector or "Unknown"),
                    "count": int(len(group)),
                    "advancers": int((moves > 0).sum()),
                    "decliners": int((moves < 0).sum()),
                    "average_1d_pct": round(float(moves.mean()), 3) if moves.notna().any() else None,
                    "average_volume_ratio_20d": round(float(volumes.mean()), 3)
                    if volumes.notna().any()
                    else None,
                }
            )
        sector_breadth.sort(key=lambda row: number(row.get("average_1d_pct")), reverse=True)
    result = {
        "summary": summarize(rows, len(universe)),
        "sector_breadth": sector_breadth,
        "leaders": leaders,
        "laggards": laggards,
        "alerts": alerts,
        "rows": rows,
    }
    result["summary"]["stale_rows_excluded"] = stale_rows_excluded
    return result


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build live/delayed breadth and volume snapshots.")
    parser.add_argument("--tracked-universe", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--nifty500-universe", type=Path, default=DEFAULT_NIFTY500)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--min-screen-coverage", type=float, default=0.70)
    parser.add_argument("--min-nifty500-coverage", type=float, default=0.70)
    args = parser.parse_args()

    screen = load_universe(args.tracked_universe, "screen")
    nifty500 = load_universe(args.nifty500_universe, "nifty500")
    screen_snapshot = build_universe_snapshot(screen, args.batch_size)
    nifty500_snapshot = build_universe_snapshot(nifty500, args.batch_size)

    screen_coverage = number(screen_snapshot["summary"].get("coverage_pct")) / 100
    nifty500_coverage = number(nifty500_snapshot["summary"].get("coverage_pct")) / 100
    if screen_coverage < args.min_screen_coverage or nifty500_coverage < args.min_nifty500_coverage:
        raise RuntimeError(
            "Live snapshot coverage was insufficient: "
            f"screen {screen_coverage:.0%}, Nifty 500 {nifty500_coverage:.0%}. "
            "The previous snapshot was retained."
        )

    payload = {
        "generated_at": dt.datetime.now(MARKET_TIMEZONE).isoformat(timespec="seconds"),
        "data_layer": "Live/delayed raw LTP versus official previous raw close",
        "source_note": "Yahoo Finance 5-minute quotes where available; daily close is a labelled fallback.",
        "tracked_screen": screen_snapshot,
        "nifty500": nifty500_snapshot,
    }
    atomic_json(args.output, payload)
    print(
        "Live market snapshot: "
        f"screen {screen_snapshot['summary']['downloaded_count']}/{screen_snapshot['summary']['source_count']}, "
        f"Nifty 500 {nifty500_snapshot['summary']['downloaded_count']}/{nifty500_snapshot['summary']['source_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
