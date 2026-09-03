#!/usr/bin/env python3
"""Exchange-official daily price history from the NSE equity bhavcopy.

Yahoo Finance is unreliable from GitHub-hosted runners: the multi-ticker
endpoint silently drops a large share of NSE symbols, which starved the Nifty
500 rotation of coverage and stalled the whole dashboard. The NSE bhavcopy is
one authoritative file per session with ~100% coverage, so this module keeps a
small rolling window of it in the repo and serves it to the rotation scripts in
the same shape ``yfinance.download`` would return.

Layout on disk (only the current month's file changes on a normal run, so past
months stay byte-stable and the repo does not bloat)::

    data/nse_prices/2026-08.csv.gz   d,symbol,open,high,low,close,prev_close,volume
    data/nse_prices/2026-09.csv.gz
    data/nse_prices/corp_actions.json   detected split/bonus factors (audit trail)

Index levels are stored alongside stocks under the pseudo-symbols
``__NIFTY50__`` and ``__NIFTY500__`` (from ``ind_close_all``).

Split / bonus adjustment: the NSE corporate-actions API gives the exact
published ratio; the bhavcopy price series (its ex-date PREV_CLOSE is left
*unadjusted* next to the adjusted close) is used to confirm the action really
happened and on which session, and to catch demergers the API does not express
as a ratio. If the API is unreachable the ratios committed last run are reused.

CLI::

    python scripts/nse_price_history.py --data-dir data/nse_prices --lookback-days 430
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

MONS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SERIES_KEEP = {"EQ", "BE", "BZ", "SM", "ST"}
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
COLS = ["d", "symbol", "open", "high", "low", "close", "prev_close", "volume"]

NIFTY50_SYMBOL = "__NIFTY50__"
NIFTY500_SYMBOL = "__NIFTY500__"
INDEX_ALIASES = {
    "^NSEI": NIFTY50_SYMBOL,
    "NIFTY 50": NIFTY50_SYMBOL,
    "^CRSLDX": NIFTY500_SYMBOL,
    "^CNX500": NIFTY500_SYMBOL,
    "NIFTY 500": NIFTY500_SYMBOL,
}



# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*", "Referer": "https://www.nseindia.com/all-reports"})
    for u in ("https://www.nseindia.com", "https://www.nseindia.com/all-reports"):
        try:
            s.get(u, timeout=15)
        except requests.RequestException:
            pass
    return s


def _get(s: requests.Session, url: str) -> bytes | None:
    for attempt in range(3):
        try:
            r = s.get(url, timeout=35)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(2 + attempt * 3)
    return None


def _num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _parse_full(raw: bytes, d: dt.date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().upper() for c in df.columns]
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"].isin(SERIES_KEEP)]
    if df.empty:
        return df
    return pd.DataFrame({
        "d": d.isoformat(), "symbol": df["SYMBOL"].astype(str).str.strip(),
        "open": _num(df["OPEN_PRICE"]), "high": _num(df["HIGH_PRICE"]), "low": _num(df["LOW_PRICE"]),
        "close": _num(df["CLOSE_PRICE"]), "prev_close": _num(df["PREV_CLOSE"]),
        "volume": _num(df["TTL_TRD_QNTY"]),
    })


def _parse_oldzip(raw: bytes, d: dt.date) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(z.open(name))
    df.columns = [c.strip().upper() for c in df.columns]
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"].isin(SERIES_KEEP)]
    if df.empty:
        return df
    return pd.DataFrame({
        "d": d.isoformat(), "symbol": df["SYMBOL"].astype(str).str.strip(),
        "open": _num(df["OPEN"]), "high": _num(df["HIGH"]), "low": _num(df["LOW"]),
        "close": _num(df["CLOSE"]), "prev_close": _num(df["PREVCLOSE"]),
        "volume": _num(df["TOTTRDQTY"]),
    })


def _parse_udiff(raw: bytes, d: dt.date) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(z.open(name))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["FinInstrmTp"].astype(str).isin(["STK", "EQ"])]
    df["SctySrs"] = df["SctySrs"].astype(str).str.strip()
    df = df[df["SctySrs"].isin(SERIES_KEEP)]
    if df.empty:
        return df
    return pd.DataFrame({
        "d": d.isoformat(), "symbol": df["TckrSymb"].astype(str).str.strip(),
        "open": _num(df["OpnPric"]), "high": _num(df["HghPric"]), "low": _num(df["LwPric"]),
        "close": _num(df["ClsPric"]), "prev_close": _num(df["PrvsClsgPric"]),
        "volume": _num(df["TtlTradgVol"]),
    })


def fetch_equity_day(s: requests.Session, d: dt.date) -> pd.DataFrame | None:
    m = MONS[d.month - 1]
    tries = [
        (f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv", _parse_full),
        (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{d:%Y}/{m}/cm{d.day:02d}{m}{d:%Y}bhav.csv.zip", _parse_oldzip),
        (f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip", _parse_udiff),
    ]
    for url, fn in tries:
        raw = _get(s, url)
        if not raw:
            continue
        try:
            df = fn(raw, d)
            if len(df):
                return df
        except (ValueError, KeyError, zipfile.BadZipFile):
            continue
    return None


def fetch_index_day(s: requests.Session, d: dt.date) -> pd.DataFrame | None:
    raw = _get(s, f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{d:%d%m%Y}.csv")
    if not raw:
        return None
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except ValueError:
        return None
    df.columns = [c.strip() for c in df.columns]
    name = df["Index Name"].astype(str).str.strip().str.upper()
    want = {"NIFTY 50": NIFTY50_SYMBOL, "NIFTY 500": NIFTY500_SYMBOL}
    out = []
    for label, sym in want.items():
        row = df[name == label]
        if row.empty:
            continue
        r = row.iloc[0]
        out.append({
            "d": d.isoformat(), "symbol": sym,
            "open": _num(pd.Series([r.get("Open Index Value")])).iloc[0],
            "high": _num(pd.Series([r.get("High Index Value")])).iloc[0],
            "low": _num(pd.Series([r.get("Low Index Value")])).iloc[0],
            "close": _num(pd.Series([r.get("Closing Index Value")])).iloc[0],
            "prev_close": float("nan"), "volume": 0.0,
        })
    return pd.DataFrame(out) if out else None


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def _month_path(data_dir: Path, ym: str) -> Path:
    return data_dir / f"{ym}.csv.gz"


def _read_all(data_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(data_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].csv.gz")):
        try:
            frames.append(pd.read_csv(p))
        except (OSError, ValueError):
            continue
    if not frames:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(frames, ignore_index=True)
    return df[[c for c in COLS if c in df.columns]]


def _write_month(data_dir: Path, ym: str, df: pd.DataFrame) -> None:
    df = df[df["d"].str.startswith(ym)].sort_values(["d", "symbol"])
    df = df.drop_duplicates(["d", "symbol"], keep="last")
    # mtime=0 keeps unchanged months byte-stable so the repo does not churn.
    df.to_csv(
        _month_path(data_dir, ym),
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def update_store(data_dir: Path, lookback_days: int = 430, sleep: float = 0.35) -> dict:
    """Fetch any missing recent sessions into the monthly store. Returns a summary."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_all(data_dir)
    have_days = set(existing["d"].astype(str)) if len(existing) else set()

    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)
    wanted = [
        start + dt.timedelta(days=i)
        for i in range((today - start).days + 1)
        if (start + dt.timedelta(days=i)).weekday() < 5
    ]
    missing = [d for d in wanted if d.isoformat() not in have_days]

    s = make_session()
    fetched_rows: list[pd.DataFrame] = []
    got_days = 0
    missing_days: list[str] = []
    for i, d in enumerate(missing):
        eq = fetch_equity_day(s, d)
        if eq is None:
            missing_days.append(d.isoformat())
            continue
        idx = fetch_index_day(s, d)
        part = pd.concat([eq] + ([idx] if idx is not None else []), ignore_index=True)
        fetched_rows.append(part)
        got_days += 1
        time.sleep(sleep)
        if i and i % 120 == 0:
            s = make_session()

    if fetched_rows:
        parts = ([existing] if len(existing) else []) + fetched_rows
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.dropna(subset=["d", "symbol"])
        new_rows = pd.concat(fetched_rows, ignore_index=True)
        touched_months = sorted({str(d)[:7] for d in new_rows["d"]})
        for ym in touched_months:
            _write_month(data_dir, ym, combined)

    # trim monthly files that fell entirely outside the window
    keep_from = (today - dt.timedelta(days=lookback_days + 40)).isoformat()[:7]
    for p in sorted(data_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].csv.gz")):
        if p.stem.replace(".csv", "") < keep_from:
            p.unlink()

    final = _read_all(data_dir)
    summary = {
        "sessions_in_store": int(final["d"].nunique()) if len(final) else 0,
        "date_range": [str(final["d"].min()), str(final["d"].max())] if len(final) else [None, None],
        "symbols": int(final["symbol"].nunique()) if len(final) else 0,
        "sessions_fetched": got_days,
        "sessions_missing": missing_days,
    }
    summary.update(_refresh_corp_actions(data_dir, final, s))
    return summary


# --------------------------------------------------------------------------- #
# corporate-action ratios (NSE corporate-actions API, primary) + audit cross-check
# --------------------------------------------------------------------------- #
CA_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
    "&from_date={frm}&to_date={to}"
)
# temporary rights / partly-paid tickers - a real symbol never carries these
_SKIP_SUFFIXES = {"RE", "RE1", "RE2", "RE3", "PP", "PP1", "BL", "NP", "W", "WA", "PART"}


def _skip_symbol(sym: str) -> bool:
    return "-" in sym and sym.rsplit("-", 1)[-1].upper() in _SKIP_SUFFIXES


def _f(x) -> float | None:
    try:
        return float(str(x).strip().strip("."))
    except (TypeError, ValueError):
        return None


def classify_ratio(subject: str) -> float | None:
    """Price multiplier for pre-ex history from a corporate-action subject line.

    "Face Value Split From Rs 10 To Re 1" -> 0.1 ;  "Bonus 1:1" -> 0.5
    Returns None for dividends / anything without a clean split or bonus ratio.
    """
    tl = subject.strip().lower()
    is_split = any(k in tl for k in ("split", "sub-division", "sub division", "subdivision"))
    is_bonus = "bonus" in tl
    if is_split:
        m = re.search(r"from\s+r[se]\.?\s*(\d+(?:\.\d+)?).*?\bto\s+r[se]\.?\s*(\d+(?:\.\d+)?)", tl)
        if m:
            frm, to = _f(m.group(1)), _f(m.group(2))
            if frm and to:
                return to / frm
        m = re.search(r"split[^0-9:]*?(\d+)\s*[:/]\s*(\d+)", tl)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return min(a, b) / max(a, b)
    if is_bonus:
        m = re.search(r"bonus[^0-9]*?(\d+)\s*[:/]\s*(\d+)", tl)
        if m:
            new, held = int(m.group(1)), int(m.group(2))
            return held / (held + new)
    return None


def fetch_corp_actions(session: requests.Session, start: dt.date, end: dt.date) -> tuple[dict, dict, int]:
    """-> ({SYMBOL: [[ex_date, factor]]}, {SYMBOL: {ex_date, ...} any action}, chunks_failed)."""
    session.get(
        "https://www.nseindia.com/companies-listing/corporate-filings-actions", timeout=15
    )
    out: dict[str, list] = {}
    any_dates: dict[str, set] = {}
    failed = 0
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=90), end)
        url = CA_URL.format(frm=cur.strftime("%d-%m-%Y"), to=chunk_end.strftime("%d-%m-%Y"))
        rows = None
        for attempt in range(4):
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                rows = r.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(3 + attempt * 3)
        if rows is None:
            failed += 1
            cur = chunk_end + dt.timedelta(days=1)
            continue
        for x in rows if isinstance(rows, list) else []:
            ex = x.get("exDate")
            sym = str(x.get("symbol", "")).strip().upper()
            if not sym or not ex or _skip_symbol(sym):
                continue
            try:
                ex_iso = dt.datetime.strptime(ex, "%d-%b-%Y").date().isoformat()
            except ValueError:
                continue
            any_dates.setdefault(sym, set()).add(ex_iso)
            factor = classify_ratio(str(x.get("subject", "")))
            if factor is None or not (0 < factor < 1):
                continue
            out.setdefault(sym, [])
            if [ex_iso, round(factor, 6)] not in out[sym]:
                out[sym].append([ex_iso, round(factor, 6)])
        cur = chunk_end + dt.timedelta(days=1)
        time.sleep(0.5)
    for sym in out:
        out[sym].sort()
    return out, any_dates, failed


def _big_moves(g: pd.DataFrame) -> list[tuple[str, float]]:
    """Sessions where close/prev_close is a big NON-reverting step. -> [(date, ratio)]."""
    c = g["close"].to_numpy(dtype=float)
    p = g["prev_close"].to_numpy(dtype=float)
    dates = g["d"].tolist()
    hits: list[tuple[str, float]] = []
    for i in range(len(g)):
        if not (p[i] and p[i] > 0) or c[i] <= 0:
            continue
        r = c[i] / p[i]
        if 0.72 <= r <= 1.40:
            continue
        if i + 1 < len(g) and c[i + 1] > 0 and abs((c[i + 1] / c[i]) - (1.0 / r)) < 0.15:
            continue  # snapped back next session -> bad print, not a corp action
        hits.append((dates[i], r))
    return hits


def _resolve_symbol(g: pd.DataFrame, api_events: list, api_any_dates: set) -> tuple[list[list], list[str]]:
    """Reconcile published ratios with the actual price series for one symbol.

    * An API split/bonus is applied only if the real price step near its ex-date
      matches the ratio (kills stray/duplicate API rows like a phantom split on a
      stock that never moved).
    * A big non-reverting price step with NO matching API row is treated as a
      corporate action at the observed ratio - this catches demergers and the
      occasional split the API worded in a way we could not parse.
    """
    moves = _big_moves(g)
    used_moves: set[str] = set()
    resolved: list[list] = []
    notes: list[str] = []

    for ex_iso, factor in sorted({(d, f) for d, f in api_events}):
        ex = dt.date.fromisoformat(ex_iso)
        best = None
        for md, mr in moves:
            if abs((dt.date.fromisoformat(md) - ex).days) <= 3:
                if best is None or abs(mr - factor) < abs(best[1] - factor):
                    best = (md, mr)
        if best and (0.55 * factor <= best[1] <= 1.6 * factor):
            resolved.append([best[0], round(factor, 6)])
            used_moves.add(best[0])
        elif best:
            notes.append(f"api {ex_iso} x{factor:.3f} rejected (price moved x{best[1]:.2f})")
        else:
            notes.append(f"api {ex_iso} x{factor:.3f} had no matching price step")

    for md, mr in moves:
        if md in used_moves or mr >= 1.0:
            continue  # only price-reducing events; up-moves are not corp actions
        near_api = any(
            abs((dt.date.fromisoformat(md) - dt.date.fromisoformat(a)).days) <= 5
            for a in api_any_dates
        )
        if not near_api:
            notes.append(f"unadjusted move {md} x{mr:.3f} (no corp action on record)")
            continue
        resolved.append([md, round(mr, 6)])
        notes.append(f"detected {md} x{mr:.3f} (corp action on record, ratio unparsed)")

    # collapse anything within 9 days (NSE PREV_CLOSE can lag a session)
    resolved.sort()
    collapsed: list[list] = []
    for d, f in resolved:
        if collapsed and (dt.date.fromisoformat(d) - dt.date.fromisoformat(collapsed[-1][0])).days < 9:
            continue
        collapsed.append([d, f])
    return collapsed, notes


def _refresh_corp_actions(data_dir: Path, df: pd.DataFrame, session: requests.Session) -> dict:
    """Rebuild corp_actions.json = {SYMBOL: [[ex_date, price_factor], ...]}.

    Published ratios (NSE corporate-actions API) are the source of truth for the
    exact number; the price series is the source of truth for whether and when it
    actually happened. If the API is unreachable we keep the committed ratios and
    still run the price-only detector so demergers are not missed.
    """
    path = data_dir / "corp_actions.json"
    existing: dict = json.loads(path.read_text()) if path.exists() else {}

    today = dt.date.today()
    try:
        fresh, any_dates, chunks_failed = fetch_corp_actions(
            session, today - dt.timedelta(days=460), today
        )
    except requests.RequestException:
        fresh, any_dates, chunks_failed = {}, {}, -1

    if df.empty:
        return {"corp_action_symbols": len(existing), "corp_actions_fetched": 0,
                "corp_action_chunks_failed": chunks_failed, "corp_action_notes": []}

    stocks = df[~df["symbol"].str.startswith("__")]
    resolved: dict[str, list] = {}
    all_notes: list[str] = []
    for sym, g in stocks.sort_values("d").groupby("symbol"):
        if _skip_symbol(sym):
            continue
        g = g.drop_duplicates("d")
        api_events = list(fresh.get(sym, [])) or list(existing.get(sym, []))
        events, notes = _resolve_symbol(g, api_events, any_dates.get(sym, set()))
        if events:
            resolved[sym] = events
        for n in notes:
            all_notes.append(f"{sym}: {n}")

    if fresh or not existing:
        path.write_text(json.dumps(resolved, indent=1, sort_keys=True))
        out_map = resolved
    else:
        out_map = existing

    return {
        "corp_action_symbols": len(out_map),
        "corp_actions_fetched": sum(len(v) for v in fresh.values()),
        "corp_action_chunks_failed": chunks_failed,
        "corp_action_notes": sorted(all_notes)[:60],
    }


def _yahoo_frame(g: pd.DataFrame, events: list[tuple[str, float]]) -> pd.DataFrame:
    g = g.sort_values("d")
    idx = pd.to_datetime(g["d"])
    o = g["open"].to_numpy(dtype=float)
    h = g["high"].to_numpy(dtype=float)
    lo = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)
    v = g["volume"].to_numpy(dtype=float)
    cumf = pd.Series(1.0, index=range(len(g))).to_numpy()
    dvals = idx.to_numpy()
    for ex_date, factor in events:
        cumf[dvals < pd.Timestamp(ex_date).to_datetime64()] *= factor
    frame = pd.DataFrame(
        {
            "Open": o * cumf,
            "High": h * cumf,
            "Low": lo * cumf,
            "Close": c,
            "Adj Close": c * cumf,
            "Volume": v / cumf,
        },
        index=idx,
    )
    frame.index.name = "Date"
    return frame


def load_frames(data_dir: Path, months: int = 15, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame[Open,High,Low,Close,Adj Close,Volume]} from the store.

    Stock keys are Yahoo-style ("RELIANCE.NS"); index keys "^NSEI", "^CRSLDX",
    "^CNX500" all resolve to the stored Nifty 50 / Nifty 500 levels.
    """
    data_dir = Path(data_dir)
    df = _read_all(data_dir)
    if df.empty:
        return {}
    df = df.dropna(subset=["d", "symbol", "close"])
    cutoff = (dt.date.today() - dt.timedelta(days=int(months * 31))).isoformat()
    df = df[df["d"] >= cutoff]

    ca_path = data_dir / "corp_actions.json"
    ca = json.loads(ca_path.read_text()) if ca_path.exists() else {}

    wanted_syms: set[str] | None = None
    if tickers is not None:
        wanted_syms = set()
        for t in tickers:
            up = str(t).strip().upper()
            wanted_syms.add(INDEX_ALIASES.get(up, up[:-3] if up.endswith(".NS") else up))

    out: dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol"):
        if wanted_syms is not None and sym not in wanted_syms:
            continue
        if _skip_symbol(sym):
            continue
        events = [(d, float(f)) for d, f in ca.get(sym, [])]
        frame = _yahoo_frame(g, events)
        if sym == NIFTY50_SYMBOL:
            out["^NSEI"] = frame
        elif sym == NIFTY500_SYMBOL:
            out["^CRSLDX"] = frame
            out["^CNX500"] = frame
        else:
            out[f"{sym}.NS"] = frame
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/nse_prices"))
    ap.add_argument("--lookback-days", type=int, default=430)
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    summary = update_store(args.data_dir, args.lookback_days, args.sleep)
    print(json.dumps(summary, indent=2))

    # Only hard-fail when the store is unusable (empty and nothing fetched).
    if summary["sessions_in_store"] < 60:
        print("nse_price_history: store still has fewer than 60 sessions", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
