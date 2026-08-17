#!/usr/bin/env python3
"""
Fetch mutual fund NAVs from AMFI and calculate short-period NAV returns for a
watchlist of schemes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import pandas as pd
import requests


ROOT = Path(os.environ.get("MARKET_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mutual_fund_navs"
DEFAULT_WATCHLIST = ROOT / "outputs" / "mutual_fund_watchlist.json"
NAV_ALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
NAV_HISTORY_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def load_watchlist(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("Mutual fund watchlist must be a JSON list.")
    rows = []
    for item in payload:
        if isinstance(item, str):
            rows.append({"label": item, "query": item})
        elif isinstance(item, dict) and item.get("query"):
            rows.append({
                "label": str(item.get("label") or item["query"]),
                "query": str(item["query"]),
                "scheme_code": str(item.get("scheme_code") or "").strip(),
            })
    return rows


def fetch_text(url: str, params: dict[str, str] | None = None) -> str:
    response = requests.get(
        url,
        params=params,
        timeout=45,
        headers={
            "User-Agent": "Mozilla/5.0 mutual-fund-nav-fetcher",
            "Accept": "text/plain,text/html,*/*",
        },
    )
    response.raise_for_status()
    return response.text


def parse_nav_text(text: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if ";" in line]
    if not lines:
        return pd.DataFrame()
    data = pd.read_csv(io.StringIO("\n".join(lines)), sep=";", engine="python")
    data.columns = [str(column).strip() for column in data.columns]
    rename = {
        "Scheme Code": "scheme_code",
        "Scheme Name": "scheme_name",
        "Net Asset Value": "nav",
        "Date": "date",
    }
    data = data.rename(columns={key: value for key, value in rename.items() if key in data.columns})
    if "scheme_code" in data:
        data["scheme_code"] = data["scheme_code"].astype(str).str.strip()
    if "scheme_name" in data:
        data["scheme_name"] = data["scheme_name"].astype(str).str.strip()
    if "nav" in data:
        data["nav"] = pd.to_numeric(data["nav"], errors="coerce")
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True)
    return data.dropna(subset=["scheme_code", "scheme_name", "nav"])


def match_scheme(nav_all: pd.DataFrame, query: str, scheme_code: str = "") -> pd.Series | None:
    if nav_all.empty:
        return None
    if scheme_code:
        pinned = nav_all[nav_all["scheme_code"].astype(str) == str(scheme_code)]
        return pinned.iloc[0] if len(pinned) == 1 else None
    query_norm = normalize(query)
    scheme_norm = nav_all["scheme_name"].map(normalize)
    exact = nav_all[scheme_norm == query_norm]
    if not exact.empty:
        return exact.iloc[0]
    tokens = [token for token in query_norm.split() if token not in {"fund", "plan", "option"}]
    if not tokens:
        return None
    mask = pd.Series(True, index=nav_all.index)
    for token in tokens:
        mask &= scheme_norm.str.contains(rf"\b{re.escape(token)}\b", regex=True, na=False)
    matched = nav_all[mask]
    if not matched.empty:
        growth = matched[matched["scheme_name"].str.contains("growth", case=False, na=False)]
        direct = growth[growth["scheme_name"].str.contains("direct", case=False, na=False)]
        if not direct.empty:
            return direct.iloc[0]
        if not growth.empty:
            return growth.iloc[0]
        return matched.iloc[0]
    return None


def amfi_date(value: dt.date) -> str:
    return value.strftime("%d-%b-%Y")


def fetch_history(start: dt.date, end: dt.date) -> pd.DataFrame:
    text = fetch_text(NAV_HISTORY_URL, {"frmdt": amfi_date(start), "todt": amfi_date(end)})
    return parse_nav_text(text)


def return_for(history: pd.DataFrame, scheme_code: str, days: int) -> float | None:
    data = history[history["scheme_code"].astype(str) == str(scheme_code)].dropna(subset=["date", "nav"]).sort_values("date")
    if len(data) <= days:
        return None
    latest = float(data.iloc[-1]["nav"])
    previous = float(data.iloc[-days - 1]["nav"])
    if previous <= 0:
        return None
    return (latest / previous - 1) * 100


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch AMFI mutual fund NAVs for dashboard watchlist.")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-days", type=int, default=90)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    watchlist = load_watchlist(args.watchlist)
    try:
        nav_all = parse_nav_text(fetch_text(NAV_ALL_URL))
    except requests.RequestException as exc:
        raise SystemExit(f"Could not reach AMFI current NAV feed: {exc}") from exc
    start = dt.date.today() - dt.timedelta(days=args.history_days)
    try:
        history = fetch_history(start, dt.date.today())
    except requests.RequestException as exc:
        raise SystemExit(f"Could not reach AMFI historical NAV feed: {exc}") from exc

    rows = []
    for item in watchlist:
        matched = match_scheme(nav_all, item["query"], item.get("scheme_code", ""))
        if matched is None:
            rows.append({"label": item["label"], "query": item["query"], "matched": False})
            continue
        scheme_code = str(matched["scheme_code"])
        rows.append(
            {
                "label": item["label"],
                "query": item["query"],
                "matched": True,
                "scheme_code": scheme_code,
                "scheme_name": matched["scheme_name"],
                "nav": round(float(matched["nav"]), 4),
                "nav_date": matched["date"].date().isoformat() if pd.notna(matched.get("date")) else "",
                "return_1d_pct": return_for(history, scheme_code, 1),
                "return_5d_pct": return_for(history, scheme_code, 5),
                "return_20d_pct": return_for(history, scheme_code, 20),
                "return_60d_pct": return_for(history, scheme_code, 60),
            }
        )

    data = pd.DataFrame(rows)
    for column in ["return_1d_pct", "return_5d_pct", "return_20d_pct", "return_60d_pct"]:
        if column in data:
            data[column] = data[column].map(lambda value: round(float(value), 2) if pd.notna(value) else None)
    stamp = dt.date.today().isoformat()
    output = args.output_dir / f"{stamp}-mutual-fund-navs.csv"
    data.to_csv(output, index=False)
    print(f"Matched funds: {int(data.get('matched', pd.Series(dtype=bool)).sum()) if not data.empty else 0}")
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
