#!/usr/bin/env python3
"""
Fetch recent market news wires around AI-led selling, broader market weakness,
and Indian equity-market context.

The script uses public RSS feeds and writes:
  - a readable Markdown digest
  - a CSV for sorting/filtering
  - a JSON file with the raw structured items
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path


ROOT = Path(os.environ.get("MARKET_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "market_news_wires"

THEME_KEYWORDS = {
    "AI / tech risk": [
        "artificial intelligence",
        "semiconductor",
        "chip",
        "nvidia",
        "tech selloff",
        "technology stocks",
    ],
    "market selling": [
        "selloff",
        "sell-off",
        "falls",
        "fall",
        "slumps",
        "drops",
        "declines",
        "correction",
        "risk-off",
        "volatility",
    ],
    "India market": [
        "nifty",
        "sensex",
        "india",
        "indian shares",
        "midcap",
        "smallcap",
        "mutual fund",
        "fii",
        "fpi",
    ],
    "macro / flows": [
        "fed",
        "yields",
        "dollar",
        "rupee",
        "crude",
        "oil",
        "tariff",
        "earnings",
        "flows",
    ],
}

KEYWORD_PATTERNS = {
    keyword: re.compile(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])")
    for keywords in THEME_KEYWORDS.values()
    for keyword in keywords
}
KEYWORD_PATTERNS["ai"] = re.compile(r"(?<![a-z0-9])ai(?![a-z0-9])")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def google_news_rss(query: str) -> str:
    encoded = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"


def feed_urls(days: int) -> list[str]:
    when = f"when:{days}d"
    queries = [
        f'(AI OR "artificial intelligence" OR Nvidia OR semiconductor OR technology) '
        f'(selloff OR "market fall" OR "stocks fall" OR correction) {when}',
        f'(Nifty OR Sensex OR "Indian shares" OR midcap OR smallcap) '
        f'(falls OR selloff OR correction OR "risk off") {when}',
        f'("mutual funds" OR SIP OR AMFI OR FII OR FPI) '
        f'(India OR equity OR market) {when}',
        f'("broader market" OR midcap OR smallcap) '
        f'(selling OR correction OR fall OR volatility) India {when}',
    ]
    return [google_news_rss(query) for query in queries]


def parse_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(dt.timezone.utc)
        return parsed.isoformat(timespec="seconds")
    except Exception:
        return clean_text(value)


def fetch_url(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 market-news-wire-fetcher",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_rss(xml_bytes: bytes, source_url: str) -> list[dict[str, object]]:
    root = ET.fromstring(xml_bytes)
    channel_title = clean_text(root.findtext("./channel/title")) or urllib.parse.urlparse(source_url).netloc
    rows: list[dict[str, object]] = []
    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title"))
        link = clean_text(item.findtext("link"))
        description = clean_text(item.findtext("description"))
        published = parse_date(item.findtext("pubDate"))
        source = clean_text(item.findtext("source")) or channel_title
        if title or link:
            rows.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "published": published,
                    "source": source,
                    "feed": channel_title,
                    "source_url": source_url,
                }
            )
    return rows


def classify(row: dict[str, object]) -> tuple[list[str], int]:
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    themes: list[str] = []
    score = 0
    for theme, keywords in THEME_KEYWORDS.items():
        hits = [keyword for keyword in keywords if KEYWORD_PATTERNS[keyword].search(text)]
        if theme == "AI / tech risk" and KEYWORD_PATTERNS["ai"].search(text):
            hits.append("ai")
        if hits:
            themes.append(theme)
            score += len(hits)
    if "AI / tech risk" in themes and "market selling" in themes:
        score += 5
    if "India market" in themes and "market selling" in themes:
        score += 4
    if "mutual fund" in text or "sip" in text or "amfi" in text:
        score += 3
    return themes, score


def dedupe(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        key = str(row.get("link") or row.get("title", "")).strip().lower()
        key = re.sub(r"[?#].*$", "", key)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = ["published", "score", "themes", "source", "title", "link", "description", "feed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(rows: list[dict[str, object]], path: Path, limit: int) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# AI and Market-Selling News Wire Digest",
        "",
        f"Generated: {now}",
        "",
        "Focus: AI/technology risk, broader market selling, Indian equities, flows, mutual funds, and macro triggers.",
        "",
        "## Highest-Relevance Wires",
        "",
    ]
    for index, row in enumerate(rows[:limit], start=1):
        themes = ", ".join(row.get("themes", []))
        lines.extend(
            [
                f"### {index}. {row.get('title', '')}",
                "",
                f"- Source: {row.get('source', '')}",
                f"- Published: {row.get('published', '')}",
                f"- Themes: {themes}",
                f"- Score: {row.get('score', '')}",
                f"- Link: {row.get('link', '')}",
                "",
            ]
        )
        description = str(row.get("description", "")).strip()
        if description:
            lines.extend([description, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch AI/market-selling news wires from public RSS feeds.")
    parser.add_argument("--days", type=int, default=7, help="Look back this many days in Google News searches.")
    parser.add_argument("--limit", type=int, default=40, help="Number of top articles to show in the Markdown digest.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=20, help="Network timeout per feed.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for url in feed_urls(args.days):
        try:
            rows.extend(parse_rss(fetch_url(url, args.timeout), url))
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    rows = dedupe(rows)
    for row in rows:
        themes, score = classify(row)
        row["themes"] = themes
        row["score"] = score
    rows = [row for row in rows if row["score"] > 0]
    rows.sort(key=lambda item: (int(item.get("score", 0)), str(item.get("published", ""))), reverse=True)

    stamp = dt.date.today().isoformat()
    csv_path = args.output_dir / f"{stamp}-ai-market-wires.csv"
    json_path = args.output_dir / f"{stamp}-ai-market-wires.json"
    md_path = args.output_dir / f"{stamp}-ai-market-wires.md"
    latest_md = args.output_dir / "latest-ai-market-wires.md"

    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(rows, md_path, args.limit)
    write_markdown(rows, latest_md, args.limit)

    print(f"Fetched relevant wires: {len(rows)}")
    print(f"Markdown digest: {md_path}")
    print(f"Latest digest: {latest_md}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if errors:
        print()
        print("Feeds with errors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
