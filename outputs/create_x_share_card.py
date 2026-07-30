#!/usr/bin/env python3
"""
Create a public share-card page for the latest X draft.

This is the free no-API route:
  1. Copy the draft chart into docs/x/<draft-slug>/
  2. Create an HTML page with X/Twitter card metadata
  3. Create an x.com intent URL that includes the share page URL

After git push, X can render the chart as a large link preview card.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

from PIL import Image


ROOT = Path("/Users/anshumanomjhunjhunwala/Documents/Codex/2026-07-17/making-a-lightweight")
DOCS_DIR = ROOT / "docs"
LATEST_DRAFT_DIR = ROOT / "work" / "x_post_drafts" / "latest"
DEFAULT_PUBLIC_BASE = "https://hvj7p7shkv-hub.github.io/market-dashboard"
MAX_X_CHARS = 280
CARD_WIDTH = 1200
CARD_HEIGHT = 628
CARD_PADDING = 18


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "share"


def truncate(value: str, limit: int) -> str:
    value = re.sub(r"[ \t]+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def find_chart(draft_dir: Path) -> Path | None:
    for name in ("chart.jpg", "chart.jpeg", "chart.png"):
        path = draft_dir / name
        if path.exists():
            return path
    return None


def create_preview_card(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        inner_width = CARD_WIDTH - CARD_PADDING * 2
        inner_height = CARD_HEIGHT - CARD_PADDING * 2
        scale = min(inner_width / image.width, inner_height / image.height)
        resized = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), "white")
        x = (CARD_WIDTH - resized.width) // 2
        y = (CARD_HEIGHT - resized.height) // 2
        card.paste(resized, (x, y))
        destination.parent.mkdir(parents=True, exist_ok=True)
        card.save(destination, "JPEG", quality=88, optimize=True)


def create_page_chart(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        image.convert("RGB").save(destination, "JPEG", quality=90, optimize=True)


def html_page(title: str, description: str, image_url: str, page_url: str, long_text: str) -> str:
    title_e = html.escape(title)
    desc_e = html.escape(description)
    image_e = html.escape(image_url)
    page_e = html.escape(page_url)
    body_text = html.escape(long_text).replace("\n", "<br>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_e}</title>
  <link rel="canonical" href="{page_e}">
  <meta name="description" content="{desc_e}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="{title_e}">
  <meta property="og:description" content="{desc_e}">
  <meta property="og:url" content="{page_e}">
  <meta property="og:image" content="{image_e}">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="628">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{title_e}">
  <meta name="twitter:description" content="{desc_e}">
  <meta name="twitter:image" content="{image_e}">
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #161816; background: #f7f8f4; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 40px 18px 56px; }}
    h1 {{ font-size: clamp(34px, 5vw, 64px); margin: 0 0 12px; letter-spacing: 0; }}
    .date {{ color: #626960; font-size: 18px; margin-bottom: 24px; }}
    img {{ width: 100%; height: auto; border: 1px solid #d9ded7; background: white; display: block; }}
    .note {{ font-size: 20px; line-height: 1.55; margin-top: 28px; max-width: 780px; }}
    a {{ color: #0c6b4f; }}
  </style>
</head>
<body>
  <main>
    <h1>{title_e}</h1>
    <div class="date">Technical note</div>
    <img src="chart.jpg" alt="{title_e} chart">
    <div class="note">{body_text}</div>
    <p><a href="../../">Open market dashboard</a></p>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a public X card page from the latest X draft.")
    parser.add_argument("--draft-dir", type=Path, default=LATEST_DRAFT_DIR)
    parser.add_argument("--public-base", default=DEFAULT_PUBLIC_BASE)
    parser.add_argument("--slug-suffix", help="Optional suffix to create a fresh share URL, e.g. v2.")
    parser.add_argument("--copy-intent", action="store_true", help="Copy x.com intent URL to clipboard.")
    parser.add_argument("--open-intent", action="store_true", help="Open x.com intent URL.")
    args = parser.parse_args()

    metadata_path = args.draft_dir / "metadata.json"
    post_path = args.draft_dir / "post.txt"
    long_path = args.draft_dir / "post_long.md"
    if not metadata_path.exists() or not post_path.exists():
        raise SystemExit("Latest X draft not found. Run outputs/create_x_post_draft_from_bear.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stock = str(metadata.get("stock") or "Stock Note").strip()
    date = str(metadata.get("date") or "").strip()
    post_text = post_path.read_text(encoding="utf-8").strip()
    long_text = long_path.read_text(encoding="utf-8").strip() if long_path.exists() else post_text
    chart = find_chart(args.draft_dir)
    if chart is None:
        raise SystemExit("No chart image found in latest draft.")

    share_slug = f"{date}-{slugify(stock)}" if date else slugify(stock)
    if args.slug_suffix:
        share_slug = f"{share_slug}-{slugify(args.slug_suffix)}"
    share_dir = DOCS_DIR / "x" / share_slug
    share_dir.mkdir(parents=True, exist_ok=True)
    image_dest = share_dir / "chart.jpg"
    create_page_chart(chart, image_dest)
    card_dest = share_dir / "x-card.jpg"
    create_preview_card(chart, card_dest)

    public_base = args.public_base.rstrip("/")
    page_url = f"{public_base}/x/{share_slug}/"
    image_url = f"{public_base}/x/{share_slug}/{card_dest.name}"
    description = truncate(post_text.replace("\n", " "), 190)
    (share_dir / "index.html").write_text(
        html_page(stock, description, image_url, page_url, long_text),
        encoding="utf-8",
    )

    available_for_text = max(0, MAX_X_CHARS - len(page_url) - 2)
    tweet_text = f"{truncate(post_text, available_for_text)}\n\n{page_url}".strip()
    intent_url = f"https://x.com/intent/tweet?text={quote(tweet_text)}"
    (share_dir / "tweet_text.txt").write_text(tweet_text + "\n", encoding="utf-8")
    (share_dir / "tweet_intent_url.txt").write_text(intent_url + "\n", encoding="utf-8")
    (share_dir / "share_url.txt").write_text(page_url + "\n", encoding="utf-8")

    if args.copy_intent:
        subprocess.run(["pbcopy"], input=intent_url, text=True, check=False)
    if args.open_intent:
        subprocess.run(["open", intent_url], check=False)

    print(f"Created share card page: {share_dir / 'index.html'}")
    print(f"Public URL after git push: {page_url}")
    print(f"X compose URL: {intent_url}")
    if args.copy_intent:
        print("Copied X compose URL to clipboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
