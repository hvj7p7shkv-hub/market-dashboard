#!/usr/bin/env python3
"""
Create an X-ready draft package from an exported Bear watchlist note.

This does not post to X. It creates:
  - post.txt
  - post_long.md
  - chart image copy, if available
  - metadata.json

The latest draft is also mirrored to work/x_post_drafts/latest so a macOS
Shortcut can always read the same files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/Users/anshumanomjhunjhunwala/Documents/Codex/2026-07-17/making-a-lightweight")
BEAR_EXPORT_DIR = ROOT / "outputs" / "bear_dashboard_notes"
DRAFT_DIR = ROOT / "work" / "x_post_drafts"
MAX_X_CHARS = 280
TARGET_POST_CHARS = 205


def comparable(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "draft"


def latest_export() -> Path | None:
    files = sorted(BEAR_EXPORT_DIR.glob("*bear-watchlist-notes.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_notes(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    notes = data.get("notes", []) if isinstance(data, dict) else data
    return notes if isinstance(notes, list) else []


def find_note(notes: list[dict], stock: str) -> dict | None:
    target = comparable(stock)
    exact = [note for note in notes if comparable(str(note.get("title", ""))) == target]
    if exact:
        return exact[0]
    contains = [note for note in notes if target in comparable(str(note.get("title", "")))]
    return contains[0] if len(contains) == 1 else None


def extract_latest_observation(note_text: str, requested_date: str | None) -> tuple[str | None, str | None, str]:
    pattern = re.compile(
        r"^## (?P<date>\d{4}-\d{2}-\d{2})\n(?P<body>.*?)(?=^---\s*$|^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(note_text))
    if requested_date:
        matches = [match for match in matches if match.group("date") == requested_date]
    if not matches:
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", note_text)
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE).strip()
        return requested_date, None, cleaned

    match = matches[0]
    date = match.group("date")
    body = match.group("body").strip()
    action_match = re.search(r"^Action:\s*(.+)$", body, flags=re.MULTILINE)
    observation_match = re.search(r"^Observation:\s*(.*)", body, flags=re.MULTILINE | re.DOTALL)
    action = action_match.group(1).strip() if action_match else None
    observation = observation_match.group(1).strip() if observation_match else body
    observation = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", observation)
    observation = re.sub(r"#stocks/[^\s]+", "", observation).strip()
    observation = re.sub(r"\n{3,}", "\n\n", observation)
    return date, action, observation


def truncate_for_x(text: str, limit: int = MAX_X_CHARS) -> str:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def short_plan_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r"^watching\s+for\s+", "Watch for ", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\s+before\s+considering\s+action\.?$", ".", sentence, flags=re.IGNORECASE)
    return sentence[:1].upper() + sentence[1:] if sentence else sentence


def has_plan_marker(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(re.search(r"\b(watch|wait|retest|entry|support)\b|follow-through", lowered))


def compress_why(sentence: str) -> str:
    lowered = sentence.lower()
    if "volume" in lowered and "breakout" in lowered:
        return "Volume spike on breakout adds confirmation."
    if "volume" in lowered:
        return "Volume action supports the move."
    return truncate_for_x(sentence, 90)


def compress_plan(sentence: str) -> str:
    sentence = short_plan_sentence(sentence)
    sentence = re.sub(r"\s+or\s+a\s+controlled\s+", " or ", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\s+of\s+the\s+breakout\s+zone", " of breakout zone", sentence, flags=re.IGNORECASE)
    return truncate_for_x(sentence, 90)


def structured_note(title: str, date: str, action: str | None, observation: str) -> tuple[str, str]:
    sentences = split_sentences(observation)
    if not sentences:
        sentences = [observation.strip()] if observation.strip() else []

    setup = sentences[0] if sentences else ""
    why = ""
    plan = ""

    for sentence in sentences[1:]:
        lowered = sentence.lower()
        if not why and any(word in lowered for word in ("volume", "supported", "strength", "confidence", "relative")):
            why = sentence
        if not plan and has_plan_marker(sentence):
            plan = short_plan_sentence(sentence)

    if not why and len(sentences) > 1:
        why = sentences[1]
    if not plan and len(sentences) > 2:
        plan = short_plan_sentence(sentences[-1])
    if not plan and action:
        plan = f"Status: {action}."

    short_lines = [title, ""]
    if setup:
        short_lines.append(f"Setup: {setup}")
    if why and why != setup:
        short_lines.extend(["", f"Why it matters: {why}"])
    if plan and plan not in (setup, why):
        short_lines.extend(["", f"Plan: {plan}"])

    short_text = "\n".join(short_lines).strip()
    if len(short_text) > TARGET_POST_CHARS:
        compact_lines = [title, ""]
        if setup:
            compact_lines.append(f"Setup: {truncate_for_x(setup, 95)}")
        if why and why != setup:
            compact_lines.extend(["", f"Why: {compress_why(why)}"])
        if plan and plan != setup:
            compact_lines.extend(["", f"Plan: {compress_plan(plan)}"])
        short_text = "\n".join(compact_lines).strip()

    long_lines = [
        f"# {title}",
        "",
        f"Date: {date}",
    ]
    if action:
        long_lines.append(f"Action: {action}")
    long_lines.extend(["", "## Setup", setup or observation.strip()])
    if why and why != setup:
        long_lines.extend(["", "## Why It Matters", why])
    if plan and plan not in (setup, why):
        long_lines.extend(["", "## Plan", plan])
    long_lines.extend(["", "## Full Observation", observation.strip()])

    return truncate_for_x(short_text), "\n".join(long_lines).strip()


def first_image_path(note: dict, export_path: Path) -> Path | None:
    images = note.get("images") or []
    if not images:
        return None
    asset_path = images[0].get("asset_path") if isinstance(images[0], dict) else None
    if not asset_path:
        return None
    path = export_path.parent / asset_path
    return path if path.exists() else None


def mirror_latest(source_dir: Path) -> None:
    latest_dir = DRAFT_DIR / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(source_dir, latest_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an X draft package from a Bear watchlist note export.")
    parser.add_argument("--stock", required=True, help="Stock note title to draft from.")
    parser.add_argument("--date", help="Observation date to use, yyyy-mm-dd. Defaults to latest dated block.")
    parser.add_argument("--export", type=Path, help="Specific Bear export JSON. Defaults to latest.")
    parser.add_argument("--copy-text", action="store_true", help="Copy post.txt to the macOS clipboard.")
    parser.add_argument("--open-folder", action="store_true", help="Open the draft folder in Finder.")
    args = parser.parse_args()

    export_path = args.export or latest_export()
    if export_path is None or not export_path.exists():
        raise SystemExit("No Bear watchlist export found. Run outputs/export_bear_watchlist_notes_for_dashboard.py first.")

    notes = load_notes(export_path)
    note = find_note(notes, args.stock)
    if note is None:
        raise SystemExit(
            f"Could not find {args.stock!r} in {export_path}. "
            "Refresh the Bear watchlist export, then rerun this command."
        )

    title = str(note.get("title") or args.stock).strip()
    date, action, observation = extract_latest_observation(str(note.get("note") or ""), args.date)
    date = date or args.date or dt.date.today().isoformat()
    post_text, long_text = structured_note(title, date, action, observation)

    draft_path = DRAFT_DIR / f"{date}-{slugify(title)}"
    draft_path.mkdir(parents=True, exist_ok=True)
    (draft_path / "post.txt").write_text(post_text + "\n", encoding="utf-8")
    (draft_path / "post_long.md").write_text(long_text + "\n", encoding="utf-8")

    copied_image = None
    image_path = first_image_path(note, export_path)
    if image_path:
        copied_image = draft_path / f"chart{image_path.suffix.lower()}"
        shutil.copy2(image_path, copied_image)

    metadata = {
        "stock": title,
        "date": date,
        "action": action,
        "source_export": str(export_path),
        "draft_dir": str(draft_path),
        "post_chars": len(post_text),
        "image": str(copied_image) if copied_image else "",
    }
    (draft_path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    shortcut_readme = "\n".join(
        [
            "# X Draft Package",
            "",
            "Use post.txt as the X text.",
            "Attach chart image if present.",
            "",
            f"Text file: {draft_path / 'post.txt'}",
            f"Image file: {copied_image or 'No image found'}",
            "",
        ]
    )
    (draft_path / "README.md").write_text(shortcut_readme, encoding="utf-8")
    mirror_latest(draft_path)

    if args.copy_text:
        subprocess.run(["pbcopy"], input=post_text, text=True, check=False)
    if args.open_folder:
        subprocess.run(["open", str(draft_path)], check=False)

    print(f"Created X draft: {draft_path}")
    print(f"Post chars: {len(post_text)}")
    if copied_image:
        print(f"Image: {copied_image}")
    if args.copy_text:
        print("Copied post text to clipboard.")
    print()
    print(post_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
