import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dashboard_skips_newer_empty_wire_snapshot(tmp_path):
    builder = load_script("build_market_dashboard")
    older = tmp_path / "2026-08-24-ai-market-wires.csv"
    older.write_text("published,title\n2026-08-24T10:00:00Z,Valid wire\n", encoding="utf-8")
    newer = tmp_path / "2026-08-25-ai-market-wires.csv"
    newer.write_text("published,title\n", encoding="utf-8")
    builder.WIRES_DIR = tmp_path

    assert builder.find_latest_wires() == older


def test_fetcher_does_not_overwrite_snapshot_when_all_feeds_fail(tmp_path, monkeypatch):
    fetcher = load_script("fetch_ai_market_news_wires")
    existing = tmp_path / "2026-08-26-ai-market-wires.csv"
    existing.write_text("published,title\n2026-08-26T08:00:00Z,Existing wire\n", encoding="utf-8")
    monkeypatch.setattr(fetcher, "feed_urls", lambda days: ["https://unavailable.invalid/feed"])
    monkeypatch.setattr(fetcher, "fetch_url", lambda url, timeout: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(sys, "argv", ["fetch", "--output-dir", str(tmp_path)])

    assert fetcher.main() == 1
    assert "Existing wire" in existing.read_text(encoding="utf-8")
