import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "nifty500_rotation_analysis.py"
SPEC = importlib.util.spec_from_file_location("nifty500_rotation_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prefers_broad_constituent_session_when_benchmark_is_one_day_ahead():
    data = pd.DataFrame(
        {"downloaded": [True] * 9 + [False], "last_date": ["2026-08-24"] * 9 + [None]}
    )

    assert MODULE.aligned_market_date(data, "2026-08-25", 0.85) == "2026-08-24"


def test_keeps_benchmark_session_without_sufficient_constituent_consensus():
    data = pd.DataFrame(
        {"downloaded": [True] * 8 + [False] * 2, "last_date": ["2026-08-24"] * 8 + [None, None]}
    )

    assert MODULE.aligned_market_date(data, "2026-08-25", 0.85) == "2026-08-25"
