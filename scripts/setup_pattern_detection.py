#!/usr/bin/env python3
"""
Detect actionable chart setups from daily OHLCV data.

The scanner is intentionally heuristic. It does not decide trades; it surfaces
stocks that deserve a human chart review after the market closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class SetupHit:
    name: str
    score: float
    quality: str
    commentary: str
    tags: list[str]
    pivot_level: float | None = None
    breakout_date: str | None = None
    volume_ratio: float | None = None
    retest_gap_pct: float | None = None
    flag_depth_pct: float | None = None
    impulse_return_pct: float | None = None
    neckline_level: float | None = None


EMPTY_RESULT = {
    "setup_score": 0.0,
    "setup_quality": "No clear setup",
    "primary_setup": "",
    "setup_tags": "",
    "setup_pivot_level": None,
    "setup_breakout_date": None,
    "setup_volume_ratio": None,
    "setup_retest_gap_pct": None,
    "setup_flag_depth_pct": None,
    "setup_impulse_return_pct": None,
    "setup_neckline_level": None,
    "setup_commentary": "",
}


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _date(value: object) -> str | None:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return None


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    close = _series(frame, "Adj Close")
    if close.dropna().empty:
        close = _series(frame, "Close")
    high = _series(frame, "High").reindex(close.index)
    low = _series(frame, "Low").reindex(close.index)
    open_ = _series(frame, "Open").reindex(close.index)
    volume = _series(frame, "Volume").reindex(close.index).fillna(0)
    data = pd.DataFrame(
        {
            "open": open_.fillna(close),
            "high": high.fillna(close),
            "low": low.fillna(close),
            "close": close,
            "volume": volume,
        }
    ).dropna(subset=["close"])
    return data.sort_index()


def _atr(data: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _pivot_indices(series: pd.Series, direction: str, left: int = 4, right: int = 4) -> list[int]:
    values = series.to_numpy()
    pivots: list[int] = []
    for idx in range(left, len(values) - right):
        window = values[idx - left : idx + right + 1]
        if not len(window) or not math.isfinite(values[idx]):
            continue
        if direction == "high" and values[idx] >= max(window):
            pivots.append(idx)
        elif direction == "low" and values[idx] <= min(window):
            pivots.append(idx)
    return pivots


def _cluster_level(
    data: pd.DataFrame,
    pivot_indexes: Iterable[int],
    current: float,
    tolerance_pct: float = 2.5,
    max_above_current_pct: float = 10.0,
) -> tuple[float | None, int, list[int]]:
    pivots = [(idx, float(data["high"].iloc[idx])) for idx in pivot_indexes]
    candidates: list[tuple[float, int, list[int], float]] = []
    for idx, value in pivots:
        if value <= 0:
            continue
        if value > current * (1 + max_above_current_pct / 100):
            continue
        cluster = [(other_idx, other_value) for other_idx, other_value in pivots if abs(other_value / value - 1) * 100 <= tolerance_pct]
        if len(cluster) < 2:
            continue
        level = sum(item[1] for item in cluster) / len(cluster)
        closeness_penalty = abs(current / level - 1) * 100
        recency_bonus = max(item[0] for item in cluster) / max(len(data), 1)
        score = len(cluster) * 10 + recency_bonus - closeness_penalty
        candidates.append((level, len(cluster), [item[0] for item in cluster], score))
    if not candidates:
        return None, 0, []
    level, count, indexes, _ = max(candidates, key=lambda item: item[3])
    return level, count, indexes


def _volume_ratio(data: pd.DataFrame, idx: int, window: int = 20) -> float | None:
    if idx < 1 or idx >= len(data):
        return None
    start = max(0, idx - window)
    baseline = data["volume"].iloc[start:idx].replace(0, math.nan).dropna()
    if baseline.empty:
        return None
    avg = float(baseline.mean())
    if avg <= 0:
        return None
    return float(data["volume"].iloc[idx] / avg)


def _breakout_retest(data: pd.DataFrame) -> SetupHit | None:
    if len(data) < 90:
        return None
    latest = float(data["close"].iloc[-1])
    pivots = _pivot_indices(data["high"].tail(180), "high")
    offset = max(len(data) - 180, 0)
    pivot_indexes = [idx + offset for idx in pivots]
    pivot, touches, touch_indexes = _cluster_level(data, pivot_indexes, latest, max_above_current_pct=8)
    if pivot is None or touches < 2:
        return None

    atr = _atr(data).iloc[-1]
    recent_start = max(max(touch_indexes) + 1 if touch_indexes else 0, len(data) - 70)
    breakout_idx = None
    breakout_volume = None
    for idx in range(recent_start, len(data)):
        close = float(data["close"].iloc[idx])
        prev_close = float(data["close"].iloc[idx - 1]) if idx else close
        vol_ratio = _volume_ratio(data, idx)
        crossed = close >= pivot * 1.012 and prev_close <= pivot * 1.025
        decisive = vol_ratio is not None and vol_ratio >= 1.35
        if crossed and decisive:
            breakout_idx = idx
            breakout_volume = vol_ratio
            break
    if breakout_idx is None:
        return None

    latest_low = float(data["low"].iloc[-1])
    retest_gap = min(abs(latest / pivot - 1), abs(latest_low / pivot - 1)) * 100
    near_pivot = retest_gap <= 4.0 or (math.isfinite(float(atr)) and abs(latest - pivot) <= atr * 1.5)
    holding_pivot = latest >= pivot * 0.97
    if not (near_pivot and holding_pivot):
        return None

    pullback_volume = float(data["volume"].iloc[-5:].mean()) if len(data) >= 5 else float(data["volume"].iloc[-1])
    breakout_abs_volume = float(data["volume"].iloc[breakout_idx])
    quiet_retest = breakout_abs_volume > 0 and pullback_volume < breakout_abs_volume * 0.75
    score = 2.7 + min((breakout_volume or 1) / 2, 1.2)
    if quiet_retest:
        score += 0.8
    if latest > pivot:
        score += 0.4
    return SetupHit(
        name="Breakout retest",
        score=score,
        quality="Actionable watch" if score >= 3.5 else "Watch",
        commentary=(
            "Prior resistance has been cleared on higher volume and price is back near that pivot. "
            "This is the kind of retest zone where follow-through can matter more than chasing a stretched candle."
        ),
        tags=["breakout", "retest", "pivot", "volume"],
        pivot_level=pivot,
        breakout_date=_date(data.index[breakout_idx]),
        volume_ratio=breakout_volume,
        retest_gap_pct=retest_gap,
    )


def _flag_setup(data: pd.DataFrame) -> SetupHit | None:
    if len(data) < 90:
        return None
    recent = data.tail(80).copy()
    latest = float(recent["close"].iloc[-1])
    sma50 = data["close"].rolling(50).mean().iloc[-1]
    if math.isfinite(float(sma50)) and latest < sma50:
        return None

    best: SetupHit | None = None
    for impulse_lookback in (55, 45, 35, 25):
        window = recent.tail(impulse_lookback)
        if len(window) < 20:
            continue
        search = window.iloc[:-5]
        start_label = search["close"].idxmin()
        start_pos = window.index.get_loc(start_label)
        after_start = window.iloc[start_pos:]
        peak_label = after_start["close"].idxmax()
        peak_pos = window.index.get_loc(peak_label)
        if peak_pos >= len(window) - 4:
            continue
        start_close = float(window["close"].iloc[start_pos])
        peak_close = float(window["close"].iloc[peak_pos])
        if start_close <= 0:
            continue
        impulse_return = (peak_close / start_close - 1) * 100
        if impulse_return < 18:
            continue
        consolidation = window.iloc[peak_pos:]
        if not 5 <= len(consolidation) <= 24:
            continue
        low_since_peak = float(consolidation["low"].min())
        depth = (peak_close / low_since_peak - 1) * 100 if low_since_peak > 0 else 999
        if depth > 22:
            continue
        still_tight = latest >= peak_close * 0.82 and latest <= peak_close * 1.06
        if not still_tight:
            continue
        impulse_volume = float(window["volume"].iloc[max(0, start_pos): peak_pos + 1].mean())
        flag_volume = float(consolidation["volume"].mean())
        volume_contracting = impulse_volume > 0 and flag_volume <= impulse_volume * 0.9
        high_tight = impulse_return >= 45 and depth <= 18
        score = 3.4 if high_tight else 2.4
        score += min(impulse_return / 80, 0.8)
        score += 0.6 if volume_contracting else 0.0
        score -= 0.4 if depth > 18 else 0.0
        quality = "Actionable watch" if score >= 3.4 else "Watch"
        name = "High-tight flag" if high_tight else "Bullish flag"
        hit = SetupHit(
            name=name,
            score=score,
            quality=quality,
            commentary=(
                "Price has made a sharp impulse move and is now consolidating near the highs. "
                "A controlled pullback with quieter volume keeps the continuation setup alive."
            ),
            tags=["flag", "continuation", "uptrend"] + (["high-tight"] if high_tight else []),
            pivot_level=peak_close,
            breakout_date=_date(peak_label),
            volume_ratio=flag_volume / impulse_volume if impulse_volume > 0 else None,
            flag_depth_pct=depth,
            impulse_return_pct=impulse_return,
        )
        if best is None or hit.score > best.score:
            best = hit
    return best


def _inverse_head_shoulders(data: pd.DataFrame) -> SetupHit | None:
    if len(data) < 140:
        return None
    recent = data.tail(180)
    lows = _pivot_indices(recent["low"], "low", left=5, right=5)
    if len(lows) < 3:
        return None
    best: SetupHit | None = None
    for left_idx in lows[:-2]:
        for head_idx in [idx for idx in lows if idx > left_idx]:
            for right_idx in [idx for idx in lows if idx > head_idx]:
                left_low = float(recent["low"].iloc[left_idx])
                head_low = float(recent["low"].iloc[head_idx])
                right_low = float(recent["low"].iloc[right_idx])
                if not (head_low < left_low * 0.94 and head_low < right_low * 0.94):
                    continue
                shoulder_gap = abs(left_low / right_low - 1) * 100 if right_low else 999
                if shoulder_gap > 16:
                    continue
                if head_idx - left_idx < 12 or right_idx - head_idx < 12:
                    continue
                left_peak = float(recent["high"].iloc[left_idx:head_idx].max())
                right_peak = float(recent["high"].iloc[head_idx:right_idx].max())
                neckline = (left_peak + right_peak) / 2
                latest = float(recent["close"].iloc[-1])
                distance = (latest / neckline - 1) * 100 if neckline else -999
                if distance < -6:
                    continue
                score = 2.6 + max(min(distance / 3, 0.8), -0.4)
                hit = SetupHit(
                    name="Possible inverse H&S",
                    score=score,
                    quality="Actionable watch" if distance >= -2 else "Watch",
                    commentary=(
                        "A possible inverse head-and-shoulders base is visible. The setup improves only if price clears "
                        "or holds near the neckline with better volume."
                    ),
                    tags=["inverse-head-and-shoulders", "base", "reversal"],
                    pivot_level=neckline,
                    neckline_level=neckline,
                    retest_gap_pct=abs(distance),
                )
                if best is None or hit.score > best.score:
                    best = hit
    return best


def _head_shoulders_risk(data: pd.DataFrame) -> SetupHit | None:
    if len(data) < 140:
        return None
    recent = data.tail(180)
    highs = _pivot_indices(recent["high"], "high", left=5, right=5)
    if len(highs) < 3:
        return None
    best: SetupHit | None = None
    for left_idx in highs[:-2]:
        for head_idx in [idx for idx in highs if idx > left_idx]:
            for right_idx in [idx for idx in highs if idx > head_idx]:
                left_high = float(recent["high"].iloc[left_idx])
                head_high = float(recent["high"].iloc[head_idx])
                right_high = float(recent["high"].iloc[right_idx])
                if not (head_high > left_high * 1.05 and head_high > right_high * 1.05):
                    continue
                shoulder_gap = abs(left_high / right_high - 1) * 100 if right_high else 999
                if shoulder_gap > 16:
                    continue
                if head_idx - left_idx < 12 or right_idx - head_idx < 12:
                    continue
                left_trough = float(recent["low"].iloc[left_idx:head_idx].min())
                right_trough = float(recent["low"].iloc[head_idx:right_idx].min())
                neckline = (left_trough + right_trough) / 2
                latest = float(recent["close"].iloc[-1])
                distance = (latest / neckline - 1) * 100 if neckline else 999
                if distance > 5:
                    continue
                score = -3.0 if distance <= 0 else -2.0
                hit = SetupHit(
                    name="Head-and-shoulders risk",
                    score=score,
                    quality="Risk review",
                    commentary=(
                        "A possible head-and-shoulders top is forming or breaking. This is a caution signal, not an entry setup."
                    ),
                    tags=["head-and-shoulders", "risk", "distribution"],
                    pivot_level=neckline,
                    neckline_level=neckline,
                    retest_gap_pct=abs(distance),
                )
                if best is None or hit.score < best.score:
                    best = hit
    return best


def _to_result(hit: SetupHit | None, extras: list[SetupHit]) -> dict[str, object]:
    if hit is None:
        return dict(EMPTY_RESULT)
    tags = list(dict.fromkeys([*hit.tags, *(tag for extra in extras for tag in extra.tags)]))
    commentary = hit.commentary
    if extras:
        extra_names = ", ".join(extra.name for extra in extras if extra.name != hit.name)
        if extra_names:
            commentary = f"{commentary} Secondary patterns: {extra_names}."
    return {
        "setup_score": _round(hit.score),
        "setup_quality": hit.quality,
        "primary_setup": hit.name,
        "setup_tags": ", ".join(tags),
        "setup_pivot_level": _round(hit.pivot_level),
        "setup_breakout_date": hit.breakout_date,
        "setup_volume_ratio": _round(hit.volume_ratio),
        "setup_retest_gap_pct": _round(hit.retest_gap_pct),
        "setup_flag_depth_pct": _round(hit.flag_depth_pct),
        "setup_impulse_return_pct": _round(hit.impulse_return_pct),
        "setup_neckline_level": _round(hit.neckline_level),
        "setup_commentary": commentary,
    }


def scan_setups(frame: pd.DataFrame) -> dict[str, object]:
    data = _prepare(frame)
    if len(data) < 90:
        return dict(EMPTY_RESULT)
    hits = [
        hit
        for hit in [
            _head_shoulders_risk(data),
            _breakout_retest(data),
            _flag_setup(data),
            _inverse_head_shoulders(data),
        ]
        if hit is not None
    ]
    if not hits:
        return dict(EMPTY_RESULT)
    risk_hits = [hit for hit in hits if hit.quality == "Risk review"]
    if risk_hits:
        primary = min(risk_hits, key=lambda hit: hit.score)
        others = [hit for hit in hits if hit is not primary and hit.score > 0]
        return _to_result(primary, others)
    primary = max(hits, key=lambda hit: hit.score)
    others = [hit for hit in hits if hit is not primary and hit.score >= 2.0]
    return _to_result(primary, others)
