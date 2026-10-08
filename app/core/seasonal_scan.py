"""曜日×時間帯の季節性アノマリー網羅スキャン(Phase9)。

app.core.discovery の設計思想(有限の仮説を総当たりでt検定+多重検定補正して統計的な
足切りを通過したものだけを見せる)を、ローソク足パターンではなく「曜日×時間帯バケット×
方向」という有限の仮説空間に適用する。各仮説の"hit"は該当する曜日・時間帯バケットに
入った最初のバー(立ち上がり、app.core.seasonal_template._signal_seasonalのenteringと
同じ考え方)とし、その次バー始値エントリー→forward_bars本後の終値決済のpips損益について
片側t検定を行う。discovery.pyは変更禁止のため、_one_sided_p_value/_forward_returns_pips
相当のロジックはこのモジュール内に独立実装する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.core.engine import _pip_size

DIRECTIONS: list[int] = [1, -1]
WEEKDAYS: list[int] = list(range(7))

# scan_seasonalityはsymbol未指定でも使えるよう固定値を使う(discovery.scan_patternsと同じ流儀)。
# t検定のp値/q値は正のスケーリングに対して不変なため、この値の選択はpassed判定に影響しない。
_DEFAULT_PIP_SIZE = 0.0001


def _one_sided_p_value(pips: np.ndarray) -> float:
    """discovery._one_sided_p_valueと同じ考え方(片側t検定、n<2または分散0はp=1.0)。"""
    if len(pips) < 2:
        return 1.0
    std = float(np.std(pips, ddof=1))
    if std == 0.0:
        return 1.0
    result = stats.ttest_1samp(pips, popmean=0.0, alternative="greater")
    return float(result.pvalue)


def _forward_returns_pips(
    df: pd.DataFrame,
    hit_mask: pd.Series,
    direction: int,
    forward_bars: int,
    pip_size: float,
) -> np.ndarray:
    """discovery._forward_returns_pipsと同じ考え方(次バー始値エントリー→forward_bars本後終値決済)。"""
    n = len(df)
    hit_positions = np.flatnonzero(hit_mask.to_numpy())
    exit_positions = hit_positions + 1 + forward_bars
    valid = hit_positions[exit_positions < n]
    if valid.size == 0:
        return np.empty(0, dtype=float)

    open_arr = df["open"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    entry = open_arr[valid + 1]
    exit_ = close_arr[valid + 1 + forward_bars]

    if direction == 1:
        pips = (exit_ - entry) / pip_size
    else:
        pips = (entry - exit_) / pip_size
    return pips


def _window_entering_mask(ts: pd.Series, weekday: int, hour_start: int, hour_end: int) -> pd.Series:
    """weekday×[hour_start, hour_end)の窓に入った最初のバーのみTrue。

    hour_start/hour_endはscan_seasonality側で常に単日内([0,24]の範囲)に収まるように
    構成されるため、seasonal_templateにある日またぎ分岐は不要。

    bool dtypeのSeriesに.shift(1)すると境界のNaNでobject dtypeに昇格し、続く
    .fillna(False)がFutureWarning(silent downcasting)を出すため、numpy配列で
    「直前バーがFalseで当バーがTrue」の立ち上がり判定を組む
    (seasonal_template._entering_maskと同じ考え方)。
    """
    weekday_match = ts.dt.dayofweek == weekday
    hour_match = (ts.dt.hour >= hour_start) & (ts.dt.hour < hour_end)
    hit_arr = (weekday_match & hour_match).to_numpy()
    prev_arr = np.concatenate(([False], hit_arr[:-1]))
    return pd.Series(hit_arr & ~prev_arr, index=ts.index)


def scan_seasonality(
    df: pd.DataFrame,
    hour_bucket_size: int = 4,
    forward_bars: int = 4,
    min_trades: int = 30,
    fdr_q: float = 0.1,
    symbol: str | None = None,
) -> dict:
    """曜日(0-6)×時間帯バケット×方向[1,-1]の全組合せを仮説として総当たり検証する。

    時間帯バケットは0時から hour_bucket_size 時間刻みで24時間を分割する
    (例: hour_bucket_size=4 なら [0,4),[4,8),...,[20,24) の6個)。

    n_trades < min_trades の仮説はt検定を実施せずp_value/q_value=None, passed=Falseのまま
    候補に残す(除外しない: discovery.scan_patternsと同じ誠実性の要件)。FDR補正
    (Benjamini-Hochberg)はp_valueが計算できた仮説のみを対象に行う。

    symbolを渡すとJPYペア判定込みの正しいpip_sizeでmean_pipsを算出する
    (未指定時は非JPYペア想定の0.0001。p_value/q_value/passedはpip_sizeのスケーリングに
    不変なため合否判定には影響しない)。
    """
    pip_size = _pip_size(symbol) if symbol is not None else _DEFAULT_PIP_SIZE
    ts = df["timestamp"]
    hour_starts = list(range(0, 24, hour_bucket_size))

    candidates: list[dict] = []
    for weekday in WEEKDAYS:
        for hour_start in hour_starts:
            hour_end = min(hour_start + hour_bucket_size, 24)
            entering = _window_entering_mask(ts, weekday, hour_start, hour_end)
            for direction in DIRECTIONS:
                pips = _forward_returns_pips(df, entering, direction, forward_bars, pip_size)
                n_trades = int(pips.size)
                candidates.append(
                    {
                        "weekday": weekday,
                        "hour_start": hour_start,
                        "hour_end": hour_end,
                        "direction": direction,
                        "n_trades": n_trades,
                        "mean_pips": float(np.mean(pips)) if n_trades > 0 else 0.0,
                        "p_value": _one_sided_p_value(pips) if n_trades >= min_trades else None,
                        "q_value": None,
                        "passed": False,
                    }
                )

    testable = [c for c in candidates if c["p_value"] is not None]
    if testable:
        raw_pvalues = np.array([c["p_value"] for c in testable], dtype=float)
        adjusted = stats.false_discovery_control(raw_pvalues, method="bh")
        for c, q in zip(testable, adjusted):
            c["q_value"] = float(q)
            c["passed"] = bool(q < fdr_q)

    candidates.sort(key=lambda c: (c["q_value"] is None, c["q_value"] if c["q_value"] is not None else 0.0))

    return {
        "n_hypotheses": len(candidates),
        "n_trades_min_threshold": min_trades,
        "fdr_q": fdr_q,
        "candidates": candidates,
    }
