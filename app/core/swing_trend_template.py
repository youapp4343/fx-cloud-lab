"""スワップ狙いスイングトレンドテンプレート(swing_trend)。

MAクロス(fast/slow)の順張りに方向制限(direction_limit)を付けたもの。
プラススワップ方向のみエントリーし、長期保有(SL/TP広め+トレーリング任意)で
トレンド+スワップ+リベートの三重取りを狙う運用の検証用。
direction_limit: 0=両方向, 1=買いのみ, -1=売りのみ。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import sma
from app.core.strategy_model import Strategy


def _signal_swing_trend(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    fast_period = max(2, int(strategy.params["fast_period"].value))
    slow_period = max(3, int(strategy.params["slow_period"].value))
    dl = int(strategy.params["direction_limit"].value)

    fast = sma(df["close"], fast_period)
    slow = sma(df["close"], slow_period)
    pf, ps = fast.shift(1), slow.shift(1)
    golden = (pf <= ps) & (fast > slow)
    dead = (pf >= ps) & (fast < slow)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dl >= 0:
        signal[golden.fillna(False)] = 1
    if dl <= 0:
        signal[dead.fillna(False)] = -1
    return signal


templates.register(
    "swing_trend",
    defaults={
        "fast_period": 10.0,
        "slow_period": 50.0,
        "direction_limit": 1.0,
        "sl_pips": 150.0,
        "tp_pips": 400.0,
        "lot": 0.1,
    },
    signal_fn=_signal_swing_trend,
)
