"""ボラティリティ急拡大への追随テンプレート(ニュースAPI無しでの現実的な代替戦略)。

指標発表時刻を正確に予測する代わりに、短期ATRが長期ATRに対して急拡大したタイミング
(=何らかの材料でボラティリティが急変した可能性が高い局面)を捉え、直近レンジの
ブレイク方向に追随する。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, donchian_channel
from app.core.strategy_model import Strategy


def _signal_vol_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    atr_fast = int(strategy.params["atr_fast"].value)
    atr_slow = int(strategy.params["atr_slow"].value)
    vol_ratio = float(strategy.params["vol_ratio"].value)
    range_bars = int(strategy.params["range_bars"].value)

    atr_fast_series = atr(df["high"], df["low"], df["close"], atr_fast)
    atr_slow_series = atr(df["high"], df["low"], df["close"], atr_slow)
    upper, lower = donchian_channel(df["high"], df["low"], range_bars)

    vol_hit = (atr_fast_series / atr_slow_series) >= vol_ratio
    breakout_up = vol_hit & (df["close"] > upper)
    breakout_down = vol_hit & (df["close"] < lower)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[breakout_up.fillna(False)] = 1
    signal[breakout_down.fillna(False)] = -1
    return signal


templates.register(
    "vol_breakout",
    defaults={
        "atr_fast": 5,
        "atr_slow": 20,
        "vol_ratio": 2.0,
        "range_bars": 10,
        "sl_pips": 30,
        "tp_pips": 60,
        "lot": 0.1,
    },
    signal_fn=_signal_vol_breakout,
)
