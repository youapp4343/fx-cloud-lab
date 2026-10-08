"""遅延スパイクフェード(spike_fade_delayed)。

spike_fadeのリアルティック検証で「急変の瞬間はスプレッド最悪でエッジが消える」ことが
実証されたため、急変検知からdelay_bars本待ってスプレッド正常化後にエントリーする改良版。
仮説: 行き過ぎの揺り戻しは数時間継続するため、遅延してもエッジの一部が残る。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _signal_spike_fade_delayed(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    atr_period = max(2, int(strategy.params["atr_period"].value))
    lookback = max(1, int(strategy.params["lookback_bars"].value))
    mult = float(strategy.params["spike_mult"].value)
    delay = max(0, int(strategy.params["delay_bars"].value))

    a = atr(df["high"], df["low"], df["close"], atr_period)
    move = df["close"] - df["close"].shift(lookback)
    spike_up = (move > a * mult).shift(delay)
    spike_down = (move < -a * mult).shift(delay)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[spike_up.fillna(False).astype(bool)] = -1
    signal[spike_down.fillna(False).astype(bool)] = 1
    return signal


templates.register(
    "spike_fade_delayed",
    defaults={
        "atr_period": 14.0,
        "lookback_bars": 3.0,
        "spike_mult": 2.5,
        "delay_bars": 4.0,
        "sl_pips": 30.0,
        "tp_pips": 30.0,
        "lot": 0.1,
    },
    signal_fn=_signal_spike_fade_delayed,
)
