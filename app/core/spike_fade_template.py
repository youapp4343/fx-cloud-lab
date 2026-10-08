"""スパイクフェードテンプレート(spike_fade)。

直近lookback_bars本の値動き(close変化)がATRのspike_mult倍を超える「急騰/急落」を検知し、
行き過ぎの反対方向へ逆張りする(ボラティリティの刈り取り)。

誠実性の注記: 実際の急変時はスプレッド拡大・スリッページ増大・約定拒否が起きるため、
固定スプレッドのバックテストは楽観側に歪む。実運用前に必ずデモで約定品質を確認すること。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _signal_spike_fade(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    atr_period = max(2, int(strategy.params["atr_period"].value))
    lookback = max(1, int(strategy.params["lookback_bars"].value))
    mult = float(strategy.params["spike_mult"].value)

    a = atr(df["high"], df["low"], df["close"], atr_period)
    move = df["close"] - df["close"].shift(lookback)
    spike_up = move > a * mult
    spike_down = move < -a * mult

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[spike_up.fillna(False)] = -1   # 急騰→売り(フェード)
    signal[spike_down.fillna(False)] = 1  # 急落→買い
    return signal


templates.register(
    "spike_fade",
    defaults={
        "atr_period": 14.0,
        "lookback_bars": 3.0,
        "spike_mult": 3.0,
        "sl_pips": 30.0,
        "tp_pips": 30.0,
        "lot": 0.1,
    },
    signal_fn=_signal_spike_fade,
)
