"""エンベロープ逆張り平均回帰(envelope_reversion)テンプレート。

bb_reversionと同型のバンドタッチ逆張りだが、バンドがSMA±固定%(envelopes)であり
標準偏差に追随しない。ボラティリティ非追随のため静かな相場で機能しやすく、
荒れ相場ではバンドが広がらず刺さりやすい、という異なるリスク特性を持つ。

先読み回避: envelopesはバーiまでの過去windowのみを使うためシフト不要。
シグナル自体はシフトせず、engine.run_backtestのshift(1)が次バー始値執行を保証する
(bb_reversion等と同じ規律)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import envelopes
from app.core.strategy_model import Strategy


def _signal_envelope_reversion(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """バーiの終値が上バンド以上なら売り、下バンド以下なら買い(平均回帰の逆張り)。"""
    period = max(2, int(strategy.params["period"].value))
    deviation_pct = abs(float(strategy.params["deviation_pct"].value))
    upper, lower = envelopes(df["close"], period, deviation_pct)

    touch_upper = df["close"] >= upper
    touch_lower = df["close"] <= lower

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[touch_upper.fillna(False)] = -1
    signal[touch_lower.fillna(False)] = 1
    return signal


templates.register(
    "envelope_reversion",
    defaults={
        "period": 20.0,
        "deviation_pct": 0.15,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_envelope_reversion,
)
