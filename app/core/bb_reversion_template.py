"""ボリンジャーバンド逆張り平均回帰(bb_reversion)テンプレート。

app/core/engine.pyの_signal_rsi_reversalと同じ「逆張り」思想だが統計的根拠が異なる:
RSIは値幅モメンタムをオシレーター化した相対指標で判定するのに対し、本テンプレートは
価格の標準偏差(ボリンジャーバンド=SMA±sigma*std)に基づく統計的な乖離度で判定する。

先読み回避: bollinger_bandsはバーiまでの過去windowのみを使うためシフト不要。
シグナル自体はシフトせず、engine.run_backtestのshift(1)が次バー始値執行を保証する
(既存の_signal_ma_cross等と同じ規律)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import bollinger_bands
from app.core.strategy_model import Strategy


def _signal_bb_reversion(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """バーiの終値が上バンド以上なら売り、下バンド以下なら買い(平均回帰の逆張り)。"""
    period = int(strategy.params["period"].value)
    sigma = float(strategy.params["sigma"].value)
    _, upper, lower = bollinger_bands(df["close"], period, sigma)

    touch_upper = df["close"] >= upper
    touch_lower = df["close"] <= lower

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[touch_upper.fillna(False)] = -1
    signal[touch_lower.fillna(False)] = 1
    return signal


templates.register(
    "bb_reversion",
    defaults={
        "period": 20.0,
        "sigma": 2.0,
        "sl_pips": 20.0,
        "tp_pips": 40.0,
        "lot": 0.1,
    },
    signal_fn=_signal_bb_reversion,
)
