"""オシレーター合議(コンフルエンス)逆張りテンプレート(osc_confluence)。

6オシレーター(RSI/Stochastic%K/CCI/Williams%R/DeMarker/Momentum)それぞれの
「売られすぎ/買われすぎ」判定を集計し、min_agree個以上が同時に一致した状態へ
遷移したバー(エッジ)で逆張りシグナルを出す。単体では弱いオシレーターでも、
複数の同時極値は稀で強いフィルタになりうる、という仮説の検証用。

ゾーンは各オシレーターの標準的既定値で固定(RSI30/70, Stoch20/80, CCI±100,
WPR-80/-20, DeMarker0.3/0.7, Momentum99.7/100.3)。periodは全オシレーター共通。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import cci, demarker, momentum, rsi, stochastic, williams_r
from app.core.strategy_model import Strategy


def _signal_osc_confluence(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    period = max(2, int(strategy.params["period"].value))
    min_agree = max(1, min(6, int(strategy.params["min_agree"].value)))

    r = rsi(df["close"], period)
    k, _ = stochastic(df["high"], df["low"], df["close"], period, 3, 3)
    c = cci(df["high"], df["low"], df["close"], period)
    w = williams_r(df["high"], df["low"], df["close"], period)
    dm = demarker(df["high"], df["low"], period)
    mo = momentum(df["close"], period)

    oversold = (
        (r <= 30).astype(int) + (k <= 20).astype(int) + (c <= -100).astype(int)
        + (w <= -80).astype(int) + (dm <= 0.3).astype(int) + (mo <= 99.7).astype(int)
    )
    overbought = (
        (r >= 70).astype(int) + (k >= 80).astype(int) + (c >= 100).astype(int)
        + (w >= -20).astype(int) + (dm >= 0.7).astype(int) + (mo >= 100.3).astype(int)
    )

    buy_state = oversold >= min_agree
    sell_state = overbought >= min_agree
    # 状態への遷移バーのみ発火(滞在中の連続発火を防ぐ。NaN比較はFalseになるため安全)
    buy_edge = buy_state & ~buy_state.shift(1).fillna(False).astype(bool)
    sell_edge = sell_state & ~sell_state.shift(1).fillna(False).astype(bool)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[buy_edge] = 1
    signal[sell_edge] = -1
    return signal


templates.register(
    "osc_confluence",
    defaults={
        "period": 14.0,
        "min_agree": 4.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_osc_confluence,
)
