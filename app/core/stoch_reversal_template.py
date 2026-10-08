"""ストキャスティクスの%K/%Dクロスをゾーン内で逆張りする(stoch_reversal)テンプレート。

GitHubオープンソースEAコーパスで頻出(15/187本)のStochastic型の典型形。
%Kが%Dを上抜け、かつ%Kが売られすぎゾーン(k < lower)にあるバーで買い、
%Kが%Dを下抜け、かつ%Kが買われすぎゾーン(k > upper)にあるバーで売る。
lower/upperが逆転した設定でもクラッシュしない(シグナル条件が満たされにくくなるだけ)。

先読み回避: シグナル自体はシフトせず、engine.run_backtestのshift(1)が
次バー始値執行を保証する(既存の_signal_rsi_reversal等と同じ規律)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import stochastic
from app.core.strategy_model import Strategy


def _signal_stoch_reversal(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """ゾーン内での%K/%Dクロスを逆張りシグナルにする。"""
    # 0/負値でのクラッシュ防止(過去監査で頻出): k_periodは2以上、
    # d_period/slowingは1以上にクリップする(1はrolling(1).mean()=恒等で有効)
    k_period = max(2, int(strategy.params["k_period"].value))
    d_period = max(1, int(strategy.params["d_period"].value))
    slowing = max(1, int(strategy.params["slowing"].value))
    lower = float(strategy.params["lower"].value)
    upper = float(strategy.params["upper"].value)

    k, d = stochastic(df["high"], df["low"], df["close"], k_period, d_period, slowing)
    prev_k = k.shift(1)
    prev_d = d.shift(1)
    cross_up_in_oversold = (prev_k <= prev_d) & (k > d) & (k < lower)
    cross_down_in_overbought = (prev_k >= prev_d) & (k < d) & (k > upper)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[cross_up_in_oversold.fillna(False)] = 1
    signal[cross_down_in_overbought.fillna(False)] = -1
    return signal


templates.register(
    "stoch_reversal",
    defaults={
        "k_period": 14.0,
        "d_period": 3.0,
        "slowing": 3.0,
        "lower": 20.0,
        "upper": 80.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_stoch_reversal,
)
