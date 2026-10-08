"""MACDライン/シグナルラインのクロスで順張りする(macd_cross)テンプレート。

GitHubオープンソースEAコーパスで最頻出(29/187本)のMACD型の典型形。
zero_filter != 0 のときはゼロライン位置をトレンド方向フィルタとして使い、
ゼロラインより上のゴールデンクロスのみ買い・下のデッドクロスのみ売りに制限する。

先読み回避: シグナル自体はシフトせず、engine.run_backtestのshift(1)が
次バー始値執行を保証する(既存の_signal_ma_cross等と同じ規律)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import macd
from app.core.strategy_model import Strategy


def _signal_macd_cross(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """MACDラインがシグナルラインを上抜けたバーで買い、下抜けたバーで売り。"""
    # 0/負値でのクラッシュ防止(過去監査で頻出): period系は2以上にクリップする
    fast_period = max(2, int(strategy.params["fast_period"].value))
    slow_period = max(2, int(strategy.params["slow_period"].value))
    signal_period = max(2, int(strategy.params["signal_period"].value))
    zero_filter = float(strategy.params["zero_filter"].value)

    macd_line, signal_line, _ = macd(df["close"], fast_period, slow_period, signal_period)
    prev_macd = macd_line.shift(1)
    prev_signal = signal_line.shift(1)
    golden = (prev_macd <= prev_signal) & (macd_line > signal_line)
    dead = (prev_macd >= prev_signal) & (macd_line < signal_line)
    if zero_filter != 0.0:
        golden = golden & (macd_line > 0.0)
        dead = dead & (macd_line < 0.0)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[golden.fillna(False)] = 1
    signal[dead.fillna(False)] = -1
    return signal


templates.register(
    "macd_cross",
    defaults={
        "fast_period": 12.0,
        "slow_period": 26.0,
        "signal_period": 9.0,
        "zero_filter": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_macd_cross,
)
