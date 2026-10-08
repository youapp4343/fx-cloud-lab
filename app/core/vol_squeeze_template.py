"""ボラティリティスクイーズ後のブレイクアウト追随テンプレート(Phase10: 新規指標・戦略)。

vol_breakout_template.pyは「ATR fast/slow比の瞬間的な倍率」でボラティリティ拡大を検知するのに
対し、本テンプレートは「ATRが自身の直近履歴(percentile_lookback本)の分布の中で相対的に低い
水準(スクイーズ)にあったあとの拡大」という異なる着眼点でブレイクアウトを捉える。

ブレイクの瞬間は既にATRが上昇し始めておりatr_percentile_rank自体はスクイーズ閾値を超えている
可能性が高いため、「今スクイーズ中」ではなく「直近squeeze_confirm_bars本以内にスクイーズ状態が
一度でもあったか」を条件にする。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, donchian_channel
from app.core.novel_indicators import atr_percentile_rank
from app.core.strategy_model import Strategy


def _signal_vol_squeeze_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    # atr()はewm(alpha=1/period)のためperiod<=0でZeroDivisionError、rolling(lookback)は
    # lookback<=0で空windowになりatr_percentile_rank内部でIndexErrorになる(FABLE監査で発見、
    # optimizerのrange探索で0/負値グリッドを踏むとクラッシュしうる)。他パラメータと同様に
    # max(1, ...)でクリップする。
    atr_period = max(1, int(strategy.params["atr_period"].value))
    percentile_lookback = max(1, int(strategy.params["percentile_lookback"].value))
    squeeze_threshold = float(strategy.params["squeeze_threshold"].value)
    squeeze_confirm_bars = max(1, int(strategy.params["squeeze_confirm_bars"].value))
    range_bars = max(1, int(strategy.params["range_bars"].value))

    a = atr(df["high"], df["low"], df["close"], atr_period)
    pct_rank = atr_percentile_rank(a, percentile_lookback)
    is_squeezed = pct_rank <= squeeze_threshold
    # bool Seriesのままrolling().max()すると環境によりdtypeが揺れるため、int化してから
    # 0との大小比較でboolに戻す(NaN区間は0扱いでFalseになり、FutureWarningも出ない)。
    was_squeezed = is_squeezed.astype(int).rolling(squeeze_confirm_bars).max().fillna(0) > 0

    upper, lower = donchian_channel(df["high"], df["low"], range_bars)
    breakout_up = df["close"] > upper
    breakout_down = df["close"] < lower

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(was_squeezed & breakout_up).fillna(False)] = 1
    signal[(was_squeezed & breakout_down).fillna(False)] = -1
    return signal


templates.register(
    "vol_squeeze_breakout",
    defaults={
        "atr_period": 14.0,
        "percentile_lookback": 100.0,
        "squeeze_threshold": 0.2,
        "squeeze_confirm_bars": 10.0,
        "range_bars": 20.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_vol_squeeze_breakout,
)
