"""CCI±200折返し逆張り(手法#9)テンプレート。USDJPY H4想定。

CCI(cci_period)が売られすぎ/買われすぎの極値(既定±200、標準的な±100より深い
オーバーシュート水準)まで達したのち、そこから極値内側へ折り返した瞬間を逆張りの
トリガーとする(「極値到達」自体ではなく「極値からの復帰」で仕掛けるため、極値に
張り付いたままの連続シグナルにはならない。osc_reversal/rsi_reversalと同じ
「回復/反落」の型)。

近似(重要): 原手法の決済(SMA30タッチ、または直近スイングのフィボナッチ50%到達)は
本Stage1では機械化を簡略化し、engineの固定pips決済(sl_pips/tp_pips)で代替する。

先読み規律: CCIはindicators.cci(rollingのみ)で確定バー+自分の過去だけから計算。
折返し判定の.shift(1)は自分の直前バーとの比較のみ(先読みではない)。翌バー始値執行の
shiftはengine.run_backtest側が行い、テンプレ内に翌バー参照はない。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import cci
from app.core.strategy_model import Strategy


def _signal_cci_reversal(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    cci_period = max(2, int(p["cci_period"].value))  # 既定50
    upper = float(p["upper"].value)                  # 買われすぎ側の極値(既定200)
    lower = float(p["lower"].value)                  # 売られすぎ側の極値(既定200。-lowerとして使う)
    dir_mode = int(p["dir_mode"].value)               # 0=both,1=long,2=short

    c = cci(df["high"], df["low"], df["close"], cci_period)
    prev_c = c.shift(1)

    # 売られすぎ(-lower以下)からの折返し上抜け → 買い
    long_raw = (prev_c <= -lower) & (c > -lower)
    # 買われすぎ(upper以上)からの折返し下抜け → 売り
    short_raw = (prev_c >= upper) & (c < upper)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_raw.fillna(False)] = 1
    if dir_mode in (0, 2):
        signal[short_raw.fillna(False)] = -1
    return signal


templates.register(
    "cci_reversal",
    defaults={
        "cci_period": 50.0, "upper": 200.0, "lower": 200.0, "dir_mode": 0.0,
        "sl_pips": 40.0, "tp_pips": 60.0, "lot": 0.1,
    },
    signal_fn=_signal_cci_reversal,
)
