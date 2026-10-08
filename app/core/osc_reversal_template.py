"""オシレーター逆張り統合テンプレート(osc_reversal)。

CCI / Williams %R / DeMarker / Momentum の4オシレーターを osc_index で選択する
ゾーン回復型逆張り(rsi_reversalと同じ「極値ゾーンから戻った瞬間」のエッジ検知)。
lower/upper はオシレーター毎に規約が異なるため osc_index に応じた既定ゾーンを
内部で持ち、lower/upper=0(デフォルト)のときはそれを使う(明示指定で上書き可)。

osc_index: 0=CCI(±100), 1=WilliamsR(-80/-20), 2=DeMarker(0.3/0.7), 3=Momentum(99.7/100.3)
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import cci, demarker, momentum, williams_r
from app.core.strategy_model import Strategy

_DEFAULT_ZONES = {0: (-100.0, 100.0), 1: (-80.0, -20.0), 2: (0.3, 0.7), 3: (99.7, 100.3)}


def _signal_osc_reversal(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    osc_index = max(0, min(3, int(strategy.params["osc_index"].value)))
    period = max(2, int(strategy.params["period"].value))
    lower = float(strategy.params["lower"].value)
    upper = float(strategy.params["upper"].value)
    if lower == 0.0 and upper == 0.0:
        lower, upper = _DEFAULT_ZONES[osc_index]

    if osc_index == 0:
        osc = cci(df["high"], df["low"], df["close"], period)
    elif osc_index == 1:
        osc = williams_r(df["high"], df["low"], df["close"], period)
    elif osc_index == 2:
        osc = demarker(df["high"], df["low"], period)
    else:
        osc = momentum(df["close"], period)

    prev = osc.shift(1)
    recover = (prev <= lower) & (osc > lower)
    fall = (prev >= upper) & (osc < upper)
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[recover.fillna(False)] = 1
    signal[fall.fillna(False)] = -1
    return signal


templates.register(
    "osc_reversal",
    defaults={
        "osc_index": 0.0,
        "period": 14.0,
        "lower": 0.0,
        "upper": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_osc_reversal,
)
