"""拡張オシレーター合議テンプレート(osc_confluence_x、14種)。

osc_confluence(6種)にTradingView定番8種(StochRSI/MFI/UltimateOsc/CMO/TSI/
Fisher/ConnorsRSI/CMF)を加えた14オシレーターの合議。min_agree個以上が同時に
売られすぎ/買われすぎ一致へ遷移したバーで逆張り。既存osc_confluenceは6種のまま
不変(min_agreeの母数意味が変わるため別テンプレートとして提供)。

ゾーン(標準値): RSI30/70, Stoch20/80, CCI±100, WPR-80/-20, DeMarker0.3/0.7,
Momentum99.7/100.3, StochRSI20/80, MFI20/80, UO30/70, CMO±50, TSI±25,
Fisher±1.5, ConnorsRSI10/90, CMF±0.2。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import (
    chaikin_money_flow, cci, cmo, connors_rsi, demarker, fisher_transform,
    mfi, momentum, rsi, stoch_rsi, stochastic, tsi, ultimate_oscillator, williams_r,
)
from app.core.strategy_model import Strategy


def _signal_osc_confluence_x(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    period = max(2, int(strategy.params["period"].value))
    min_agree = max(1, min(14, int(strategy.params["min_agree"].value)))

    h, l, c, v = df["high"], df["low"], df["close"], df.get("volume")
    if v is None:
        v = pd.Series(0.0, index=df.index)

    r = rsi(c, period)
    k, _ = stochastic(h, l, c, period, 3, 3)
    cc = cci(h, l, c, period)
    w = williams_r(h, l, c, period)
    dm = demarker(h, l, period)
    mo = momentum(c, period)
    sr = stoch_rsi(c, period, period, 3)
    mf = mfi(h, l, c, v, period)
    uo = ultimate_oscillator(h, l, c)
    cm = cmo(c, period)
    ts = tsi(c)
    fi = fisher_transform(h, l, max(5, period // 2))
    cr = connors_rsi(c)
    cf = chaikin_money_flow(h, l, c, v, period)

    zones = [  # (系列, 売られすぎ閾値, 買われすぎ閾値)
        (r, 30, 70), (k, 20, 80), (cc, -100, 100), (w, -80, -20), (dm, 0.3, 0.7),
        (mo, 99.7, 100.3), (sr, 20, 80), (mf, 20, 80), (uo, 30, 70), (cm, -50, 50),
        (ts, -25, 25), (fi, -1.5, 1.5), (cr, 10, 90), (cf, -0.2, 0.2),
    ]
    oversold = sum((s <= lo).astype(int) for s, lo, _ in zones)
    overbought = sum((s >= hi).astype(int) for s, _, hi in zones)

    buy_state = oversold >= min_agree
    sell_state = overbought >= min_agree
    buy_edge = buy_state & ~buy_state.shift(1).fillna(False).astype(bool)
    sell_edge = sell_state & ~sell_state.shift(1).fillna(False).astype(bool)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[buy_edge] = 1
    signal[sell_edge] = -1
    return signal


templates.register(
    "osc_confluence_x",
    defaults={
        "period": 14.0,
        "min_agree": 8.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_osc_confluence_x,
)
