"""Supertrend フリップ 順張り テンプレート(note EA記事 Rank2「ATR EA」の機械化)。

出典: note.com/aimjey EAランキング2位。15分足USDJPY、ATR(14)+Supertrend(乗数1.0)。
close が mid+mult*ATR を上抜けたら買い(=Supertrendが上昇にフリップ)、mid-mult*ATRを
下抜けたら売り。報告PF 1.48(始値)/1.30(全tick)。OOS分割なし・上位10選抜=選択バイアス注意。

標準Supertrend: final upper/lower を再帰更新し、closeが前バーのバンドを抜けた時に
トレンド方向がフリップする。フリップした初回バーでシグナル(再アーム)。決済は engine sl/tp
(記事: SL4.0=400pips, TP5.0=500pips)。

先読み規律: ATR/バンドは確定バーまでの過去のみ、再帰は index<=i のみ参照。執行shiftはengine。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _signal_supertrend(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    period = max(2, int(p["atr_period"].value))
    mult = float(p["mult"].value)
    dir_mode = int(p["dir_mode"].value)

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    a = atr(df["high"], df["low"], df["close"], period).to_numpy(float)
    hl2 = (high + low) / 2.0
    bu = hl2 + mult * a
    bl = hl2 - mult * a
    n = len(df)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)  # +1 up, -1 down
    sig = np.zeros(n, dtype=int)
    for i in range(n):
        if np.isnan(bu[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]):
            fu[i] = bu[i]; fl[i] = bl[i]; trend[i] = 1
            continue
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (bl[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        # トレンド判定: 前バーがdownで close が上バンド上抜け→up、その逆→down
        if trend[i - 1] == 1:
            trend[i] = -1 if close[i] < fl[i] else 1
        else:
            trend[i] = 1 if close[i] > fu[i] else -1
        if trend[i] == 1 and trend[i - 1] == -1:
            sig[i] = 1   # 上昇フリップ初回
        elif trend[i] == -1 and trend[i - 1] == 1:
            sig[i] = -1  # 下降フリップ初回
    if dir_mode == 1:
        sig[sig < 0] = 0
    elif dir_mode == 2:
        sig[sig > 0] = 0
    return pd.Series(sig, index=df.index, dtype=int)


templates.register(
    "supertrend",
    defaults={"atr_period": 14.0, "mult": 1.0, "dir_mode": 0.0,
              "sl_pips": 400.0, "tp_pips": 500.0, "lot": 0.1},
    signal_fn=_signal_supertrend,
)
