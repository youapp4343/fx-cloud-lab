"""GMMA完全整列 順張り テンプレート(note EA記事 Rank1「GMMA EA」の機械化)。

出典: note.com/aimjey EAランキング1位。15分足USDJPY、短期SMA6本[10,15,20,25,30,40]と
長期SMA6本[45,45,50,55,60,95]。全短期が対応する全長期を上回ったら買い、下回ったら売り。
報告PF 1.26(始値)/1.33(全tick)。ただしOOS分割なし・上位10選抜=選択バイアスに注意。

エントリー: 6ペア(SMA10>SMA45, 15>45, 20>50, 25>55, 30>60, 40>95)が全て成立=完全上昇整列
の初回バーで買い(再アーム)。全て逆=完全下降整列の初回で売り。決済は engine の sl/tp
(記事: SL4.0=400pips, TP5.0=500pips のワイドストップ順張り)。

先読み規律: SMAは確定バーまでの過去のみ、再アーム shift(1) は自分の過去参照。執行shiftはengine。
"""
from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import sma
from app.core.strategy_model import Strategy

_SHORT = [10, 15, 20, 25, 30, 40]
_LONG = [45, 45, 50, 55, 60, 95]


def _signal_gmma_stack(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    dir_mode = int(strategy.params["dir_mode"].value)
    close = df["close"]
    sS = [sma(close, p) for p in _SHORT]
    sL = [sma(close, p) for p in _LONG]
    up = sS[0] > sL[0]
    dn = sS[0] < sL[0]
    for i in range(1, 6):
        up = up & (sS[i] > sL[i])
        dn = dn & (sS[i] < sL[i])
    up = up.fillna(False); dn = dn.fillna(False)
    long_sig = up & (~up.shift(1).fillna(False))
    short_sig = dn & (~dn.shift(1).fillna(False))
    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "gmma_stack",
    defaults={"dir_mode": 0.0, "sl_pips": 400.0, "tp_pips": 500.0, "lot": 0.1},
    signal_fn=_signal_gmma_stack,
)
