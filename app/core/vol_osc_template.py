"""出来高オシレーター(vol_osc)テンプレート — MFI/OBVダイバージェンス/Force/CMF(2026-07-18)。

「出来高=大口の足跡」仮説の体系検証。各指標は標準用法で事前登録:
- ind=1 MFI(14): >80でショート / <20でロング(逆張り)
- ind=2 OBVダイバージェンス: 価格20本高値更新かつOBV未更新→ショート、ミラーでロング
- ind=3 Force Index(13EMA): ゼロ上抜け→ロング、下抜け→ショート(順張り)
- ind=4 CMF(20): 符号の転換に追随
出口は呼び出し側の max_hold_bars(時間切れ)。tickボリューム=真の出来高でない点は前提。
先読み回避: 全てバーi確定情報、engineのshift(1)执行。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_vol_osc(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    ind = int(strategy.params["ind"].value)
    h = df["high"]; l = df["low"]; c = df["close"]
    v = df["volume"].astype(float)
    n = len(df)
    sig = pd.Series(0, index=df.index, dtype=int)

    if ind == 1:  # MFI(14)
        tp = (h + l + c) / 3
        mf = tp * v
        pos = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
        neg = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
        mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
        sig[(mfi > 80).fillna(False)] = -1
        sig[(mfi < 20).fillna(False)] = 1
    elif ind == 2:  # OBVダイバージェンス
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        ph = c.rolling(20).max()
        pl = c.rolling(20).min()
        oh = obv.rolling(20).max()
        ol = obv.rolling(20).min()
        bear = (c >= ph) & (obv < oh * 0.999 - 1e-9)   # 価格高値更新・OBV未更新
        bull = (c <= pl) & (obv > ol * 1.001 + 1e-9)
        sig[bear.fillna(False)] = -1
        sig[bull.fillna(False)] = 1
    elif ind == 3:  # Force Index(13EMA)ゼロクロス
        fi = (c.diff() * v).ewm(span=13, adjust=False).mean()
        up = (fi > 0) & (fi.shift(1) <= 0)
        dn = (fi < 0) & (fi.shift(1) >= 0)
        sig[up.fillna(False)] = 1
        sig[dn.fillna(False)] = -1
    else:  # CMF(20)符号転換
        rng = (h - l).replace(0, np.nan)
        mfm = ((c - l) - (h - c)) / rng
        cmf = (mfm * v).rolling(20).sum() / v.rolling(20).sum()
        up = (cmf > 0.05) & (cmf.shift(1) <= 0.05)
        dn = (cmf < -0.05) & (cmf.shift(1) >= -0.05)
        sig[up.fillna(False)] = 1
        sig[dn.fillna(False)] = -1
    return sig


templates.register(
    "vol_osc",
    defaults={
        "ind": 1.0,
        "sl_pips": 500.0,
        "tp_pips": 500.0,
        "max_hold_bars": 24.0,
        "lot": 0.1,
    },
    signal_fn=_signal_vol_osc,
)
