"""連続ローソク・ブレイクアウト(consec_break)テンプレート。

「N本以上の連続陰線のあと陽線が出て、その連続の最後の陰線の高値を超えたらロング」
(ミラー: N本連続陽線→陰線が最後の陽線の安値割れでショート)。値動きの荒いgold(XAUUSD)の
H1ブレイクアウトを主眼。RR例 TP100pips/SL50pips(pip=0.1なら$10/$5)。

トリガー(先読み無し):
- long: 当該足が陽線(close>open) かつ 直前まで min_consec 本以上の連続陰線 かつ
  当該足の高値 > 直前(=最後の陰線)の高値。
- short: 対称(連続陽線→陰線が直前陽線の安値割れ)。
- side: 1=ロングのみ(既定) / -1=ショートのみ / 0=両方。

SL/TPは engine の sl_pips/tp_pips(固定pips)。シグナルは当該確定足の情報のみ使用、shiftはengine側。
近似: 「最後の陰線の高値超え」を当該足の高値ブレイクで検知し執行は次足始値(engine shift(1))。
真のバイストップ執行との差(ブレイク水準〜次足始値のスリッページ)は許容。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _run_length(mask: np.ndarray) -> np.ndarray:
    """各位置で終わる連続Trueの本数。"""
    out = np.zeros(len(mask), dtype=int)
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        out[i] = c
    return out


def _signal_consec_break(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    min_consec = max(1, int(p["min_consec"].value))
    side = int(p["side"].value) if "side" in p else 1

    # ATR動的SL/TP(任意): atr_tp_mult>0 なら ATR×倍率でsl_price/tp_price列を返す(非定常補正)。
    atr_tp_mult = float(p["atr_tp_mult"].value) if "atr_tp_mult" in p else 0.0
    atr_sl_mult = float(p["atr_sl_mult"].value) if "atr_sl_mult" in p else 0.0
    atr_period = max(2, int(p["atr_period"].value)) if "atr_period" in p else 14
    use_atr = atr_tp_mult > 0 and atr_sl_mult > 0

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    a = atr(df["high"], df["low"], df["close"], atr_period).to_numpy(float) if use_atr else None

    red = c < o
    black = c > o
    run_red = _run_length(red)
    run_black = _run_length(black)

    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan)
    tp_price = np.full(n, np.nan)
    for i in range(1, n):
        sig = 0
        if side >= 0 and black[i] and run_red[i - 1] >= min_consec and h[i] > h[i - 1]:
            sig = 1
        elif side <= 0 and red[i] and run_black[i - 1] >= min_consec and low[i] < low[i - 1]:
            sig = -1
        if sig == 0:
            continue
        signal[i] = sig
        if use_atr and not np.isnan(a[i]):
            sl_price[i] = c[i] - sig * atr_sl_mult * a[i]
            tp_price[i] = c[i] + sig * atr_tp_mult * a[i]

    if use_atr:
        return pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price}, index=df.index)
    return pd.Series(signal, index=df.index, dtype=int)


templates.register(
    "consec_break",
    defaults={
        "min_consec": 2.0,
        "side": 1.0,          # 1=long / -1=short / 0=both
        "sl_pips": 50.0,      # gold pip=0.1 → $5
        "tp_pips": 100.0,     # → $10 (RR1:2)
        "atr_period": 14.0,
        "atr_sl_mult": 0.0,   # >0 かつ atr_tp_mult>0 でATR動的SL/TP(非定常補正)
        "atr_tp_mult": 0.0,
        "lot": 0.1,
    },
    signal_fn=_signal_consec_break,
)
