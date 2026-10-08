"""120SMA + Williams %R テンプレート(YouTube解析レポート #8、検証優先順位1位)。

出典: C:\\CodexProject\\YOUTUBE_kaiseki\\artifacts\\scalping_methods_10_detailed.md #8
「Williams %Rを使う5分足手法」。時間足M5。SMA(120)で大方向を判定し、その方向への
過熱解消(Williams %R の±極値からの戻り)を順張りエントリーのトリガーにする。

ロング: close>SMA120(上昇方向) かつ WPR(30)が-90を下抜けた後、-90を上抜けて確定
  (2バーパターン: bar[i-1]<=-90 and bar[i]>-90、動画の「その後-90を上抜けて足が確定する」
  をそのまま数値化)。次バー始値で買い(engineのshift(1)が担当、テンプレ側はshiftしない)。
ショートはミラー: close<SMA120 かつ WPRが-10を上抜けた後、-10を下抜けて確定。
SL=直近安値/高値(lookback=10本、ATR×sl_buffer_atrぶん外側)、TP=SL幅×1.5(動的sl_price/tp_price)。

先読み規律: SMA/WPR/ATRは全て確定バーまでのrolling/ewm。2バークロス判定はwpr.shift(1)との
比較のみで未来を参照しない。sl_price/tp_priceは信号バー時点のrolling(10)安値/高値とATRから
計算する値(信号バー確定時点で既知)であり、シグナル列と一緒にengine側でshift(1)されるため、
実際の使用は「シグナル確定バーの次バー」時点になる(先読みなし)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, sma, williams_r
from app.core.strategy_model import Strategy


def _signal_sma120_wpr(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    sma_period = max(2, int(p["sma_period"].value))
    wpr_period = max(2, int(p["wpr_period"].value))
    level_low = float(p["level_low"].value)    # -90 (売られすぎ極値)
    level_high = float(p["level_high"].value)  # -10 (買われすぎ極値)
    sl_lookback = max(2, int(p["sl_lookback"].value))
    sl_buffer_atr = float(p["sl_buffer_atr"].value)
    rr = float(p["rr"].value)
    side = int(p["side"].value)  # 0=both, 1=long_only, -1=short_only

    h, l, c = df["high"], df["low"], df["close"]
    ma = sma(c, sma_period)
    wpr = williams_r(h, l, c, wpr_period)
    a = atr(h, l, c, 14)

    trend_up = c > ma
    trend_dn = c < ma
    wpr_prev = wpr.shift(1)
    # 2バークロス確定パターン(先読みなし: 自分の過去shiftのみ参照)
    cross_up = (wpr_prev <= level_low) & (wpr > level_low)
    cross_dn = (wpr_prev >= level_high) & (wpr < level_high)

    long_sig = (trend_up & cross_up).fillna(False)
    short_sig = (trend_dn & cross_dn).fillna(False)
    if side > 0:
        short_sig &= False
    elif side < 0:
        long_sig &= False

    recent_low = l.rolling(sl_lookback).min()
    recent_high = h.rolling(sl_lookback).max()

    n = len(df)
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    li = long_sig.to_numpy()
    si = short_sig.to_numpy()
    ca = c.to_numpy()
    rl = recent_low.to_numpy()
    rh = recent_high.to_numpy()
    aa = a.to_numpy()

    sig[li] = 1
    sig[si] = -1

    sl_l = rl - sl_buffer_atr * aa
    sl_s = rh + sl_buffer_atr * aa
    slp[li] = sl_l[li]
    tpp[li] = ca[li] + rr * (ca[li] - sl_l[li])
    slp[si] = sl_s[si]
    tpp[si] = ca[si] - rr * (sl_s[si] - ca[si])

    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


templates.register(
    "sma120_wpr",
    defaults={
        "sma_period": 120.0,
        "wpr_period": 30.0,
        "level_low": -90.0,
        "level_high": -10.0,
        "sl_lookback": 10.0,
        "sl_buffer_atr": 0.1,
        "rr": 1.5,
        "side": 0.0,
        "sl_pips": 15.0,
        "tp_pips": 22.5,
        "max_hold_bars": 96.0,
        "lot": 0.1,
    },
    signal_fn=_signal_sma120_wpr,
)
