"""SMA10/30 + RSI(8) テンプレート(YouTube解析レポート #9、優先度4位)。

出典: C:\\CodexProject\\YOUTUBE_kaiseki\\artifacts\\scalping_methods_10_detailed.md #9
「SMAとRSIを使った1分足スキャルピング」。動画内検証は2023-04-24〜04-27の4日間・
35トレード・勝率45.7%・PF1.09(タイトルの勝率80%は非再現)。本テンプレはこの
低いPF・小標本を自前の複数年・複数ペアデータで再検証するための機械化。

数値化した曖昧語:
- 「SMA10>SMA30」→ そのまま(トレンド方向フィルタ)。
- 「RSIが40を下回った後40を再上抜け」→ 2バークロス確定パターン(T1のWPR判定と同じ
  構造): rsi.shift(1)<=40 かつ rsi>40。
- 「陽線で確定したら買う」→ クロス確定バー自身がclose>open(同バーの陽線確定)。
- 「損切りは直近安値の少し下」→ SL=直近安値(lookback=10本)-buffer×ATR。
- 「利確は1.5R」→ TP=SL幅×1.5(動的sl_price/tp_price)。

先読み規律: SMA/RSI/ATRは全て確定バーまでのrolling/ewm。2バークロス判定は
rsi.shift(1)の自分の過去参照のみ、陽線確定は当バー自身のOHLC(バー確定時点で既知)。
sl_price/tp_priceは信号バー確定時点で既知のrolling(lookback)安値/高値とATRから
計算し、シグナル列と一緒にengine側でshift(1)されるため実際の使用は信号バーの次バー
(先読みなし)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, rsi, sma
from app.core.strategy_model import Strategy


def _signal_sma1030_rsi(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    sma_fast = max(2, int(p["sma_fast"].value))    # 10
    sma_slow = max(2, int(p["sma_slow"].value))     # 30
    rsi_period = max(2, int(p["rsi_period"].value))  # 8
    level_low = float(p["level_low"].value)    # 40
    level_high = float(p["level_high"].value)  # 60
    sl_lookback = max(2, int(p["sl_lookback"].value))  # 10
    sl_buffer_atr = float(p["sl_buffer_atr"].value)
    rr = float(p["rr"].value)
    side = int(p["side"].value)  # 0=both, 1=long_only, -1=short_only

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    s_f = sma(c, sma_fast)
    s_s = sma(c, sma_slow)
    r = rsi(c, rsi_period)
    a = atr(h, l, c, 14)

    trend_up = s_f > s_s
    trend_dn = s_f < s_s
    r_prev = r.shift(1)
    cross_up = (r_prev <= level_low) & (r > level_low)
    cross_dn = (r_prev >= level_high) & (r < level_high)
    bull_candle = c > o
    bear_candle = c < o

    long_sig = (trend_up & cross_up & bull_candle).fillna(False)
    short_sig = (trend_dn & cross_dn & bear_candle).fillna(False)
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
    "sma1030_rsi",
    defaults={
        "sma_fast": 10.0,
        "sma_slow": 30.0,
        "rsi_period": 8.0,
        "level_low": 40.0,
        "level_high": 60.0,
        "sl_lookback": 10.0,
        "sl_buffer_atr": 0.1,
        "rr": 1.5,
        "side": 0.0,
        "sl_pips": 8.0,
        "tp_pips": 12.0,
        "max_hold_bars": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_sma1030_rsi,
)
