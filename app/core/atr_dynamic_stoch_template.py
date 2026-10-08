"""atr_dynamic_stoch(EMA3本配列+StochRSIクロス+ATR動的SL/TP)テンプレート
(YouTube解析レポート set2 #5由来、動画内検証の忠実再現)。

出典: scripts/scalp_set2_prereg.md T2。M5。EMA(8)>EMA(14)>EMA(50)の順配列(ロング、
ショートはミラー)中に、StochRSI(rsi_period=14, stoch_period=14, smoothK=3,
smoothD=3)のK線がD線を確定バーで上抜け(ロング)/下抜け(ショート)したらエントリー。
SL=3×ATR(14)、TP=2×ATR(14)(RR3:2=リスクの方が大きい構成、動画の忠実再現のため
事後修正しない)。6ペア=6検定。

StochRSI: indicators.stoch_rsi はK平滑のみを返すため、本テンプレでD線(Kの3期間SMA)
まで独自に計算する(RSI→ストキャスティクス変換→3期間SMA×2段、TradingView定義)。

先読み規律: EMA/RSI/StochRSI/ATRは全て確定バーiまでの情報のみのewm/rollingで計算。
ゴールデン/デッドクロスは「前バーのK<=D かつ 当バーのK>D」の確定判定(shift(1)の
自分の過去参照のみ)。sl_price/tp_priceは信号バーi確定時点で既知のclose[i]と
ATR[i]から計算し、シグナル列と一緒にengine側でshift(1)されるため実際の使用は
信号バーの次バー(先読みなし、ema3_fractal_templateと同じパターン)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, rsi
from app.core.strategy_model import Strategy


def _signal_atr_dynamic_stoch(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ema_fast = max(2, int(p["ema_fast"].value))     # 8
    ema_mid = max(2, int(p["ema_mid"].value))        # 14
    ema_slow = max(2, int(p["ema_slow"].value))       # 50
    rsi_period = max(2, int(p["rsi_period"].value))   # 14
    stoch_period = max(2, int(p["stoch_period"].value))  # 14
    smooth_k = max(1, int(p["smooth_k"].value))       # 3
    smooth_d = max(1, int(p["smooth_d"].value))       # 3
    sl_atr_mult = float(p["sl_atr_mult"].value)        # 3.0
    tp_atr_mult = float(p["tp_atr_mult"].value)        # 2.0
    side = int(p["side"].value)

    h, l, c = df["high"], df["low"], df["close"]
    e_f = ema(c, ema_fast)
    e_m = ema(c, ema_mid)
    e_s = ema(c, ema_slow)
    a = atr(h, l, c, 14)

    stack_up = (e_f > e_m) & (e_m > e_s)
    stack_dn = (e_f < e_m) & (e_m < e_s)

    # StochRSI K/D(TradingView定義): RSI→期間内レンジ位置(0-100)→3期間SMA(K)→3期間SMA(D)
    r = rsi(c, rsi_period)
    lo = r.rolling(stoch_period).min()
    hi = r.rolling(stoch_period).max()
    stoch_rng = hi - lo
    raw = (100.0 * (r - lo) / stoch_rng).where(stoch_rng != 0)
    k_line = raw.rolling(smooth_k).mean()
    d_line = k_line.rolling(smooth_d).mean()

    prev_k, prev_d = k_line.shift(1), d_line.shift(1)
    golden = (prev_k <= prev_d) & (k_line > d_line)
    dead = (prev_k >= prev_d) & (k_line < d_line)

    long_sig = (stack_up & golden.fillna(False)).fillna(False)
    short_sig = (stack_dn & dead.fillna(False)).fillna(False)
    if side > 0:
        short_sig &= False
    elif side < 0:
        long_sig &= False

    n = len(df)
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    li = long_sig.to_numpy()
    si = short_sig.to_numpy()
    ca = c.to_numpy()
    aa = a.to_numpy()

    sig[li] = 1
    sig[si] = -1
    slp[li] = ca[li] - sl_atr_mult * aa[li]
    tpp[li] = ca[li] + tp_atr_mult * aa[li]
    slp[si] = ca[si] + sl_atr_mult * aa[si]
    tpp[si] = ca[si] - tp_atr_mult * aa[si]

    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


templates.register(
    "atr_dynamic_stoch",
    defaults={
        "ema_fast": 8.0,
        "ema_mid": 14.0,
        "ema_slow": 50.0,
        "rsi_period": 14.0,
        "stoch_period": 14.0,
        "smooth_k": 3.0,
        "smooth_d": 3.0,
        "sl_atr_mult": 3.0,
        "tp_atr_mult": 2.0,
        "side": 0.0,
        "sl_pips": 30.0,   # フォールバック(sl_price無効時のみ使用)
        "tp_pips": 20.0,   # フォールバック(tp_price無効時のみ使用)
        "max_hold_bars": 200.0,   # 安全弁(事前登録に明記なし、RR3:2の動的決済が主)
        "lot": 0.1,
    },
    signal_fn=_signal_atr_dynamic_stoch,
)
