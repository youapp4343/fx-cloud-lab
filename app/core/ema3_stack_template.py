"""EMA50/100/150 パーフェクトオーダー押し目テンプレート(YouTube解析レポート #2、優先度2位)。

出典: C:\\CodexProject\\YOUTUBE_kaiseki\\artifacts\\scalping_methods_10_detailed.md #2
「EMA3本を使った1分足戦略」。時間足M1。EMA50/100/150が全て順配列(かつ各々が
slope_lb本前より上昇/下降=横向き除外)の状態で、価格がEMA100まで押し、EMA50を
上抜けた時点で買う。

数値化した曖昧語:
- 「3本が上向きで短期から順に並ぶ」→ EMA50>EMA100>EMA150 かつ 各EMAがslope_lb(5)本前
  より上昇(=傾きゲート。これにより「EMAが横向き/交錯」ケースは自動的に発火しない)。
- 「価格がEMA100まで押し、反発する」→ 直近touch_lookback(5)本以内に low<=EMA100。
- 「EMA50を上抜いた時点で買う」(上抜け確定)→ 2バークロス: close.shift(1)<=EMA50.shift(1)
  かつ close>EMA50(継続保持ではなく突破の瞬間のみ発火、再発火を防ぐ)。
- 「損切りはEMA100の下」→ SL=EMA100(信号バー時点の値)-buffer×ATR。押し基準バーの厳密な
  EMA100値ではなく信号バー時点の値で近似(touch_lookback=5本以内なのでほぼ同水準、docstring明記)。
- 「利確は損切り幅の最低2倍」→ TP=SL幅×2(動的sl_price/tp_price)。

先読み規律: 全指標は確定バーまでのewm/rolling。クロス判定はclose.shift(1)/EMA.shift(1)の
自分の過去参照のみ。sl_price/tp_priceは信号バー確定時点で既知の値であり、シグナル列と
一緒にengine側でshift(1)されるため実際の使用は信号バーの次バー(先読みなし)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema
from app.core.strategy_model import Strategy


def _signal_ema3_stack(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ema_fast = max(2, int(p["ema_fast"].value))    # 50
    ema_mid = max(2, int(p["ema_mid"].value))       # 100
    ema_slow = max(2, int(p["ema_slow"].value))     # 150
    slope_lb = max(1, int(p["slope_lb"].value))     # 5
    touch_lookback = max(1, int(p["touch_lookback"].value))  # 5
    sl_buffer_atr = float(p["sl_buffer_atr"].value)
    rr = float(p["rr"].value)
    side = int(p["side"].value)  # 0=both, 1=long_only, -1=short_only

    h, l, c = df["high"], df["low"], df["close"]
    e_f = ema(c, ema_fast)
    e_m = ema(c, ema_mid)
    e_s = ema(c, ema_slow)
    a = atr(h, l, c, 14)

    stack_up = (
        (e_f > e_m) & (e_m > e_s)
        & (e_f > e_f.shift(slope_lb)) & (e_m > e_m.shift(slope_lb)) & (e_s > e_s.shift(slope_lb))
    )
    stack_dn = (
        (e_f < e_m) & (e_m < e_s)
        & (e_f < e_f.shift(slope_lb)) & (e_m < e_m.shift(slope_lb)) & (e_s < e_s.shift(slope_lb))
    )

    touched_up = (l <= e_m).rolling(touch_lookback).max().fillna(0) > 0
    touched_dn = (h >= e_m).rolling(touch_lookback).max().fillna(0) > 0

    cross_up = (c.shift(1) <= e_f.shift(1)) & (c > e_f)
    cross_dn = (c.shift(1) >= e_f.shift(1)) & (c < e_f)

    long_sig = (stack_up & touched_up & cross_up).fillna(False)
    short_sig = (stack_dn & touched_dn & cross_dn).fillna(False)
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
    em = e_m.to_numpy()
    aa = a.to_numpy()

    sig[li] = 1
    sig[si] = -1

    sl_l = em - sl_buffer_atr * aa
    sl_s = em + sl_buffer_atr * aa
    slp[li] = sl_l[li]
    tpp[li] = ca[li] + rr * (ca[li] - sl_l[li])
    slp[si] = sl_s[si]
    tpp[si] = ca[si] - rr * (sl_s[si] - ca[si])

    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


templates.register(
    "ema3_stack",
    defaults={
        "ema_fast": 50.0,
        "ema_mid": 100.0,
        "ema_slow": 150.0,
        "slope_lb": 5.0,
        "touch_lookback": 5.0,
        "sl_buffer_atr": 0.1,
        "rr": 2.0,
        "side": 0.0,
        "sl_pips": 15.0,
        "tp_pips": 30.0,
        "max_hold_bars": 96.0,
        "lot": 0.1,
    },
    signal_fn=_signal_ema3_stack,
)
