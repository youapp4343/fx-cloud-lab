"""EMA20/50/100 + Williams Fractal 押し目テンプレート(YouTube解析レポート #7、優先度3位)。

出典: C:\\CodexProject\\YOUTUBE_kaiseki\\artifacts\\scalping_methods_10_detailed.md #7
「フラクタル指標と3本の移動平均線」。時間足M1またはM5(データ側で選択、テンプレは
時間足非依存)。EMA20>EMA50>EMA100の順配列中に価格がEMA20またはEMA50へ押し、
確定した上向きWilliams Fractalが出現したら買う。

Williams Fractal(標準5本窓)の先読み対策(レポート#7の警告「Fractalは2本分の遅延を
認めている」への厳密対応):
  上向きFractal(bar[k]が安値の極小)は、窓[k-2, k-1, k, k+1, k+2]の中央bar[k]が
  最安値のときに成立するが、右側2本(k+1, k+2)が未確定だと判定できない。
  本実装ではバーインデックスiで「bar[i-2]が[i-4..i]の中央最小」を判定し、これは
  bar[i]まで(=自分の過去とバーiの終値まで)の情報だけで確定する(k=i-2, 右側2本=i-1,i)。
  つまり `up_fractal[i]` はバーi確定時点で初めて真偽が決まり、それより前のバーでは
  絶対に分からない(=2本ラグを正しく反映、先読みなし)。

数値化した曖昧語:
- 「EMA20>EMA50>EMA100の順配列」→ 傾きゲートなし(レポート原文に傾き条件の明記なし)、
  順序のみ(横向き/交錯時は順序不成立のため自然に発火しない)。
- 「価格がEMA20またはEMA50まで押した」→ 直近lookback(5)本以内に low<=EMA20 または low<=EMA50。
- 「緑の上向きFractalが表示されたら買う」→ 上記2本ラグ確定のup_fractal[i]==Trueが
  信号バーiで成立。
- 「損切りはEMA20/EMA50基準で異なる」→ レポート#7の注意通り原典でも押し基準MAにより
  分岐するが、本Stage1では「押し安値(lookback本)-ATR×0.1の外側で統一」に簡略化する
  (ユーザー事前登録仕様の指定通り)。TP=SL幅×1.5。

先読み規律: EMAは確定バーまでのewm。Fractalはshift(1)〜shift(4)の自分の過去参照のみ
(上記の通りバーi確定時点で判明)。sl_price/tp_priceは信号バー確定時点で既知の
rolling(lookback)安値/高値とATRから計算し、シグナル列と一緒にengine側でshift(1)
されるため実際の使用は信号バーの次バー(先読みなし)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema
from app.core.strategy_model import Strategy


def _signal_ema3_fractal(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ema_fast = max(2, int(p["ema_fast"].value))    # 20
    ema_mid = max(2, int(p["ema_mid"].value))       # 50
    ema_slow = max(2, int(p["ema_slow"].value))     # 100
    touch_lookback = max(1, int(p["touch_lookback"].value))  # 5
    sl_lookback = max(2, int(p["sl_lookback"].value))  # 10
    sl_buffer_atr = float(p["sl_buffer_atr"].value)
    rr = float(p["rr"].value)
    side = int(p["side"].value)  # 0=both, 1=long_only, -1=short_only

    h, l, c = df["high"], df["low"], df["close"]
    e_f = ema(c, ema_fast)
    e_m = ema(c, ema_mid)
    e_s = ema(c, ema_slow)
    a = atr(h, l, c, 14)

    stack_up = (e_f > e_m) & (e_m > e_s)
    stack_dn = (e_f < e_m) & (e_m < e_s)

    touched_up = ((l <= e_f) | (l <= e_m)).rolling(touch_lookback).max().fillna(0) > 0
    touched_dn = ((h >= e_f) | (h >= e_m)).rolling(touch_lookback).max().fillna(0) > 0

    # Williams Fractal(標準5本窓、中央=i-2、確定=i時点)。docstring参照。
    lo2 = l.shift(2)
    up_fractal = (lo2 < l.shift(4)) & (lo2 < l.shift(3)) & (lo2 < l.shift(1)) & (lo2 < l)
    hi2 = h.shift(2)
    down_fractal = (hi2 > h.shift(4)) & (hi2 > h.shift(3)) & (hi2 > h.shift(1)) & (hi2 > h)

    long_sig = (stack_up & touched_up & up_fractal).fillna(False)
    short_sig = (stack_dn & touched_dn & down_fractal).fillna(False)
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
    "ema3_fractal",
    defaults={
        "ema_fast": 20.0,
        "ema_mid": 50.0,
        "ema_slow": 100.0,
        "touch_lookback": 5.0,
        "sl_lookback": 10.0,
        "sl_buffer_atr": 0.1,
        "rr": 1.5,
        "side": 0.0,
        "sl_pips": 15.0,
        "tp_pips": 22.5,
        "max_hold_bars": 96.0,
        "lot": 0.1,
    },
    signal_fn=_signal_ema3_fractal,
)
