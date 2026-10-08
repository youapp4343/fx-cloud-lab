"""fractal_rsi50_fixed10(EMA21/50/200+RSI50ライン+Williams Fractal, TP固定10pips)
テンプレート(YouTube解析レポート set2 #4由来)。

出典: scripts/scalp_set2_prereg.md T3。既存 ema3_fractal_template.py との違い:
EMA期間21/50/200(既存は20/50/100)、RSIフィルタが50ライン基準(既存はMAタッチ+乖離
条件)、TP=10pips絶対固定(既存はSL幅×RR)。M1+M5(2TF、テンプレ自体はTF非依存で
6ペア×2TF=12検定として渡された時間足データをそのまま使う)。

条件(バーi確定情報のみ):
- EMA21>EMA50>EMA200の順配列(ロング) かつ RSI(14)>=50
  ショートはミラー: EMA21<EMA50<EMA200 かつ RSI(14)<=50
- 確定Williams Fractal(標準5本窓、中央=i-2、2本ラグを正しく反映)。
  ロングは上向きFractal(安値の極小=反発足)、ショートは下向きFractal(高値の極大)。
  判定式はema3_fractal_template.pyのdocstringで検証済みのものをそのまま流用する
  (bar[i-2]が[i-4..i]の中央最小/最大 ⇔ バーi確定時点で初めて真偽が決まる=先読みなし)。
- SL=フラクタルバー(=中央bar[i-2])の安値/高値。lo2=l.shift(2)/hi2=h.shift(2)は
  バーi時点で既に確定済みの値のため、そのままsl_price列に使える。
- TP=10pips絶対固定(tp_pipsパラメータのみ指定、tp_price列は返さない)。

先読み規律: EMA/RSIは確定バーまでのewm。Fractalはshift(1)〜shift(4)の自分の過去
参照のみ。sl_priceはシグナル列と一緒にengine側でshift(1)されるため、実際の使用は
信号バーの次バー(先読みなし)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import ema, rsi
from app.core.strategy_model import Strategy


def _signal_fractal_rsi50_fixed10(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ema_fast = max(2, int(p["ema_fast"].value))    # 21
    ema_mid = max(2, int(p["ema_mid"].value))       # 50
    ema_slow = max(2, int(p["ema_slow"].value))     # 200
    rsi_period = max(2, int(p["rsi_period"].value))  # 14
    side = int(p["side"].value)

    h, l, c = df["high"], df["low"], df["close"]
    e_f = ema(c, ema_fast)
    e_m = ema(c, ema_mid)
    e_s = ema(c, ema_slow)
    r = rsi(c, rsi_period)

    stack_up = (e_f > e_m) & (e_m > e_s)
    stack_dn = (e_f < e_m) & (e_m < e_s)

    # Williams Fractal(標準5本窓、中央=i-2、確定=i時点)。ema3_fractal_template.py参照。
    lo2 = l.shift(2)
    up_fractal = (lo2 < l.shift(4)) & (lo2 < l.shift(3)) & (lo2 < l.shift(1)) & (lo2 < l)
    hi2 = h.shift(2)
    down_fractal = (hi2 > h.shift(4)) & (hi2 > h.shift(3)) & (hi2 > h.shift(1)) & (hi2 > h)

    long_sig = (stack_up & (r >= 50.0) & up_fractal).fillna(False)
    short_sig = (stack_dn & (r <= 50.0) & down_fractal).fillna(False)
    if side > 0:
        short_sig &= False
    elif side < 0:
        long_sig &= False

    n = len(df)
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    li = long_sig.to_numpy()
    si = short_sig.to_numpy()
    lo2a = lo2.to_numpy()
    hi2a = hi2.to_numpy()

    sig[li] = 1
    sig[si] = -1
    slp[li] = lo2a[li]   # ロング: フラクタルバー(i-2)の安値
    slp[si] = hi2a[si]   # ショート: フラクタルバー(i-2)の高値

    return pd.DataFrame({"signal": sig, "sl_price": slp}, index=df.index)


templates.register(
    "fractal_rsi50_fixed10",
    defaults={
        "ema_fast": 21.0,
        "ema_mid": 50.0,
        "ema_slow": 200.0,
        "rsi_period": 14.0,
        "side": 0.0,
        "sl_pips": 15.0,   # フォールバック(sl_price無効時のみ使用)
        "tp_pips": 10.0,   # 固定TP(事前登録どおり)
        "max_hold_bars": 200.0,   # 安全弁(事前登録に明記なし)
        "lot": 0.1,
    },
    signal_fn=_signal_fractal_rsi50_fixed10,
)
