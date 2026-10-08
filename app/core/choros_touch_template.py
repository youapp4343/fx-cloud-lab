"""Choros/Dance MA初回タッチ テンプレート(裁量トレーダー記事の機械化シリーズ 手法#8)。

出典: Forex Factory "Choros"/"Dance"。10EMA/35SMA/50EMAの並びでトレンドを判定し、
上昇トレンドで価格が上からこれらMAへ戻った初回タッチで順張り買い(下降は逆)。掲示板では
「35SMAへの初回タッチは反発候補、2回目以降は反発力が落ちるので50EMAまで待つ」という
初回優位が使われる。機械化できる核 = *3本MAスタックでのトレンド定義 + 指定MAへの初回タッチ
再アーム* を実装する。ラウンドナンバー/上位足の重なり/日次利益目標での打ち切り等の裁量は捨てる。

ma_pullback と同じ構造(トレンドレジーム & MA帯タッチ & 張り付き連続発火の再アーム)を、
単線から3本MAスタックのトレンド確認に拡張したもの。押し目買いは約定価格が有利という
構造上の狙いも同じ。

先読み規律: signal_fnはshiftしない生シグナル(1/-1/0)。翌バー始値執行のshiftはengine側。
再アームの `.shift(1)` は自分の過去のみ参照で先読みでない。MA/ATRウォームアップのNaNは
比較でFalseに落ち、最終マスクを .fillna(False) してから代入する。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, sma
from app.core.strategy_model import Strategy


def _signal_choros_touch(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    fast_p = max(2, int(p["fast_ema"].value))    # 既定10EMA
    mid_p = max(2, int(p["mid_sma"].value))      # 既定35SMA
    slow_p = max(2, int(p["slow_ema"].value))    # 既定50EMA
    touch_ma = int(p["touch_ma"].value)          # 0=fast,1=mid(既定),2=slow のどのMAへのタッチで入るか
    atr_period = max(2, int(p["atr_period"].value))
    touch_atr_mult = float(p["touch_atr_mult"].value)
    dir_mode = int(p["dir_mode"].value)          # 0=both, 1=long_only, 2=short_only

    fast = ema(df["close"], fast_p)
    mid = sma(df["close"], mid_p)
    slow = ema(df["close"], slow_p)
    chosen = {0: fast, 1: mid, 2: slow}.get(touch_ma, mid)
    band = touch_atr_mult * atr(df["high"], df["low"], df["close"], atr_period)

    # スタック順(強いトレンド): 上昇=close>fast>mid>slow、下降=close<fast<mid<slow
    up_trend = (df["close"] > fast) & (fast > mid) & (mid > slow)
    dn_trend = (df["close"] < fast) & (fast < mid) & (mid < slow)
    low_touch = df["low"] <= (chosen + band)     # 上昇中に指定MA帯へ押し戻した
    high_touch = df["high"] >= (chosen - band)   # 下降中に指定MA帯へ戻した

    long_raw = (up_trend & low_touch).fillna(False)
    short_raw = (dn_trend & high_touch).fillna(False)
    # 初回タッチ: MA帯に張り付く間の連続発火を防ぎ、離れて戻った最初のバーのみ発火
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "choros_touch",
    defaults={
        "fast_ema": 10.0, "mid_sma": 35.0, "slow_ema": 50.0, "touch_ma": 1.0,
        "atr_period": 14.0, "touch_atr_mult": 0.5, "dir_mode": 0.0,
        "sl_pips": 25.0, "tp_pips": 50.0, "lot": 0.1,
    },
    signal_fn=_signal_choros_touch,
)
