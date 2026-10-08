"""rollover_marubozu(時間帯クラスタリング仮説)テンプレート(YouTube解析レポート set2 #10由来)。

出典: scripts/scalp_set2_prereg.md T1。「H1/M30/M15の足切り替わり直後、坊主頭
(実体側に髭がほぼ無い)の足が出たらその方向にモメンタムが続きやすい」という、
EMA配列+押し目クラスとは別系統の仮説(時刻構造×値動きの複合)。

M1データ前提。境界は毎時00分(H1)/00,30分(M30)/00,15,30,45分(M15)の3種、
すべてUTC絶対時刻でTZ変換は行わない(データのtimestampがそのままUTC)。
本テンプレは1本でboundary_mode(0=H1,1=M30,2=M15)を切り替えて3インスタンス化する
(事前登録どおり境界ごとに別テンプレインスタンスとして6ペア×3境界=18検定)。

坊主頭定義(数値化、事前登録どおり固定・スイープ対象外):
  レンジ = high-low、実体 = |close-open|
  エントリー方向側の髭(=陽線なら上ヒゲ high-close、陰線なら下ヒゲ close-low)
  <= レンジの10%、かつ 実体 >= レンジの60%。
  陽線坊主頭→ロング、陰線坊主頭→ショート(色で方向が決まるため side フィルタは
  デフォルト無効=0、両方向発火)。

境界判定: そのバー自身のtimestamp.minuteが境界集合に属する(=足切り替わり直後の
最初のM1バーそのもの)。バー確定時点の自分自身の情報のみで判定できるため先読みなし。
エントリー: 坊主頭バー確定 → engine側のshift(1)で自動的に次バー始値エントリーになる
(本テンプレでは追加のshiftを行わない、既存テンプレと同じ規律)。

TP=3pips固定(tp_pipsパラメータのみ指定、tp_price列は返さない)。
SL=坊主頭バーの反対側極値(ロング=そのバーのlow、ショート=そのバーのhigh)を
sl_price列で返す(信号バー自身の値、engine側の1回のshiftでバーi確定情報→
次バー執行という既存の先読み回避パターンに従う)。max_hold_bars=30(M1で30分)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _boundary_mask(minute: pd.Series, boundary_mode: int) -> pd.Series:
    if boundary_mode <= 0:  # H1: 毎時00分
        return minute == 0
    if boundary_mode == 1:  # M30: 00,30分
        return minute.isin([0, 30])
    return minute.isin([0, 15, 30, 45])  # M15: 00,15,30,45分


def _signal_rollover_marubozu(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    boundary_mode = int(round(float(p["boundary_mode"].value)))
    wick_max_frac = float(p["wick_max_frac"].value)   # 0.10 固定
    body_min_frac = float(p["body_min_frac"].value)    # 0.60 固定
    side = int(p["side"].value)  # 0=both, 1=long_only, -1=short_only

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    body = (c - o).abs()
    has_range = rng > 0

    bull = c > o
    bear = c < o
    up_wick = h - c   # 陽線: エントリー方向(上)側の髭
    dn_wick = c - l   # 陰線: エントリー方向(下)側の髭

    marubozu_long = bull & has_range & (up_wick <= wick_max_frac * rng) & (body >= body_min_frac * rng)
    marubozu_short = bear & has_range & (dn_wick <= wick_max_frac * rng) & (body >= body_min_frac * rng)

    minute = df["timestamp"].dt.minute
    boundary = _boundary_mask(minute, boundary_mode)

    long_sig = (marubozu_long & boundary).fillna(False)
    short_sig = (marubozu_short & boundary).fillna(False)
    if side > 0:
        short_sig &= False
    elif side < 0:
        long_sig &= False

    n = len(df)
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    li = long_sig.to_numpy()
    si = short_sig.to_numpy()
    la = l.to_numpy()
    ha = h.to_numpy()

    sig[li] = 1
    sig[si] = -1
    slp[li] = la[li]   # ロング: 坊主頭バー自身のlow(反対側極値)
    slp[si] = ha[si]   # ショート: 坊主頭バー自身のhigh(反対側極値)

    return pd.DataFrame({"signal": sig, "sl_price": slp}, index=df.index)


templates.register(
    "rollover_marubozu",
    defaults={
        "boundary_mode": 0.0,   # 0=H1(毎時00分) / 1=M30(00,30分) / 2=M15(00,15,30,45分)
        "wick_max_frac": 0.10,
        "body_min_frac": 0.60,
        "side": 0.0,
        "sl_pips": 15.0,   # フォールバック(sl_price無効時のみ使用)
        "tp_pips": 3.0,    # 固定TP(事前登録どおり)
        "max_hold_bars": 30.0,
        "lot": 0.1,
    },
    signal_fn=_signal_rollover_marubozu,
)
