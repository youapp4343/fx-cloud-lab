"""サポート/レジスタンスでの反発ローソク足 テンプレート(X教育インフォグラフィックの機械化)。

出典: X @FxTraderCourse のGOLD解説画像4枚。①大きな流れ(HH/HL・Fib)②調整か転換か
(チャネル)は裁量の「見極め」で機械化不能(公開システム17手法の結論と同じ核)。機械化できる
のは③④のみ:
  ③ 反発サイン = ピンバー / 包み足 / ハラミ / だまし抜け
  ④ シナリオ = 反発足を直近安値サポートで確認 → ロング、SL=直近安値、TP=直近高値
本テンプレはこの核 = *直近レンジ安値(サポート)で強気反発足 → 順張り(上位トレンド整合)* を実装。
狙いは「反発足単体では優位が無い(candle_pattern既検証)ものを、サポート近接 + トレンド文脈で
ゲートすれば救えるか」= 記事が主張する『文脈(見極め)こそがエッジ』仮説の直接検証。

サポート/レジスタンスは donchian_channel(直近lookback本の高安、内部で shift(1) 済み=因果的)で
近似する。detect_swings は order 本先読みで pivot 確定するため、その確定ラグを厳密に扱わないと
先読みになる。donchian は当バーを含まない過去N本のmax/minなので構造的に先読みが無く安全。
「直近安値でのSL/直近高値でのTP」は本Stage1では固定 sl_pips/tp_pips で近似(エントリーの優位
判定が目的)。構造的SL/TP(sl_price/tp_price列)はStage2の忠実版で扱う。

先読み規律: signal_fnはshiftしない生シグナル。反発足・donchian・SMAはすべて確定バー+自分の
過去のみ。再アームの .shift(1) は自分の過去参照で先読みでない。執行shiftはengine側。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, sma
from app.core.patterns import (
    bearish_engulfing,
    bullish_engulfing,
    inside_bar,
    pin_bar_bearish,
    pin_bar_bullish,
)
from app.core.strategy_model import Strategy


def _bool(s: pd.Series, index) -> pd.Series:
    return s.reindex(index).fillna(False).astype(bool)


def _signal_reversal_at_support(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    lookback = max(3, int(p["lookback"].value))            # 直近安値/高値の参照本数
    support_atr = float(p["support_atr"].value)            # サポート近接許容(×ATR)
    atr_period = max(2, int(p["atr_period"].value))
    use_trend = int(p["use_trend"].value)                  # 1=上位トレンド整合を要求
    trend_period = max(2, int(p["trend_period"].value))
    dir_mode = int(p["dir_mode"].value)                    # 0=both,1=long,2=short

    high, low, close = df["high"], df["low"], df["close"]
    prior_low = low.shift(1).rolling(lookback).min()       # 直近lookback本の安値(当バー除外=因果的)
    prior_high = high.shift(1).rolling(lookback).max()
    band = support_atr * atr(high, low, close, atr_period)

    idx = df.index
    bull_rev = (_bool(pin_bar_bullish(df), idx) | _bool(bullish_engulfing(df), idx)
                | _bool(inside_bar(df), idx))              # ハラミ=inside_barで代理(文脈で方向決定)
    bear_rev = (_bool(pin_bar_bearish(df), idx) | _bool(bearish_engulfing(df), idx)
                | _bool(inside_bar(df), idx))

    near_support = (low <= prior_low + band).fillna(False)     # 直近安値サポート帯に接触
    near_resist = (high >= prior_high - band).fillna(False)    # 直近高値レジ帯に接触

    if use_trend:
        trend = sma(close, trend_period)
        up_ok = (close > trend).fillna(False)
        dn_ok = (close < trend).fillna(False)
    else:
        up_ok = dn_ok = pd.Series(True, index=idx)

    long_raw = (bull_rev & near_support & up_ok).fillna(False)
    short_raw = (bear_rev & near_resist & dn_ok).fillna(False)
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))   # 帯張り付き連発を抑制
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=idx, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "reversal_at_support",
    defaults={
        "lookback": 20.0, "support_atr": 0.5, "atr_period": 14.0,
        "use_trend": 1.0, "trend_period": 50.0, "dir_mode": 0.0,
        "sl_pips": 25.0, "tp_pips": 50.0, "lot": 0.1,
    },
    signal_fn=_signal_reversal_at_support,
)
