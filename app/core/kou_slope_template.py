"""MA傾き順張り押し目テンプレート(@BassistFX_Kou「MAの傾きに沿ってポジションを建てる」の機械化)。

出典: @BassistFX_Kou のインフォグラフィック。核となる主張は「移動平均線の傾き(トレンド)
方向にのみ仕掛け、その傾きに沿って建てる」。原典は買い4種・売り4種のエントリー型を挙げる:
  (a)MAが横ばい→わずかに上向いた瞬間のクロス、(b)MA上向き中のクロス、
  (c)MalへのタッチからのMA反発、(d)MAからの乖離。
これらに共通する頑健な核 = 「傾きゲートを掛けたMA反発(押し目・戻り目)」だけを機械化する。
残る変種(横ばい→上向き転換のクロス、上向き中のクロス、乖離)は本テンプレの
「傾き上向き × MA帯タッチ × MA上へ引け戻し × 続伸」条件に畳み込まれる(近似として明記)。

核ロジック:
  ma = sma(close, ma_period)。傾き = ma と ma.shift(slope_lb) の大小。
  LONG: 傾き上向き AND 直近(当バーor前バー)で安値がMA帯(+band*ATR)へ押し戻り接触 AND
        当バーがMA上へ引け戻し(close>ma) AND 続伸(close>前close)。
  SHORT: 上記の鏡像(傾き下向き・高値がMA帯へ戻り接触・close<ma・続落)。
  再アーム: 反発の初回バーのみ発火(MA帯に張り付く間の連続発火を抑止)。

近似(docstring明記): 原典が触れる「ボラ拡大→収縮→再開のリズム」「損を限定しトレンドに乗る」
は、本Stage1では傾きゲート付きMA反発 + 固定SL/TP(sl_pips=30/tp_pips=60)で簡略化している。
MAタッチは「当バーまたは直前バーでの接触」で代理し、押し目の深さ・戻りの角度は問わない。

先読み規律: signal_fnはshiftしない生シグナル(1/-1/0のpd.Series)を返すのみ。翌バー始値執行の
shiftはengine.run_backtest側(raw_signal.shift(1))が担う。各バーは自バーOHLC + 自分の過去
(shift(1)/shift(slope_lb)で得る過去値)のみ参照し、未来バーは一切見ない。傾き比較・タッチの
当バーor前バー判定・続伸比較・再アームの.shiftは全て自分の過去参照。MA/ATRウォームアップの
NaNは比較でFalseに落ち(最終マスクは.fillna(False)後に代入)、シグナル無しとなる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, sma
from app.core.strategy_model import Strategy


def _signal_kou_slope(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ma_period = max(2, int(p["ma_period"].value))       # トレンド判定MAの期間
    slope_lb = max(1, int(p["slope_lb"].value))         # 傾きを測る遡及本数
    band_atr = float(p["band_atr"].value)               # MA帯の許容幅(×ATR)
    atr_period = max(2, int(p["atr_period"].value))
    dir_mode = int(p["dir_mode"].value)                 # 0=both, 1=long, 2=short

    high, low, close = df["high"], df["low"], df["close"]
    ma = sma(close, ma_period)
    band = band_atr * atr(high, low, close, atr_period)

    slope_up = ma > ma.shift(slope_lb)   # MA傾き上向き(トレンドフォロー方向)
    slope_dn = ma < ma.shift(slope_lb)   # MA傾き下向き

    touch_up = low <= (ma + band)        # 上昇中に安値がMA帯へ下から押し戻り接触
    touch_dn = high >= (ma - band)       # 下降中に高値がMA帯へ上から戻り接触
    # 当バーまたは直前バーでの接触(自分の過去参照のみ、先読みなし)
    touched_up = touch_up | touch_up.shift(1).fillna(False)
    touched_dn = touch_dn | touch_dn.shift(1).fillna(False)

    resumed_up = (close > ma) & (close > close.shift(1))  # MA上へ引け戻し & 続伸(反発再開)
    resumed_dn = (close < ma) & (close < close.shift(1))  # MA下へ引け戻し & 続落

    long_raw = (slope_up & touched_up & resumed_up).fillna(False)
    short_raw = (slope_dn & touched_dn & resumed_dn).fillna(False)
    # 再アーム: 反発の初回バーのみ発火(連続発火を1回に絞る)
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "kou_slope",
    defaults={
        "ma_period": 50.0, "slope_lb": 10.0, "band_atr": 0.5, "atr_period": 14.0,
        "dir_mode": 0.0, "sl_pips": 30.0, "tp_pips": 60.0, "lot": 0.1,
    },
    signal_fn=_signal_kou_slope,
)
