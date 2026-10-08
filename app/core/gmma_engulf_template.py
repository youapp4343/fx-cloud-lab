"""GMMA(グッピー複合移動平均)押し目 + 大陽線包み足 反転買いテンプレート。

出典:「大陽線の包み足でGMMA押し目買い」系の解説記事。GMMAは短期群EMA[3,5,8,10,12,15]
と長期群EMA[30,35,40,45,50,60]の2群のリボンで、短期群は投機筋の勢い、長期群は
実需筋のトレンドを表すとされる。長期群が上向きに開いて並ぶ(=強いトレンド)局面で、
価格が一時的に長期群まで押し込まれたところへ「前バーの陰線を丸ごと飲み込む大陽線
(bullish engulfing)」が出た瞬間を反転買いのトリガーとする。

機械化した核(このテンプレの発火条件):
  - トレンド文脈: 長期群が上向き整列 = ema30 > ema60 かつ ema60 自体が上向き
    (ema60 > ema60.shift(slope_lb))。GMMA長期群6本すべての広がり判定ではなく、
    最も速い(30)・最も遅い(60)の2本の大小関係で「長期群が上向きに開いている」
    ことの代理とする(近似)。
  - 押し目(長期群への接触): 当バーの安値が長期群ゾーンへ到達 = low <= ema(close, zone_ema)。
    zone_ema既定30(長期群の最上端=浅い接触)。45等に上げると長期群の内側〜下端まで
    踏み込む深い押しに近似できる(パラメータ化)。
  - トリガー: app.core.patterns.bullish_engulfing(df)(前バーの陰線実体を当バーの陽線
    実体が完全に飲み込む)が当バーで確定。
  - SHORT は鏡像(下降整列 ema30<ema60 & ema60下向き、高値が長期群ゾーンへ到達、
    bearish_engulfing)。dir_modeは既定1(買いのみ、記事が買い限定のため)。

近似(要注記):
  - GMMA短期群EMA[3,5,8,10,12,15]は発火条件のゲートには使わない。本テンプレの
    「反転トリガー」は大陽線包み足という単一のプライスアクション条件で代替しており、
    短期群の収束・反転(モメンタム転換)を別条件として要求しない近似である。
  - SL=「包み足の安値」/「60EMA下」、TP=次の節目、はStage1では扱わずengineの
    固定sl_pips/tp_pipsで近似する(既定 sl_pips=30, tp_pips=60)。
  - 「長期群への深い接触」はema(zone_ema)という1本の代表線への到達で近似しており、
    長期群6本すべてへの到達具合(何本目まで刺さったか)は区別しない。

先読み規律: signal_fnはシフトしない生シグナル(1/-1/0のpd.Series)を返すのみ。全ての
EMAは確定バーcloseからのewm計算で当該バーまでの情報のみを使う。bullish_engulfing/
bearish_engulfingはapp.core.patternsの既存関数(前バーとの比較のみ、先読みなし)。
トレンド傾き判定のema60.shift(slope_lb)、再アームのraw & (~raw.shift(1))はいずれも
自分の過去参照のみで先読みでない。翌バー始値執行のシフトはengine.run_backtest側
(raw_signal.shift(1))が担うため、本テンプレート内では一切シフトしない。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import ema
from app.core.patterns import bearish_engulfing, bullish_engulfing
from app.core.strategy_model import Strategy


def _signal_gmma_engulf(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    zone_ema = max(2, int(p["zone_ema"].value))   # 押し目接触判定に使う長期群ゾーン線の期間(既定30=長期群上端)
    slope_lb = max(1, int(p["slope_lb"].value))   # ema60傾き判定の遡及本数
    dir_mode = int(p["dir_mode"].value)            # 0=both, 1=long, 2=short(記事は買い限定のため既定1)

    high, low, close = df["high"], df["low"], df["close"]

    ema30 = ema(close, 30)        # GMMA長期群の最速線(上端)
    ema60 = ema(close, 60)        # GMMA長期群の最遅線(下端)
    zone = ema(close, zone_ema)   # 押し目/戻り目の接触判定に使うゾーン線

    trend_up = (ema30 > ema60) & (ema60 > ema60.shift(slope_lb))  # 長期群上向き整列 & ema60自体も上向き
    trend_dn = (ema30 < ema60) & (ema60 < ema60.shift(slope_lb))  # 鏡像(下降整列)

    touch_up = low <= zone     # 押し目: 安値が長期群ゾーンへ到達
    touch_dn = high >= zone    # 戻り目: 高値が長期群ゾーンへ到達

    bull_engulf = bullish_engulfing(df)  # 前バー陰線を当バー陽線が丸ごと飲む大陽線
    bear_engulf = bearish_engulfing(df)

    long_raw = (trend_up & touch_up & bull_engulf).fillna(False)
    short_raw = (trend_dn & touch_dn & bear_engulf).fillna(False)
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))   # 再アーム: 条件成立の初回バーのみ発火
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "gmma_engulf",
    defaults={
        "zone_ema": 30.0, "slope_lb": 5.0, "dir_mode": 1.0,
        "sl_pips": 30.0, "tp_pips": 60.0, "lot": 0.1,
    },
    signal_fn=_signal_gmma_engulf,
)
