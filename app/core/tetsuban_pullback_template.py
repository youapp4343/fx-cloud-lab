"""鉄板パターン① 4時間足レベルの押し目・戻り目 テンプレート(247pマニュアルの機械化)。

出典: 「私が安定して利益を上げられるようになった4つの鉄板パターン」(100億円トレーダー)。
著者が最も得意とし『これだけで23億まで増やしたトレーダーがいる』と述べる看板パターン。
本質は多時間足20SMAトレンドフォローの押し目買い/戻り売りで、エントリーの核(p100-136)は:

  「上位足方向へ発生した押し目・戻り目の中で、下位足(執行足)が上位足と同じ方向に
   ダウ理論のトレンド転換/継続を確定させた瞬間(=直近スイング高値/安値の更新)で仕掛ける」

= 単なるMA接触(既存 ma_pullback=null)ではなく、押し目形成後の *ダウ継続ブレイク* が
トリガである点が差別化。マニュアルのMA設定(1時間足チャート: 青20 / 赤80=4時間足20SMA相当
/ 黄480=日足20SMA相当)をそのまま採用し、mid(80)=中期トレンド、slow(480)=大きな流れの
フィルタとする。パターン②(1時間足レベル)③(日足レベル)は同一機構をTF/MA階層で読み替えた
ものなので、本テンプレをTF・mid/slow期間ちがいで走らせれば同機構を検証できる。

近似(docstring明記): 「押し目形成」は直近pull_lb本で価格がmid帯へ接触したことで代理、
「ダウ継続トレンド転換」は直近dow_lb本のスイング高安の更新(donchian, 内部shift(1)=因果)で
代理する。SL=押し安値/戻り高値、TP=トレンド継続の構造的決済はStage1では固定pipsで近似。

先読み規律: SMA/ATR/donchianは確定バー+自分の過去のみ。donchianは当バー除外済みで
先読みなし。トリガの.shiftは自分の過去参照のみ。執行shiftはengine側、テンプレ内shiftなし。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, donchian_channel, sma
from app.core.strategy_model import Strategy


def _signal_tetsuban_pullback(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    mid_period = max(2, int(p["mid_period"].value))     # 既定80(=上位足20SMA相当)
    slow_period = max(2, int(p["slow_period"].value))   # 既定480(=大きな流れ20SMA相当)
    slope_lb = max(1, int(p["slope_lb"].value))         # mid傾きの参照本数
    pull_lb = max(1, int(p["pull_lb"].value))           # 押し目形成(mid帯接触)の遡及本数
    band_atr = float(p["band_atr"].value)               # mid帯の許容幅(×ATR)
    dow_lb = max(2, int(p["dow_lb"].value))             # ダウ継続ブレイクのスイング参照本数
    atr_period = max(2, int(p["atr_period"].value))
    dir_mode = int(p["dir_mode"].value)                 # 0=both,1=long,2=short

    high, low, close = df["high"], df["low"], df["close"]
    mid = sma(close, mid_period)
    slow = sma(close, slow_period)
    band = band_atr * atr(high, low, close, atr_period)
    prior_high, prior_low = donchian_channel(high, low, dow_lb)  # 当バー除外の直近スイング高安

    trend_up = (mid > mid.shift(slope_lb)) & (close > slow)       # 中期上昇 & 大きな流れ上
    trend_dn = (mid < mid.shift(slope_lb)) & (close < slow)
    pulled_up = low.rolling(pull_lb).min() <= (mid + band)        # 直近で押し目(mid帯へ接触)
    pulled_dn = high.rolling(pull_lb).max() >= (mid - band)       # 直近で戻り目
    break_up = high > prior_high                                  # ダウ継続: 直近高値更新
    break_dn = low < prior_low                                    # ダウ継続: 直近安値更新

    long_raw = (trend_up & pulled_up & break_up).fillna(False)
    short_raw = (trend_dn & pulled_dn & break_dn).fillna(False)
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))      # 連続ブレイク発火を1回に
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "tetsuban_pullback",
    defaults={
        "mid_period": 80.0, "slow_period": 480.0, "slope_lb": 10.0,
        "pull_lb": 20.0, "band_atr": 0.5, "dow_lb": 10.0, "atr_period": 14.0,
        "dir_mode": 0.0, "sl_pips": 40.0, "tp_pips": 80.0, "lot": 0.1,
    },
    signal_fn=_signal_tetsuban_pullback,
)
