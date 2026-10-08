"""リクイディティ・スイープ逆張り テンプレート(SMC/ICT「manipulationの終わり」の機械化)。

出典: ICT Power of 3(Accumulation→Manipulation→Distribution)/ SMC。ALCHEMIST等の
「ブレイクを追わず、manipulation(=ストップ狩り)が終わった場所で仕掛ける」という主張の、
機械化できる唯一の核 = *直近スイング高安を一瞬抜いて(=流動性を刈って)即反転・回帰した
リジェクション* を実装する。OB/IDM/構造判定などの裁量ラベリングは機械化不能なので捨てる。

ロジック:
- prior_high/prior_low = donchian(直近lookback本の高安、内部shift(1)=当バー除外で因果的)。
- 売り(上の流動性を刈って失敗): 当バー high>prior_high(スイープ) かつ close<prior_high(回帰)
  かつ 上ヒゲ拒否(close が当バーレンジ下部= (close-low)/(high-low) <= 1-rej_frac)→ -1。
- 買い(下の流動性を刈って失敗): low<prior_low かつ close>prior_low かつ 下ヒゲ拒否
  ((close-low)/(high-low) >= rej_frac)→ +1。
これは「ブレイク失敗の逆張り」(turtle_soup と同系)に *ヒゲ拒否* を足してSMCの
「スイープ&リクレイム」に忠実化したもの。ALCHEMISTはXAUUSD特化なので既定SL/TPは
ゴールドのスケール寄り(検証スクリプト側で銘柄別に上書きする)。

先読み規律: donchianは当バー除外で因果的、判定は当バーOHLC+自分の過去のみ。再アームの
.shift(1)は過去参照のみ。執行shiftはengine側、テンプレ内shiftなし。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import donchian_channel
from app.core.strategy_model import Strategy


def _signal_liquidity_sweep(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    lookback = max(3, int(p["lookback"].value))
    rej_frac = float(p["rej_frac"].value)   # ヒゲ拒否の強さ(0.5=実体が半分より上/下)
    dir_mode = int(p["dir_mode"].value)     # 0=both,1=long,2=short

    high, low, close = df["high"], df["low"], df["close"]
    prior_high, prior_low = donchian_channel(high, low, lookback)
    rng = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / rng          # 0=安値引け,1=高値引け

    swept_low = (low < prior_low) & (close > prior_low)      # 下を刈って回帰
    swept_high = (high > prior_high) & (close < prior_high)  # 上を刈って回帰
    long_raw = (swept_low & (close_pos >= rej_frac)).fillna(False)          # 下ヒゲ拒否
    short_raw = (swept_high & (close_pos <= (1.0 - rej_frac))).fillna(False)  # 上ヒゲ拒否
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "liquidity_sweep",
    defaults={
        "lookback": 20.0, "rej_frac": 0.5, "dir_mode": 0.0,
        "sl_pips": 200.0, "tp_pips": 300.0, "lot": 0.1,  # 既定はゴールドスケール寄り
    },
    signal_fn=_signal_liquidity_sweep,
)
