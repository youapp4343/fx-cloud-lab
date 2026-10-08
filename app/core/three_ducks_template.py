"""Three Ducks(3羽のアヒル)テンプレート — MTF SMA60 アラインメントの単一系列近似。

出典: Forex Factory "3 Ducks Trading System"(Captain_Currency)。H4/H1/M15 の3つの時間足で
それぞれ SMA(close,60) を見て、価格が上位2足のSMA60より上(下)にあり、実行足でSMA60を
上抜け(下抜け)したらエントリーする、時間足の合意(アヒルが3羽一列)を要求する順張り手法。

単一系列MTF近似(重要): 本テンプレートは1つの時間足の df しか受け取らないため、上位足を
別途読まず、同一系列の *入れ子SMA* で階層を近似する:
- sma_fast = SMA(close, base_period)            … 実行足(自TF)相当。
- sma_mid  = SMA(close, base_period*mid_mult)   … 1つ上の時間足の SMA60 相当(既定 mid_mult=4)。
- sma_slow = SMA(close, base_period*slow_mult)  … 2つ上の時間足の SMA60 相当(既定 slow_mult=16)。
実TFがM15なら mid≒H1・slow≒H4 の SMA60 に概ね対応する。これは近似であり、実足の
確定タイミング・ギャップ・営業時間差は反映しない(真のMTFではない点に注意)。

- LONG:  close > sma_slow かつ close > sma_mid かつ close が sma_fast を上抜け。
- SHORT: 上記の完全なミラー。

先読み規律: signal_fn はshiftしない生シグナル(1/-1/0)を返す。翌バー始値執行のshiftは
engine.run_backtest 側が担う。sma_fastクロス検出の `.shift(1)` は自分の過去バーのみを参照
するため先読みではない。クロスは本質的に単発イベントのため追加の再アームは不要。
SMAウォームアップのNaNは比較でFalseに落ち無シグナルとなる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import sma
from app.core.strategy_model import Strategy


def _signal_three_ducks(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    base_period = max(2, int(p["base_period"].value))
    mid_mult = max(1.0, float(p["mid_mult"].value))
    slow_mult = max(1.0, float(p["slow_mult"].value))
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    mid_period = max(2, int(base_period * mid_mult))
    slow_period = max(2, int(base_period * slow_mult))

    close = df["close"]
    sma_fast = sma(close, base_period)
    sma_mid = sma(close, mid_period)
    sma_slow = sma(close, slow_period)

    # 実行足SMAのクロス検出(自分の過去バー参照)。
    prev_close, prev_fast = close.shift(1), sma_fast.shift(1)
    cross_up = (close > sma_fast) & (prev_close <= prev_fast)
    cross_dn = (close < sma_fast) & (prev_close >= prev_fast)

    long_sig = ((close > sma_slow) & (close > sma_mid) & cross_up).fillna(False)
    short_sig = ((close < sma_slow) & (close < sma_mid) & cross_dn).fillna(False)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "three_ducks",
    defaults={
        "base_period": 60.0, "mid_mult": 4.0, "slow_mult": 16.0, "dir_mode": 0.0,
        "sl_pips": 30.0, "tp_pips": 30.0, "lot": 0.1,
    },
    signal_fn=_signal_three_ducks,
)
