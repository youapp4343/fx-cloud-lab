"""TDIクロス テンプレート(裁量トレーダー記事の機械化シリーズ 手法#7)。

出典: Forex Factory "Trading Made Simple"(TDI: Traders Dynamic Index)。price_line(緑)が
signal_line(赤)を上抜けたら買い/下抜けたら売り、という合議系オシレーターのクロス手法。
記事本体では「緑線の傾き(時計の12-2時/4-6時方向)」「平均足の色」「ストキャス方向」等の
裁量条件が重なるが、機械化できる核 = *TDIクロス + 市場ベースライン(50 or market_base)を
挟むゲート* だけを実装する。傾き・平均足・「クロス後1-2本目のみ」等の裁量は捨てる。

記事の重大注意点(リペイント): 配布版TDI指標には過去線が後から書き換わる版があり、
遡って見るときれいなクロスに見える。本テンプレートの `indicators.tdi` は確定バーのRSIから
rolling計算する純粋関数でリペイントしない(indicators.py の tdi docstring 参照)。

先読み規律: signal_fnはshiftしない生シグナル(1/-1/0)を返す。翌バー始値執行のshiftは
engine.run_backtest側が担う。クロス検出のための `.shift(1)` は自分の過去バーのみを参照
するため先読みではない(ma_pullback/macd_cross と同じ流儀)。ウォームアップNaNは比較で
Falseに落ち無シグナルとなる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import tdi
from app.core.strategy_model import Strategy


def _signal_tdi_cross(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    rsi_period = max(2, int(p["rsi_period"].value))
    price_period = max(1, int(p["price_period"].value))
    signal_period = max(1, int(p["signal_period"].value))
    band_period = max(2, int(p["band_period"].value))
    band_mult = float(p["band_mult"].value)
    gate = int(p["gate"].value)          # 0=なし, 1=50線(RSI中立)基準, 2=market_base(BB中央)基準
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    green, red, base, _upper, _lower = tdi(
        df["close"], rsi_period, price_period, signal_period, band_period, band_mult
    )
    gp, rp = green.shift(1), red.shift(1)
    cross_up = (green > red) & (gp <= rp)
    cross_dn = (green < red) & (gp >= rp)

    if gate == 1:
        long_gate, short_gate = green > 50.0, green < 50.0
    elif gate == 2:
        long_gate, short_gate = green > base, green < base
    else:
        long_gate = short_gate = pd.Series(True, index=df.index)

    long_sig = (cross_up & long_gate).fillna(False)
    short_sig = (cross_dn & short_gate).fillna(False)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "tdi_cross",
    defaults={
        "rsi_period": 13.0, "price_period": 2.0, "signal_period": 7.0,
        "band_period": 34.0, "band_mult": 1.618, "gate": 1.0, "dir_mode": 0.0,
        "sl_pips": 20.0, "tp_pips": 30.0, "lot": 0.1,
    },
    signal_fn=_signal_tdi_cross,
)
