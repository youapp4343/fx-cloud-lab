"""キリ番オーバーシュート逆張り(手法#15/#16)テンプレート。USDJPY想定。

ラウンドナンバー(キリ番)は指値/逆指値が集中しやすく、一時的に突き抜けたのち
反発しやすいという市場のアノマリーに基づく。当バー終値に最も近いキリ番
rn = round(close/grid_price)*grid_price を求め、当バーの高値/安値がその rn を
overshoot_pips 分だけ突き抜けた(オーバーシュート)のち、終値がその突き抜け水準の
内側へ戻った(=行き過ぎの否定)バーを逆張りのトリガーとする。

grid_price は価格単位そのもの(JPYペア既定0.5=50pips刻み、.00/.50ライン)。pipは
symbolが'JPY'で終わるかの簡易判定(0.01/0.0001、ema10_easy等と同じ簡易規約。
app/core/symbols.pyの正典とは非JPY特例で不一致だが対象通貨では実用上十分)。
非JPYペアで使う場合はgrid_priceを対象ペアのpipスケールに合わせて明示指定すること
(既定値0.5はUSDJPY想定であり、他ペアにそのまま使うと刻み幅が不適切になる)。

近似(重要): 原手法は本来ティック単位の値動きに対する裁量判断だが、本テンプレは
OHLC確定足(high/low/closeのみ)で近似する。「オーバーシュート→反転」を1本のバー内の
値幅(高安と終値の関係)で代理しており、実際のティックでの突き抜けとその後の反転の
順序(バー内のどちらが先に付いたか)まではOHLCから再現できない。

再アーム: rn付近に張り付いて毎バー条件成立し続ける連続発火を防ぐため、
raw & ~raw.shift(1) で成立した最初のバーのみ採用する(turtle_soup/liquidity_sweepと
同じ「1レベルにつき1回」の近似規律。同一rnに張り付いたままの連続発火をこれで抑える。
rnがバーをまたいで変化した場合も、条件が一度Falseに落ちれば自然に再アームされる)。

先読み規律: rn/オーバーシュート判定は当バー自身のOHLCのみから計算し未来参照なし。
再アームの.shift(1)は自分の過去バーとの比較のみ。翌バー始値執行のshiftはengine側が行う。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_roundnum_fade(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    grid_price = max(1e-9, float(p["grid_price"].value))  # 既定0.5(USDJPY 50pips刻み)。0/負値でのクラッシュ防止
    overshoot_pips = float(p["overshoot_pips"].value)      # 既定10
    dir_mode = int(p["dir_mode"].value)                    # 0=both,1=long,2=short

    pip = 0.01 if strategy.symbol.upper().endswith("JPY") else 0.0001
    overshoot_dist = overshoot_pips * pip

    high, low, close = df["high"], df["low"], df["close"]
    rn = (close / grid_price).round() * grid_price  # 当バー終値に最も近いキリ番

    upper_th = rn + overshoot_dist
    lower_th = rn - overshoot_dist

    short_raw = ((high >= upper_th) & (close < upper_th)).fillna(False)  # 上に突き抜けて戻り → 売り
    long_raw = ((low <= lower_th) & (close > lower_th)).fillna(False)    # 下に突き抜けて戻り → 買い

    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "roundnum_fade",
    defaults={
        "grid_price": 0.5, "overshoot_pips": 10.0, "dir_mode": 0.0,
        "sl_pips": 15.0, "tp_pips": 25.0, "lot": 0.1,
    },
    signal_fn=_signal_roundnum_fade,
)
