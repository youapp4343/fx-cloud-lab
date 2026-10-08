"""タートル・スープ+ワン テンプレート(20バーブレイクアウト「失敗」の逆張り)。

出典: Linda Bradford Raschke / Laurence Connors "Street Smarts" の "Turtle Soup" および
"Turtle Soup Plus One"。タートルズの20日ブレイクアウト順張りを逆手に取り、N=20バーの
新高値/新安値を一瞬つけた直後にレンジ内へ戻る=ブレイクの「失敗」を確認して、抜けた方向と
逆に張る短期反転手法。検証可能な核だけを機械化する:

- 買い: 当バーの安値が「直近20バーの最安値(当バー除く)」を下抜けして新安値をつけ、
  かつ当バーの終値がその直近20バー最安値以上へ戻した(=下抜け失敗)ら +1。
- 売り: 当バーの高値が「直近20バーの最高値(当バー除く)」を上抜けし、かつ当バーの終値が
  その直近20バー最高値以下へ戻した(=上抜け失敗)ら -1(買いの鏡)。

エントリー様式の近似(重要): 原法はブレイク失敗を確認したのち、直近安値/高値を再ブレイク
する逆指値〈ストップ〉で建てる。当エンジンは「シグナルの翌バー始値の成行」しか扱えないため、
セットアップを確認したバーに +1/-1 を出し、engineのshift(1)によって翌バー始値で執行する形で
近似する。これが原法の "Plus One"(翌足での押し戻りエントリー)に相当する近似である。

省略した選別条件: 原法の「ブレイクされた直近安値/高値は3本以上前に形成されたものであること
(=直近数本の連続した安値更新の途中ではないこと)」という条件は、pandasで綺麗に機械化しづらい
ため本テンプレートでは緩和(省略)している。そのぶんシグナルは増え、質は原法よりばらつく。
手仕舞いも原法の3系統(A:建値+α / B:トレーリング / C:時間切れ)をengineの sl/tp/max_hold_bars
へ単純化して集約している(近似)。

再アーム(連続発火抑制): レンジ外へ張り付く間の毎バー発火を防ぐため、raw & ~raw.shift(1) で
条件成立の最初のバーのみを採用する(ma_pullback等と同じ流儀)。

先読み規律: donchian_channel は内部で shift(1) 済み(=当バーを除く直近N本の最高値/最安値)で
あり、当バーの high/low/close とそのまま比較して先読みにならない。signal_fn は shift しない
生シグナル(1/-1/0)を返し、翌バー始値執行のための shift(1) は engine 側が担う。ウォームアップ
期間のNaNは比較で False となりシグナル無しになる(最終マスクは .fillna(False) 済み)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import donchian_channel
from app.core.strategy_model import Strategy


def _signal_turtle_soup(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    lookback = max(2, int(p["lookback"].value))
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    # donchian_channel は内部で shift(1) 済み → (直近lookback本の最高値, 最安値)。いずれも当バー除外。
    prior_high, prior_low = donchian_channel(df["high"], df["low"], lookback)

    # 買い: 当バー安値が直近最安値を下抜け(新安値)し、終値はその最安値以上へ戻す(下抜け失敗)
    long_raw = ((df["low"] < prior_low) & (df["close"] >= prior_low)).fillna(False)
    # 売り: 当バー高値が直近最高値を上抜けし、終値はその最高値以下へ戻す(上抜け失敗)
    short_raw = ((df["high"] > prior_high) & (df["close"] <= prior_high)).fillna(False)

    # 再アーム: 条件が続く間の連続発火を抑え、成立した最初のバーのみ発火させる
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "turtle_soup",
    defaults={
        "lookback": 20.0,
        "dir_mode": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 45.0,
        "max_hold_bars": 20.0,  # engineが自動認識(A/B/Cの時間切れ手仕舞いに相当)
        "lot": 0.1,
    },
    signal_fn=_signal_turtle_soup,
)
