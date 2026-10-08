"""ダブル・ボリンジャーバンド(Double BB)トレンドゾーン テンプレート。

出典: Kathy Lien らが広めた "Double Bollinger Bands" 手法。同一SMA(既定20)に対して
σ=1 と σ=2 の2組のバンドを描き、価格がどのゾーンにいるかでトレンドの強弱を判定する:
  - +1σ〜+2σ の帯 = 「買いゾーン(健全な上昇トレンド)」
  - -1σ〜-2σ の帯 = 「売りゾーン(健全な下降トレンド)」
  - -1σ〜+1σ の内側 = レンジ(新規建てしない中立ゾーン)
本テンプレートはこの核を機械化する。

レジーム/エントリー:
  - LONG: 終値が +1σ(u1)を上抜けて買いゾーンに入った最初のバー(再アーム、初回のみ)。
  - SHORT: 終値が -1σ(l1)を下抜けて売りゾーンに入った最初のバー(同上)。
  - 内側(l1..u1)では新規建てしない。close>u1 の間は買い専用ゾーン、close<l1 の間は
    売り専用ゾーンとなり、逆方向の新規は構造的に出ない(short は close<l1<u1 が必要なため)。

σ=2 バンド(u2/l2)の役割: 買い/売りゾーンの外縁。終値が既に +2σ を超えている(=1本で
2σ超の急伸)ような過伸張局面は「追いかけ」になるため新規建てを見送る(買いゾーンを
+1σ〜+2σ に限定)。原典の "candle body closes back inside" 的な過熱回避の主旨を、
バンド間ゾーン判定として取り込んだもの。これにより double_bb の名の通り2組のバンドが
共に判定に効く。

近似・簡略化(重要):
  - 決済ルール「ローソク実体の75%が +1σ を割り込んだら手仕舞い」は再現せず、engine 側の
    固定 sl_pips / tp_pips / max_hold_bars に単純化して委譲している(トレンド追随の
    トレーリング的手仕舞いは max_hold_bars による時間切れで近似)。

先読み規律: signal_fn はシフトしない生シグナル(1/-1/0)を返すのみ。翌バー始値執行の
shift は engine.run_backtest 側が担う。bollinger_bands はバー i までの過去 window のみを
使うためシフト不要。再アーム用 shift(1) は自分の過去バーのみ参照(先読みではない)。
ウォームアップ NaN は比較で False に落ち、シグナル無しとなる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import bollinger_bands
from app.core.strategy_model import Strategy


def _signal_double_bb(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    period = max(2, int(p["bb_period"].value))
    sigma1 = float(p["sigma1"].value)
    sigma2 = float(p["sigma2"].value)
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    close = df["close"]
    _, u1, l1 = bollinger_bands(close, period, sigma1)  # 内側バンド(±1σ)
    _, u2, l2 = bollinger_bands(close, period, sigma2)  # 外側バンド(±2σ)

    # 買い/売りゾーン = 内側バンドと外側バンドの間(健全なトレンド帯、過伸張は除外)。
    in_long_zone = (close > u1) & (close < u2)
    in_short_zone = (close < l1) & (close > l2)
    # 再アーム: ゾーンに入った最初のバーのみ発火。ゾーン内に留まる間の連続発火を防ぐ。
    long_sig = (in_long_zone & (~in_long_zone.shift(1).fillna(False))).fillna(False)
    short_sig = (in_short_zone & (~in_short_zone.shift(1).fillna(False))).fillna(False)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "double_bb",
    defaults={
        "bb_period": 20.0, "sigma1": 1.0, "sigma2": 2.0, "dir_mode": 0.0,
        "sl_pips": 50.0, "tp_pips": 100.0, "lot": 0.1, "max_hold_bars": 48.0,
    },
    signal_fn=_signal_double_bb,
)
