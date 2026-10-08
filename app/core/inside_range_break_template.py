"""インサイドレンジ・ブレイク テンプレート(GOLD日足の値動き例の機械化)。

対象パターン: 目立つ上ヒゲピンバー → 数日レンジ内に収束(インサイド) → レンジ上限を
実体(終値)でブレイク。vol_squeeze_breakout_template.py がATRのpercentile rankで
ボラティリティの「スクイーズ→拡大」を検知するのに対し、本テンプレートは価格の高安
そのものが直近マザーレンジ内に収束しているかを直接見る、より価格アクション寄りの近縁手法。

近似・注記(Stage1、割り切り):
- 起点となる「目立つ上ヒゲのピンバー」は検出せず、マザーレンジ形成→収束→終値ブレイク
  という構造の核のみを実装する(ピンバー起点の厳密な絞り込みは省略)。
- 「上ヒゲを実体で否定」は終値ベースのブレイク(close > mother_high / close < mother_low)
  で近似する。高安(ヒゲ)がレンジを一瞬超えてもcloseがレンジ内に収まればブレイク未確定
  というロジックになる。
- マザーレンジ(range_lb本)・インサイド収束判定(inside_n本)はいずれも当バーを除外した
  過去N本(shift(1).rolling)で計算するcausalな定義(indicators.donchian_channelと同じ
  規律)。ブレイクの成否判定のみ当バーのcloseを使う(確定バーの終値を見るだけで先読みではない)。

先読み規律: signal_fnはshiftしない生シグナル。マザーレンジ・収束はすべて確定バー+
自分の過去のみ(shift(1).rolling)。再アームの`.shift(1)`は自分の過去参照で先読みでない。
執行shiftはengine側(engine.run_backtest内でactionable = raw_signal.shift(1))。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_inside_range_break(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    range_lb = max(2, int(p["range_lb"].value))   # マザーレンジの参照本数
    inside_n = max(1, int(p["inside_n"].value))   # 収束確認に要する直近本数
    dir_mode = int(p["dir_mode"].value)            # 0=both, 1=long, 2=short

    high, low, close = df["high"], df["low"], df["close"]

    # マザーレンジ: 当バーを除外した直近range_lb本の高安(donchian_channelと同じcausal定義)
    mother_high = high.shift(1).rolling(range_lb).max()
    mother_low = low.shift(1).rolling(range_lb).min()

    # インサイド収束: 当バーを除外した直近inside_n本の高安がすべてmotherレンジ内に
    # 収まっているか(=その期間の最高値がmother_high未満 かつ 最安値がmother_low超)
    inside_high = high.shift(1).rolling(inside_n).max()
    inside_low = low.shift(1).rolling(inside_n).min()
    converged = ((inside_high < mother_high) & (inside_low > mother_low)).fillna(False)

    # ブレイク: 当バーの終値がmotherレンジ上限/下限を実体で上抜け/下抜け(収束確認込み)
    breakout_up = (converged & (close > mother_high)).fillna(False)
    breakout_down = (converged & (close < mother_low)).fillna(False)

    # 再アーム: ブレイク状態が連続する間は初回バーのみシグナル化(張り付き連発を抑制)
    up_sig = breakout_up & (~breakout_up.shift(1).fillna(False))
    down_sig = breakout_down & (~breakout_down.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[up_sig] = 1
    if dir_mode in (0, 2):
        signal[down_sig] = -1
    return signal


templates.register(
    "inside_range_break",
    defaults={
        "range_lb": 10.0,
        "inside_n": 3.0,
        "dir_mode": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_inside_range_break,
)
