"""ローソク足パターン検出器(app/core/patterns.py)を戦略テンプレート化する。

pattern_index は PATTERN_NAMES(PATTERN_REGISTRYのキーをソートした固定順序リスト)への
整数インデックスであり、StrategyParam.value が float 限定という制約下でパターン選択という
カテゴリ変数を表現するための設計。sl_pips/tp_pips/lot と同じ通常のfloatパラメータとして
UIに表示され、range/stepを指定すれば既存のグリッドサーチ/ランダムサーチの対象にできるため、
「どのローソク足パターンが最も優位性があるか」を横断的に検証できる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.patterns import PATTERN_REGISTRY
from app.core.strategy_model import Strategy

PATTERN_NAMES: list[str] = sorted(PATTERN_REGISTRY.keys())


def _direction_for(pattern_name: str) -> int:
    """パターン名から売買方向を判定する(1=買い, -1=売り, 0=方向不定で単体取引しない)。"""
    if "bullish" in pattern_name or pattern_name.endswith("_up"):
        return 1
    if "bearish" in pattern_name or pattern_name.endswith("_down"):
        return -1
    return 0


def _signal_candle_pattern(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    pattern_index = int(strategy.params["pattern_index"].value)
    pattern_index = max(0, min(pattern_index, len(PATTERN_NAMES) - 1))  # range探索でのout-of-rangeを端にクリップ
    pattern_name = PATTERN_NAMES[pattern_index]

    # direction_overrideは任意パラメータ(trailing_pips等と同じ流儀でdefaultsには含めない)。
    # 発掘パイプラインが「データが示す方向」とパターン名の命名規約が矛盾する場合に、
    # 命名規約由来の方向を明示的に上書きするために使う(app.core.discovery.promote_to_strategy参照)。
    override_param = strategy.params.get("direction_override")
    if override_param is not None and override_param.value != 0:
        direction = 1 if override_param.value > 0 else -1
    else:
        direction = _direction_for(pattern_name)

    signal = pd.Series(0, index=df.index, dtype=int)
    if direction == 0:
        return signal  # 中立パターン(inside_bar等)はoverride指定が無ければ単体では取引しない

    hits = PATTERN_REGISTRY[pattern_name](df)
    signal[hits.fillna(False)] = direction
    return signal


templates.register(
    "candle_pattern",
    defaults={
        "pattern_index": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_candle_pattern,
)
