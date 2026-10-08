"""戦略テンプレートのレジストリ。

新しいテンプレートを追加する場合は、そのテンプレートを実装するモジュールの中で
`register()` を呼び、`app/core/__init__.py` にそのモジュールの import 文を追加する
(パッケージ初期化時に登録処理が確実に走るようにするため)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union

import pandas as pd

# シグナル関数は 1/-1/0 の pd.Series を返す(既存3テンプレートと同じ)か、
# "signal"列(必須)に加えて"sl_price"/"tp_price"/"confidence"等の任意列を持つ
# pd.DataFrame を返す(構造的SL/TPや信頼度ベースのロットサイジング向け、Phase4以降で利用)。
SignalResult = Union[pd.Series, pd.DataFrame]
SignalFn = Callable[[Any, pd.DataFrame], SignalResult]


@dataclass(frozen=True)
class TemplateSpec:
    defaults: Dict[str, float]
    signal_fn: SignalFn
    exportable: bool = True  # False の場合、MQL EA生成の対象から除外する(Phase3で使用)


_REGISTRY: Dict[str, TemplateSpec] = {}


def register(name: str, defaults: Dict[str, float], signal_fn: SignalFn, exportable: bool = True) -> None:
    _REGISTRY[name] = TemplateSpec(defaults=dict(defaults), signal_fn=signal_fn, exportable=exportable)


def get(name: str) -> TemplateSpec:
    if name not in _REGISTRY:
        raise ValueError(f"unknown template: {name}")
    return _REGISTRY[name]


def names() -> List[str]:
    return sorted(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def defaults_for(name: str) -> Dict[str, float]:
    return dict(get(name).defaults)
