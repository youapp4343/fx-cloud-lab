"""浅い決定木からif-thenルールを抽出し、戦略テンプレート化する(Phase8: ML特徴量スキャン, Layer B)。

app/core/discovery.py と同じ誠実性の設計思想: sklearnモデルそのものを戦略シグナルとして直接
使うこと(`ml_signal`のようなテンプレート)はしない(EA化不能・過学習の温床のため)。代わりに
浅い決定木(max_depth<=3程度)から葉ノードの条件を抽出し、明示的なif-thenルールとして人間が
検証・EA化できる形に昇格させる。

extract_tree_rulesが返す条件は特徴量×不等号×しきい値のAND連結(可変長)だが、既存の
StrategyParamはfloat限定のフラットなパラメータしか表現できないため、複数条件をそのまま
テンプレート化するのはparamsスキーマの制約上困難。そのため`feature_rule`テンプレートは
「最重要と判定された単一条件」に単純化したv1として実装する(複数条件AND対応はスキーマ拡張が
必要な将来課題)。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from app.core import templates
from app.core.features import FEATURE_NAMES, build_features
from app.core.strategy_model import Strategy


def extract_tree_rules(
    features: pd.DataFrame, target: pd.Series, feature_names: list[str], max_depth: int = 3
) -> list[dict]:
    """浅いDecisionTreeClassifierを学習し、葉ノードごとのif-thenルールを抽出する。

    targetは二値化して学習する(target > 0 を1、target <= 0 を0)。NaN行(特徴量またはtargetが
    欠損)は学習前にdropnaで除外する。各葉について、根からの分岐条件をAND連結したもの・
    その葉の多数派クラス・落ちた訓練サンプル数・的中率(多数派クラスの割合)を集める。

    戻り値: [{"conditions": [{"feature":str,"op":"<="or">","threshold":float}, ...],
              "predicted_class": 1 or 0, "n_samples": int, "precision": float}, ...]
    precision降順。n_samplesが全体の1%未満の葉は偶然のノイズに乗った不安定な分岐である
    可能性が高く信頼できないため除外する。
    """
    combined = pd.concat([features, target.rename("__target__")], axis=1).dropna()
    if len(combined) == 0:
        return []
    feature_names = list(feature_names)
    X = combined[feature_names]
    y = (combined["__target__"] > 0).astype(int)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    tree = clf.tree_
    classes = clf.classes_

    min_samples = len(X) * 0.01  # 全体の1%未満の葉は信頼できないため除外する

    rules: list[dict[str, Any]] = []

    def _walk(node_id: int, conditions: list[dict[str, Any]]) -> None:
        if tree.children_left[node_id] == -1:  # 葉ノード(sklearn規約: 葉はchildren_left=-1)
            n_samples = int(tree.n_node_samples[node_id])
            if n_samples < min_samples:
                return
            class_counts = tree.value[node_id][0]
            best_idx = int(np.argmax(class_counts))
            precision = float(class_counts[best_idx] / class_counts.sum())
            rules.append(
                {
                    "conditions": conditions,
                    "predicted_class": int(classes[best_idx]),
                    "n_samples": n_samples,
                    "precision": precision,
                }
            )
            return

        feature_name = feature_names[tree.feature[node_id]]
        threshold = float(tree.threshold[node_id])
        _walk(
            tree.children_left[node_id],
            conditions + [{"feature": feature_name, "op": "<=", "threshold": threshold}],
        )
        _walk(
            tree.children_right[node_id],
            conditions + [{"feature": feature_name, "op": ">", "threshold": threshold}],
        )

    _walk(0, [])
    rules.sort(key=lambda r: r["precision"], reverse=True)
    return rules


def _signal_feature_rule(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """単一の特徴量条件をif-thenルールとして売買シグナルに変換する(v1: 単一条件のみ対応)。

    feature_index: FEATURE_NAMES(build_features()の固定列順序リスト)へのインデックス
    (app.core.candle_template のpattern_indexと同じ設計: range/step指定でのグリッドサーチ対象に
    するため、floatパラメータでカテゴリ変数を表現する)。
    direction: 1なら「feature > threshold で買い、feature <= threshold で売り」、
    -1なら符号反転(「feature > threshold で売り、feature <= threshold で買い」)。
    NaN(ウォームアップ期間等で特徴量が未確定)の行はシグナル0(取引なし)とする。
    """
    feature_index = int(strategy.params["feature_index"].value)
    feature_index = max(0, min(feature_index, len(FEATURE_NAMES) - 1))  # range探索でのout-of-rangeを端にクリップ
    feature_name = FEATURE_NAMES[feature_index]
    threshold = float(strategy.params["threshold"].value)
    direction = float(strategy.params["direction"].value)

    col = build_features(df)[feature_name]
    valid = col.notna()
    above = col > threshold

    base_signal = pd.Series(0, index=df.index, dtype=int)
    base_signal[valid & above] = 1
    base_signal[valid & ~above] = -1

    direction_sign = 1 if direction > 0 else -1
    return base_signal * direction_sign


templates.register(
    "feature_rule",
    defaults={
        "feature_index": 0.0,
        "threshold": 0.0,
        "direction": 1.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_feature_rule,
)
