"""ML特徴量スキャン(Phase8: データ駆動戦略発掘の高度版、Layer B)。

app/core/discovery.py(Phase5)と同じ誠実性の設計思想を貫く: 相関係数(IC)やfeature importanceは
あくまで「示唆」であり、直接の売買根拠にはしない。sklearnモデルそのものを戦略シグナルとして
直接使うこと(EA化不能・過学習の温床になるため)はせず、モデルから浅い決定木のルールを抽出し
明示的なif-thenルールとして戦略に昇格させる(app.core.feature_rule_template.extract_tree_rules)。
このモジュールはIC計算とfeature importance計算という「発見の道具」のみを提供する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import TimeSeriesSplit


def compute_ic(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """各特徴量列について、targetとのSpearman順位相関係数(情報係数IC)を計算する。

    NaN行は特徴量×target両方揃っている行のみで計算する(列ごとに独立してdropna相当の
    マスクを適用する。特徴量ごとにNaNの発生期間が異なりうるため、全列一括dropnaより
    各列で使えるサンプルを最大限activateする)。

    誠実性ノート: 為替の短期予測におけるICは絶対値0.02程度でも大きい方であり、0.1を超えることは
    稀、というのが一般的な理解である。ここで算出されるIC値を過大評価しないこと。
    """
    records: list[dict[str, object]] = []
    for col in features.columns:
        valid = features[col].notna() & target.notna()
        x = features.loc[valid, col]
        y = target.loc[valid]
        if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
            ic = float("nan")  # 定数列・データ不足はSpearman相関が定義できない
        else:
            ic, _p = stats.spearmanr(x, y)
            ic = float(ic)
        records.append({"feature": col, "ic": ic, "abs_ic": float("nan") if np.isnan(ic) else abs(ic)})

    result = pd.DataFrame(records, columns=["feature", "ic", "abs_ic"])
    return result.sort_values("abs_ic", ascending=False, na_position="last").reset_index(drop=True)


def _purged_splits(n_samples: int, n_splits: int, embargo_bars: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplitの各分割から、訓練窓末尾embargo_bars本を除いた(train_idx, test_idx)を返す。

    sklearnのTimeSeriesSplitにはpurge/embargo機能が無いため自前実装する。時系列特徴量
    (リターンラグ・RSI等)は隣接バー間で自己相関を持つため、訓練直後のバーをそのまま検証に
    使うとリーク気味の楽観評価になる。TimeSeriesSplitのtrain_idxは常に0始まりの連続した
    昇順インデックス(位置)なので、末尾embargo_bars件をスライスで除くだけで
    「訓練窓の末尾と検証窓の先頭の間の空白」を作れる。
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in splitter.split(np.zeros(n_samples)):
        if embargo_bars > 0:
            train_idx = train_idx[:-embargo_bars] if len(train_idx) > embargo_bars else train_idx[:0]
        result.append((train_idx, test_idx))
    return result


def compute_feature_importance(
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    embargo_bars: int = 10,
    mode: str = "regression",
) -> dict:
    """時系列purged CVでfeature importanceを計算する。

    mode="regression"はRandomForestRegressor(スコア=R²)、mode="classification"は
    RandomForestClassifier(スコア=accuracy)を使う(targetがbuild_target(mode="direction")の
    ような離散値の場合はclassificationを指定する)。NaN行(特徴量またはtargetが欠損)は
    分割前にdropnaで除外する。分割・embargoは_purged_splits参照。

    戻り値: {"importances": DataFrame(columns=["feature","importance"], importance降順),
             "fold_scores": list[float], "mean_score": float, "n_folds": int}
    """
    if mode not in ("regression", "classification"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'regression' or 'classification')")

    combined = pd.concat([features, target.rename("__target__")], axis=1).dropna()
    feature_cols = list(features.columns)
    X = combined[feature_cols].reset_index(drop=True)
    y = combined["__target__"].reset_index(drop=True)

    importances_sum = np.zeros(len(feature_cols), dtype=float)
    fold_scores: list[float] = []

    for train_idx, test_idx in _purged_splits(len(X), n_splits, embargo_bars):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue  # embargoで訓練側が空になった極端な小データの分割はスキップする
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if mode == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))

        importances_sum += model.feature_importances_
        fold_scores.append(float(score))

    n_folds = len(fold_scores)
    if n_folds == 0:
        raise ValueError("有効な分割がありません(n_splits/embargo_barsに対しデータ数が不足しています)")

    importances = (
        pd.DataFrame({"feature": feature_cols, "importance": importances_sum / n_folds})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "importances": importances,
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "n_folds": n_folds,
    }


def summarize_ml_scan(ic_result: pd.DataFrame, importance_result: dict) -> dict:
    """UI向けの誠実性サマリ。

    mean_cv_score(R²ないしaccuracy基準のスコア)が低い場合に「予測力は無い」と誇張せず、
    かといって僅かなプラスを過大評価もしない、正直な一言判定(verdict)を添える。
    為替の短期予測ではR²が僅かでも正であること自体は珍しくないため、閾値は控えめに置く。
    """
    mean_score = float(importance_result["mean_score"])
    if mean_score < 0.02:
        verdict = "予測力はほぼ無い(ノイズ域)"
    else:
        verdict = "弱いが測定可能な予測力あり(過信は禁物)"

    return {
        "top_features_by_ic": ic_result.head(10).to_dict("records"),
        "top_features_by_importance": importance_result["importances"].head(10).to_dict("records"),
        "mean_cv_score": mean_score,
        "verdict": verdict,
    }
