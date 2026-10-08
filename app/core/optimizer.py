"""パラメータ最適化(グリッドサーチ + ランダムサーチ + ウォークフォワード検証 + IS/OOS検証)。"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from app.core import engine
from app.core.metrics import calculate_metrics
from app.core.strategy_model import Strategy

MAX_COMBINATIONS = 5000
WALK_FORWARD_GRID_THRESHOLD = 1000


def _build_axes(strategy: Strategy) -> Dict[str, List[float]]:
    """range/stepが設定されたパラメータについて、軸ごとの候補値リストを作る。"""
    axes: Dict[str, List[float]] = {}
    for key, param in strategy.params.items():
        if param.range is not None and param.step is not None:
            lo, hi, step = param.range[0], param.range[1], param.step
            if step <= 0:
                raise ValueError(f"param '{key}' の step は正の値である必要があります")
            if hi < lo:
                raise ValueError(
                    f"param '{key}' の range が逆転しています(min={lo} > max={hi})"
                )
            values: List[float] = []
            n_steps = int(round((hi - lo) / step))
            for i in range(n_steps + 1):
                v = lo + i * step
                if v > hi + 1e-9:
                    break
                values.append(round(v, 10))
            if not values:
                values = [lo]
            axes[key] = values
        else:
            axes[key] = [param.value]
    return axes


def _axes_total(axes: Dict[str, List[float]]) -> int:
    total = 1
    for values in axes.values():
        total *= len(values)
    return total


def build_param_grid(strategy: Strategy) -> List[Dict[str, float]]:
    """range/stepが設定されたパラメータの直積で、試行パラメータのリストを作る。"""
    axes = _build_axes(strategy)
    keys = list(axes.keys())
    total = _axes_total(axes)
    if total > MAX_COMBINATIONS:
        raise ValueError(
            f"組合せ数が上限({MAX_COMBINATIONS})を超えています: {total}件。range/stepを見直してください。"
        )

    combos: List[Dict[str, float]] = []
    for values_tuple in itertools.product(*(axes[k] for k in keys)):
        combos.append(dict(zip(keys, values_tuple)))
    return combos


def random_param_grid(strategy: Strategy, n_iter: int, seed: int = 42) -> List[Dict[str, float]]:
    """axesの候補値から各軸独立に一様ランダムサンプリングし、n_iter件の組合せを作る。

    n_iterが全組合せ数(直積)以上の場合は、重複を避けるためbuild_param_gridの全数を返す。
    """
    if n_iter <= 0:
        raise ValueError("n_iter は正の整数である必要があります")
    if n_iter > MAX_COMBINATIONS:
        raise ValueError(f"n_iter が上限({MAX_COMBINATIONS})を超えています: {n_iter}件。")

    axes = _build_axes(strategy)
    keys = list(axes.keys())
    total = _axes_total(axes)
    if n_iter >= total:
        return build_param_grid(strategy)

    rng = np.random.default_rng(seed)
    combos: List[Dict[str, float]] = []
    for _ in range(n_iter):
        combos.append({k: float(rng.choice(axes[k])) for k in keys})
    return combos


def _split_is_oos(df: pd.DataFrame, is_oos_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split_idx = int(n * is_oos_split)
    is_df = df.iloc[:split_idx].reset_index(drop=True)
    oos_df = df.iloc[split_idx:].reset_index(drop=True)
    if len(is_df) == 0 or len(oos_df) == 0:
        raise ValueError(
            f"is_oos_split={is_oos_split} ではIS/OOSのどちらかが空になります(全{n}本)。データ量か比率を見直してください。"
        )
    return is_df, oos_df


def _apply_params(strategy: Strategy, combo: Dict[str, float]) -> Strategy:
    trial = strategy.model_copy(deep=True)
    for key, value in combo.items():
        trial.params[key].value = value
    return trial


def _evaluate_combo(
    strategy: Strategy,
    combo: Dict[str, float],
    is_df: pd.DataFrame,
    oos_df: Optional[pd.DataFrame],
    spread_pips: float,
    slippage_pips: float,
    commission_per_lot: float,
) -> Dict[str, Any]:
    """1組合せをIS(必要ならOOSも)でBTしてmetrics化する。"""
    trial = _apply_params(strategy, combo)

    is_result = engine.run_backtest(
        trial,
        is_df,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_per_lot=commission_per_lot,
    )
    is_metrics = calculate_metrics(is_result)

    if oos_df is not None:
        oos_result = engine.run_backtest(
            trial,
            oos_df,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            commission_per_lot=commission_per_lot,
        )
        oos_metrics = calculate_metrics(oos_result)
        return {"params": combo, "is_metrics": is_metrics, "oos_metrics": oos_metrics}
    return {"params": combo, "metrics": is_metrics}


def _sort_results_desc(results: List[Dict[str, Any]]) -> None:
    def _sort_key(entry: Dict[str, Any]) -> float:
        metrics = entry.get("metrics") or entry.get("is_metrics") or {}
        return metrics.get("net_profit", 0.0)

    results.sort(key=_sort_key, reverse=True)


def _run_search(
    combos: List[Dict[str, float]],
    strategy: Strategy,
    df: pd.DataFrame,
    spread_pips: float,
    slippage_pips: float,
    commission_per_lot: float,
    is_oos_split: Optional[float],
) -> Dict[str, Any]:
    if is_oos_split is not None:
        is_df, oos_df = _split_is_oos(df, is_oos_split)
    else:
        is_df, oos_df = df, None

    results = [
        _evaluate_combo(strategy, combo, is_df, oos_df, spread_pips, slippage_pips, commission_per_lot)
        for combo in combos
    ]
    _sort_results_desc(results)

    return {"total_combinations": len(combos), "results": results}


def run_grid_search(
    strategy: Strategy,
    df: pd.DataFrame,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
    commission_per_lot: float = 0.0,
    is_oos_split: Optional[float] = None,
) -> Dict[str, Any]:
    combos = build_param_grid(strategy)
    return _run_search(combos, strategy, df, spread_pips, slippage_pips, commission_per_lot, is_oos_split)


def run_random_search(
    strategy: Strategy,
    df: pd.DataFrame,
    n_iter: int,
    seed: int = 42,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
    commission_per_lot: float = 0.0,
    is_oos_split: Optional[float] = None,
) -> Dict[str, Any]:
    combos = random_param_grid(strategy, n_iter, seed)
    return _run_search(combos, strategy, df, spread_pips, slippage_pips, commission_per_lot, is_oos_split)


def _walk_forward_bounds(
    n: int, train_bars: int, test_bars: int, mode: Literal["rolling", "anchored"]
) -> List[tuple[int, int, int, int]]:
    """(train_start, train_end, test_start, test_end)の窓境界をtest_bars刻みで前進させながら列挙する。"""
    bounds: List[tuple[int, int, int, int]] = []
    k = 0
    while True:
        if mode == "anchored":
            train_start = 0
            train_end = train_bars + k * test_bars
        else:
            train_start = k * test_bars
            train_end = train_start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        if test_end > n:
            break
        bounds.append((train_start, train_end, test_start, test_end))
        k += 1
    return bounds


def run_walk_forward(
    strategy: Strategy,
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    mode: Literal["rolling", "anchored"] = "rolling",
    n_iter: Optional[int] = None,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
    commission_per_lot: float = 0.0,
) -> Dict[str, Any]:
    """訓練窓で最良パラメータを選び、直後のテスト窓で検証する処理をスライドさせながら繰り返す。"""
    if mode not in ("rolling", "anchored"):
        raise ValueError(f"mode は 'rolling' か 'anchored' である必要があります: {mode!r}")
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars/test_bars は正の整数である必要があります")

    df = df.reset_index(drop=True)
    n = len(df)
    bounds = _walk_forward_bounds(n, train_bars, test_bars, mode)
    if not bounds:
        raise ValueError(
            f"train_bars={train_bars}, test_bars={test_bars} では窓を1つも作れません(全{n}本)。"
        )

    # 組合せ候補はstrategyのparam定義のみに依存し窓データには依存しないため、ループの外で1度だけ作る。
    axes = _build_axes(strategy)
    if _axes_total(axes) > WALK_FORWARD_GRID_THRESHOLD:
        effective_n_iter = n_iter if n_iter is not None else WALK_FORWARD_GRID_THRESHOLD
        combos = random_param_grid(strategy, effective_n_iter)
    else:
        combos = build_param_grid(strategy)

    windows: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []
    all_equity: List[Dict[str, Any]] = []
    combined_initial_balance = 10000.0

    for train_start, train_end, test_start, test_end in bounds:
        if not combos:
            continue
        train_df = df.iloc[train_start:train_end].reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        train_evals = [
            _evaluate_combo(strategy, combo, train_df, None, spread_pips, slippage_pips, commission_per_lot)
            for combo in combos
        ]
        best = max(train_evals, key=lambda e: e["metrics"]["net_profit"])

        trial = _apply_params(strategy, best["params"])
        test_result = engine.run_backtest(
            trial,
            test_df,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            commission_per_lot=commission_per_lot,
        )
        test_metrics = calculate_metrics(test_result)

        all_trades.extend(test_result["trades"])
        all_equity.extend(test_result["equity_curve"])
        combined_initial_balance = test_result["initial_balance"]

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "best_params": best["params"],
                "train_metrics": best["metrics"],
                "test_metrics": test_metrics,
            }
        )

    combined_test_metrics = calculate_metrics(
        {"trades": all_trades, "equity_curve": all_equity, "initial_balance": combined_initial_balance}
    )

    return {"windows": windows, "combined_test_metrics": combined_test_metrics}
