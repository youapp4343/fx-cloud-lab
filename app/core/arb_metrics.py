"""マルチシンボル裁定戦略固有の統計指標(arb_stats)を算出する。

スキーマの意味論は docs/arb_spec.md §11 が正規仕様。このモジュールは
run_arb_backtest(app/core/arb_engine.py)の戻り値(result辞書)のみから算出する
(ArbConfig自体や生OHLCデータへはアクセスしない)。
"""

from __future__ import annotations

from typing import Any, Dict, List


def _group_trades_by_basket(trades: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    """position_id単位でtradesをグルーピングする(挿入順=バスケット決済順を保持)。"""
    baskets: Dict[Any, List[Dict[str, Any]]] = {}
    for t in trades:
        baskets.setdefault(t["position_id"], []).append(t)
    return baskets


def calculate_arb_stats(result: dict) -> dict:
    """result(run_arb_backtestの戻り値)からarb_stats(docs/arb_spec.md §11)を算出する。"""
    trades: List[Dict[str, Any]] = result.get("trades") or []
    equity_curve: List[Dict[str, Any]] = result.get("equity_curve") or []

    baskets = _group_trades_by_basket(trades)
    basket_count = len(baskets)

    basket_profits: Dict[Any, float] = {
        pid: sum(float(t["profit"]) for t in group) for pid, group in baskets.items()
    }
    basket_win_rate = (
        sum(1 for p in basket_profits.values() if p > 0) / basket_count if basket_count else 0.0
    )

    # exit_kindはバスケット内全レッグで同一(§7)なので任意の1レッグ(先頭)から取る
    basket_exit_kinds: Dict[Any, str] = {pid: group[0]["exit_kind"] for pid, group in baskets.items()}

    exit_kind_distribution: Dict[str, int] = {}
    for kind in basket_exit_kinds.values():
        exit_kind_distribution[kind] = exit_kind_distribution.get(kind, 0) + 1

    converge_count = exit_kind_distribution.get("converge", 0)
    convergence_rate = converge_count / basket_count if basket_count else 0.0

    # 合算profitが厳密に負(< 0)のconvergeバスケットのみ計上する(profit==0はloss扱いしない)
    converge_loss_count = sum(
        1
        for pid, kind in basket_exit_kinds.items()
        if kind == "converge" and basket_profits[pid] < 0
    )
    converge_loss_rate = converge_loss_count / converge_count if converge_count else 0.0

    profit_sums_by_kind: Dict[str, float] = {}
    for pid, kind in basket_exit_kinds.items():
        profit_sums_by_kind[kind] = profit_sums_by_kind.get(kind, 0.0) + basket_profits[pid]
    avg_profit_by_exit_kind: Dict[str, float] = {
        kind: profit_sums_by_kind[kind] / exit_kind_distribution[kind]
        for kind in profit_sums_by_kind
    }

    # 保有バー数: equity_curveのtimestamp→位置の辞書による位置差で算出する
    # (バー数の正規定義。時間差からの換算はしない。§11)
    ts_to_idx: Dict[Any, int] = {row["timestamp"]: idx for idx, row in enumerate(equity_curve)}
    bars_held: List[int] = []
    for pid, group in baskets.items():
        # 全レッグは同一バーで同時に建ち同時に決済される(§2.2/§2.1)ため任意レッグでよい
        open_ts = group[0]["entry_time"]
        close_ts = group[0]["exit_time"]
        if open_ts in ts_to_idx and close_ts in ts_to_idx:
            bars_held.append(ts_to_idx[close_ts] - ts_to_idx[open_ts])
    avg_hold_bars = (sum(bars_held) / len(bars_held)) if bars_held else 0.0
    max_hold_bars_held = max(bars_held) if bars_held else 0

    # leg_pnl_offset_degree: 各バスケットの 1 - |Σ_k profit_k| / Σ_k |profit_k| の単純平均。
    # レッグ2本未満のバスケット(相殺概念が定義できない)とΣ|profit_k|==0のバスケットは除外。
    offset_degrees: List[float] = []
    for pid, group in baskets.items():
        if len(group) < 2:
            continue
        abs_sum = sum(abs(float(t["profit"])) for t in group)
        if abs_sum == 0:
            continue
        offset_degrees.append(1.0 - abs(basket_profits[pid]) / abs_sum)
    leg_pnl_offset_degree = (sum(offset_degrees) / len(offset_degrees)) if offset_degrees else 0.0

    return {
        "basket_count": basket_count,
        "basket_win_rate": basket_win_rate,
        "exit_kind_distribution": exit_kind_distribution,
        "convergence_rate": convergence_rate,
        "converge_loss_count": converge_loss_count,
        "converge_loss_rate": converge_loss_rate,
        "avg_profit_by_exit_kind": avg_profit_by_exit_kind,
        "avg_hold_bars": avg_hold_bars,
        "max_hold_bars_held": max_hold_bars_held,
        "leg_pnl_offset_degree": leg_pnl_offset_degree,
        "max_gross_exposure_lots": float(result.get("max_gross_exposure_lots", 0.0)),
        "divergence_stats": result.get("divergence_stats") or {},
        "skipped_entries": result.get("skipped_entries")
        or {"beta_degenerate": 0, "balance_nonpositive": 0},
    }
