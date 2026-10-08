"""ナンピン戦略固有の統計指標(nanpin_stats)を算出する。

スキーマの意味論は docs/nanpin_spec.md §11 が正規仕様。このモジュールは
run_nanpin_backtest(app/core/nanpin_engine.py)の戻り値(result辞書)のみから算出する
(NanpinConfig自体や生OHLCデータへはアクセスしない)。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _group_trades_by_basket(trades: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    """position_id単位でtradesをグルーピングする(挿入順=バスケット決済順を保持)。"""
    baskets: Dict[Any, List[Dict[str, Any]]] = {}
    for t in trades:
        baskets.setdefault(t["position_id"], []).append(t)
    return baskets


def calculate_nanpin_stats(result: dict) -> dict:
    """result(run_nanpin_backtestの戻り値)からnanpin_stats(§11)を算出する。"""
    trades: List[Dict[str, Any]] = result.get("trades") or []
    stopout_events: List[Dict[str, Any]] = result.get("stopout_events") or []
    skip_events: List[Dict[str, Any]] = result.get("skip_events") or []
    equity_curve: List[Dict[str, Any]] = result.get("equity_curve") or []
    initial_balance = float(result.get("initial_balance", 10000.0))

    baskets = _group_trades_by_basket(trades)
    basket_count = len(baskets)

    basket_profits: Dict[Any, float] = {
        pid: sum(float(t["profit"]) for t in group) for pid, group in baskets.items()
    }
    basket_win_rate = (
        sum(1 for p in basket_profits.values() if p > 0) / basket_count if basket_count else 0.0
    )

    layer_counts: Dict[Any, int] = {pid: len(group) for pid, group in baskets.items()}
    max_layers_reached = max(layer_counts.values()) if layer_counts else 0

    layer_distribution: Dict[str, int] = {}
    for count in layer_counts.values():
        key = str(count)
        layer_distribution[key] = layer_distribution.get(key, 0) + 1

    lot_sums: Dict[Any, float] = {pid: sum(float(t["lot"]) for t in group) for pid, group in baskets.items()}
    max_total_lot_held = max(lot_sums.values()) if lot_sums else 0.0

    # --- max_floating_loss ---
    # equity_curve は§2.3のmark-to-market評価(close基準)。バスケット決済(trades)の
    # 事象からrealized balanceの推移を逆算し、equity - realized_balance で含み損益を復元する。
    ts_to_idx: Dict[Any, int] = {row["timestamp"]: idx for idx, row in enumerate(equity_curve)}

    close_events: List[tuple] = []  # (exit_time, そのbasketのprofit合計) を時刻昇順に並べる
    for pid, group in baskets.items():
        close_events.append((group[0]["exit_time"], basket_profits[pid]))
    close_events.sort(key=lambda e: e[0])

    running_balance = initial_balance
    event_idx = 0
    best_loss_amount = 0.0
    best_loss_pct = 0.0
    best_loss_ts: Optional[Any] = None

    for row in equity_curve:
        ts = row["timestamp"]
        while event_idx < len(close_events) and close_events[event_idx][0] <= ts:
            running_balance += close_events[event_idx][1]
            event_idx += 1
        floating = float(row["equity"]) - running_balance
        loss = max(0.0, -floating)
        if loss > best_loss_amount:
            best_loss_amount = loss
            best_loss_pct = (loss / running_balance * 100.0) if running_balance != 0 else 0.0
            best_loss_ts = ts

    max_floating_loss = {
        "amount": best_loss_amount,
        "pct_of_balance": best_loss_pct,
        "timestamp": str(best_loss_ts) if best_loss_ts is not None else "",
    }

    # --- min_margin_level_pct ---
    # エンジン(run_nanpin_backtest)がバスケット保有中の全バーで追跡した値をそのまま使う
    # (FABLE監査対応: stopoutに至らなかった「紙一重の生存」ケースも捕捉するため、
    # エンジン側で毎バーのrunning minをresult["min_margin_level_pct"]として公開する設計に変更)。
    # 旧形式のresult(このキーが無い保存済みJSON)への後方互換として、キー欠落時のみ
    # stopout_eventsのmargin_level_at_stopout最小値で近似する(stopout 0回ならNone)。
    if "min_margin_level_pct" in result:
        min_margin_level_pct: Optional[float] = result["min_margin_level_pct"]
    else:
        margin_levels = [
            e["margin_level_at_stopout"] for e in stopout_events if e.get("margin_level_at_stopout") is not None
        ]
        min_margin_level_pct = min(margin_levels) if margin_levels else None

    stopout_count = len(stopout_events)
    stopout_total_loss = sum(float(e["loss_amount"]) for e in stopout_events)

    profit_before_first_stopout: Optional[float] = None
    if stopout_events:
        first_stopout_ts = stopout_events[0]["timestamp"]
        profit_before_first_stopout = sum(
            float(t["profit"]) for t in trades if t["exit_time"] < first_stopout_ts
        )

    bars_held: List[int] = []
    for pid, group in baskets.items():
        layer0 = next((t for t in group if t.get("layer_index") == 0), group[0])
        open_ts = layer0["entry_time"]
        close_ts = group[0]["exit_time"]
        if open_ts in ts_to_idx and close_ts in ts_to_idx:
            bars_held.append(ts_to_idx[close_ts] - ts_to_idx[open_ts])

    avg_basket_bars = (sum(bars_held) / len(bars_held)) if bars_held else 0.0
    max_basket_bars_held = max(bars_held) if bars_held else 0

    skipped_additions = {
        "atr_spike": sum(1 for e in skip_events if e.get("rule") == "atr_spike"),
        "velocity": sum(1 for e in skip_events if e.get("rule") == "velocity"),
    }

    return {
        "basket_count": basket_count,
        "basket_win_rate": basket_win_rate,
        "max_layers_reached": max_layers_reached,
        "layer_distribution": layer_distribution,
        "max_total_lot_held": max_total_lot_held,
        "max_floating_loss": max_floating_loss,
        "min_margin_level_pct": min_margin_level_pct,
        "stopout_count": stopout_count,
        "stopout_total_loss": stopout_total_loss,
        "profit_before_first_stopout": profit_before_first_stopout,
        "avg_basket_bars": avg_basket_bars,
        "max_basket_bars_held": max_basket_bars_held,
        "skipped_additions": skipped_additions,
    }
