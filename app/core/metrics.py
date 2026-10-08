"""バックテスト結果からパフォーマンス指標を計算する。"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _empty_metrics() -> Dict[str, Any]:
    return {
        "net_profit": 0.0,
        "total_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "max_drawdown": {"amount": 0.0, "pct": 0.0},
        "recovery_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_consecutive_losses": 0,
        "total_pips": 0.0,
        "expectancy_pips": 0.0,
        "fill_count": 0,
        "position_count": 0,
        "return_pct": 0.0,
        "win_rate_by_position": 0.0,
    }


def _position_key(trade: Dict[str, Any], index: int) -> Any:
    """ポジション単位の集計キーを返す。

    position_idが無い/Noneの古い形式のtradesは「1trade=1position」とみなし、
    そのtrade自身のインデックスをキーにフォールバックする。実際のposition_id値
    (1始まりのint)とインデックス値(0始まりのint)が偶然一致してもポジションが
    誤って合算されないよう、タプルで名前空間を分けている。
    """
    position_id = trade.get("position_id")
    if position_id is not None:
        return ("id", position_id)
    return ("idx", index)


def calculate_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    trades: List[Dict[str, Any]] = result.get("trades") or []
    equity_curve: List[Dict[str, Any]] = result.get("equity_curve") or []

    if not trades:
        return _empty_metrics()

    profits = [float(t["profit"]) for t in trades]
    net_profit = sum(profits)
    total_trades = len(trades)

    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]

    win_rate = len(wins) / total_trades

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0

    equity_values = [float(e["equity"]) for e in equity_curve]
    max_dd_amount = 0.0
    max_dd_pct = 0.0
    if equity_values:
        peak = equity_values[0]
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd_amount:
                max_dd_amount = drawdown
            if peak > 0:
                dd_pct = drawdown / peak * 100.0
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

    recovery_factor = (net_profit / max_dd_amount) if max_dd_amount > 0 else 0.0

    # 簡易シャープレシオ: トレードごとのリターン系列の平均/標準偏差×sqrt(トレード数)
    if len(profits) > 1:
        mean = float(np.mean(profits))
        std = float(np.std(profits, ddof=1))
        sharpe_ratio = (mean / std) * float(np.sqrt(len(profits))) if std > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    max_consecutive_losses = 0
    streak = 0
    for p in profits:
        if p < 0:
            streak += 1
            max_consecutive_losses = max(max_consecutive_losses, streak)
        else:
            streak = 0

    # 可変ロット・分割決済対応: フィル(=trades要素)単位に加え、ポジション単位の指標を追加する。
    total_pips = sum(float(t["pips"]) for t in trades)
    fill_count = total_trades
    expectancy_pips = total_pips / fill_count

    position_profits: Dict[Any, float] = {}
    for idx, t in enumerate(trades):
        key = _position_key(t, idx)
        position_profits[key] = position_profits.get(key, 0.0) + float(t["profit"])
    position_count = len(position_profits)
    position_wins = sum(1 for p in position_profits.values() if p > 0)
    win_rate_by_position = position_wins / position_count

    initial_balance = result.get("initial_balance", 10000.0)
    return_pct = (net_profit / initial_balance * 100.0) if initial_balance else 0.0

    return {
        "net_profit": net_profit,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": {"amount": max_dd_amount, "pct": max_dd_pct},
        "recovery_factor": recovery_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_consecutive_losses": max_consecutive_losses,
        "total_pips": total_pips,
        "expectancy_pips": expectancy_pips,
        "fill_count": fill_count,
        "position_count": position_count,
        "return_pct": return_pct,
        "win_rate_by_position": win_rate_by_position,
    }
