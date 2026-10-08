"""ナンピン向け資金管理2モード + 月次メトリクス(エンジン非改変の後処理)。

run_nanpin_backtest の result["trades"] を入力に、以下を計算する:
- モードA(原資回収逃がし withdraw_principal): 運用残高が初期原資+利益で initial×2 に
  達したら初期原資分を退避(出金)、運用は初期原資で継続。ローリスク・利益確保型。
- モードB(複利3倍リセット compound_reset): 複利で運用し initial×reset_multiple に達したら
  達成分を退避して運用を initial に戻す(利益を刈って再スタート)。
- 月次: 月あたり平均取引回数・平均月利。

いずれも「トレード単位の事後シミュレーション」であり、証拠金維持率・同時保有の制約は
無視する近似(nanpin_engineのequityとは別粒度)。ナンピンのmax_floating_lossは別途必読。
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

CAPITAL_MGMT_DISCLAIMER = (
    "資金管理はトレード決済列の事後シミュレーションです。ナンピンの含み損(max_floating_loss)は"
    "反映されず、退避判定は実現損益ベースです。ロスカット・証拠金維持率と併読してください。"
)


def apply_capital_mode(
    trades: List[Dict[str, Any]],
    initial_balance: float = 10000.0,
    mode: str = "withdraw_principal",
    reset_multiple: float = 3.0,
) -> Dict[str, Any]:
    """mode: 'withdraw_principal'(原資回収逃がし) / 'compound_reset'(複利N倍リセット) / 'plain'(素)。"""
    if not trades:
        return {"final_operating": initial_balance, "withdrawn_total": 0.0,
                "final_total": initial_balance, "reset_count": 0, "ruined": False,
                "curve": [], "disclaimer": CAPITAL_MGMT_DISCLAIMER}

    tr = sorted(trades, key=lambda t: pd.Timestamp(t["exit_time"]))
    operating = initial_balance
    withdrawn = 0.0
    reset_count = 0
    ruined = False
    curve: List[Dict[str, Any]] = []

    for t in tr:
        if ruined:
            break
        operating += float(t["profit"])
        if operating <= 0:
            ruined = True
            curve.append({"time": str(t["exit_time"]), "operating": operating, "withdrawn": withdrawn})
            break
        if mode == "withdraw_principal" and operating >= 2 * initial_balance:
            # 初期原資分を退避、運用は初期原資で継続(超過利益は運用に残す)
            withdrawn += initial_balance
            operating -= initial_balance
        elif mode == "compound_reset" and operating >= reset_multiple * initial_balance:
            # 達成分(initial超過)を退避し、運用を初期原資へリセット
            withdrawn += operating - initial_balance
            operating = initial_balance
            reset_count += 1
        curve.append({"time": str(t["exit_time"]), "operating": operating, "withdrawn": withdrawn})

    return {
        "final_operating": operating,
        "withdrawn_total": withdrawn,
        "final_total": operating + withdrawn,
        "reset_count": reset_count,
        "ruined": ruined,
        "curve": curve,
        "disclaimer": CAPITAL_MGMT_DISCLAIMER,
    }


def monthly_stats(trades: List[Dict[str, Any]], initial_balance: float = 10000.0) -> Dict[str, Any]:
    """月あたり平均取引回数・平均月利(%)を返す。月利は当月profit合計/初期残高。"""
    if not trades:
        return {"months": 0, "avg_trades_per_month": 0.0, "avg_monthly_return_pct": 0.0,
                "total_trades": 0}
    df = pd.DataFrame(trades)
    df["ym"] = pd.to_datetime(df["exit_time"]).dt.to_period("M")
    g = df.groupby("ym")
    counts = g.size()
    profits = g["profit"].sum()
    n_months = len(counts)
    return {
        "months": int(n_months),
        "avg_trades_per_month": float(counts.mean()),
        "avg_monthly_return_pct": float((profits / initial_balance * 100.0).mean()),
        "total_trades": int(len(df)),
    }
