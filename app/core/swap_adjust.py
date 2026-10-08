"""スワップポイント後処理(エンジン非改変)。

エンジンはスワップ未対応のため、trades列に対して保有日数×日次スワップpipsを加算する
近似を提供する。注意: 水曜3倍付与(週5営業日で7日分)は暦日数ベース計算なら概ね等価。
スワップレートは変動するため、検証時点の値での近似であり将来を保証しない。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def apply_swap(
    trades: List[Dict[str, Any]],
    swap_long_pips_per_day: float,
    swap_short_pips_per_day: float,
    pip_value_per_lot: float = 10.0,
) -> Dict[str, Any]:
    """各tradeに保有日数×スワップpipsを加算した合計を返す(元trades非変異)。

    戻り値: {"swap_pips_total", "swap_profit_total", "trades_with_swap":
             [{**t, "swap_pips", "profit_with_swap"}...]}
    """
    out = []
    swap_pips_total = 0.0
    swap_profit_total = 0.0
    for t in trades:
        days = max(0.0, (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 86400.0)
        rate = swap_long_pips_per_day if t["side"] == "long" else swap_short_pips_per_day
        sp = days * rate
        sprofit = sp * pip_value_per_lot * t["lot"]
        swap_pips_total += sp
        swap_profit_total += sprofit
        out.append({**t, "swap_pips": sp, "profit_with_swap": t["profit"] + sprofit})
    return {
        "swap_pips_total": swap_pips_total,
        "swap_profit_total": swap_profit_total,
        "trades_with_swap": out,
    }
