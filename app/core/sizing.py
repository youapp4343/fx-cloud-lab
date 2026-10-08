"""資金管理(ロットサイジング)ロジック。

money が None または sizing='fixed' の場合は必ず fixed_lot をそのまま返し、
既存の固定ロット動作(丸め・クリップ無し)を完全に維持する。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.strategy_model import MoneyManagement

_LARGE_R = 1e9  # 損失トレードが無い場合のRの代用値(0除算回避)


def _clip(lot: float, money: MoneyManagement) -> float:
    lot = max(money.min_lot, min(money.max_lot, lot))
    lot = round(lot, 2)
    # round(2)がmin_lotの小数3桁目以下を切り捨ててmin_lot未満になるケース(例: min_lot=0.015)を防ぐ。
    return max(money.min_lot, lot)


def _risk_pct_lot(balance: float, risk_pct: float, sl_pips: float, pip_value_per_lot: float) -> float:
    denom = sl_pips * pip_value_per_lot
    if denom <= 0:
        return 0.0
    return (balance * risk_pct) / denom


def compute_lot(
    money: Optional[MoneyManagement],
    fixed_lot: float,
    balance: float,
    sl_pips: float,
    pip_value_per_lot: float,
    confidence: float,
    closed_trades: List[Dict[str, Any]],
) -> float:
    """money が None または sizing='fixed' なら fixed_lot をそのまま返す(既存動作)。

    それ以外は money.sizing に応じてロットを計算し、money.min_lot/max_lot でクリップして返す。
    closed_trades は「このバックテスト内でこの時点までに確定済みのtradesリスト」
    (未来のトレードを含まないこと=ルックアヘッド禁止)。
    """
    if money is None or money.sizing == "fixed":
        return fixed_lot

    if money.sizing == "risk_pct":
        assert money.risk_pct is not None  # モデルのvalidatorで保証済み
        raw_lot = _risk_pct_lot(balance, money.risk_pct, sl_pips, pip_value_per_lot)
        return _clip(raw_lot, money)

    if money.sizing == "confidence_tiers":
        tiers = sorted(money.tiers or [], key=lambda t: t.min_confidence, reverse=True)
        chosen = money.min_lot
        for tier in tiers:
            if confidence >= tier.min_confidence:
                chosen = tier.lot
                break
        return _clip(chosen, money)

    if money.sizing == "kelly":
        window = closed_trades[-money.kelly_window :] if money.kelly_window > 0 else closed_trades
        if len(window) < money.kelly_min_trades or not window:
            return fixed_lot

        profits = [float(t["profit"]) for t in window]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        win_rate = len(wins) / len(profits)

        if not losses:
            r = _LARGE_R
        else:
            avg_win = (sum(wins) / len(wins)) if wins else 0.0
            avg_loss = abs(sum(losses) / len(losses))
            r = (avg_win / avg_loss) if avg_loss > 0 else _LARGE_R

        # r<=0 (勝ちトレードの平均利益が0以下)はポジションを持つ根拠が無いため f を負値として扱う
        f = (win_rate - (1.0 - win_rate) / r) if r > 0 else -1.0
        if f <= 0:
            return _clip(money.min_lot, money)

        kelly_risk_pct = f * money.kelly_fraction
        raw_lot = _risk_pct_lot(balance, kelly_risk_pct, sl_pips, pip_value_per_lot)
        return _clip(raw_lot, money)

    return fixed_lot
