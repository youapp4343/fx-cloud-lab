"""Strategy から MQL4/MQL5 EA ソースコードを生成する(Phase3)。

analyze() はファイルI/Oを行わない純粋関数で、エクスポート可否判定と
candle_pattern のパターン名/取引方向の解決のみを行う。
実際の Jinja2 レンダリング(app/templates_mql/*.j2 依存)は generate() が担う。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2

from app.core import templates
from app.core.candle_template import PATTERN_NAMES, _direction_for
from app.core.engine import EXIT_SPEC_VERSION
from app.core.strategy_model import Strategy

SUPPORTED_DIALECTS = ("mq4", "mq5")
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates_mql"

# v1でEA生成に対応するテンプレート名(将来テンプレートが増えても対象を明示的に絞る)。
_V1_TEMPLATES = ("ma_cross", "rsi_reversal", "breakout", "candle_pattern", "night_scalp")


class ExportBlocked(ValueError):
    """reasons: list[str] を保持する。エクスポート不可の全理由を列挙する(1つ目で即raiseしない)。"""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass
class ExportPlan:
    template: str
    resolved: dict          # candle_patternなら{"pattern_name":..., "pattern_fn":..., "direction":...}、他は{}
    blocked_reasons: list[str]
    warnings: list[str]


@dataclass
class GeneratedEA:
    filename: str            # "{strategy.name}.{dialect}"
    code: str
    warnings: list[str]
    spec_version: str


def _pascal_case(pattern_name: str) -> str:
    return "".join(word.capitalize() for word in pattern_name.split("_"))


def _magic_for(name: str) -> int:
    """戦略名から決定的な正整数magic番号を生成する(hash()はプロセス毎に非決定的なため不使用)。"""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_000_000_000


def _resolve_candle_pattern(strategy: Strategy) -> tuple[dict, list[str]]:
    pattern_index = int(strategy.params["pattern_index"].value)
    pattern_index = max(0, min(pattern_index, len(PATTERN_NAMES) - 1))  # range探索でのout-of-rangeを端にクリップ
    pattern_name = PATTERN_NAMES[pattern_index]

    override_param = strategy.params.get("direction_override")
    if override_param is not None and override_param.value != 0:
        direction = 1 if override_param.value > 0 else -1
    else:
        direction = _direction_for(pattern_name)

    if direction == 0:
        reason = (
            f"パターン'{pattern_name}'は中立(方向不定)でdirection_overrideも未指定のため、"
            "EA化する取引方向を決定できません"
        )
        return {}, [reason]

    resolved = {
        "pattern_name": pattern_name,
        "pattern_fn": "Pattern_" + _pascal_case(pattern_name),
        "direction": direction,
    }
    return resolved, []


def analyze(strategy: Strategy) -> ExportPlan:
    """生成可否とfeature検出のみを行う(ファイルI/Oなし、純粋関数)。

    問題を1つ見つけても即returnせず、全チェックを行ってからまとめて返す。
    """
    reasons: list[str] = []
    resolved: dict = {}

    spec = templates.get(strategy.template)
    if not spec.exportable:
        reasons.append(f"テンプレート'{strategy.template}'はexportable=falseのためEA出力に対応していません")
    if strategy.template not in _V1_TEMPLATES:
        reasons.append(
            f"テンプレート'{strategy.template}'はv1未対応テンプレートです"
            f"(v1対応テンプレート: {', '.join(_V1_TEMPLATES)})"
        )

    if strategy.params.get("trailing_pips") is not None:
        reasons.append("trailing_pipsはv1のEA出力では未対応です")
    if strategy.params.get("max_hold_bars") is not None:
        reasons.append("max_hold_barsはv1のEA出力では未対応です")
    if strategy.filters and strategy.filters.trade_hours and len(strategy.filters.trade_hours) == 2:
        # engine.py の _hour_allowed は len!=2 なら実質フィルタ無効として扱うため、
        # それに合わせて長さ2の場合のみ「有効なtrade_hoursフィルタ」としてブロックする。
        reasons.append("trade_hoursフィルタはv1のEA出力では未対応です")
    if strategy.filters and strategy.filters.vol_regime is not None:
        reasons.append("vol_regimeゲートはv1のEA出力では未対応です")
    if strategy.symbol.upper() in ("XAUUSD", "XAGUSD"):
        # 生成EAは実行時にDigitsからpip換算するため金銀(pip=0.1/0.01の特殊仕様)では
        # sl_pips/tp_pipsの意味がバックテストとずれる(Wave5 案E)。v1では出力ブロック。
        reasons.append("金銀(XAUUSD/XAGUSD)はv1のEA出力では未対応です(pip定義がFXと異なるため)")
    if strategy.money is not None and strategy.money.sizing != "fixed":
        reasons.append(f"sizing='{strategy.money.sizing}'はv1のEA出力では未対応です(固定ロットのみ対応)")
    if strategy.money is not None and strategy.money.partial_exits:
        reasons.append("分割決済(partial_exits)はv1のEA出力では未対応です")

    if strategy.template == "candle_pattern":
        candle_resolved, candle_reasons = _resolve_candle_pattern(strategy)
        resolved = candle_resolved
        reasons.extend(candle_reasons)

    return ExportPlan(template=strategy.template, resolved=resolved, blocked_reasons=reasons, warnings=[])


def _template_context(strategy: Strategy, plan: ExportPlan) -> dict[str, Any]:
    if strategy.template == "ma_cross":
        return {
            "fast_period": int(strategy.params["fast_period"].value),
            "slow_period": int(strategy.params["slow_period"].value),
        }
    if strategy.template == "rsi_reversal":
        return {
            "period": int(strategy.params["period"].value),
            "lower": float(strategy.params["lower"].value),
            "upper": float(strategy.params["upper"].value),
        }
    if strategy.template == "breakout":
        return {"lookback_bars": int(strategy.params["lookback_bars"].value)}
    if strategy.template == "night_scalp":
        # daily_target_pips/daily_loss_pipsはテンプレートparamsに無い任意拡張
        # (バックテスト側はdaily_guardで後処理、EA側は入力で内蔵)。未指定は-1=無効。
        dtp = strategy.params.get("daily_target_pips")
        dlp = strategy.params.get("daily_loss_pips")
        return {
            "hour_start": int(strategy.params["hour_start"].value) % 24,
            # hour_end=24は「日末まで」の正当値(%24でラップしない、seasonalの教訓)
            "hour_end": max(0, min(24, int(strategy.params["hour_end"].value))),
            "bb_period": max(2, int(strategy.params["bb_period"].value)),
            "bb_sigma": float(strategy.params["bb_sigma"].value),
            "daily_target_pips": float(dtp.value) if dtp is not None else -1.0,
            "daily_loss_pips": float(dlp.value) if dlp is not None else -1.0,
        }
    if strategy.template == "candle_pattern":
        return {
            "resolved_pattern_fn": plan.resolved["pattern_fn"],
            "resolved_direction": int(plan.resolved["direction"]),
        }
    return {}


def generate(strategy: Strategy, dialect: str) -> GeneratedEA:
    """analyze()した上でJinja2レンダリングしコード文字列を返す(ファイル書き込みはしない)。"""
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(f"unsupported dialect: {dialect!r} (supported: {SUPPORTED_DIALECTS})")

    plan = analyze(strategy)
    if plan.blocked_reasons:
        raise ExportBlocked(plan.blocked_reasons)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)
    template = env.get_template(f"{strategy.template}.{dialect}.j2")

    max_spread_pips = (
        strategy.filters.max_spread_pips
        if strategy.filters and strategy.filters.max_spread_pips is not None
        else -1
    )

    ctx: dict[str, Any] = {
        "symbol": strategy.symbol,
        "timeframe": strategy.timeframe,
        "strategy_name": strategy.name,
        "template_name": strategy.template,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_version": EXIT_SPEC_VERSION,
        "magic": _magic_for(strategy.name),
        "lot": float(strategy.params["lot"].value),
        "sl_pips": float(strategy.params["sl_pips"].value),
        "tp_pips": float(strategy.params["tp_pips"].value),
        "max_spread_pips": float(max_spread_pips),
    }
    ctx.update(_template_context(strategy, plan))

    rendered = template.render(**ctx)
    return GeneratedEA(
        filename=f"{strategy.name}.{dialect}",
        code=rendered,
        warnings=plan.warnings,
        spec_version=EXIT_SPEC_VERSION,
    )
