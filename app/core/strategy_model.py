from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from app.core import templates
from app.core.validation import validate_slug


class StrategyParam(BaseModel):
    value: float
    range: Optional[List[float]] = None
    step: Optional[float] = None


class NewsFilter(BaseModel):
    mode: Literal["avoid", "only"] = "avoid"
    before_min: int = 30
    after_min: int = 30
    min_impact: Literal["high", "medium", "low"] = "high"
    currencies: Optional[List[str]] = None  # None = symbolから自動判定(例: EURUSD -> ["EUR","USD"])

    @model_validator(mode="after")
    def _check_minutes(self) -> "NewsFilter":
        if self.before_min < 0 or self.after_min < 0:
            raise ValueError("before_min/after_min は0以上である必要があります")
        return self


class MtfConfirm(BaseModel):
    timeframe: str  # 上位足の時間足(例"H4")
    trend_period: int = 50  # 上位足closeのSMA期間(トレンド方向判定用)
    mode: Literal["with_trend", "against_trend"] = "with_trend"

    @field_validator("timeframe")
    @classmethod
    def _check_timeframe(cls, v: str) -> str:
        return validate_slug(v, "mtf_confirm.timeframe")

    @field_validator("trend_period")
    @classmethod
    def _check_trend_period(cls, v: int) -> int:
        if v < 2:
            raise ValueError("trend_period は2以上である必要があります")
        return v


class VolRegimeGate(BaseModel):
    """ボラティリティレジームゲート(Wave5 案D)。ATR百分位順位が範囲外の期間は新規
    エントリーを止める汎用フィルタ。全テンプレート横断で使える(engine側で統合)。"""

    atr_period: int = 14
    percentile_lookback: int = 200
    max_percentile: Optional[float] = 0.9  # rank > max で新規停止(高ボラ回避)
    min_percentile: Optional[float] = None  # rank < min で新規停止(低ボラ回避)

    @field_validator("atr_period")
    @classmethod
    def _check_atr_period(cls, v: int) -> int:
        if v < 1:
            raise ValueError("vol_regime.atr_period は1以上である必要があります")
        return v

    @field_validator("percentile_lookback")
    @classmethod
    def _check_lookback(cls, v: int) -> int:
        if v < 2:
            raise ValueError("vol_regime.percentile_lookback は2以上である必要があります")
        return v

    @field_validator("max_percentile", "min_percentile")
    @classmethod
    def _check_pct(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("vol_regime の percentile は0.0〜1.0である必要があります")
        return v

    @model_validator(mode="after")
    def _check_at_least_one(self) -> "VolRegimeGate":
        if self.max_percentile is None and self.min_percentile is None:
            raise ValueError("vol_regime は max_percentile / min_percentile の少なくとも一方が必須です")
        return self


class StrategyFilters(BaseModel):
    trade_hours: Optional[List[int]] = None
    max_spread_pips: Optional[float] = None
    news: Optional[NewsFilter] = None
    mtf_confirm: Optional[MtfConfirm] = None
    vol_regime: Optional[VolRegimeGate] = None


class Tier(BaseModel):
    min_confidence: float
    lot: float


class PartialExit(BaseModel):
    at_pips: float
    close_ratio: float  # 0<close_ratio<=1


class MoneyManagement(BaseModel):
    sizing: Literal["fixed", "risk_pct", "confidence_tiers", "kelly"] = "fixed"
    risk_pct: Optional[float] = None          # 1トレードの許容損失(残高比、例0.01=1%)
    kelly_fraction: float = 0.25              # フルKellyへの係数
    kelly_window: int = 30                    # 直近何件の確定済みトレードでW/Rを推定するか
    kelly_min_trades: int = 20                # これ未満はbase_lotを使う
    tiers: Optional[List[Tier]] = None
    min_lot: float = 0.01
    max_lot: float = 5.0
    partial_exits: Optional[List[PartialExit]] = None   # このタスクではスキーマのみ、engineでの解釈は次タスク
    breakeven_after_first_tp: bool = False               # 同上

    @field_validator("kelly_fraction")
    @classmethod
    def _check_kelly_fraction(cls, v: float) -> float:
        if not (0 < v <= 0.5):
            raise ValueError("kelly_fraction は 0 より大きく 0.5 以下である必要があります")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "MoneyManagement":
        if self.min_lot <= 0 or self.max_lot <= 0 or self.min_lot > self.max_lot:
            raise ValueError("min_lot/max_lot が不正です(0より大きく、min_lot<=max_lotである必要があります)")
        if self.sizing == "risk_pct" and self.risk_pct is None:
            raise ValueError("sizing='risk_pct' の場合 risk_pct は必須です")
        if self.risk_pct is not None and not (0 < self.risk_pct <= 0.2):
            raise ValueError("risk_pct は 0 より大きく 0.2(=20%) 以下である必要があります")
        if self.sizing == "confidence_tiers" and not self.tiers:
            raise ValueError("sizing='confidence_tiers' の場合 tiers は必須です")
        if self.tiers:
            for t in self.tiers:
                if t.lot <= 0:
                    raise ValueError("tiers の lot は正の値である必要があります")
        if self.partial_exits:
            total = sum(p.close_ratio for p in self.partial_exits)
            if total > 1.0 + 1e-9:
                raise ValueError(f"partial_exits の close_ratio 合計が1.0を超えています: {total}")
            for p in self.partial_exits:
                if not (0 < p.close_ratio <= 1.0):
                    raise ValueError("partial_exits の close_ratio は 0〜1.0 の範囲である必要があります")
                if p.at_pips <= 0:
                    raise ValueError("partial_exits の at_pips は正の値である必要があります")
            at_pips_list = [p.at_pips for p in self.partial_exits]
            if at_pips_list != sorted(at_pips_list):
                raise ValueError("partial_exits は at_pips の昇順で指定してください")
        return self


class Strategy(BaseModel):
    name: str
    template: str
    symbol: str
    timeframe: str
    params: Dict[str, StrategyParam]
    filters: Optional[StrategyFilters] = None
    money: Optional[MoneyManagement] = None  # None = 従来通りの固定ロット動作(後方互換)

    @field_validator("name", "symbol", "timeframe")
    @classmethod
    def _check_slug(cls, v: str, info) -> str:
        return validate_slug(v, info.field_name)

    @field_validator("template")
    @classmethod
    def _check_template_registered(cls, v: str) -> str:
        if not templates.is_registered(v):
            raise ValueError(f"unknown template: {v!r} (registered: {templates.names()})")
        return v

    @model_validator(mode="after")
    def _check_required_params(self) -> "Strategy":
        required = templates.get(self.template).defaults.keys()
        missing = [k for k in required if k not in self.params]
        if missing:
            raise ValueError(
                f"template '{self.template}' に必要なパラメータが不足しています: {missing}"
            )
        return self


def default_params(template: str) -> Dict[str, StrategyParam]:
    defaults = templates.defaults_for(template)
    return {key: StrategyParam(value=value) for key, value in defaults.items()}
