"""ナンピンマーチンゲール戦略の設定スキーマ。

挙動(バー内判定順序・コストモデル・result辞書のスキーマ等)の正規仕様は
docs/nanpin_spec.md にある。このモジュールは設定値の構造とバリデーションのみを担い、
バックテストロジックは持たない(それは app/core/nanpin_engine.py が実装する)。

app/core/engine.py・app/core/strategy_model.py とは完全に独立しており、どちらにも
依存しない(Strategy/StrategyParam等は一切importしない)。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from app.core.validation import validate_slug

# docs/nanpin_spec.md §10 の固定免責文言。一字一句このまま result["disclaimer"] に
# 設定すること(nanpin_engine.py側で再入力・言い換えしない)。文言を厳密に固定するのは、
# この警告が「ロスカット0回=安全」という誤読(生存者バイアス・テールリスクの過小評価)を
# 防ぐためのものであり、実装ごとに言い回しが揺れると警告としての実効性が下がるため。
NANPIN_DISCLAIMER = (
    "ナンピンマーチンゲール系は『高勝率・小さな利益の積み上げ+稀に壊滅的損失』という"
    "統計的性質を持ちます。このバックテストでロスカットが0回でも、それは検証期間内に"
    "破綻条件が出現しなかったことを意味するだけで、安全性の証明ではありません"
    "(テールリスクは常に観測期間の外側に存在します)。win_rate・profit_factorの高さを"
    "単独で評価しないでください。max_floating_loss / theoretical_max_loss / "
    "ロスカット履歴を必ず併読してください。"
)


class LotRule(BaseModel):
    mode: Literal["multiplier", "additive", "flat"] = "multiplier"
    base_lot: float = 0.01
    multiplier: float = 2.0
    add_lot: float = 0.01
    max_lot_per_order: float = 10.0
    max_total_lot: float = 20.0

    @field_validator("base_lot")
    @classmethod
    def _check_base_lot(cls, v: float) -> float:
        # 0.005未満はround(lot, 2)で0.00に丸まり、加重平均・必要証拠金計算のゼロ除算を
        # 引き起こす(FABLE監査で発見)ため、最小取引単位0.01を下限とする。
        if v < 0.01:
            raise ValueError(f"lot_rule.base_lot は0.01以上である必要があります: {v!r}")
        return v

    @field_validator("multiplier")
    @classmethod
    def _check_multiplier(cls, v: float) -> float:
        # mode!="multiplier" でも常に範囲検証する(将来modeを切り替えても
        # 安全な範囲に収まっていることを保証するため)。
        if not (1.0 <= v <= 3.0):
            raise ValueError(f"lot_rule.multiplier は1.0以上3.0以下である必要があります: {v!r}")
        return v

    @field_validator("add_lot")
    @classmethod
    def _check_add_lot(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"lot_rule.add_lot は0以上である必要があります: {v!r}")
        return v

    @field_validator("max_lot_per_order")
    @classmethod
    def _check_max_lot_per_order(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"lot_rule.max_lot_per_order は正の値である必要があります: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_max_total_lot(self) -> "LotRule":
        if self.max_total_lot < self.base_lot:
            raise ValueError(
                "lot_rule.max_total_lot は base_lot 以上である必要があります: "
                f"max_total_lot={self.max_total_lot!r}, base_lot={self.base_lot!r}"
            )
        return self


class SpacingRule(BaseModel):
    mode: Literal["fixed", "geometric", "additive"] = "fixed"
    base_gap_pips: float = 20.0
    gap_ratio: float = 1.5
    gap_add_pips: float = 10.0

    @field_validator("base_gap_pips")
    @classmethod
    def _check_base_gap_pips(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"spacing.base_gap_pips は正の値である必要があります: {v!r}")
        return v

    @field_validator("gap_ratio")
    @classmethod
    def _check_gap_ratio(cls, v: float) -> float:
        if not (0.5 <= v <= 3.0):
            raise ValueError(f"spacing.gap_ratio は0.5以上3.0以下である必要があります: {v!r}")
        return v

    @field_validator("gap_add_pips")
    @classmethod
    def _check_gap_add_pips(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"spacing.gap_add_pips は0以上である必要があります: {v!r}")
        return v


class SkipRule(BaseModel):
    # enabled=Trueの意味論(defer方式)は docs/nanpin_spec.md §6 を正規仕様とする。
    # ここでは閾値の整合性のみを検証する。
    enabled: bool = False
    atr_period: int = 14
    atr_skip_pips: Optional[float] = None
    velocity_bars: int = 3
    velocity_skip_pips: Optional[float] = None

    @field_validator("atr_period")
    @classmethod
    def _check_atr_period(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"skip.atr_period は2以上である必要があります: {v!r}")
        return v

    @field_validator("velocity_bars")
    @classmethod
    def _check_velocity_bars(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"skip.velocity_bars は1以上である必要があります: {v!r}")
        return v

    @field_validator("atr_skip_pips")
    @classmethod
    def _check_atr_skip_pips(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"skip.atr_skip_pips は正の値である必要があります: {v!r}")
        return v

    @field_validator("velocity_skip_pips")
    @classmethod
    def _check_velocity_skip_pips(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"skip.velocity_skip_pips は正の値である必要があります: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_enabled_requires_threshold(self) -> "SkipRule":
        if self.enabled and self.atr_skip_pips is None and self.velocity_skip_pips is None:
            raise ValueError(
                "skip.enabled=True の場合、atr_skip_pips / velocity_skip_pips の"
                "少なくとも一方を設定してください"
            )
        return self


class RiskGuard(BaseModel):
    max_layers: int = 8
    stopout_mode: Literal["equity_dd", "margin_level"] = "equity_dd"
    max_floating_loss_pct: float = 50.0
    leverage: float = 25.0
    stopout_level_pct: float = 100.0
    basket_sl_pips: Optional[float] = None
    max_basket_bars: Optional[int] = None

    @field_validator("max_layers")
    @classmethod
    def _check_max_layers(cls, v: int) -> int:
        if not (1 <= v <= 15):
            raise ValueError(f"risk.max_layers は1以上15以下である必要があります: {v!r}")
        return v

    @field_validator("max_floating_loss_pct")
    @classmethod
    def _check_max_floating_loss_pct(cls, v: float) -> float:
        if not (0 < v <= 100):
            raise ValueError(
                f"risk.max_floating_loss_pct は0より大きく100以下である必要があります: {v!r}"
            )
        return v

    @field_validator("leverage")
    @classmethod
    def _check_leverage(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"risk.leverage は正の値である必要があります: {v!r}")
        return v

    @field_validator("stopout_level_pct")
    @classmethod
    def _check_stopout_level_pct(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"risk.stopout_level_pct は正の値である必要があります: {v!r}")
        return v

    @field_validator("basket_sl_pips")
    @classmethod
    def _check_basket_sl_pips(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"risk.basket_sl_pips は正の値である必要があります: {v!r}")
        return v

    @field_validator("max_basket_bars")
    @classmethod
    def _check_max_basket_bars(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError(f"risk.max_basket_bars は1以上である必要があります: {v!r}")
        return v


class NanpinConfig(BaseModel):
    name: str
    symbol: str = "EURUSD"
    timeframe: str = "M15"
    entry: Literal["always_long", "always_short", "ma_direction", "rsi_counter"] = "always_long"
    ma_period: int = 200
    rsi_period: int = 14
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    tp_pips: float = 15.0
    reentry_wait_bars: int = 0
    lot_rule: LotRule = LotRule()
    spacing: SpacingRule = SpacingRule()
    skip: SkipRule = SkipRule()
    risk: RiskGuard = RiskGuard()

    @field_validator("name", "symbol", "timeframe")
    @classmethod
    def _check_slug(cls, v: str, info) -> str:
        return validate_slug(v, info.field_name)

    @field_validator("tp_pips")
    @classmethod
    def _check_tp_pips(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tp_pips は正の値である必要があります: {v!r}")
        return v

    @field_validator("ma_period")
    @classmethod
    def _check_ma_period(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"ma_period は2以上である必要があります: {v!r}")
        return v

    @field_validator("rsi_period")
    @classmethod
    def _check_rsi_period(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"rsi_period は2以上である必要があります: {v!r}")
        return v

    @field_validator("reentry_wait_bars")
    @classmethod
    def _check_reentry_wait_bars(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"reentry_wait_bars は0以上である必要があります: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_rsi_bounds(self) -> "NanpinConfig":
        if not (0 < self.rsi_lower < self.rsi_upper < 100):
            raise ValueError(
                "rsi_lower/rsi_upper が不正です(0 < rsi_lower < rsi_upper < 100 である"
                f"必要があります): rsi_lower={self.rsi_lower!r}, rsi_upper={self.rsi_upper!r}"
            )
        return self
