"""マルチシンボル裁定(アービトラージ)戦略の設定スキーマ。

挙動(データ整列規約・バー内判定順序・シグナル計算式・コストモデル・result辞書の
スキーマ等)の正規仕様は docs/arb_spec.md にある。このモジュールは設定値の構造と
バリデーションのみを担い、バックテストロジックは持たない(それは app/core/arb_engine.py
がPhase2で実装する)。

app/core/engine.py・app/core/nanpin_model.py・app/core/strategy_model.py とは完全に
独立しており、どちらにも依存しない(Strategy/NanpinConfig等は一切importしない)。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from app.core.validation import validate_slug

# docs/arb_spec.md §10 の固定免責文言。一字一句このまま result["disclaimer"] に設定する
# こと(arb_engine.py側で再入力・言い換えしない)。文言を厳密に固定するのは、この警告が
# 「バックテストで乖離から利益が出た=現実に裁定利益が取れる」という誤読(単一データ
# ソース・確定バーでは執行の同時性/片側約定リスクを再現できないことの過小評価)を防ぐ
# ためのものであり、実装ごとに言い回しが揺れると警告としての実効性が下がるため。
ARB_DISCLAIMER = (
    "本ツールの裁定バックテストは単一データソース(dukascopy BID)の確定バーのみを"
    "用いており、実際の裁定執行を支配する要素(レッグ間の約定時差・片側約定リスク・"
    "板の厚み・業者間の価格差・執行遅延)を再現できません。三角裁定の乖離は現代の市場"
    "ではHFTが数ミリ秒で解消する領域にあり、本結果で乖離が観測されても『個人が取れる"
    "利益』の証明にはなりません(現実的なスプレッドを課すと利益が消えることの確認こそが"
    "本検証の目的です)。統計的裁定のペア相関・共和分関係は恒久的ではなく、検証期間で"
    "有効でも将来突然崩壊し得ます(両レッグ同時損失があり得ます)。レッグ執行コストは"
    "本モデルより悪化する方向にしか外れません。win_rate・profit_factorを単独で評価せず、"
    "converge_loss_rate・exit_kind分布・max_gross_exposure・stop_z/時間切れ決済の損失を"
    "必ず併読してください。"
)


def _quote_currency(symbol: str) -> str:
    """6文字FXシンボルのクオート通貨(末尾3文字、大文字化)を返す。

    beta_notionalのロット式(docs/arb_spec.md §3-A-5)は「クオート通貨建て名目額の一致」
    を前提とするため、この関数の呼び出し元は6文字シンボルであることを先に検証すること。
    """
    return symbol[-3:].upper()


class StatArbParams(BaseModel):
    # 意味論は docs/arb_spec.md §3-A。symbols[0]=leg0、symbols[1]=leg1。
    symbols: List[str] = ["EURUSD", "GBPUSD"]
    lookback: int = 200
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    hedge_mode: Literal["rolling_ols", "fixed_1"] = "rolling_ols"
    lot_sizing: Literal["equal_lots", "beta_notional", "manual"] = "equal_lots"
    manual_lots: Optional[List[float]] = None

    @field_validator("symbols")
    @classmethod
    def _check_symbols(cls, v: List[str]) -> List[str]:
        if len(v) != 2:
            raise ValueError(
                f"stat_arb.symbols はちょうど2シンボルである必要があります: {v!r}"
            )
        for s in v:
            validate_slug(s, "stat_arb.symbols")
        if v[0].upper() == v[1].upper():
            raise ValueError(
                f"stat_arb.symbols の2シンボルは相異なる必要があります: {v!r}"
            )
        return v

    @field_validator("lookback")
    @classmethod
    def _check_lookback(cls, v: int) -> int:
        if not (30 <= v <= 10000):
            raise ValueError(
                f"stat_arb.lookback は30以上10000以下である必要があります: {v!r}"
            )
        return v

    @field_validator("manual_lots")
    @classmethod
    def _check_manual_lots_values(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        # lot_sizing!="manual" でも指定されていれば常に検証する(nanpin_model の
        # multiplier と同じ「将来モードを切り替えても安全」規律)。
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError(
                f"stat_arb.manual_lots はちょうど2要素である必要があります: {v!r}"
            )
        for lot in v:
            if lot < 0.01:
                raise ValueError(
                    f"stat_arb.manual_lots の各要素は0.01以上である必要があります: {v!r}"
                )
        return v

    @model_validator(mode="after")
    def _check_z_thresholds(self) -> "StatArbParams":
        if not (0 <= self.exit_z < self.entry_z < self.stop_z):
            raise ValueError(
                "z閾値が不正です(0 <= exit_z < entry_z < stop_z である必要があります): "
                f"exit_z={self.exit_z!r}, entry_z={self.entry_z!r}, stop_z={self.stop_z!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_manual_lots_required(self) -> "StatArbParams":
        if self.lot_sizing == "manual" and self.manual_lots is None:
            raise ValueError(
                "stat_arb.lot_sizing='manual' の場合、manual_lots(2要素、各0.01以上)を"
                "指定してください"
            )
        return self

    @model_validator(mode="after")
    def _check_beta_notional_quote_currency(self) -> "StatArbParams":
        # docs/arb_spec.md §3-A-5: beta_notionalのロット式はクオート通貨建て名目額の
        # 一致を前提とするため、両シンボルのクオート通貨(末尾3文字)一致を必須とする。
        # 末尾3文字をクオート通貨とみなせるのは6文字シンボルのみなので長さも検証する。
        if self.lot_sizing != "beta_notional":
            return self
        for s in self.symbols:
            if len(s) != 6:
                raise ValueError(
                    "stat_arb.lot_sizing='beta_notional' は6文字シンボル"
                    f"(例: EURUSD)のみ対応です: {s!r}"
                )
        if _quote_currency(self.symbols[0]) != _quote_currency(self.symbols[1]):
            raise ValueError(
                "stat_arb.lot_sizing='beta_notional' は両シンボルのクオート通貨が一致"
                "している必要があります(ロット式がクオート通貨建て名目額の一致を前提と"
                f"するため): {self.symbols!r}"
            )
        return self


class TriangularParams(BaseModel):
    # 意味論は docs/arb_spec.md §3-B。leg0=cross_symbol、leg1/leg2=leg_symbols。
    cross_symbol: str = "GBPJPY"
    leg_symbols: List[str] = ["GBPUSD", "USDJPY"]
    composition: Literal["product", "quotient"] = "product"
    entry_dev_pips: float = 5.0
    exit_dev_pips: float = 1.0
    stop_dev_pips: float = 20.0

    @field_validator("cross_symbol")
    @classmethod
    def _check_cross_symbol(cls, v: str) -> str:
        return validate_slug(v, "triangular.cross_symbol")

    @field_validator("leg_symbols")
    @classmethod
    def _check_leg_symbols(cls, v: List[str]) -> List[str]:
        if len(v) != 2:
            raise ValueError(
                f"triangular.leg_symbols はちょうど2シンボルである必要があります: {v!r}"
            )
        for s in v:
            validate_slug(s, "triangular.leg_symbols")
        return v

    @model_validator(mode="after")
    def _check_symbols_distinct(self) -> "TriangularParams":
        upper = [self.cross_symbol.upper()] + [s.upper() for s in self.leg_symbols]
        if len(set(upper)) != 3:
            raise ValueError(
                "triangular の cross_symbol / leg_symbols は3シンボルすべて相異なる"
                f"必要があります: cross_symbol={self.cross_symbol!r}, "
                f"leg_symbols={self.leg_symbols!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_dev_thresholds(self) -> "TriangularParams":
        if not (0 <= self.exit_dev_pips < self.entry_dev_pips < self.stop_dev_pips):
            raise ValueError(
                "dev閾値が不正です(0 <= exit_dev_pips < entry_dev_pips < stop_dev_pips "
                f"である必要があります): exit_dev_pips={self.exit_dev_pips!r}, "
                f"entry_dev_pips={self.entry_dev_pips!r}, "
                f"stop_dev_pips={self.stop_dev_pips!r}"
            )
        return self


class ArbRiskGuard(BaseModel):
    # 意味論は docs/arb_spec.md §2.1-a/d・§2.2条件2。max_hold_barsはOptionalではない
    # (裁定バスケットの無期限保有は「収束を待ち続けて破綻」の典型経路のため必須)。
    max_hold_bars: int = 500
    max_floating_loss_pct: float = 50.0
    reentry_wait_bars: int = 0

    @field_validator("max_hold_bars")
    @classmethod
    def _check_max_hold_bars(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"risk.max_hold_bars は1以上である必要があります: {v!r}")
        return v

    @field_validator("max_floating_loss_pct")
    @classmethod
    def _check_max_floating_loss_pct(cls, v: float) -> float:
        if not (0 < v <= 100):
            raise ValueError(
                f"risk.max_floating_loss_pct は0より大きく100以下である必要があります: {v!r}"
            )
        return v

    @field_validator("reentry_wait_bars")
    @classmethod
    def _check_reentry_wait_bars(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"risk.reentry_wait_bars は0以上である必要があります: {v!r}")
        return v


class ArbConfig(BaseModel):
    # デフォルトのシンボルに NZDUSD を使わないこと(リサンプル済みデータ0行の既知問題、
    # docs/arb_spec.md §0.3)。
    name: str
    timeframe: str = "M15"
    mode: Literal["stat_arb", "triangular"] = "stat_arb"
    base_lot: float = 0.10
    stat_arb: StatArbParams = StatArbParams()
    triangular: TriangularParams = TriangularParams()
    spread_overrides: Dict[str, float] = {}
    risk: ArbRiskGuard = ArbRiskGuard()

    @field_validator("name", "timeframe")
    @classmethod
    def _check_slug(cls, v: str, info) -> str:
        return validate_slug(v, info.field_name)

    @field_validator("base_lot")
    @classmethod
    def _check_base_lot(cls, v: float) -> float:
        # 0.01未満はround(lot, 2)で0.00に丸まり得る(最小取引単位割れ)ため下限0.01。
        if v < 0.01:
            raise ValueError(f"base_lot は0.01以上である必要があります: {v!r}")
        return v

    @field_validator("spread_overrides")
    @classmethod
    def _check_spread_overrides(cls, v: Dict[str, float]) -> Dict[str, float]:
        for symbol, pips in v.items():
            validate_slug(symbol, "spread_overrides のキー")
            if pips <= 0:
                raise ValueError(
                    f"spread_overrides の値は正の値である必要があります: "
                    f"{symbol!r}={pips!r}"
                )
        return v

    def active_symbols(self) -> List[str]:
        """現在のmodeで実際に使用するシンボルをleg_index順に返す。

        docs/arb_spec.md 付録A: stat_arb → symbols の順(leg0, leg1)、
        triangular → [cross_symbol] + leg_symbols の順(leg0, leg1, leg2)。
        エンジンのデータ整列(§1.1)・APIのデータロード(§13.1)はこの順序を正とする。
        """
        if self.mode == "stat_arb":
            return list(self.stat_arb.symbols)
        return [self.triangular.cross_symbol] + list(self.triangular.leg_symbols)
