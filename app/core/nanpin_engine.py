"""ナンピンマーチンゲール戦略のバックテストエンジン。

正規仕様: docs/nanpin_spec.md。このファイルは同文書の §2〜§9 を実装する
(バー内判定順序・コストモデル・result辞書スキーマ・theoretical_max_loss・warnings)。
この順序を変更する場合は仕様書と NANPIN_SPEC_VERSION を同時に更新すること
(docs/exit_rules_spec.md の EXIT_SPEC_VERSION 運用を踏襲)。

app/core/engine.py とは完全に独立しており、import しない・変更しない(docs/nanpin_spec.md §0)。
_pip_size / _pip_value_per_lot は app/core/symbols.py(シンボル仕様の単一の真実、Wave5 案E)へ
委譲し、engine.py の同名関数と同一ロジックを保つ(§2.2-b, §5)。

スキップ条件(§3-C)は「defer(保留)方式」を採用する: スキップされたレベルは消費されず、
後続バーで条件解除かつ価格到達が続けば約定する。「永久に諦める」でも「レベルを飛ばして
次に進める」でもない(§6)。これは複数ありうる合理的なモデルの中から本実装が選んだ定義であり、
実際のブローカー/EAの挙動を保証するものではない。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.core import symbols
from app.core.indicators import atr, rsi, sma
from app.core.nanpin_model import NANPIN_DISCLAIMER, LotRule, NanpinConfig, SpacingRule

NANPIN_SPEC_VERSION = "1"  # docs/nanpin_spec.md 参照


def _pip_size(symbol: str) -> float:
    """pip定義: app/core/symbols.py へ委譲(engine.py の _pip_size と同一ロジック、§2.2-b)。"""
    return symbols.pip_size(symbol)


def _pip_value_per_lot(symbol: str, pip_size: float, price: float) -> float:
    """1lotあたりのpip価値をUSD建てで概算する(app/core/symbols.py へ委譲、§5)。

    engine.py の _pip_value_per_lot と同一ロジック。契約サイズ(FX=100,000通貨、
    XAU=100oz、XAG=5,000oz)とJPYクオートのprice割りUSD換算近似は symbols.py 側が持つ。
    金銀はUSD建てのため価格に依存せず XAU=$10/lot、XAG=$50/lot 固定になる。
    """
    return symbols.pip_value_per_lot(symbol, price, pip_size_override=pip_size)


def _raw_entry_signal(config: NanpinConfig, df: pd.DataFrame) -> pd.Series:
    """§3-A: エントリートリガーの生シグナル(shift前)。"""
    if config.entry == "always_long":
        return pd.Series(1, index=df.index, dtype=int)
    if config.entry == "always_short":
        return pd.Series(-1, index=df.index, dtype=int)
    if config.entry == "ma_direction":
        ma = sma(df["close"], config.ma_period)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[df["close"] > ma] = 1
        sig[df["close"] < ma] = -1
        return sig
    if config.entry == "rsi_counter":
        r = rsi(df["close"], config.rsi_period)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[r < config.rsi_lower] = 1
        sig[r > config.rsi_upper] = -1
        return sig
    raise ValueError(f"unknown entry mode: {config.entry!r}")  # NanpinConfig の Literal 型で到達不能なはず


def _raw_lot(lot_rule: LotRule, m: int) -> float:
    """§3-B: レイヤーm(0始まり)の素のロット raw_lot_m。"""
    if lot_rule.mode == "multiplier":
        return lot_rule.base_lot * (lot_rule.multiplier ** m)
    if lot_rule.mode == "additive":
        return lot_rule.base_lot + lot_rule.add_lot * m
    if lot_rule.mode == "flat":
        return lot_rule.base_lot
    raise ValueError(f"unknown lot_rule mode: {lot_rule.mode!r}")  # Literal型で到達不能なはず


def _lot_for_layer(lot_rule: LotRule, m: int) -> float:
    """§3-B: raw_lot_m を max_lot_per_order でクリップし小数第2位に丸める(min_lotクリップは行わない)。"""
    raw = _raw_lot(lot_rule, m)
    return round(min(raw, lot_rule.max_lot_per_order), 2)


def _gap_pips(spacing: SpacingRule, m: int) -> float:
    """§3-B: レイヤーm(m>=1)を追加する際の直前レベルからのギャップ(pips)。"""
    if spacing.mode == "fixed":
        return spacing.base_gap_pips
    if spacing.mode == "geometric":
        return spacing.base_gap_pips * (spacing.gap_ratio ** (m - 1))
    if spacing.mode == "additive":
        return spacing.base_gap_pips + spacing.gap_add_pips * (m - 1)
    raise ValueError(f"unknown spacing mode: {spacing.mode!r}")  # Literal型で到達不能なはず


def _build_level_ladder(config: NanpinConfig, entry_price: float, side: int, pip_size: float) -> List[float]:
    """§3-B: レベル価格ラダー levels[0..max_layers-1] を構築する(バスケット開始時に1回だけ)。

    levels[0] = entry_price(レイヤー0の実約定価格)。以後 levels[m] = levels[m-1] - side*gap_m*pip_size。
    実際の約定/defer結果に関わらず不変(§2.2-a)。
    """
    levels = [entry_price]
    for m in range(1, config.risk.max_layers):
        gap = _gap_pips(config.spacing, m)
        levels.append(levels[m - 1] - side * gap * pip_size)
    return levels


def _check_skip(
    config: NanpinConfig,
    atr_pips: Optional[pd.Series],
    velocity_pips: Optional[pd.Series],
    i: int,
) -> Optional[str]:
    """§3-C: バー i でスキップ条件が成立するか判定する(atr_spike優先、成立すればvelocityは見ない)。"""
    if not config.skip.enabled:
        return None
    if config.skip.atr_skip_pips is not None and atr_pips is not None:
        a_val = atr_pips.iloc[i]
        if pd.notna(a_val) and a_val >= config.skip.atr_skip_pips:
            return "atr_spike"
    if config.skip.velocity_skip_pips is not None and velocity_pips is not None:
        v_val = velocity_pips.iloc[i]
        if pd.notna(v_val) and v_val >= config.skip.velocity_skip_pips:
            return "velocity"
    return None


def _theoretical_max_loss(config: NanpinConfig, initial_balance: float) -> float:
    """§8: 設定と initial_balance のみから静的に計算する(価格データ不使用)。

    注意(FABLE監査で明確化): この値は「ロスカット発動水準での損失」であり、定義上
    initial_balance を超えない(equity_dd: balance×pct/100はpct<=100で常にbalance以下、
    margin_level: max(balance-非負値, 0)も常にbalance以下)。「全レイヤー到達時の素の損失」
    は別指標 _full_ladder_loss が担う(§9の警告1はそちらを比較対象にする)。
    """
    pip_size = _pip_size(config.symbol)
    # quote_ccy分岐(Wave5 案E): JPY末尾判定だと金銀クロス等が誤ってJPY扱いになりうるため、
    # symbols.pyのクオート通貨で判定する。金銀(USD建て)はnominal_price=1.0に落ちる。
    spec = symbols.get_spec(config.symbol)
    nominal_price = 150.0 if spec.quote_ccy == "JPY" else 1.0

    # §3-Bのロットラダーを max_layers 分構築し、max_total_lot を超える直前で打ち止める
    # (§2.2-a-1の上限ロジックと同一の考え方)。
    total_lot_effective = 0.0
    for m in range(config.risk.max_layers):
        candidate = _lot_for_layer(config.lot_rule, m)
        if total_lot_effective + candidate > config.lot_rule.max_total_lot:
            break
        total_lot_effective += candidate

    if config.risk.stopout_mode == "equity_dd":
        # ロットラダーに依存しない: 含み損がbalanceのmax_floating_loss_pct%に達したら発動する
        # 定義そのものであり、何層積み増していようと価格が十分逆行すれば必ずこの金額で発動する。
        return initial_balance * config.risk.max_floating_loss_pct / 100.0

    # margin_level モード(誠実性ノート: nominal_priceは仮の定数であり、対円クロス等では
    # 実際の必要証拠金と大きく乖離しうる近似値。桁感把握用の粗い目安であり精密な数値ではない。
    # 金銀はnominal_price=1.0のため名目額をさらに過小評価する=損失見積りは保守側に倒れる)。
    required_margin = total_lot_effective * spec.contract_size * nominal_price / config.risk.leverage
    return max(initial_balance - required_margin * config.risk.stopout_level_pct / 100.0, 0.0)


def _full_ladder_loss(config: NanpinConfig) -> float:
    """§8b: 全レイヤー約定後、さらに1ギャップ分逆行した時点での素の含み損(USD概算、静的計算)。

    ロスカットが存在しなかった場合にラダー構造が自然に抱えうる損失額であり、
    「構造的に破綻し得る設定」(§9警告1)の判定に使う。theoretical_max_loss(ロスカット
    発動水準)とは意味が異なる: あちらは定義上initial_balanceを超えないが、こちらは
    ロット拡大×レイヤー数×間隔次第でいくらでも大きくなる(FABLE監査で§9警告1が
    到達不能だった問題の修正として導入)。pip価値はnominal_price近似(§8と同じ限界)。
    """
    # quote_ccy分岐+契約サイズ対応(Wave5 案E): nominal_price近似でのpip価値を
    # symbols.pip_value_per_lot と同一式で計算する(JPYクオート=0.01*100,000/150で従来と同値、
    # 金銀はUSD建て契約サイズにより XAU=$10/lot・XAG=$50/lot 固定になる)。
    nominal_price = 150.0 if symbols.get_spec(config.symbol).quote_ccy == "JPY" else 1.0
    pip_value_nominal = symbols.pip_value_per_lot(config.symbol, price=nominal_price)

    # 有効レイヤー(max_total_lotで打ち止め)とその累積ギャップ(エントリーからの距離pips)
    lots: List[float] = []
    cum_gaps: List[float] = []
    total_lot = 0.0
    cum_gap = 0.0
    for m in range(config.risk.max_layers):
        candidate = _lot_for_layer(config.lot_rule, m)
        if total_lot + candidate > config.lot_rule.max_total_lot:
            break
        if m >= 1:
            cum_gap += _gap_pips(config.spacing, m)
        lots.append(candidate)
        cum_gaps.append(cum_gap)
        total_lot += candidate

    if not lots:
        return 0.0

    # 最終レイヤーからさらに1ギャップ先を「価格が到達した最悪点」とする
    worst_point = cum_gaps[-1] + _gap_pips(config.spacing, len(lots))
    return sum(lot * (worst_point - g) * pip_value_nominal for lot, g in zip(lots, cum_gaps))


def run_nanpin_backtest(
    config: NanpinConfig,
    df: pd.DataFrame,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
    commission_per_lot: float = 0.0,
    initial_balance: float = 10000.0,
    vol_gate_max_pct: Optional[float] = None,
    vol_gate_min_pct: Optional[float] = None,
    vol_gate_atr_period: int = 14,
    vol_gate_lookback: int = 200,
    vol_gate_blocks_adds: bool = False,
    ma_filter_period: Optional[int] = None,
    osc_filter: Optional[str] = None,
    osc_filter_period: int = 14,
) -> Dict[str, Any]:
    """vol_gate_min/max_pct: ATR百分位がこの範囲外(荒れすぎ/凪ぎすぎ)で新規開始を止める。
    vol_gate_blocks_adds=True で追加ナンピンも同ゲートで保留(defer、荒れ時は塩漬けを深追いしない)。
    ma_filter_period: エントリー方向がSMA(period)トレンドと一致する時のみ新規開始。
    osc_filter: 'rsi'等。売られすぎ/買われすぎゾーンでのみ新規開始(方向整合)。"""
    pip_size = _pip_size(config.symbol)
    # margin_levelモードの必要証拠金計算用(Wave5 案E)。FXは従来通り100,000で挙動不変、
    # 金銀は契約サイズ(XAU=100oz/XAG=5,000oz)×時価が正しいUSD建て名目額になる。
    contract_size = symbols.get_spec(config.symbol).contract_size
    theoretical_max_loss = _theoretical_max_loss(config, initial_balance)
    full_ladder_loss = _full_ladder_loss(config)

    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []
    stopout_events: List[Dict[str, Any]] = []
    skip_events: List[Dict[str, Any]] = []
    warnings: List[str] = []
    min_margin_level_pct: Optional[float] = None  # margin_levelモード時のみ更新(§11、FABLE監査対応)

    df = df.reset_index(drop=True)
    n = len(df)

    if n > 0:
        # §3-A エントリートリガー(shift(1)規律、engine.pyと同一)
        raw_entry = _raw_entry_signal(config, df)
        actionable_entry = raw_entry.shift(1).fillna(0).astype(int)

        # ボラレジームゲート(Wave5 案D、新規バスケット開始のみ制限。既存バスケットの
        # ナンピン追加・決済には不干渉)。ATR百分位rank>max_pctの高ボラ期は新規開始を止める。
        # rank.shift(1)+NaNは不明=停止側(engine.pyのvol_regimeと同思想)。
        vol_gate_ok = None
        if vol_gate_max_pct is not None or vol_gate_min_pct is not None:
            from app.core.novel_indicators import atr_percentile_rank

            _a = atr(df["high"], df["low"], df["close"], vol_gate_atr_period)
            _rank = atr_percentile_rank(_a, vol_gate_lookback).shift(1)
            vgo = _rank.notna()
            if vol_gate_max_pct is not None:
                vgo = vgo & (_rank <= vol_gate_max_pct)  # 荒れすぎ回避
            if vol_gate_min_pct is not None:
                vgo = vgo & (_rank >= vol_gate_min_pct)  # 凪ぎすぎ回避
            vol_gate_ok = vgo

        # MAフィルター: エントリー方向がSMAトレンドと一致する時のみ新規開始(shift(1)確定バー)
        ma_filter_up = None
        if ma_filter_period is not None:
            _ma = sma(df["close"], max(2, ma_filter_period))
            ma_filter_up = (df["close"] > _ma).shift(1).fillna(False).astype(bool)

        # オシレーターフィルター: 売られすぎ(買い方向)/買われすぎ(売り方向)ゾーンのみ新規開始
        osc_buy_ok = osc_sell_ok = None
        if osc_filter is not None:
            if osc_filter == "rsi":
                _o = rsi(df["close"], osc_filter_period).shift(1)
                osc_buy_ok = (_o <= 30).fillna(False).astype(bool)
                osc_sell_ok = (_o >= 70).fillna(False).astype(bool)

        # §3-C スキップ条件系列(shift(1)規律)。閾値未設定/skip無効なら計算しない。
        if config.skip.enabled and config.skip.atr_skip_pips is not None:
            atr_price = atr(df["high"], df["low"], df["close"], config.skip.atr_period)
            atr_pips: Optional[pd.Series] = (atr_price / pip_size).shift(1)
        else:
            atr_pips = None

        if config.skip.enabled and config.skip.velocity_skip_pips is not None:
            raw_velocity = (df["close"] - df["close"].shift(config.skip.velocity_bars)).abs() / pip_size
            velocity_pips: Optional[pd.Series] = raw_velocity.shift(1)
        else:
            velocity_pips = None

        balance = initial_balance
        basket: Optional[Dict[str, Any]] = None
        # バックテスト全体を通じて保持する状態(§2.0)。初回バスケットはreentry_wait_bars制約を受けない。
        last_basket_close_bar_index: Optional[int] = None

        equity_curve.append({"timestamp": df.iloc[0]["timestamp"], "equity": balance})

        def _entry_cost_price(raw: float, side: int) -> float:
            # §5: エントリー・追加時は spread+slippage を不利方向へ加算(engine.py _open_price と同一)。
            cost = (spread_pips + slippage_pips) * pip_size
            return raw + cost if side == 1 else raw - cost

        def _exit_cost_price(raw: float, side: int) -> float:
            # §5: 決済時は slippage のみを不利方向へ加算(engine.py _close_price と同一、例外なし)。
            cost = slippage_pips * pip_size
            return raw - cost if side == 1 else raw + cost

        def _pip_value(price: float) -> float:
            return _pip_value_per_lot(config.symbol, pip_size, price)

        def _close_basket(exit_price: float, exit_time: Any, exit_kind: str) -> None:
            """バスケット内の全レイヤーを exit_price で一括決済し、trades に1レイヤー1レコードで追記する(§7)。"""
            nonlocal balance, basket
            assert basket is not None
            side = basket["side"]
            for layer in basket["layers"]:
                if side == 1:
                    pips = (exit_price - layer["entry_price"]) / pip_size
                else:
                    pips = (layer["entry_price"] - exit_price) / pip_size
                profit = pips * _pip_value(exit_price) * layer["lot"] - commission_per_lot * layer["lot"]
                trades.append(
                    {
                        "entry_time": layer["entry_time"],
                        "exit_time": exit_time,
                        "side": "long" if side == 1 else "short",
                        "entry_price": layer["entry_price"],
                        "exit_price": exit_price,
                        "lot": layer["lot"],
                        "pips": pips,
                        "profit": profit,
                        "exit_kind": exit_kind,
                        "position_id": basket["position_id"],
                        "layer_index": layer["layer_index"],
                    }
                )
                balance += profit
            basket = None

        for i in range(1, n):
            row = df.iloc[i]
            ts = row["timestamp"]
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])

            # §2.1 新規バスケット判定(バー i の2.2処理が始まる前の時点でバスケット無しの場合のみ)
            # balance<=0(口座破綻)後は新規エントリーを停止する: equity_dd閾値が負になると
            # 含み益でも即stopoutする退化ループに入り、stopout_countやtradesが無意味に膨張して
            # マーチンの損失を過小・過大に歪めるため(FABLE監査で発見)。
            if basket is None and balance > 0:
                sig = int(actionable_entry.iloc[i])
                reentry_ok = (
                    last_basket_close_bar_index is None
                    or (i - last_basket_close_bar_index - 1) >= config.reentry_wait_bars
                )
                vol_ok = vol_gate_ok is None or bool(vol_gate_ok.iloc[i])
                # MA/oscフィルター(新規開始のみ、方向整合)
                filt_ok = True
                if sig != 0:
                    if ma_filter_up is not None:
                        filt_ok = filt_ok and (bool(ma_filter_up.iloc[i]) if sig == 1 else not bool(ma_filter_up.iloc[i]))
                    if osc_buy_ok is not None:
                        filt_ok = filt_ok and (bool(osc_buy_ok.iloc[i]) if sig == 1 else bool(osc_sell_ok.iloc[i]))
                if sig != 0 and reentry_ok and vol_ok and filt_ok:
                    side = sig
                    entry_price = _entry_cost_price(o, side)
                    levels = _build_level_ladder(config, entry_price, side, pip_size)
                    lot0 = _lot_for_layer(config.lot_rule, 0)
                    basket = {
                        "side": side,
                        "layers": [
                            {"entry_price": entry_price, "entry_time": ts, "lot": lot0, "layer_index": 0}
                        ],
                        "avg_price": entry_price,
                        "levels": levels,
                        "next_level_index": 1,
                        "cumulative_lot": lot0,
                        "layer_count": 1,
                        "cap_reached": False,
                        "basket_open_bar_index": i,
                        "position_id": str(ts),
                    }
                    # 2.1が成立した場合も同一バー内でそのまま2.2の判定に進む(engine.pyが新規建玉直後に
                    # 同一バーの高安でSL/TP判定を行うのと同じ規律、§2.1末尾)。

            # §2.2 バスケット保有時の判定(2.1で今開いたばかりのバスケットも対象)
            if basket is not None:
                side = basket["side"]

                # --- a. ナンピン追加判定(不利方向優先)。b/c/d/eより必ず先に評価し、その結果
                # (追加後の状態)を踏まえてb以降を評価する(§2.2の保守的規律、§2.5)。
                # 「価格が逆行してナンピン追加とロスカットの両方が同一バーで成立しうる状況では、
                # まず追加を反映した上でロスカット水準を再評価する」という意図的な設計である。
                added_this_bar = False
                if not basket["cap_reached"]:
                    while True:
                        # 1. 上限チェック(このレベルを試す前に毎回行う、§2.2-a-1)
                        if basket["layer_count"] >= config.risk.max_layers:
                            basket["cap_reached"] = True
                            break
                        candidate_lot = _lot_for_layer(config.lot_rule, basket["next_level_index"])
                        if basket["cumulative_lot"] + candidate_lot > config.lot_rule.max_total_lot:
                            basket["cap_reached"] = True
                            break

                        # 2. 窓開け／到達判定(指値約定モデル)
                        level = basket["levels"][basket["next_level_index"]]
                        if side == 1:
                            window_gap = o <= level
                        else:
                            window_gap = o >= level

                        if window_gap:
                            raw_fill = o
                            reached = True
                        else:
                            reached = (l <= level) if side == 1 else (h >= level)
                            raw_fill = level

                        if not reached:
                            break  # 未到達。このバーの追加判定はここまで(レベルを飛ばして次を見ない)

                        # 3. スキップ判定(§3-C)+ボラゲートによる追加保留(vol_gate_blocks_adds)
                        skip_rule = _check_skip(config, atr_pips, velocity_pips, i)
                        if skip_rule is None and vol_gate_blocks_adds and vol_gate_ok is not None \
                                and not bool(vol_gate_ok.iloc[i]):
                            skip_rule = "vol_gate"  # 荒れ/凪ぎで追加ナンピンをdefer(塩漬け深追い回避)
                        if skip_rule is not None:
                            # defer方式(§6): スキップされたレベルは next_level_index を進めず
                            # 消費しない。永久に諦めるのでも、レベルを飛ばして次に強制的に
                            # 進めるのでもなく、後続バーで条件解除かつ価格到達が続けば約定する。
                            skip_events.append(
                                {"timestamp": ts, "rule": skip_rule, "level_index": basket["next_level_index"]}
                            )
                            break  # このバーの追加判定はここまで

                        # 4. 約定(trades にはまだ追記しない。バスケット決済時に全レイヤーまとめて追記、§7)
                        fill_price = _entry_cost_price(raw_fill, side)
                        layer_index = basket["next_level_index"]
                        basket["layers"].append(
                            {"entry_price": fill_price, "entry_time": ts, "lot": candidate_lot, "layer_index": layer_index}
                        )
                        basket["cumulative_lot"] += candidate_lot
                        basket["layer_count"] += 1
                        # avg_priceは実約定済みレイヤーのみから再計算する(levelsラダー自体は再計算しない)
                        basket["avg_price"] = (
                            sum(ly["entry_price"] * ly["lot"] for ly in basket["layers"]) / basket["cumulative_lot"]
                        )
                        basket["next_level_index"] += 1
                        added_this_bar = True
                        # 同一バーで次のレベルも継続してチェックする(while継続)

                closed = False

                # --- b. 強制ロスカット判定(aの追加を反映済みの状態で評価) ---
                if basket is not None:
                    mark = l if side == 1 else h
                    floating_pnl = sum(
                        side * (mark - ly["entry_price"]) / pip_size * _pip_value(mark) * ly["lot"]
                        for ly in basket["layers"]
                    )
                    margin_level_pct: Optional[float] = None
                    if config.risk.stopout_mode == "equity_dd":
                        stopout = (-floating_pnl) >= balance * config.risk.max_floating_loss_pct / 100.0
                    else:
                        required_margin = basket["cumulative_lot"] * contract_size * mark / config.risk.leverage
                        margin_level_pct = (balance + floating_pnl) / required_margin * 100.0
                        stopout = margin_level_pct < config.risk.stopout_level_pct
                        # §11: バスケット保有中の全バーでの最小証拠金維持率をエンジン側で追跡する
                        # (stopoutに至らなかった紙一重の生存ケースも捕捉するため、FABLE監査対応)。
                        if min_margin_level_pct is None or margin_level_pct < min_margin_level_pct:
                            min_margin_level_pct = margin_level_pct

                    if stopout:
                        layers_closed = basket["layer_count"]
                        exit_price = _exit_cost_price(mark, side)
                        _close_basket(exit_price, ts, "stopout")
                        stopout_events.append(
                            {
                                "timestamp": ts,
                                "price": exit_price,
                                "layers_closed": layers_closed,
                                "loss_amount": -floating_pnl,
                                "balance_after": balance,
                                "margin_level_at_stopout": margin_level_pct,
                            }
                        )
                        last_basket_close_bar_index = i
                        closed = True
                        if balance <= 0:
                            warnings.append(
                                "口座破綻: 残高が0以下になったため、以降の新規エントリーを停止しました"
                            )

                # --- c. バスケットSL判定 ---
                if not closed and basket is not None and config.risk.basket_sl_pips is not None:
                    if side * (basket["avg_price"] - c) >= config.risk.basket_sl_pips * pip_size:
                        exit_price = _exit_cost_price(c, side)
                        _close_basket(exit_price, ts, "basket_sl")
                        last_basket_close_bar_index = i
                        closed = True

                # --- d. バスケットTP判定(このバーで a が1件も約定していない場合のみ、§2.5) ---
                # 追加が起きたバーはTPを見送る: 「ナンピンで深追いした直後にすぐ利確する」という
                # 平均取得単価まで戻ってからの利確という本来の意図に反する決済を避けるための、
                # 意図的な保守的規律である(到達していても決済せず翌バー以降に持ち越す)。
                if not closed and basket is not None and not added_this_bar:
                    tp_price = basket["avg_price"] + side * config.tp_pips * pip_size
                    tp_hit = (h >= tp_price) if side == 1 else (l <= tp_price)
                    if tp_hit:
                        exit_price = _exit_cost_price(tp_price, side)
                        _close_basket(exit_price, ts, "basket_tp")
                        last_basket_close_bar_index = i
                        closed = True

                # --- e. 時間切れ判定 ---
                if not closed and basket is not None and config.risk.max_basket_bars is not None:
                    if i - basket["basket_open_bar_index"] >= config.risk.max_basket_bars:
                        exit_price = _exit_cost_price(c, side)
                        _close_basket(exit_price, ts, "time")
                        last_basket_close_bar_index = i
                        closed = True

            # §2.3 equity_curveへの記録(毎バー、2.1/2.2の判定後に1回)
            if basket is not None:
                side = basket["side"]
                # mark-to-market: stopout判定(b)がlow/high(悲観側)を使うのに対し、equityは
                # close(実勢時価評価)を使う。目的が異なる意図的な使い分け(§2.3)。
                floating_pnl_close = sum(
                    side * (c - ly["entry_price"]) / pip_size * _pip_value(c) * ly["lot"]
                    for ly in basket["layers"]
                )
                equity_curve.append({"timestamp": ts, "equity": balance + floating_pnl_close})
            else:
                equity_curve.append({"timestamp": ts, "equity": balance})

        # §2.4 最終バー処理(ループ終了後)
        if basket is not None:
            last_row = df.iloc[-1]
            side = basket["side"]
            exit_price = _exit_cost_price(float(last_row["close"]), side)
            _close_basket(exit_price, last_row["timestamp"], "eod")
            equity_curve[-1]["equity"] = balance

    # §9 動的警告(3条件は独立、該当すれば全て追加する)
    # 警告1はfull_ladder_loss(全レイヤー約定+1ギャップ逆行時の素の損失)を比較対象にする。
    # 旧実装はtheoretical_max_loss(ロスカット発動水準)を使っていたが、それは定義上
    # initial_balanceを超えないため警告が数学的に到達不能だった(FABLE監査で発見・修正)。
    if full_ladder_loss > initial_balance:
        warnings.append("全レイヤー到達時の想定損失が初期残高を超えています(構造的に破綻し得る設定)")
    if len(stopout_events) >= 1:
        warnings.append(
            f"期間内に{len(stopout_events)}回のロスカットが発生しました。"
            "profit_factorは生存期間の値であり継続性を保証しません"
        )
    if config.risk.stopout_mode == "equity_dd" and config.risk.max_floating_loss_pct >= 80:
        warnings.append("ロスカット水準が深すぎます(実質SLなし運用に近い)")

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "initial_balance": initial_balance,
        "warnings": warnings,
        "stopout_events": stopout_events,
        "skip_events": skip_events,
        "theoretical_max_loss": theoretical_max_loss,
        "full_ladder_loss": full_ladder_loss,
        "min_margin_level_pct": min_margin_level_pct,
        "disclaimer": NANPIN_DISCLAIMER,  # §10: 再入力せずそのまま代入
    }
