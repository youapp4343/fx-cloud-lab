"""マルチシンボル裁定(アービトラージ)戦略のバックテストエンジン。

正規仕様: docs/arb_spec.md(Arb Spec v1)。このファイルは同文書の §1〜§10 を実装する
(データ整列規約・バー内判定順序・シグナル定義・コストモデル・result辞書スキーマ・
theoretical_max_loss・動的警告・固定免責文言)。挙動を変更する場合は仕様書と
ARB_SPEC_VERSION を同時に更新すること(docs/nanpin_spec.md の NANPIN_SPEC_VERSION
運用を踏襲)。

app/core/engine.py・app/core/nanpin_engine.py とは完全に独立しており、importしない・
変更しない(docs/arb_spec.md §0)。_pip_size / _pip_value_per_lot は app/core/symbols.py
(シンボル仕様の単一の真実、Wave5 案E)への委譲であり、engine.py の同名関数と同一定義を
保つ(§1.0。engine.py 本体は従来どおりimport禁止のまま)。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core import symbols
from app.core.arb_model import ARB_DISCLAIMER, ArbConfig

ARB_SPEC_VERSION = "1"  # docs/arb_spec.md 参照


def _pip_size(symbol: str) -> float:
    """pip定義: app/core/symbols.py へ委譲(docs/arb_spec.md §1.0)。

    app/core/engine.py の _pip_size と同一定義(両者とも symbols.py が単一の真実)。
    """
    return symbols.pip_size(symbol)


def _pip_value_per_lot(symbol: str, pip_size: float, price: float) -> float:
    """1lotあたりのpip価値をUSD建てで概算する(app/core/symbols.py へ委譲)。

    app/core/engine.py の _pip_value_per_lot と同一定義(docs/arb_spec.md §1.0)。
    契約サイズ(FX=100,000通貨、XAU=100oz、XAG=5,000oz)とJPYクオートのprice割り
    USD換算近似は symbols.py 側が持つ。
    """
    return symbols.pip_value_per_lot(symbol, price, pip_size_override=pip_size)


def _align_frames(
    symbols: List[str], dfs: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """§1.1: timestamp重複行を先勝ちでdropし、全シンボル共通timestampのみをinner joinする。

    ffill / bfill / merge_asof 等による補完・近傍マッチは明示的に禁止(§1.1)。片側
    シンボルだけ欠けたバーを直前値で埋めると「存在しない裁定機会(偽の乖離)の捏造」に
    なるため、欠損バーは捨てる(保守側)。列は open_<SYMBOL> のようにシンボル別に保持する。
    """
    rows_per_symbol: Dict[str, int] = {}
    merged: Optional[pd.DataFrame] = None
    for sym in symbols:
        d = dfs[sym][["timestamp", "open", "high", "low", "close"]].drop_duplicates(
            subset="timestamp", keep="first"
        )
        rows_per_symbol[sym] = int(len(d))
        d = d.rename(columns={c: f"{c}_{sym}" for c in ("open", "high", "low", "close")})
        merged = d if merged is None else merged.merge(d, on="timestamp", how="inner")
    assert merged is not None  # symbolsは常に2または3要素(ArbConfigのvalidatorで保証)
    merged = merged.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return merged, rows_per_symbol


def _edge_trigger(sig: pd.Series, entry_th: float, stop_th: float) -> pd.Series:
    """§3-A-4 / §3-B-2: エントリーシグナル(エッジトリガー、レベル条件ではない)。

    +1: sig が -entry_th を「新たに」下抜けたバー(かつ -stop_th 以内)
    -1: sig が +entry_th を「新たに」上抜けたバー(かつ +stop_th 以内)
    レベル条件にすると stop_z 決済直後に同方向へ再エントリーして即再決済される退化ループに
    入るため、閾値を新規に跨いだバーのみをトリガーとする(§3-A-4の必読ノート)。
    stop側の条項は「1バーで entry と stop を同時に飛び越えた(建てた瞬間に発散ストップ
    圏内)」のエントリーを禁止するもの。
    """
    prev = sig.shift(1)
    plus = (sig <= -entry_th) & (prev > -entry_th) & (sig > -stop_th)
    minus = (sig >= entry_th) & (prev < entry_th) & (sig < stop_th)
    raw = pd.Series(0, index=sig.index, dtype=int)
    raw[plus] = 1
    raw[minus] = -1
    return raw


def _empty_divergence_stats(series_name: str) -> Dict[str, Any]:
    """§6: series 以外すべて None の divergence_stats(縮退時・有効バー不足時)。"""
    return {
        "series": series_name,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "pct_bars_beyond_entry": None,
        "half_life_bars": None,
    }


def _divergence_stats(
    series_name: str,
    stats_series: pd.Series,
    entry_threshold: float,
    halflife_series: pd.Series,
) -> Dict[str, Any]:
    """§6: シグナル系列の要約統計。

    - 統計対象は shift 前の生系列の NaN 除外値。stat_arb の z は「σが0/NaNのバーは
      z_t=0 とみなす」(§3-A-3)前の NaN 保持版を渡すこと(0代入は取引判定用の規約で
      あり、ウォームアップ期間の人工的な0で記述統計を汚染しないため。§6が「NaN除外」を
      明記しているのは0代入前の系列が対象であることを含意する)。
    - half_life は AR(1) 近似(stat_arb はスプレッド s_t、triangular は dev_t が対象)。
      検証期間内の記述統計であり将来の収束を保証しない(§6誠実性ノート)。
    """
    out = _empty_divergence_stats(series_name)
    valid = stats_series.dropna()
    if len(valid) < 2:
        return out
    out["mean"] = float(valid.mean())
    out["std"] = float(valid.std(ddof=1))
    out["min"] = float(valid.min())
    out["max"] = float(valid.max())
    out["pct_bars_beyond_entry"] = float((valid.abs() >= entry_threshold).mean() * 100.0)

    prev = halflife_series.shift(1)
    pair_count = int((halflife_series.notna() & prev.notna()).sum())
    if pair_count >= 3:
        # 実質定数の系列(完全相関ペアのスプレッド等)では、AR(1)相関が浮動小数点ノイズ
        # 同士の相関になり無意味な half_life(≈0)を返してしまう。系列の変動が数値ノイズ
        # 水準しかない場合は「収束ダイナミクス無し」としてNoneのままにする(§6の
        # 0<rho<1規則の数値的堅牢化、意味論は不変)。
        _std = halflife_series.std(ddof=1)
        hl_std = 0.0 if pd.isna(_std) else float(_std)
        _mx = halflife_series.abs().max()
        hl_eps = max(1e-12, (0.0 if pd.isna(_mx) else float(_mx)) * 1e-13)
        if hl_std > hl_eps:
            rho = halflife_series.corr(prev)
            if pd.notna(rho) and 0.0 < rho < 1.0:
                out["half_life_bars"] = float(-np.log(2.0) / np.log(rho))
    return out


def run_arb_backtest(
    config: ArbConfig,
    dfs: Dict[str, pd.DataFrame],
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
    extra_leg_slippage_pips: float = 0.5,
    commission_per_lot: float = 0.0,
    initial_balance: float = 10000.0,
) -> Dict[str, Any]:
    """docs/arb_spec.md §12 の正規シグネチャ。整列(§1.1)はエンジン内で行う。"""
    symbols = config.active_symbols()  # leg_index順(付録A)
    missing = [s for s in symbols if s not in dfs]
    if missing:
        raise ValueError(
            f"dfsに必要なシンボルのデータがありません: {', '.join(missing)}"
            f"(必要なシンボル: {', '.join(symbols)}。余剰キーは無視されます)"
        )

    merged, rows_per_symbol = _align_frames(symbols, dfs)
    aligned_rows = int(len(merged))
    alignment: Dict[str, Any] = {
        "rows_per_symbol": dict(rows_per_symbol),
        "aligned_rows": aligned_rows,
        "dropped_pct_per_symbol": {
            # rows==0 のシンボルは 100.0 と定義(§1.2)
            sym: (100.0 if rows == 0 else float((rows - aligned_rows) / rows * 100.0))
            for sym, rows in rows_per_symbol.items()
        },
    }

    # §8: 価格データを使わず、メインループ前に1回だけ静的計算する。「厳密な損失上限」では
    # なく「basket_slの発動水準(初期残高基準)の目安」(発動判定は各時点のbalance基準で、
    # 最悪値mark+決済コストにより実現損失がこの値を超えることはあり得る。§8誠実性ノート)。
    theoretical_max_loss = initial_balance * config.risk.max_floating_loss_pct / 100.0

    warnings: List[str] = []
    if config.mode == "triangular":
        # §9警告1: triangularモードでは常に追加(縮退時も含む)
        warnings.append(
            "triangularモードは乖離の存在量の確認が目的であり、実運用可能性の検証では"
            "ありません(現実の三角裁定機会はHFTがミリ秒で解消します)"
        )
    for sym in symbols:
        pct = alignment["dropped_pct_per_symbol"][sym]
        if pct > 5.0:
            # §9警告2
            warnings.append(
                f"{sym}: 整列で{pct:.1f}%のバーが除外されました"
                "(シンボル間のデータ欠損が多く、検証の信頼性が低下します)"
            )

    series_name = "z" if config.mode == "stat_arb" else "dev_pips"
    skipped_entries = {"beta_degenerate": 0, "balance_nonpositive": 0}

    # §1.3: 縮退(空result)。エラーは送出せず、§7の全キーを持つ空resultを返す。
    zero_row_symbols = [s for s in symbols if rows_per_symbol[s] == 0]
    min_rows = (config.stat_arb.lookback + 3) if config.mode == "stat_arb" else 3
    if zero_row_symbols or aligned_rows < min_rows:
        if zero_row_symbols:
            warnings.append(
                f"行数0のシンボルがあるため空resultで縮退しました: "
                f"{', '.join(zero_row_symbols)}"
            )
        else:
            warnings.append(
                f"整列後のバー数が不足しているため空resultで縮退しました: "
                f"aligned_rows={aligned_rows} < 必要最小{min_rows}"
            )
        return {
            "trades": [],
            "equity_curve": [
                {"timestamp": ts, "equity": initial_balance} for ts in merged["timestamp"]
            ],
            "initial_balance": initial_balance,
            "warnings": warnings,
            "alignment": alignment,
            "divergence_stats": _empty_divergence_stats(series_name),
            "skipped_entries": skipped_entries,
            "max_gross_exposure_lots": 0.0,
            "theoretical_max_loss": theoretical_max_loss,
            "disclaimer": ARB_DISCLAIMER,  # §10: 再入力せずそのまま代入
        }

    n = aligned_rows
    timestamps: List[Any] = list(merged["timestamp"])

    # レッグ静的属性(leg_index順)。§5のコストモデルはレッグごとに独立して適用する。
    pip_sizes = [_pip_size(s) for s in symbols]
    spreads = [config.spread_overrides.get(s, spread_pips) for s in symbols]  # §5 レッグ別スプレッド
    extras = [0.0 if k == 0 else extra_leg_slippage_pips for k in range(len(symbols))]  # §5 leg_index>=1のみ

    opens = [merged[f"open_{s}"].to_numpy(dtype=float) for s in symbols]
    highs = [merged[f"high_{s}"].to_numpy(dtype=float) for s in symbols]
    lows = [merged[f"low_{s}"].to_numpy(dtype=float) for s in symbols]
    closes = [merged[f"close_{s}"].to_numpy(dtype=float) for s in symbols]

    # ------------------------------------------------------------------
    # §3 シグナル定義(全ローリング統計・シグナル・ロット計算用価格はshift(1)後に参照、§4)
    # ------------------------------------------------------------------
    beta_sig = closeA_sig = closeB_sig = None  # stat_arb用(closureから参照)
    tri_hedge_close_sig = None  # triangular用: leg2ロット式が参照するshift(1)済みclose

    if config.mode == "stat_arb":
        p = config.stat_arb
        close_a_series = merged[f"close_{p.symbols[0]}"].astype(float)
        close_b_series = merged[f"close_{p.symbols[1]}"].astype(float)
        log_a = np.log(close_a_series)
        log_b = np.log(close_b_series)
        if p.hedge_mode == "rolling_ols":
            # §3-A-2: beta = rolling_cov / rolling_var(ddofは両者でpandas既定=1、比率に影響なし)
            beta = log_a.rolling(p.lookback).cov(log_b) / log_b.rolling(p.lookback).var()
        else:  # fixed_1
            beta = pd.Series(1.0, index=merged.index)
        s_series = log_a - beta * log_b
        mu = s_series.rolling(p.lookback).mean()
        sd = s_series.rolling(p.lookback).std(ddof=1)
        # σのノイズ水準ガード: スプレッドが実質定数(完全相関ペア等)のときσは浮動小数点誤差
        # (~1e-16)の正値になる。これを「有効なσ」と扱うとzが数値ノイズだけで生成され、
        # §6統計に人工的な有効バーが混入する。スプレッド水準に対して無視できるσは
        # 「σ=0(観測不能)」と同一視する(§3-A-3の数値的堅牢化、意味論は不変)。
        s_abs_max = s_series.abs().max()
        s_scale = 0.0 if pd.isna(s_abs_max) else float(s_abs_max)  # 全NaN系列でもNaN伝播させない
        noise_eps = max(1e-12, s_scale * 1e-13)  # σ・分子共通のノイズ閾値(監査指摘: 二重定義を統一)
        sigma_valid = sd.notna() & (sd > noise_eps)
        # 分子(s−μ)の数値ノイズガード: 完全相関ペア等でスプレッドが実質定数のとき、
        # 分子・分母とも浮動小数点誤差レベル(~1e-16)になり「ノイズ÷ノイズ」のzがO(1)で振れて
        # 偶発エントリーする。分子がスプレッド水準に対して無視できる大きさなら0に落とす
        # (実注入乖離のような「ゼロノイズ基線に対する本物の乖離」は分子が大きいため影響しない。
        # §3-A-3のσ=0規則を数値的に堅牢化する実装判断、specの意味論は不変)。
        num = s_series - mu
        # NaN(ウォームアップ期間)はNaNのまま保持する(0に潰すと「不明」が「シグナル0確定」に
        # 化けて参照実装とズレる。本プロジェクトで繰り返し出たNaN→False/0の早期潰しの教訓)。
        num = num.where(num.isna() | (num.abs() > noise_eps), 0.0)
        # §6用: σが無効なバーをNaNのまま保持する生z系列(NaN除外統計の対象)
        z_stats = num / sd.where(sigma_valid)
        # §3-A-3: rolling_stdが0またはNaNのバーは z_t = 0 とみなす(エントリー不能。
        # 保有中は|z|<=exit_z扱いでconverge決済され得るが、σ=0はスプレッドが動いていない
        # 退化状態なので手仕舞いは安全側)。
        sig_series = z_stats.fillna(0.0)
        raw_entry = _edge_trigger(sig_series, p.entry_z, p.stop_z)
        exit_th, stop_th = p.exit_z, p.stop_z
        divergence_stats = _divergence_stats("z", z_stats, p.entry_z, s_series)
        beta_sig = beta.shift(1).to_numpy(dtype=float)
        closeA_sig = close_a_series.shift(1).to_numpy(dtype=float)
        closeB_sig = close_b_series.shift(1).to_numpy(dtype=float)

        # §9警告3: rolling_corr < 0.3 のバーが有効バーの10%超(shift不要の記述統計)
        corr = log_a.rolling(p.lookback).corr(log_b)
        corr_valid_count = int(corr.notna().sum())
        if corr_valid_count > 0:
            low_corr_pct = float((corr < 0.3).sum()) / corr_valid_count * 100.0
            if low_corr_pct > 10.0:
                warnings.append(
                    f"ペアの相関が不安定な期間が{low_corr_pct:.1f}%あります"
                    "(ペア選択自体が不適切な可能性)"
                )
    else:  # triangular
        t = config.triangular
        close_cross_series = merged[f"close_{t.cross_symbol}"].astype(float)
        close_la_series = merged[f"close_{t.leg_symbols[0]}"].astype(float)
        close_lb_series = merged[f"close_{t.leg_symbols[1]}"].astype(float)
        if t.composition == "product":
            synth = close_la_series * close_lb_series
        else:  # quotient
            synth = close_la_series / close_lb_series
        # §3-B-1: dev の単位は cross のpips
        dev = (close_cross_series - synth) / pip_sizes[0]
        sig_series = dev
        raw_entry = _edge_trigger(dev, t.entry_dev_pips, t.stop_dev_pips)
        exit_th, stop_th = t.exit_dev_pips, t.stop_dev_pips
        divergence_stats = _divergence_stats("dev_pips", dev, t.entry_dev_pips, dev)
        # §3-B-4: leg2ロット式が参照するshift(1)済みclose(product→legA、quotient→cross)
        if t.composition == "product":
            tri_hedge_close_sig = close_la_series.shift(1).to_numpy(dtype=float)
        else:
            tri_hedge_close_sig = close_cross_series.shift(1).to_numpy(dtype=float)

    # §4: 生系列をバーtの判定に使うのは先読み。すべてshift(1)後の値をバーiで参照する。
    sig_arr = sig_series.shift(1).to_numpy(dtype=float)  # z_sig / dev_sig
    actionable_arr = raw_entry.shift(1).fillna(0).astype(int).to_numpy()  # NaNは0扱い

    # ------------------------------------------------------------------
    # §2 バー内判定順序(正規順序)
    # ------------------------------------------------------------------
    trades: List[Dict[str, Any]] = []
    # §2.0b: バー0でequity_curveをシードする(engine.pyと同一)
    equity_curve: List[Dict[str, Any]] = [
        {"timestamp": timestamps[0], "equity": initial_balance}
    ]
    balance = initial_balance
    basket: Optional[Dict[str, Any]] = None
    last_basket_close_bar_index: Optional[int] = None
    max_gross_exposure_lots = 0.0
    bankrupt_warned = False
    tri_rounding_error_entries = 0  # §3-B-4: 丸め残余ヘッジ誤差5%超のエントリー件数

    max_hold_bars = config.risk.max_hold_bars
    max_floating_loss_pct = config.risk.max_floating_loss_pct
    reentry_wait_bars = config.risk.reentry_wait_bars

    def _entry_exec_price(k: int, raw: float, side: int) -> float:
        # §5 エントリー時: (spread_k + slippage + extra_k) を不利方向へ加算
        cost = (spreads[k] + slippage_pips + extras[k]) * pip_sizes[k]
        return raw + cost if side == 1 else raw - cost

    def _exit_exec_price(k: int, raw: float, side: int) -> float:
        # §5 決済時(converge/stop_z/basket_sl/time/eodの全て、例外なし):
        # (slippage + extra_k) のみを不利方向へ加算(スプレッドはエントリー側で片道分のみ)
        cost = (slippage_pips + extras[k]) * pip_sizes[k]
        return raw - cost if side == 1 else raw + cost

    def _floating_pnl(mark_prices: List[float]) -> float:
        """§2.1-a / §2.3 の含み損益合算(markはレッグごとの評価価格、コスト控除なし)。"""
        total = 0.0
        for leg, mark in zip(basket["legs"], mark_prices):
            k = leg["leg_index"]
            total += (
                leg["side"]
                * (mark - leg["entry_price"])
                / pip_sizes[k]
                * _pip_value_per_lot(leg["symbol"], pip_sizes[k], mark)
                * leg["lot"]
            )
        return total

    def _basket_sl_state(i: int) -> Tuple[bool, List[float]]:
        """§2.1-a: 各レッグのバー内最悪値(long→low / short→high)を「同時に成立した」と
        みなす過剰悲観評価。実際には全レッグ同時に最悪値を付けることはまず無いため、
        損失を過大評価する方向にしか外れない(保守側の虚構のみ許容、§2.0)。"""
        marks = [
            float(lows[leg["leg_index"]][i]) if leg["side"] == 1 else float(highs[leg["leg_index"]][i])
            for leg in basket["legs"]
        ]
        floating_worst = _floating_pnl(marks)
        # balanceは確定済み残高(initial_balanceではない、§2.1-a)
        return (-floating_worst) >= balance * max_floating_loss_pct / 100.0, marks

    def _open_prices(i: int) -> List[float]:
        return [float(opens[leg["leg_index"]][i]) for leg in basket["legs"]]

    def _close_basket(raw_prices: List[float], exit_time: Any, exit_kind: str) -> None:
        """全レッグ一括決済(§2.1)。1レッグ=1レコードでtradesへ追加し(§7)、
        balance += Σ profit、バスケットを空にする。"""
        nonlocal balance, basket, bankrupt_warned
        assert basket is not None
        for leg, raw in zip(basket["legs"], raw_prices):
            k = leg["leg_index"]
            side = leg["side"]
            exit_price = _exit_exec_price(k, float(raw), side)
            if side == 1:
                pips = (exit_price - leg["entry_price"]) / pip_sizes[k]
            else:
                pips = (leg["entry_price"] - exit_price) / pip_sizes[k]
            profit = (
                pips * _pip_value_per_lot(leg["symbol"], pip_sizes[k], exit_price) * leg["lot"]
                - commission_per_lot * leg["lot"]
            )
            trades.append(
                {
                    "entry_time": leg["entry_time"],
                    "exit_time": exit_time,
                    "side": "long" if side == 1 else "short",
                    "entry_price": leg["entry_price"],
                    "exit_price": exit_price,
                    "lot": leg["lot"],
                    "pips": pips,
                    "profit": profit,
                    "exit_kind": exit_kind,
                    "position_id": basket["position_id"],
                    "leg_index": k,
                    "symbol": leg["symbol"],
                }
            )
            balance += profit
        basket = None
        # §9警告6: 最初にbalance<=0へ到達した決済処理の時点で1回だけ追加(§2.2条件3)
        if balance <= 0 and not bankrupt_warned:
            warnings.append(
                "口座破綻: 残高が0以下になったため、以降の新規エントリーを停止しました"
            )
            bankrupt_warned = True

    def _compute_entry_legs(i: int) -> Optional[Dict[str, Any]]:
        """§3-A-5 / §3-B-3,4: エントリーバーiのレッグ(side, lot)列。

        退化条件に該当したら None(呼び出し側で skipped_entries["beta_degenerate"] を計上。
        triangularのロット退化も§7でスキーマを固定するため同じカウンタに計上する)。
        方向・ロットはエントリー時点で確定し、以後決済まで凍結される(§3-A-6:
        保有中のz_sigはローリング系列のため、converge決済でも損失はあり得る)。
        """
        direction = int(actionable_arr[i])
        if config.mode == "stat_arb":
            p = config.stat_arb
            if p.hedge_mode == "rolling_ols":
                b = float(beta_sig[i])
                # §3-A-5: NaN・0・|β|>10 は全lot_sizingモード共通の退化見送り
                # (βが退化している時点でスプレッド自体が信頼できない)。±infも
                # var=0由来の退化としてNaN同等に扱う。
                if not np.isfinite(b) or b == 0.0 or abs(b) > 10.0:
                    return None
            else:
                b = 1.0
            side_a = direction  # legA_side = -sign(z_sig[i]) = actionable_entry[i](§3-A-5)
            side_b = -side_a * (1 if b > 0 else -1)  # β>0なら逆方向ヘッジ、β<0なら同方向
            if p.lot_sizing == "equal_lots":
                lot_a = lot_b = config.base_lot
            elif p.lot_sizing == "beta_notional":
                # §4-2: ロット計算に使う価格もshift(1)済みclose
                lot_a = config.base_lot
                lot_b = round(abs(b) * config.base_lot * closeA_sig[i] / closeB_sig[i], 2)
                if not np.isfinite(lot_b) or lot_b < 0.01:
                    return None  # round後のlotBが最小取引単位割れ(§3-A-5)
            else:  # manual(manual_lotsの存在はpydanticで保証済み)
                lot_a, lot_b = p.manual_lots[0], p.manual_lots[1]
            return {
                "legs": [(side_a, float(lot_a)), (side_b, float(lot_b))],
                "rounding_exceeded": False,
                "frozen_beta": b,
            }

        # triangular: §3-B-3 レッグ方向の符号表(完全固定)
        #   composition  dir  leg0(cross)  leg1(legA)  leg2(legB)
        #   product      +1   long         short       short
        #   product      -1   short        long        long
        #   quotient     +1   long         short       long
        #   quotient     -1   short        long        short
        t = config.triangular
        side0 = direction
        side1 = -direction
        side2 = -direction if t.composition == "product" else direction
        # §3-B-4: leg2ロット。product→base_lot*close_legA_sig、quotient→base_lot*close_cross_sig
        ideal_lot2 = config.base_lot * float(tri_hedge_close_sig[i])
        lot2 = round(ideal_lot2, 2)
        if not np.isfinite(lot2) or lot2 < 0.01:
            return None  # round後<0.01は見送り(beta_degenerateに計上、§3-B-4)
        rounding_exceeded = ideal_lot2 > 0 and abs(lot2 - ideal_lot2) / ideal_lot2 > 0.05
        return {
            "legs": [
                (side0, float(config.base_lot)),
                (side1, float(config.base_lot)),
                (side2, float(lot2)),
            ],
            "rounding_exceeded": rounding_exceeded,
            "frozen_beta": None,
        }

    for i in range(1, n):
        ts_i = timestamps[i]
        exited_this_bar = False

        # ---- §2.1 決済判定(バーi開始時点でバスケット保有中の場合のみ)----
        # a→b→c→d は相互排他のelif連鎖(a最優先: 破滅的損失の遮断は他のどの判定よりも先、
        # かつaだけがバー内最悪値を使うため先に評価しないと楽観側を選んでしまう。§2.5)。
        if basket is not None:
            sl_hit, sl_marks = _basket_sl_state(i)
            if sl_hit:
                # a. basket_sl: 各レッグをバー内最悪値markに§5決済コストを適用して決済
                _close_basket(sl_marks, ts_i, "basket_sl")
            elif abs(sig_arr[i]) >= stop_th:
                # b. 発散ストップ(triangularでもexit_kindは"stop_z"で統一、§2.1-b)
                _close_basket(_open_prices(i), ts_i, "stop_z")
            elif abs(sig_arr[i]) <= exit_th:
                # c. 収束決済(exit_z < stop_z がpydanticで保証されるためbと同時成立しない)
                _close_basket(_open_prices(i), ts_i, "converge")
            elif i - basket["basket_open_bar_index"] >= max_hold_bars:
                # d. 時間切れ(b/c/dの執行価格はいずれもopen_k[i]: 判定材料sig[i]は
                # バーi-1 close時点の確定情報→次バー始値で執行の規律に合わせる、§2.1-d)
                _close_basket(_open_prices(i), ts_i, "time")
            if basket is None:
                exited_this_bar = True
                last_basket_close_bar_index = i

        # ---- §2.2 新規エントリー判定(バスケット無し、かつ同一バーで決済していない場合のみ。
        # 決済と新規の判定材料が同じsig[i]のため、同一バー執行は実質ドテンとなり裁定の
        # 意味論を壊す)。成立条件は番号順に評価し、途中で不成立なら以降は評価しない。----
        if basket is None and not exited_this_bar and actionable_arr[i] != 0:  # 条件1
            reentry_ok = (
                last_basket_close_bar_index is None
                or (i - last_basket_close_bar_index - 1) >= reentry_wait_bars
            )  # 条件2(nanpin_spec §2.1条件2と同一式)
            if reentry_ok:
                if balance <= 0:  # 条件3
                    skipped_entries["balance_nonpositive"] += 1
                else:
                    computed = _compute_entry_legs(i)  # 条件4
                    if computed is None:
                        skipped_entries["beta_degenerate"] += 1
                    else:
                        if computed["rounding_exceeded"]:
                            tri_rounding_error_entries += 1
                        legs = [
                            {
                                "symbol": symbols[k],
                                "side": side_k,
                                "lot": lot_k,
                                # 全レッグを同一バーiの各自のopen_k[i]に§5エントリーコストを
                                # 適用した価格で同時に建てる(§2.2)
                                "entry_price": _entry_exec_price(k, float(opens[k][i]), side_k),
                                "entry_time": ts_i,
                                "leg_index": k,
                            }
                            for k, (side_k, lot_k) in enumerate(computed["legs"])
                        ]
                        basket = {
                            "legs": legs,
                            "basket_open_bar_index": i,
                            "position_id": str(ts_i),  # nanpinと同一規約
                            # §3-A-5/§2.2: エントリー時点のβを凍結保持(方向・ロットの算出根拠。
                            # 保有中の決済判定はローリングのz_sigを使い続ける、§3-A-6)
                            "frozen_beta": computed["frozen_beta"],
                        }
                        gross = sum(leg["lot"] for leg in legs)
                        if gross > max_gross_exposure_lots:
                            max_gross_exposure_lots = gross  # §8
                        # §2.2末尾: エントリーした同一バーiで追加判定するのは§2.1-a(basket_sl)
                        # のみ(エントリーバー自身の値幅による即時ロスカットは保守側評価として
                        # 必要。b/cはエッジトリガー定義により同一バーでは構成上成立し得ず、
                        # dもmax_hold_bars>=1のため成立し得ない)。
                        sl_hit, sl_marks = _basket_sl_state(i)
                        if sl_hit:
                            _close_basket(sl_marks, ts_i, "basket_sl")
                            last_basket_close_bar_index = i

        # ---- §2.3 equity_curveへの記録(毎バー、§2.1/§2.2の判定後に1回)----
        # mark-to-marketのmarkはclose_k[i]で決済コストは控除しない(§2.1-aの悲観判定は
        # low/high、レポーティングはclose。意図的に別のmark priceを使っている。§2.3)。
        if basket is not None:
            closes_i = [float(closes[leg["leg_index"]][i]) for leg in basket["legs"]]
            equity = balance + _floating_pnl(closes_i)
        else:
            equity = balance
        equity_curve.append({"timestamp": ts_i, "equity": float(equity)})

    # ---- §2.4 最終バー処理(ループ終了後)----
    if basket is not None:
        last_i = n - 1
        raw_closes = [float(closes[leg["leg_index"]][last_i]) for leg in basket["legs"]]
        _close_basket(raw_closes, timestamps[last_i], "eod")
        # 上書きしないと最終バーだけコスト控除前の評価額が残り最終balanceと食い違う(§2.4)
        equity_curve[-1]["equity"] = float(balance)

    # ---- §9 動的警告(エンジン付与分。警告4はPhase2-CのAPI層が付与する)----
    total_skipped = skipped_entries["beta_degenerate"] + skipped_entries["balance_nonpositive"]
    if total_skipped >= 1:
        # §9警告5
        warnings.append(
            f"{total_skipped}件のエントリーを見送りました(内訳: "
            f"beta_degenerate={skipped_entries['beta_degenerate']}, "
            f"balance_nonpositive={skipped_entries['balance_nonpositive']})"
        )
    if config.mode == "triangular":
        if tri_rounding_error_entries >= 1:
            # §3-B-4: 丸め残余ヘッジ誤差(base_lotが小さいほど丸めの相対影響が大きく、
            # 「ヘッジされているつもりの裸のエクスポージャー」が残る)
            warnings.append(
                f"{tri_rounding_error_entries}件のエントリーでleg2ロットの丸め誤差が"
                "理想ヘッジ比率の5%を超えています(丸め残余の裸のエクスポージャーが残り、"
                "ヘッジ精度が低下しています)"
            )
        total_profit = sum(t_["profit"] for t_ in trades)
        effective_spread_sum = float(sum(spreads))
        if total_profit > 0 and effective_spread_sum < 3.0:
            # §9警告7
            warnings.append(
                f"利益が出ていますが、3レッグ合計{effective_spread_sum:.1f}pipsの"
                "スプレッド設定は非現実的に有利です(現実的なコストでは消える見かけの"
                "利益の可能性が高い)"
            )

    # ---- §7 result辞書(完全固定スキーマ。trades/equity_curve/initial_balanceは
    # engine.pyの戻り値と完全互換で、そのままcalculate_metricsに渡せる)----
    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "initial_balance": initial_balance,
        "warnings": warnings,
        "alignment": alignment,
        "divergence_stats": divergence_stats,
        "skipped_entries": skipped_entries,
        "max_gross_exposure_lots": max_gross_exposure_lots,
        "theoretical_max_loss": theoretical_max_loss,
        "disclaimer": ARB_DISCLAIMER,  # §10: 再入力せずそのまま代入
    }
