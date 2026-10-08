"""バックテストエンジン。

設計上の要点:
- シグナルは確定バー(shift済み)の値から算出し、エントリーは次バーの始値で行う(先読みバイアス回避)。
- SL/TPが同一バーで両方ヒットした場合はSLを優先する(保守的な想定)。

このモジュールのバー内判定順序(エントリー→SL→分割決済→TP→max_hold_bars→trailing更新)は
docs/exit_rules_spec.md の正規仕様であり、生成EA(app/core/mql_gen.py)もこの順序に従う。
この順序を変更する場合は仕様書とEXIT_SPEC_VERSIONを同時に更新すること。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core import newsfeed, sizing, templates
from app.core.indicators import atr, donchian_channel, rsi, sma
from app.core.strategy_model import Strategy

EXIT_SPEC_VERSION = "1"  # docs/exit_rules_spec.md 参照

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OHLC_DIR = BASE_DIR / "data" / "ohlc"  # mtf_confirmフィルタが上位足parquetを読むための基準ディレクトリ


from app.core import symbols  # noqa: E402 - pip定義・契約サイズの単一の真実(Wave5 案E)


def _pip_size(symbol: str) -> float:
    """pip定義: app/core/symbols.py へ委譲(XAUUSD=0.1/XAGUSD=0.01/JPYクオート=0.01/他=0.0001)。"""
    return symbols.pip_size(symbol)


def _pip_value_per_lot(symbol: str, pip_size: float, price: float) -> float:
    """1lotあたりのpip価値をUSD建てで概算する(app/core/symbols.py へ委譲)。

    契約サイズ(FX=100,000通貨、XAU=100oz、XAG=5,000oz)とJPYクオートのUSD換算近似は
    symbols.pip_value_per_lot が単一の真実として持つ。pip_size引数は既存シグネチャ互換の
    ため残し、そのまま委譲する(呼び出し元は常に _pip_size(symbol) と同値を渡す)。
    """
    return symbols.pip_value_per_lot(symbol, price, pip_size_override=pip_size)


def _symbol_currencies(symbol: str) -> List[str]:
    """symbol("EURUSD"等)を3文字ずつに分割して構成通貨コードを返す簡易ロジック。

    NewsFilter.currencies未指定時、strategy.symbolから対象通貨を自動判定するために使う。
    6文字以外の非対応シンボルはそのまま1要素のリストとして返す。
    """
    s = symbol.upper()
    if len(s) == 6:
        return [s[:3], s[3:]]
    return [s]


def _signal_ma_cross(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    fast_period = int(strategy.params["fast_period"].value)
    slow_period = int(strategy.params["slow_period"].value)
    fast = sma(df["close"], fast_period)
    slow = sma(df["close"], slow_period)
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    golden = (prev_fast <= prev_slow) & (fast > slow)
    dead = (prev_fast >= prev_slow) & (fast < slow)
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[golden.fillna(False)] = 1
    signal[dead.fillna(False)] = -1
    return signal


def _signal_rsi_reversal(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    period = int(strategy.params["period"].value)
    lower = float(strategy.params["lower"].value)
    upper = float(strategy.params["upper"].value)
    r = rsi(df["close"], period)
    prev_r = r.shift(1)
    # 下限からの回復(逆張り買い)・上限からの反落(逆張り売り)を検知する
    recover_from_lower = (prev_r <= lower) & (r > lower)
    fall_from_upper = (prev_r >= upper) & (r < upper)
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[recover_from_lower.fillna(False)] = 1
    signal[fall_from_upper.fillna(False)] = -1
    return signal


def _signal_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    lookback_bars = int(strategy.params["lookback_bars"].value)
    upper, lower = donchian_channel(df["high"], df["low"], lookback_bars)
    breakout_up = df["close"] > upper
    breakout_down = df["close"] < lower
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[breakout_up.fillna(False)] = 1
    signal[breakout_down.fillna(False)] = -1
    return signal


templates.register(
    "ma_cross",
    defaults={"fast_period": 10, "slow_period": 50, "sl_pips": 30, "tp_pips": 60, "lot": 0.1},
    signal_fn=_signal_ma_cross,
)
templates.register(
    "rsi_reversal",
    defaults={"period": 14, "lower": 30, "upper": 70, "sl_pips": 30, "tp_pips": 60, "lot": 0.1},
    signal_fn=_signal_rsi_reversal,
)
templates.register(
    "breakout",
    defaults={"lookback_bars": 20, "sl_pips": 30, "tp_pips": 60, "lot": 0.1},
    signal_fn=_signal_breakout,
)


def _generate_signals(strategy: Strategy, df: pd.DataFrame) -> templates.SignalResult:
    """テンプレート別のシグナル生成。1=買い, -1=売り, 0=シグナルなし(後方互換シム)。

    値はそのバーの終値時点で確定した情報のみを用いる(まだシフトしていない)。
    新テンプレート追加時はここを編集せず、各テンプレートのモジュールで
    `templates.register()` すること。

    戻り値は pd.Series(既存テンプレート、1/-1/0のみ)、または
    "signal"列(必須)に加え"sl_price"/"tp_price"/"confidence"列(任意)を持つ
    pd.DataFrame(構造的SL/TP・confidence伝播に対応するテンプレート用)のいずれか。
    """
    return templates.get(strategy.template).signal_fn(strategy, df)


def _value_or_none(series: Optional[pd.Series], i: int) -> Optional[float]:
    """seriesがNone、またはi番目の値がNaNならNoneを返す(構造的SL/TP未指定時のフォールバック判定用)。"""
    if series is None:
        return None
    v = series.iloc[i]
    return None if pd.isna(v) else float(v)


def _confidence_or_default(series: Optional[pd.Series], i: int) -> float:
    """seriesがNone、またはi番目の値がNaNなら1.0(デフォルトconfidence)を返す。"""
    if series is None:
        return 1.0
    v = series.iloc[i]
    return 1.0 if pd.isna(v) else float(v)


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
    commission_per_lot: float = 0.0,
    initial_balance: float = 10000.0,
) -> Dict[str, Any]:
    pip_size = _pip_size(strategy.symbol)

    money = strategy.money
    lot = float(strategy.params["lot"].value)
    sl_pips = float(strategy.params["sl_pips"].value)
    tp_pips = float(strategy.params["tp_pips"].value)
    trailing_param = strategy.params.get("trailing_pips")
    trailing_pips = float(trailing_param.value) if trailing_param is not None else None
    max_hold_bars_param = strategy.params.get("max_hold_bars")
    max_hold_bars = int(max_hold_bars_param.value) if max_hold_bars_param is not None else None
    # 指値エントリー(オプトイン): シグナル翌バーで成行せず、シグナルバー始値からoffset分
    # 有利な価格に1バー限りの指値を置く。到達すれば指値+スプレッドで約定(スリッページ0)、
    # 未到達なら破棄。平均回帰系のエントリーをテイカーからメイカー側へ寄せ、
    # 実効コストを下げるための機能(リアルティック検証での夜間スプレッド問題への対策)。
    limit_offset_param = strategy.params.get("limit_offset_pips")
    limit_offset_pips = float(limit_offset_param.value) if limit_offset_param is not None else None
    trade_hours = strategy.filters.trade_hours if strategy.filters else None
    max_spread_pips = strategy.filters.max_spread_pips if strategy.filters else None
    news_filter = strategy.filters.news if strategy.filters else None
    mtf_confirm = strategy.filters.mtf_confirm if strategy.filters else None
    vol_regime = strategy.filters.vol_regime if strategy.filters else None
    # spread_pipsはバックテスト全体で一定の想定のため、フィルタは全期間一括のON/OFFとして働く
    # (バーごとの実測スプレッドは扱っていない簡易実装)。
    spread_filter_ok = max_spread_pips is None or spread_pips <= max_spread_pips

    df = df.reset_index(drop=True)

    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if len(df) == 0:
        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "initial_balance": initial_balance,
            "warnings": warnings,
        }

    # volume列に依存するテンプレート(volume_spike/orderflow_divergence)は、volumeが
    # 全て0/欠損の場合いずれも「無言でトレード0件」または「静かなフォールバック」に
    # 縮退する(FABLE監査で発見)。原因が分かるよう汎用的に警告しておく。
    _VOLUME_DEPENDENT_TEMPLATES = {"volume_spike", "orderflow_divergence"}
    if strategy.template in _VOLUME_DEPENDENT_TEMPLATES and "volume" in df.columns:
        if not bool((df["volume"].fillna(0.0) != 0.0).any()):
            warnings.append(
                f"{strategy.template}はvolume列を使用しますが、このデータはvolumeが"
                "全て0または欠損です。シグナルが正しく機能しない可能性があります"
                "(CSVインポート元にvolume情報が含まれているか確認してください)"
            )

    # newsフィルタ用のnews_mask(バーがニュース近傍か)をバーループに入る前に1回だけ計算する。
    # カレンダー未取得(FileNotFoundError)の場合はバックテスト自体は失敗させず、
    # news_maskを全Falseにしたうえで警告を残して続行する(newsフィルタ無効化)。
    news_mask = pd.Series(False, index=df.index)
    if news_filter is not None:
        try:
            calendar = newsfeed.load_calendar()
        except FileNotFoundError as exc:
            if news_filter.mode == "only":
                # mode="only"はnews_mask全Falseだと新規エントリーが恒久的に0件になる
                # (「無効化」ではなく「全遮断」になる)ため、avoidと文言を分ける。
                warnings.append(
                    f"newsフィルタ(mode='only')がカレンダー未取得のため新規エントリーを"
                    f"全てブロックしています: {exc}"
                )
            else:
                warnings.append(f"newsフィルタを無効化しました(カレンダー未取得): {exc}")
            calendar = None
        if calendar is not None:
            currencies = news_filter.currencies or _symbol_currencies(strategy.symbol)
            news_mask = newsfeed.is_near_news(
                df["timestamp"],
                calendar,
                currencies,
                news_filter.min_impact,
                news_filter.before_min,
                news_filter.after_min,
            )
            if not bool(news_mask.any()):
                warnings.append(
                    "newsフィルタ: カレンダーは取得済みですが、指定期間・通貨・impact条件に"
                    "一致するイベントが0件でした(currencies/min_impact/データ期間を確認してください)"
                )

    # mtf_confirmフィルタ用のmtf_allow_long/mtf_allow_short(上位足トレンドと整合する方向)を
    # バーループに入る前に1回だけ計算する。上位足parquet未取得(FileNotFoundError)の場合は
    # バックテスト自体は失敗させず、両方全Trueにしたうえで警告を残して続行する(newsフィルタと同じ方針)。
    mtf_allow_long = pd.Series(True, index=df.index)
    mtf_allow_short = pd.Series(True, index=df.index)
    if mtf_confirm is not None:
        htf_path = OHLC_DIR / f"{strategy.symbol}_{mtf_confirm.timeframe.upper()}.parquet"
        try:
            htf_df = pd.read_parquet(htf_path)
        except FileNotFoundError:
            htf_df = None
            warnings.append(
                f"mtf_confirmフィルタを無効化しました(上位足{mtf_confirm.timeframe}のデータ未取得): "
                f"先にデータタブで{strategy.symbol}の{mtf_confirm.timeframe}をダウンロード・リサンプルしてください"
            )
        if htf_df is not None:
            htf_sma = sma(htf_df["close"], mtf_confirm.trend_period)
            # SMAウォームアップ中(NaN)は"不明"のままNaNとして残し、Falseに丸めない。
            # (FABLE監査で発見: ここでfillna(False)すると「トレンド不明」が「下降トレンド確定」
            # として扱われ、ウォームアップ期間中ロングだけが恒久的にブロックされる非対称バイアスに
            # なっていた。NaNのまま伝播させ、後段のknownマスクで両方向ブロックに落とす。)
            htf_trend_up = np.where(htf_sma.notna(), (htf_df["close"] > htf_sma).astype(float), np.nan)
            htf_trend_up = pd.Series(htf_trend_up, index=htf_df.index)
            # 先読み回避: 上位足バーの終値が確定するのは「バー開始時刻+上位足の時間幅」経過後
            # (donchian_channelのshift(1)・swings.confirmed_at_idxと同じ規律)。1本shiftすることで
            # 「そのバー時点で本当に確定済みだった上位足トレンド」だけを以降の参照対象にする。
            htf_trend_confirmed = htf_trend_up.shift(1)

            # merge_asof(direction="backward")で、下位足の各バーに「その時点で参照可能な直近の
            # 確定済み上位足トレンド」を割り当てる(fib_harmonic_templateのmerge_asofパターンを踏襲)。
            # _posで元のdf行順を保持し、timestampソートで一時的に並べ替えても結果を正しく戻せるようにする。
            lower_key = pd.DataFrame({"timestamp": df["timestamp"], "_pos": df.index}).sort_values(
                "timestamp", kind="stable"
            )
            htf_key = pd.DataFrame(
                {"timestamp": htf_df["timestamp"], "trend_up": htf_trend_confirmed}
            ).sort_values("timestamp", kind="stable")
            merged = pd.merge_asof(lower_key, htf_key, on="timestamp", direction="backward")
            trend_col = merged.sort_values("_pos", kind="stable")["trend_up"].reset_index(drop=True)

            # 上位足データがまだ無い期間(mergeで割り当てられなかった=NaN)は安全側で両方Falseにする。
            known = trend_col.notna()
            trend_bool = trend_col.fillna(0.0).astype(bool)
            if mtf_confirm.mode == "with_trend":
                mtf_allow_long = trend_bool & known
                mtf_allow_short = (~trend_bool) & known
            else:  # against_trend: 上位足トレンドと逆方向のみ許可(逆張り戦略向け)
                mtf_allow_long = (~trend_bool) & known
                mtf_allow_short = trend_bool & known

            if not bool(known.any()):
                # 上位足parquetは存在するが期間が下位足データと重ならない等でmergeが
                # 全く割り当てられなかった場合、無警告のまま全エントリーが恒久ブロックされる
                # (newsフィルタの「該当0件」警告と同じ思想で可視化する、FABLE監査で発見)。
                warnings.append(
                    f"mtf_confirmフィルタ: 上位足{mtf_confirm.timeframe}のデータが下位足の期間と"
                    "重なっていないため、全エントリーをブロックしています"
                )

    # ボラレジームゲート(Wave5 案D): ATR百分位順位をバーループ前に一括計算する。
    # rank.shift(1)で「その時点で確定済みの順位」のみを参照(先読み回避、シグナルもshift済み)。
    vol_regime_ok = None
    if vol_regime is not None:
        from app.core.novel_indicators import atr_percentile_rank

        a = atr(df["high"], df["low"], df["close"], vol_regime.atr_period)
        rank = atr_percentile_rank(a, vol_regime.percentile_lookback).shift(1)
        # ウォームアップNaNは「不明」=停止側(安全側、mtf_confirmのknownマスクと同思想。
        # fillna(False)で直接潰すと「不明」が「許可」に化ける非対称バイアスを避ける)。
        allowed = rank.notna()
        if vol_regime.max_percentile is not None:
            allowed = allowed & (rank <= vol_regime.max_percentile)
        if vol_regime.min_percentile is not None:
            allowed = allowed & (rank >= vol_regime.min_percentile)
        vol_regime_ok = allowed
        if not bool(allowed.any()):
            warnings.append(
                "vol_regimeゲート: 全バーがボラティリティ範囲外(またはウォームアップ)のため、"
                "全エントリーをブロックしています(percentile_lookbackがデータ長に対して大きすぎる可能性)"
            )

    signal_result = _generate_signals(strategy, df)
    if isinstance(signal_result, pd.DataFrame):
        raw_signal = signal_result["signal"]
        raw_sl_price = signal_result["sl_price"] if "sl_price" in signal_result.columns else None
        raw_tp_price = signal_result["tp_price"] if "tp_price" in signal_result.columns else None
        raw_confidence = signal_result["confidence"] if "confidence" in signal_result.columns else None
    else:
        raw_signal = signal_result
        raw_sl_price = raw_tp_price = raw_confidence = None

    actionable = raw_signal.shift(1).fillna(0).astype(int)  # 次バー始値で執行するためシフト
    actionable_sl_price = raw_sl_price.shift(1) if raw_sl_price is not None else None
    actionable_tp_price = raw_tp_price.shift(1) if raw_tp_price is not None else None
    actionable_confidence = raw_confidence.shift(1) if raw_confidence is not None else None

    balance = initial_balance
    position: Optional[Dict[str, Any]] = None  # {"side", "entry_price", "entry_time", "sl", "tp"}
    position_id_counter = 0  # 新規・ドテン再エントリーのたびに新しいIDを払い出す(run_backtest呼び出しごとにローカル)

    def _hour_allowed(ts: pd.Timestamp) -> bool:
        if not trade_hours or len(trade_hours) != 2:
            return True
        start_h, end_h = trade_hours
        if start_h <= end_h:
            return start_h <= ts.hour < end_h
        return ts.hour >= start_h or ts.hour < end_h  # 日をまたぐ範囲(例: 22時〜2時)

    def _news_allowed(i: int) -> bool:
        if news_filter is None:
            return True
        near = bool(news_mask.iloc[i])
        return not near if news_filter.mode == "avoid" else near

    def _mtf_allowed(sig: int, i: int) -> bool:
        if mtf_confirm is None:
            return True
        return bool(mtf_allow_long.iloc[i]) if sig == 1 else bool(mtf_allow_short.iloc[i])

    def _vol_regime_allowed(i: int) -> bool:
        # ボラレジームゲート(Wave5 案D): rank範囲外は新規エントリー停止。
        # ウォームアップNaNは「不明」として停止側(安全側、mtf_confirmのknownマスクと同思想)。
        if vol_regime is None:
            return True
        return bool(vol_regime_ok.iloc[i])

    def _open_price(raw_open: float, side: int) -> float:
        cost = (spread_pips + slippage_pips) * pip_size
        return raw_open + cost if side == 1 else raw_open - cost

    def _close_price(raw_price: float, side: int) -> float:
        cost = slippage_pips * pip_size
        # 決済はエントリーと逆方向の約定になるためコストは反対向きに不利にかかる
        return raw_price - cost if side == 1 else raw_price + cost

    def _new_position(
        entry_price: float,
        side: int,
        entry_time: Any,
        entry_bar_index: int,
        sl_price_override: Optional[float] = None,
        tp_price_override: Optional[float] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        nonlocal position_id_counter
        position_id_counter += 1
        position_lot = sizing.compute_lot(
            money=money,
            fixed_lot=lot,
            balance=balance,
            sl_pips=sl_pips,
            pip_value_per_lot=_pip_value_per_lot(strategy.symbol, pip_size, entry_price),
            confidence=confidence,
            closed_trades=trades,  # この時点までに確定済みのトレードのみ(ルックアヘッド禁止)
        )
        # 構造的SL/TP(sl_price/tp_price列)がエントリー価格に対して不利側(SLなら損失方向、
        # TPなら利益方向)にあることを検証する。窓開け等でシグナル生成時の想定と現在の
        # エントリー価格がずれ、overrideが逆側(=無意味な値)になっている場合はpipsベースの
        # 計算にフォールバックする(でないと「SL決済なのに利益が出る」ような矛盾が生じる)。
        sl_valid = sl_price_override is not None and side * (entry_price - sl_price_override) > 0
        tp_valid = tp_price_override is not None and side * (tp_price_override - entry_price) > 0
        sl = sl_price_override if sl_valid else entry_price - side * sl_pips * pip_size
        tp = tp_price_override if tp_valid else entry_price + side * tp_pips * pip_size
        return {
            "side": side,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "entry_bar_index": entry_bar_index,
            "sl": sl,
            "tp": tp,
            "peak_price": entry_price,  # トレーリングストップ用: 保有中の最有利価格
            "lot": position_lot,
            "lot_remaining": position_lot,  # 分割決済で減っていく残存ロット
            "partial_index": 0,  # money.partial_exitsのうち次に判定すべきレベルのインデックス
            "position_id": position_id_counter,
            "sl_is_trailing": False,
        }

    def _record_trade(
        exit_time: Any,
        exit_price: float,
        exit_kind: str,
        lot_to_close: Optional[float] = None,
    ) -> None:
        nonlocal balance, position
        assert position is not None
        side = position["side"]
        close_lot = lot_to_close if lot_to_close is not None else position["lot_remaining"]
        if side == 1:
            pips = (exit_price - position["entry_price"]) / pip_size
        else:
            pips = (position["entry_price"] - exit_price) / pip_size
        pip_value = _pip_value_per_lot(strategy.symbol, pip_size, exit_price)
        profit = pips * pip_value * close_lot - commission_per_lot * close_lot
        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": exit_time,
                "side": "long" if side == 1 else "short",
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pips": pips,
                "profit": profit,
                "lot": close_lot,
                "exit_kind": exit_kind,
                "position_id": position["position_id"],
            }
        )
        balance += profit
        # lot_remainingが尽きた場合のみポジションを完全クローズする(分割決済では一部のみ減算)。
        position["lot_remaining"] -= close_lot
        if position["lot_remaining"] <= 1e-9:
            position = None

    first_ts = df.iloc[0]["timestamp"]
    equity_curve.append({"timestamp": first_ts, "equity": balance})

    n = len(df)
    pending_limit = None  # 指値エントリーの待機注文(1バー限り有効)
    for i in range(1, n):
        row = df.iloc[i]
        sig = int(actionable.iloc[i])

        # --- 指値エントリーの約定判定(前バーで作成された注文、当バーの値幅で判定) ---
        if pending_limit is not None:
            pl = pending_limit
            pending_limit = None  # 1バー限り: 約定してもしなくても消化
            if position is None:
                touched = (
                    row["low"] <= pl["price"] if pl["side"] == 1 else row["high"] >= pl["price"]
                )
                if touched:
                    # 指値約定: スリッページなし、スプレッドのみ不利方向に反映
                    fill = pl["price"] + pl["side"] * spread_pips * pip_size
                    position = _new_position(
                        fill, pl["side"], row["timestamp"], i,
                        pl["sl_override"], pl["tp_override"], pl["confidence"],
                    )

        # filters.trade_hours/max_spread_pips/news/mtf_confirmはエントリー(新規・ドテンの両方)のみを制限する。既存ポジションのSL/TP管理には影響しない。
        if sig != 0 and spread_filter_ok and _hour_allowed(row["timestamp"]) and _news_allowed(i) and _mtf_allowed(sig, i) and _vol_regime_allowed(i):
            sl_override = _value_or_none(actionable_sl_price, i)
            tp_override = _value_or_none(actionable_tp_price, i)
            confidence = _confidence_or_default(actionable_confidence, i)
            if position is None:
                if limit_offset_pips is not None:
                    # 指値モード: 当バーでは建てず、始値からoffset分有利な価格に指値を置く
                    pending_limit = {
                        "side": sig,
                        "price": row["open"] - sig * limit_offset_pips * pip_size,
                        "sl_override": sl_override,
                        "tp_override": tp_override,
                        "confidence": confidence,
                    }
                else:
                    entry_price = _open_price(row["open"], sig)
                    position = _new_position(entry_price, sig, row["timestamp"], i, sl_override, tp_override, confidence)
            elif position["side"] != sig:
                # ドテン: 現ポジションを始値でクローズし、逆方向を新規建て
                exit_price = _close_price(row["open"], position["side"])
                _record_trade(row["timestamp"], exit_price, "doten")
                entry_price = _open_price(row["open"], sig)
                position = _new_position(entry_price, sig, row["timestamp"], i, sl_override, tp_override, confidence)

        if position is not None:
            side = position["side"]

            # SL/TP判定は当バー開始時点(前バーまでに確定済み)のSLで行う。
            # 当バーの高値/安値でトレーリングSLを先に更新してしまうと、
            # そのバー内で「有利方向に伸びてから反落した」ことを前提にした
            # 先読みバイアスになるため、判定→生存時のみ更新の順序を守る。
            sl, tp = position["sl"], position["tp"]
            if side == 1:
                sl_hit = row["low"] <= sl
                tp_hit = row["high"] >= tp
            else:
                sl_hit = row["high"] >= sl
                tp_hit = row["low"] <= tp

            if sl_hit:  # SL優先(両方ヒット時は保守的にSLを採用)
                exit_kind = "trailing" if position["sl_is_trailing"] else "sl"
                _record_trade(row["timestamp"], _close_price(sl, side), exit_kind)
            else:
                if money is not None and money.partial_exits:
                    # SLが非ヒットの場合のみ分割決済を判定する。同一バーで複数レベルに
                    # 同時到達しうる(急変動)ため、到達しなくなるまでwhileで判定を繰り返す。
                    while position is not None and position["partial_index"] < len(money.partial_exits):
                        level = money.partial_exits[position["partial_index"]]
                        target_price = position["entry_price"] + side * level.at_pips * pip_size
                        reached = row["high"] >= target_price if side == 1 else row["low"] <= target_price
                        if not reached:
                            break
                        is_first_level = position["partial_index"] == 0
                        close_lot = min(round(position["lot"] * level.close_ratio, 4), position["lot_remaining"])
                        _record_trade(
                            row["timestamp"], _close_price(target_price, side), "partial", lot_to_close=close_lot
                        )
                        if position is not None:
                            position["partial_index"] += 1
                            if is_first_level and money.breakeven_after_first_tp:
                                # 建値移動は有利方向にのみ行う(trailingとは別概念のためsl_is_trailingは変更しない)。
                                entry = position["entry_price"]
                                favorable = (side == 1 and entry > position["sl"]) or (
                                    side == -1 and entry < position["sl"]
                                )
                                if favorable:
                                    position["sl"] = entry

                if position is not None:
                    if tp_hit:
                        _record_trade(row["timestamp"], _close_price(tp, side), "tp")
                    elif max_hold_bars is not None and (i - position["entry_bar_index"]) >= max_hold_bars:
                        # 保有バー数上限に達した場合はそのバーの終値で強制決済する。
                        # trailing_pips更新より先に判定することで、trailing_pips設定時に
                        # max_hold_barsが恒久的に無効化される(elif連鎖に埋もれる)不具合を避ける。
                        _record_trade(row["timestamp"], _close_price(row["close"], side), "time")
                    elif trailing_pips is not None:
                        # 決済されなかった場合のみ、次バー以降に適用するSLを当バーの値幅で追随させる(戻さない)
                        if side == 1:
                            position["peak_price"] = max(position["peak_price"], row["high"])
                            new_sl = position["peak_price"] - trailing_pips * pip_size
                            if new_sl > position["sl"]:
                                position["sl"] = new_sl
                                position["sl_is_trailing"] = True
                        else:
                            position["peak_price"] = min(position["peak_price"], row["low"])
                            new_sl = position["peak_price"] + trailing_pips * pip_size
                            if new_sl < position["sl"]:
                                position["sl"] = new_sl
                                position["sl_is_trailing"] = True

        equity_curve.append({"timestamp": row["timestamp"], "equity": balance})

    if position is not None:
        # 期間末に残ったポジションは最終バーの終値で強制決済する
        last_row = df.iloc[-1]
        _record_trade(last_row["timestamp"], _close_price(last_row["close"], position["side"]), "eod")
        equity_curve[-1]["equity"] = balance

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "initial_balance": initial_balance,
        "warnings": warnings,
    }
