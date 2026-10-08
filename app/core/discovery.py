"""統計的スクリーニングパイプライン(Phase5: データ駆動の戦略発掘)。

「AIが無限の戦略空間から勝ち筋を発見する」という誇大な話ではなく、
PATTERN_REGISTRY の有限のローソク足パターン×方向×コンテキストという
有限個の仮説を総当たりで高速検証し、多重検定補正(Benjamini-Hochberg FDR)を
経て統計的な足切りを通過したものだけを Stage2(実バックテスト)へ回す。
「これだけ試せば偶然でもこれくらいは合格する」という期待偽陽性数を
scan_patterns/summarize_honesty で必ず併記し、誠実性を担保する。

Stage1(scan_patterns等)は「過去のある時点でパターンが出現した後、実際に
何が起きたか」を集計する完全に事後的(retrospective)な統計分析であり、
「そのバーの将来の値動きを見る」こと自体がこの分析の目的そのものである。
シミュレーション上のリアルタイム売買判断に未来の情報を使う先読みバイアスとは
別物であることに注意(forward_bars分の未来を参照するのは意図的な設計)。
Stage2(run_stage2_verification)は既存の engine.run_backtest をそのまま使うため、
そちらは shift(1)実行等の既存の先読み対策がそのまま効く。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.core.candle_template import PATTERN_NAMES
from app.core.engine import _pip_size, run_backtest
from app.core.indicators import sma
from app.core.metrics import calculate_metrics
from app.core.patterns import PATTERN_REGISTRY
from app.core.strategy_model import Strategy, StrategyParam

CONTEXTS: list[str] = ["none", "uptrend", "downtrend"]
DIRECTIONS: list[int] = [1, -1]

# scan_patternsはsymbolを受け取らないため固定値を使う。t検定のp値/q値は正のスケーリングに対して
# 不変(mean/stdが同じ係数で伸縮するだけ)なため、この値の選択は合否判定(p_value/q_value/passed)に
# 一切影響せず、候補一覧のmean_pips表示用の目安スケール(非JPYペア想定)としてのみ働く。
_DEFAULT_PIP_SIZE = 0.0001


def _context_mask(df: pd.DataFrame, context: str) -> pd.Series:
    """context: "none" | "uptrend" | "downtrend"。

    "none"は全バーTrue。"uptrend"/"downtrend"はclose と sma(close, 50) の大小関係で判定する。
    コンテキストは「そのバー時点で分かる情報」であればよく、shift(1)は不要
    (エントリー執行自体のシフトは _forward_returns_pips 側で next-bar-open として表現される)。
    """
    if context == "none":
        return pd.Series(True, index=df.index)
    trend = sma(df["close"], 50)
    if context == "uptrend":
        result = df["close"] > trend
    elif context == "downtrend":
        result = df["close"] < trend
    else:
        raise ValueError(f"unknown context: {context!r} (expected one of {CONTEXTS})")
    return result.fillna(False).astype(bool)


def _forward_returns_pips(
    df: pd.DataFrame,
    hit_mask: pd.Series,
    direction: int,
    forward_bars: int,
    pip_size: float,
) -> np.ndarray:
    """hit_mask[i]がTrueの各バーiについて、次バー始値エントリー→forward_bars本後の終値決済のpips損益を返す。

    i+1+forward_bars がデータ末尾を超えるヒットは単純に計算不能なため除外する
    (将来バーが存在しないだけであり、先読みバイアスの回避とは無関係)。
    """
    n = len(df)
    hit_positions = np.flatnonzero(hit_mask.to_numpy())
    exit_positions = hit_positions + 1 + forward_bars
    valid = hit_positions[exit_positions < n]
    if valid.size == 0:
        return np.empty(0, dtype=float)

    open_arr = df["open"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    entry = open_arr[valid + 1]
    exit_ = close_arr[valid + 1 + forward_bars]

    if direction == 1:
        pips = (exit_ - entry) / pip_size
    else:
        pips = (entry - exit_) / pip_size
    return pips


def _one_sided_p_value(pips: np.ndarray) -> float:
    """平均pips>0を対立仮説とする片側t検定(ttest_1samp, popmean=0, alternative='greater')のp値。

    direction=-1の仮説は_forward_returns_pips側で符号をすでに反転済みなので、
    「その方向に平均収益が正に偏っているか」を一貫して片側検定すればよい
    (両側検定にすると同一パターンのdirection=1/-1が常に同一p値になり、方向の優位性を区別できない)。
    n<2または分散0の退化ケースはscipyのinf/nanを避け、有意性なしとしてp_value=1.0を返す。
    """
    if len(pips) < 2:
        return 1.0
    std = float(np.std(pips, ddof=1))
    if std == 0.0:
        return 1.0
    result = stats.ttest_1samp(pips, popmean=0.0, alternative="greater")
    return float(result.pvalue)


def scan_patterns(
    df: pd.DataFrame,
    forward_bars: int = 10,
    contexts: list[str] = CONTEXTS,
    min_trades: int = 30,
    fdr_q: float = 0.1,
    symbol: str | None = None,
) -> dict:
    """PATTERN_REGISTRY全パターン×方向[1,-1]×contextsの組合せを仮説として総当たり検証する。

    n_trades < min_trades の仮説はt検定を実施せず p_value/q_value=None, passed=False のまま
    候補に残す(除外しない: 何を試して何が足切りされたかを全て見せるのが誠実性の要件のため)。
    FDR補正(Benjamini-Hochberg)はp_valueが計算できた仮説のみを対象に行う。

    symbol を渡すとJPYペア判定込みの正しいpip_sizeでmean_pipsを算出する
    (未指定時は非JPYペア想定の0.0001を使用。p_value/q_value/passedはpip_sizeの
    スケーリングに不変なため合否判定には影響しない)。
    """
    pip_size = _pip_size(symbol) if symbol is not None else _DEFAULT_PIP_SIZE
    hits_by_pattern = {name: fn(df) for name, fn in PATTERN_REGISTRY.items()}
    mask_by_context = {context: _context_mask(df, context) for context in contexts}

    candidates: list[dict] = []
    for pattern_name in PATTERN_NAMES:
        hits = hits_by_pattern[pattern_name]
        for direction in DIRECTIONS:
            for context in contexts:
                combined_mask = hits & mask_by_context[context]
                pips = _forward_returns_pips(df, combined_mask, direction, forward_bars, pip_size)
                n_trades = int(pips.size)
                candidates.append(
                    {
                        "pattern": pattern_name,
                        "direction": direction,
                        "context": context,
                        "n_trades": n_trades,
                        "mean_pips": float(np.mean(pips)) if n_trades > 0 else 0.0,
                        "p_value": _one_sided_p_value(pips) if n_trades >= min_trades else None,
                        "q_value": None,
                        "passed": False,
                    }
                )

    testable = [c for c in candidates if c["p_value"] is not None]
    if testable:
        raw_pvalues = np.array([c["p_value"] for c in testable], dtype=float)
        adjusted = stats.false_discovery_control(raw_pvalues, method="bh")
        for c, q in zip(testable, adjusted):
            c["q_value"] = float(q)
            c["passed"] = bool(q < fdr_q)

    candidates.sort(key=lambda c: (c["q_value"] is None, c["q_value"] if c["q_value"] is not None else 0.0))

    n_hypotheses = len(candidates)
    return {
        "n_hypotheses": n_hypotheses,
        "n_trades_min_threshold": min_trades,
        "fdr_q": fdr_q,
        # 「多重検定補正を全くせず p<fdr_q を閾値にしていたら、全仮説のうち偶然でも
        # これくらいは合格したはず」という補正なしの目安値(n_passed_naive等)ではなく、
        # 単純な仮説数×fdr_qの参考値。summarize_honestyの「合格集合内の期待偽発見数」
        # (=n_passed*fdr_q、BH法が実際に制御する量)とは別の指標なので混同しないこと。
        "expected_false_positives_at_q": n_hypotheses * fdr_q,
        "candidates": candidates,
    }


def _naive_direction(pattern_name: str) -> int:
    """candle_template.py内の方向判定ロジックと同じ規約の独立実装(bullish/_up→1, bearish/_down→-1, 他→0)。

    candle_template.py は変更・直接参照禁止のため、規約のみをこのモジュール内で再実装している。
    """
    if "bullish" in pattern_name or pattern_name.endswith("_up"):
        return 1
    if "bearish" in pattern_name or pattern_name.endswith("_down"):
        return -1
    return 0


def promote_to_strategy(
    candidate: dict,
    symbol: str,
    timeframe: str,
    name: str,
    sl_pips: float = 30.0,
    tp_pips: float = 60.0,
    lot: float = 0.1,
    allow_direction_override: bool = False,
) -> Strategy:
    """scan_patternsの1候補をcandle_patternテンプレートのStrategyに昇格する(保存はしない)。

    candle_patternテンプレートは各パターンの取引方向を名前規約から固定的に判定するため、
    candidateが発見した方向(データが示した方向)とその固定判定が矛盾する場合、
    デフォルト(allow_direction_override=False)ではテンプレート側が矛盾した方向の
    シグナルを出せず ValueError で自動昇格を拒否する。

    allow_direction_override=True の場合は、矛盾時に candle_pattern テンプレートの
    任意パラメータ `direction_override` を候補の発見方向で明示的にセットして昇格する
    (中立パターンで発見された方向を使う場合も同様)。実データでの検証の結果、
    ローソク足パターンの命名規約(bullish/bearish等)と統計的に発見される優位方向は
    しばしば一致しない(例: 「下抜けの後は反発しやすい」という逆張り的な優位性)ため、
    発掘パイプラインからの昇格(run_stage2_verification/APIの/promote)では
    allow_direction_override=True をデフォルトで使う。
    """
    pattern_name = candidate["pattern"]
    found_direction = candidate["direction"]
    template_direction = _naive_direction(pattern_name)

    params = {
        "pattern_index": StrategyParam(value=float(PATTERN_NAMES.index(pattern_name))),
        "sl_pips": StrategyParam(value=sl_pips),
        "tp_pips": StrategyParam(value=tp_pips),
        "lot": StrategyParam(value=lot),
    }

    if template_direction != found_direction:
        if not allow_direction_override:
            raise ValueError(
                f"発見方向とcandle_patternテンプレートの取引方向が矛盾しています: "
                f"pattern={pattern_name!r} はテンプレート上 direction={template_direction} 固定ですが、"
                f"候補は direction={found_direction} で発見されました。自動昇格できないため、"
                "手動で戦略を構築するか allow_direction_override=True を指定してください。"
            )
        params["direction_override"] = StrategyParam(value=float(found_direction))

    return Strategy(
        name=name,
        template="candle_pattern",
        symbol=symbol,
        timeframe=timeframe,
        params=params,
    )


def run_stage2_verification(
    candidates: list[dict],
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    top_n: int = 20,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
    commission_per_lot: float = 0.0,
) -> list[dict]:
    """passed=Trueの候補をq_value昇順でtop_n件に絞り、実際にengine.run_backtestで検証する。

    方向矛盾でpromote_to_strategyが失敗(ValueError)した候補もskipped_reason付きで結果に残す
    (除外して消さない)。
    """
    passed = [c for c in candidates if c.get("passed")]
    passed.sort(key=lambda c: c["q_value"])
    top = passed[:top_n]

    results: list[dict] = []
    for i, candidate in enumerate(top):
        strategy_name = f"discovery_{candidate['pattern']}_{candidate['direction']}_{candidate['context']}_{i}"
        try:
            strategy = promote_to_strategy(
                candidate,
                symbol=symbol,
                timeframe=timeframe,
                name=strategy_name,
                allow_direction_override=True,
            )
        except ValueError as exc:
            results.append({"candidate": candidate, "skipped_reason": str(exc)})
            continue

        backtest_result = run_backtest(
            strategy,
            df,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            commission_per_lot=commission_per_lot,
        )
        results.append({"candidate": candidate, "metrics": calculate_metrics(backtest_result)})

    return results


def summarize_honesty(scan_result: dict) -> dict:
    """scan_patternsの戻り値からUI表示用の誠実性サマリを作る。

    Benjamini-Hochberg法は「合格集合(q<fdr_q)の中で期待される偽発見の割合」を
    fdr_q以下に制御する手法である。よって合格集合内で偶然でも期待される偽発見数は
    n_passed * fdr_q で近似できる。
    (n_hypotheses * fdr_q は「補正を一切せず全仮説にp<fdr_qを適用した場合の期待偽陽性数」
    という別の量であり、BH合格後の集合の解釈に使うと数値が過大になり、
    本物の発見を「偶然と同水準」と誤って握りつぶしてしまう。旧実装のバグ。)
    """
    n_hypotheses = scan_result["n_hypotheses"]
    fdr_q = scan_result.get("fdr_q", 0.1)
    n_passed = sum(1 for c in scan_result["candidates"] if c["passed"])
    expected_fp = n_passed * fdr_q

    if n_passed == 0:
        verdict = "合格した候補なし(この設定では統計的に有意な優位性は見つかりませんでした)"
    else:
        verdict = (
            f"統計的に有意な候補が{n_passed}件見つかりました"
            f"(多重検定補正により、この合格集合内の偽発見は期待値で"
            f"約{expected_fp:.2f}件以下に制御されています)"
        )

    return {
        "n_hypotheses": n_hypotheses,
        "n_passed": n_passed,
        "expected_false_positives": expected_fp,
        "verdict": verdict,
    }
