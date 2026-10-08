"""メタ配分レイヤー: 複数戦略のバックテスト済みトレード列に対する動的資金配分シミュレーション。

正規仕様は docs/meta_alloc_spec.md(META_ALLOC_SPEC_VERSION と同期。いずれかを変更する
場合は両方を同時に更新すること)。

独立性の宣言: 本モジュールは app/core/engine.py / nanpin_engine.py / arb_engine.py を
import しない・変更しない。入力は run_backtest 等が出力済みの trades リスト
(entry_time / exit_time / profit / pips / lot)のみであり、価格データ・戦略定義には
一切依存しない。

ルックアヘッド禁止の構造(spec §2):
- リバランス日 t の重み計算には exit_time < t(厳密不等号)の確定トレードのみを使う
  (exit 基準。exit_time == t のトレードは「そのバー/その瞬間に確定した」情報であり
  t 時点の意思決定には使えないため窓に含めない)。
- 各トレードに適用される重みは entry_time 時点で有効だった重み。保有が次のリバランス
  日を跨いでも entry 時の重みを保持する(途中で配分変更しない)。
- ベースライン(全戦略 weight=1/N 固定)・ランダム切替対照は、重み決定関数(decider)
  だけを差し替えて同一の _simulate 経路を通す(別コードパス禁止。spec §4)。
"""
from __future__ import annotations

import bisect
import math
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

META_ALLOC_SPEC_VERSION = "1"  # docs/meta_alloc_spec.md 参照

# spec §6 の固定免責文言。一字一句このまま result["disclaimer"] に設定すること
# (再入力・言い換えしない)。NANPIN_DISCLAIMER と同じ運用方針: 「配分後の合成
# エクイティ=口座残高推移」という誤読(証拠金・同時保有制約の無視)を防ぐための
# 警告であり、言い回しが揺れると実効性が下がるため厳密に固定する。
META_ALLOC_DISCLAIMER = (
    "本結果は配分シミュレーションであり口座シミュレーションではありません。"
    "各戦略のバックテスト済みトレード損益に配分重みを乗じて合成した近似であり、"
    "証拠金・必要証拠金率・同時保有ポジション数・ロット丸め・約定競合などの"
    "口座制約を一切考慮していません。合成エクイティは実現損益(exit時点計上)"
    "ベースであり保有中の含み損益を反映しません。実運用の口座残高推移とは異なります。"
)

_REQUIRED_TRADE_KEYS = ("entry_time", "exit_time", "profit", "pips", "lot")

# pf スコアで gross_loss==0(全勝)の窓に与える有限上限。softmax の exp 計算で
# inf が nan を生むのを防ぐためのクリップ値でもある(spec §3)。
_PF_CAP = 1e6


class MetaAllocConfig(BaseModel):
    """メタ配分ハイパーパラメータ(spec §1)。値域はプラン(plan_wave5.md §3)で事前固定。"""

    lookback_days: int = 90
    rebalance_days: int = 30
    top_k: int = 5
    metric: Literal["expectancy_pips", "pf", "profit_sum"] = "expectancy_pips"
    weighting: Literal["equal_top_k", "softmax"] = "equal_top_k"
    min_trades_in_window: int = 10

    @field_validator("lookback_days", "rebalance_days", "top_k")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("1以上を指定してください")
        return v

    @field_validator("min_trades_in_window")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("0以上を指定してください")
        return v


# ---------------------------------------------------------------------------
# 入力正規化
# ---------------------------------------------------------------------------

class _NormStrategy:
    """1戦略分の正規化済みトレード列。exit_time 昇順ソート+prefix sum で
    任意のリバランス窓 [t-lookback, t) の成績を O(log n) で取れるようにする。"""

    def __init__(self, name: str, trades: List[Dict[str, Any]]) -> None:
        self.name = name
        if trades:
            df = pd.DataFrame(trades)
            missing = [k for k in _REQUIRED_TRADE_KEYS if k not in df.columns]
            if missing:
                raise ValueError(f"戦略 {name!r} の trades にキーが不足: {missing}")
            df = df.copy()
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            df["exit_time"] = pd.to_datetime(df["exit_time"])
            if bool((df["exit_time"] < df["entry_time"]).any()):
                raise ValueError(f"戦略 {name!r} に exit_time < entry_time のトレードがある")
            df = df.sort_values(["exit_time", "entry_time"], kind="mergesort").reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=list(_REQUIRED_TRADE_KEYS))
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            df["exit_time"] = pd.to_datetime(df["exit_time"])
        self.df = df
        self.exit_np = df["exit_time"].to_numpy()
        n = len(df)
        profit = df["profit"].to_numpy(dtype=float) if n else np.zeros(0)
        pips = df["pips"].to_numpy(dtype=float) if n else np.zeros(0)
        # prefix sum(先頭に0を置き、窓 [lo, hi) の合計 = cs[hi] - cs[lo])
        self.pips_cs = np.concatenate([[0.0], np.cumsum(pips)])
        self.profit_cs = np.concatenate([[0.0], np.cumsum(profit)])
        self.gross_profit_cs = np.concatenate([[0.0], np.cumsum(np.where(profit > 0, profit, 0.0))])
        self.gross_loss_cs = np.concatenate([[0.0], np.cumsum(np.where(profit < 0, -profit, 0.0))])

    def window_stats(self, t: pd.Timestamp, lookback: pd.Timedelta, min_trades: int) -> Dict[str, Any]:
        """リバランス日 t のスコア窓 [t-lookback, t)。exit_time < t 厳密(spec §2)。"""
        lo = int(np.searchsorted(self.exit_np, np.datetime64(t - lookback), side="left"))
        hi = int(np.searchsorted(self.exit_np, np.datetime64(t), side="left"))
        n = hi - lo
        gp = float(self.gross_profit_cs[hi] - self.gross_profit_cs[lo])
        gl = float(self.gross_loss_cs[hi] - self.gross_loss_cs[lo])
        if gl > 0:
            pf = gp / gl
        else:
            pf = _PF_CAP if gp > 0 else 0.0  # 全勝窓は上限値(spec §3、infはsoftmaxでnan化するため)
        return {
            "n": n,
            "expectancy_pips": (float(self.pips_cs[hi] - self.pips_cs[lo]) / n) if n else 0.0,
            "pf": pf,
            "profit_sum": float(self.profit_cs[hi] - self.profit_cs[lo]),
            "eligible": n >= min_trades,
        }


def _normalize(strategies: List[Dict[str, Any]]) -> List[_NormStrategy]:
    norm: List[_NormStrategy] = []
    seen = set()
    for s in strategies:
        if "name" not in s or "trades" not in s:
            raise ValueError("各戦略は {'name': str, 'trades': list} を持つこと")
        if s["name"] in seen:
            raise ValueError(f"戦略名が重複: {s['name']!r}")
        seen.add(s["name"])
        norm.append(_NormStrategy(str(s["name"]), list(s["trades"])))
    return norm


# ---------------------------------------------------------------------------
# 重み決定関数(decider)。全経路がこの契約で _simulate を通る(spec §4)
#   decider(t, stats) -> {name: weight}(合計1、ウォームアップ点では呼ばれない)
# ---------------------------------------------------------------------------

def _equal_weights(names: List[str]) -> Dict[str, float]:
    w = 1.0 / len(names)
    return {n: w for n in names}


def _baseline_decider(names: List[str]) -> Callable[..., Dict[str, float]]:
    """ベースライン: 全戦略 weight=1/N 固定(eligibility も成績も見ない)。"""

    def decide(t: pd.Timestamp, stats: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        return _equal_weights(names)

    return decide


def _competitive_decider(config: MetaAllocConfig, names: List[str]) -> Callable[..., Dict[str, float]]:
    """メタ配分本体: eligible(窓内トレード数>=min_trades)の上位 top_k に配分。"""

    def decide(t: pd.Timestamp, stats: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        elig = {n: s for n, s in stats.items() if s["eligible"]}
        if not elig:
            # 評価可能な戦略が1本も無い期間はウォームアップと同じ均等配分に
            # フォールバック(安全側。spec §2.3)
            return _equal_weights(names)
        # スコア降順、同点は name 昇順で決定的に(spec §3)
        ranked = sorted(elig.items(), key=lambda kv: (-kv[1][config.metric], kv[0]))
        top = ranked[: config.top_k]
        weights = {n: 0.0 for n in names}
        if config.weighting == "equal_top_k":
            w = 1.0 / len(top)
            for n, _ in top:
                weights[n] = w
        else:  # softmax
            scores = [min(s[config.metric], _PF_CAP) for _, s in top]
            m = max(scores)
            exps = [math.exp(sc - m) for sc in scores]  # max減算で数値安定化
            total = sum(exps)
            for (n, _), e in zip(top, exps):
                weights[n] = e / total
        return weights

    return decide


def _random_decider(config: MetaAllocConfig, names: List[str], rng: random.Random) -> Callable[..., Dict[str, float]]:
    """ランダム切替対照: eligibility 判定はメタ配分と同一に保ち、上位選択だけを
    ランダム k 本に置換する(「切替という行為自体の利得」を分離する対照。spec §5)。"""

    def decide(t: pd.Timestamp, stats: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        elig = sorted(n for n, s in stats.items() if s["eligible"])  # sortでseed決定性を保証
        if not elig:
            return _equal_weights(names)
        k = min(config.top_k, len(elig))
        pick = rng.sample(elig, k)
        weights = {n: 0.0 for n in names}
        for n in pick:
            weights[n] = 1.0 / k
        return weights

    return decide


# ---------------------------------------------------------------------------
# シミュレーション本体(単一計算経路)
# ---------------------------------------------------------------------------

def _empty_result(warnings: List[str]) -> Dict[str, Any]:
    return {
        "final_profit": 0.0,
        "max_drawdown": 0.0,
        "n_trades": 0,
        "portfolio_trades": [],
        "equity_curve": [],
        "weights_timeline": [],
        "warnings": warnings,
        "disclaimer": META_ALLOC_DISCLAIMER,
    }


def _simulate(
    norm: List[_NormStrategy],
    config: MetaAllocConfig,
    decider: Callable[..., Dict[str, float]],
) -> Dict[str, Any]:
    """メタ/ベースライン/ランダム対照すべてが通る唯一の経路(spec §4)。

    タイムライン: t0 = 全戦略の最初の entry_time。t0 時点(ウォームアップ)は常に
    均等配分で開始し(spec §2.3)、以後 rebalance_days ごとに decider を呼ぶ。
    トレードには entry_time が属する区間 [t_j, t_{j+1}) の重みを適用する
    (entry_time == t_j ちょうどは新しい重み側。期間跨ぎでも entry 時重み保持)。
    """
    warnings: List[str] = []
    names = [s.name for s in norm]
    if not norm:
        return _empty_result(["戦略が0件のため空の結果を返します"])
    if all(len(s.df) == 0 for s in norm):
        return _empty_result(["全戦略のトレードが0件のため空の結果を返します"])

    t0 = min(s.df["entry_time"].min() for s in norm if len(s.df))
    t_end = max(s.df["exit_time"].max() for s in norm if len(s.df))
    lookback = pd.Timedelta(days=config.lookback_days)
    step = pd.Timedelta(days=config.rebalance_days)

    times: List[pd.Timestamp] = [t0]
    weights_list: List[Dict[str, float]] = [_equal_weights(names)]  # ウォームアップ=均等
    t = t0 + step
    while t <= t_end:
        stats = {s.name: s.window_stats(t, lookback, config.min_trades_in_window) for s in norm}
        w = decider(t, stats)
        times.append(t)
        weights_list.append(w)
        t = t + step

    # 各トレードへ entry 時点の重みを適用(w=0 のトレードはポートフォリオに含めない)
    records: List[Dict[str, Any]] = []
    for s in norm:
        for row in s.df.itertuples(index=False):
            j = bisect.bisect_right(times, row.entry_time) - 1
            w = weights_list[j].get(s.name, 0.0)
            if w <= 0.0:
                continue
            records.append(
                {
                    "strategy": s.name,
                    "entry_time": row.entry_time,
                    "exit_time": row.exit_time,
                    "profit": float(row.profit),
                    "pips": float(row.pips),
                    "lot": float(row.lot),
                    "weight": w,
                    "weighted_profit": float(row.profit) * w,
                    "weighted_lot": float(row.lot) * w,
                }
            )
    # 実現損益ベースの合成: exit_time 順に確定(同時刻は entry_time→strategy で決定的に)
    records.sort(key=lambda r: (r["exit_time"], r["entry_time"], r["strategy"]))

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    equity_curve: List[Dict[str, Any]] = []
    for r in records:
        equity += r["weighted_profit"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        equity_curve.append({"timestamp": r["exit_time"], "equity": equity})

    return {
        "final_profit": equity,
        "max_drawdown": max_dd,
        "n_trades": len(records),
        "portfolio_trades": records,
        "equity_curve": equity_curve,
        "weights_timeline": [
            {"time": ts, "weights": w} for ts, w in zip(times, weights_list)
        ],
        "warnings": warnings,
        "disclaimer": META_ALLOC_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# 公開API
# ---------------------------------------------------------------------------

def run_meta_alloc(
    strategies: List[Dict[str, Any]],
    config: Optional[MetaAllocConfig] = None,
) -> Dict[str, Any]:
    """メタ配分とベースライン(全戦略均等 1/N 固定)を同一経路で計算して返す。

    主判定はベースラインとの相対改善のみ(spec §5)。improvement_maxdd は
    「ベースラインDD − メタDD」で正なら改善(DD縮小)。"""
    config = config or MetaAllocConfig()
    norm = _normalize(strategies)
    names = [s.name for s in norm]
    meta = _simulate(norm, config, _competitive_decider(config, names))
    baseline = _simulate(norm, config, _baseline_decider(names))
    return {
        "meta": meta,
        "baseline": baseline,
        "improvement_profit": meta["final_profit"] - baseline["final_profit"],
        "improvement_maxdd": baseline["max_drawdown"] - meta["max_drawdown"],
        "config": config.model_dump(),
        "disclaimer": META_ALLOC_DISCLAIMER,
    }


def run_random_control(
    strategies: List[Dict[str, Any]],
    config: Optional[MetaAllocConfig] = None,
    n_runs: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """ランダム切替対照(spec §5)。リバランス構造・eligibility・重み適用は
    メタ配分と完全同一で、top_k 選択だけをランダムに置換した n_runs 回の
    「vs ベースライン改善量」分布を返す。メタ配分の改善はこの分布の
    95パーセンタイルを超えて初めて「選択スキルあり」と主張できる。"""
    config = config or MetaAllocConfig()
    norm = _normalize(strategies)
    names = [s.name for s in norm]
    baseline = _simulate(norm, config, _baseline_decider(names))
    improvements: List[float] = []
    for r in range(n_runs):
        rng = random.Random(seed * 100003 + r)  # run毎に独立・再現可能
        res = _simulate(norm, config, _random_decider(config, names, rng))
        improvements.append(res["final_profit"] - baseline["final_profit"])
    return {
        "improvements": improvements,
        "n_runs": n_runs,
        "seed": seed,
        "baseline_final_profit": baseline["final_profit"],
        "disclaimer": META_ALLOC_DISCLAIMER,
    }


def split_half_check(
    strategies: List[Dict[str, Any]],
    config: Optional[MetaAllocConfig] = None,
) -> Dict[str, Any]:
    """split-half 頑健性チェック(spec §5)。全期間の中点時刻 mid で入力トレードを
    entry_time < mid(前半)/ entry_time >= mid(後半)に分割し、各半分で独立に
    メタ vs ベースラインを実行して改善方向(improvement_profit の符号)の一致を見る。
    exit が mid を跨ぐトレードは entry 基準で前半に属する(重み適用と同じ entry 基準)。"""
    config = config or MetaAllocConfig()
    norm = _normalize(strategies)
    non_empty = [s for s in norm if len(s.df)]
    if not non_empty:
        return {"first": None, "second": None, "consistent": None, "mid_time": None,
                "warnings": ["トレードが無いため split-half 判定不能"],
                "disclaimer": META_ALLOC_DISCLAIMER}
    t0 = min(s.df["entry_time"].min() for s in non_empty)
    t_end = max(s.df["exit_time"].max() for s in non_empty)
    mid = t0 + (t_end - t0) / 2

    def _subset(pred) -> List[Dict[str, Any]]:
        subs = []
        for s in norm:
            mask = pred(s.df["entry_time"]) if len(s.df) else []
            sub = s.df[mask] if len(s.df) else s.df
            subs.append({"name": s.name, "trades": sub.to_dict("records")})
        return subs

    halves = {}
    warnings: List[str] = []
    for label, pred in (("first", lambda e: e < mid), ("second", lambda e: e >= mid)):
        subset = _subset(pred)
        if all(len(s["trades"]) == 0 for s in subset):
            halves[label] = None
            warnings.append(f"{label} half にトレードが無い")
            continue
        r = run_meta_alloc(subset, config)
        halves[label] = {
            "improvement_profit": r["improvement_profit"],
            "improvement_maxdd": r["improvement_maxdd"],
            "meta_final_profit": r["meta"]["final_profit"],
            "baseline_final_profit": r["baseline"]["final_profit"],
            "n_trades_meta": r["meta"]["n_trades"],
        }
    if halves["first"] is None or halves["second"] is None:
        consistent: Optional[bool] = None
    else:
        a = halves["first"]["improvement_profit"]
        b = halves["second"]["improvement_profit"]
        consistent = (a > 0) == (b > 0)
    return {
        "first": halves["first"],
        "second": halves["second"],
        "consistent": consistent,
        "mid_time": mid,
        "warnings": warnings,
        "disclaimer": META_ALLOC_DISCLAIMER,
    }
