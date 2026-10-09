"""時間予算つきランダム探索(1シャード分)。

試行は2系統を抽選する:
- template: 既存テンプレート×パラメータ摂動
- gen     : lab.gen によるルール自動生成(特徴量条件のAND結合)

どちらも train → confirm → holdout の段階ゲートで評価してJSONLに追記する。
holdoutを見るのは train・confirm を両方通過した試行だけ。holdoutでは、シグナルを
日単位で巡回シフトしたプラセボと比較し、相場の地合いに乗っただけのルールを落とす。

メモリ節約: 同時に保持するのは1銘柄×1時間足のOHLCと特徴量のみ。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

warnings.simplefilter("ignore", FutureWarning)  # テンプレート側のpandas警告でログが埋まるのを防ぐ

import numpy as np
import pandas as pd
from scipy import stats

import app.core  # noqa: F401 - テンプレート登録の副作用import
from app.core import datafeed, engine, templates
from app.core.strategy_model import Strategy, StrategyFilters, StrategyParam
from lab import config, gen

BLOCK_TRIALS = 40  # 1銘柄×1時間足を読み込んだら何試行まとめて回すか


def sample_params(tpl: str, rng: random.Random) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in templates.get(tpl).defaults.items():
        # lotは損益スケールのみ。0/1はフラグの可能性が高いので触らない
        if k == "lot" or v in (0.0, 1.0):
            out[k] = v
            continue
        nv = v * rng.choice(config.PARAM_FACTORS)
        if float(v).is_integer():
            nv = float(max(1, round(nv))) if v > 0 else float(round(nv))
        out[k] = round(nv, 6)
    return out


def trial_key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def pips_of(result: Dict[str, Any]) -> np.ndarray:
    return np.array([float(t["pips"]) for t in result["trades"]], dtype=np.float64)


def pf_of(pips: np.ndarray) -> float:
    loss = -pips[pips < 0].sum()
    gain = pips[pips > 0].sum()
    if loss <= 0:
        return 99.0 if gain > 0 else 0.0
    return float(gain / loss)


def holdout_stats(pips: np.ndarray, seed: int) -> Dict[str, Any]:
    n = len(pips)
    out: Dict[str, Any] = {"ho_n": n, "ho_pf": round(pf_of(pips), 3), "ho_sum": round(float(pips.sum()), 1)}
    if n < 3 or pips.std(ddof=1) == 0:
        return out
    t = stats.ttest_1samp(pips, 0.0, alternative="greater")
    rng = np.random.default_rng(seed)
    means = pips[rng.integers(0, n, size=(config.BOOTSTRAP_N, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    equity = np.cumsum(pips)
    out.update({
        "ho_mean": round(float(pips.mean()), 3),
        "ho_p": float(t.pvalue),
        "ho_ci_lo": round(float(lo), 3),
        "ho_ci_hi": round(float(hi), 3),
        "ho_p_pos": round(float((means > 0).mean()), 4),
        "ho_sum_ex_top3": round(float(np.sort(pips)[:-3].sum()), 1),
        "ho_maxdd": round(float((np.maximum.accumulate(equity) - equity).max()), 1),
    })
    return out


def make_strategy(tpl: str, pair: str, tf: str, win: str, params: Dict[str, float]) -> Strategy:
    hours = config.WINDOWS[win]
    return Strategy(
        name="lab", template=tpl, symbol=pair, timeframe=tf,
        params={k: StrategyParam(value=v) for k, v in params.items()},
        filters=StrategyFilters(trade_hours=hours) if hours else None,
    )


def placebo_p(sig: np.ndarray, replay: Strategy, df: pd.DataFrame, cost: Dict[str, float],
              actual_sum: float, tf: str, seed: int) -> Optional[float]:
    """シグナルを日単位で巡回シフトしたプラセボ群に対する順位p値。

    時刻帯・売買方向・回数・決済ルールは保ったまま、条件と値動きの対応だけを壊す。
    実ルールの合計pipsがプラセボと大差なければ、優位性ではなく地合い(ドリフト)の寄与。
    """
    bpd = gen.BARS_PER_DAY[tf]
    n_days = len(sig) // bpd
    if n_days < 30:
        return None
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(config.PLACEBO_N):
        gen.set_replay(np.roll(sig, int(rng.integers(5, n_days - 5)) * bpd))
        if pips_of(engine.run_backtest(replay, df, **cost)).sum() >= actual_sum:
            ge += 1
    return round((1 + ge) / (config.PLACEBO_N + 1), 4)


def staged(rec: Dict[str, Any], bt: Callable[[int], np.ndarray],
           holdout_signal: Callable[[], Optional[np.ndarray]], replay: Strategy,
           parts: List[pd.DataFrame], cost: Dict[str, float], seed: int) -> Dict[str, Any]:
    tr = bt(0)
    rec.update(stage=0, tr_n=len(tr), tr_pf=round(pf_of(tr), 3))
    if len(tr) < config.TRAIN_MIN_N or rec["tr_pf"] < config.TRAIN_MIN_PF:
        return rec

    rec["stage"] = 1
    cf = bt(1)
    rec.update(cf_n=len(cf), cf_pf=round(pf_of(cf), 3))
    if len(cf) < config.CONFIRM_MIN_N or rec["cf_pf"] < config.CONFIRM_MIN_PF:
        return rec

    rec["stage"] = 2
    ho = bt(2)
    rec.update(holdout_stats(ho, seed))
    rec["ho_from"] = str(parts[2]["timestamp"].iloc[0])[:10]
    rec["ho_to"] = str(parts[2]["timestamp"].iloc[-1])[:10]
    if len(ho) >= config.HOLDOUT_MIN_N and ho.sum() > 0:
        sig = holdout_signal()
        if sig is not None:
            p = placebo_p(sig, replay, parts[2], cost, float(ho.sum()), rec["tf"], seed)
            if p is not None:
                rec["ho_placebo_p"] = p
    return rec


def run_template_trial(tpl: str, pair: str, tf: str, win: str, parts: List[pd.DataFrame],
                       rng: random.Random) -> Dict[str, Any]:
    params = sample_params(tpl, rng)
    rec: Dict[str, Any] = {"key": trial_key(tpl, pair, tf, win, params), "kind": "template",
                           "tpl": tpl, "pair": pair, "tf": tf, "win": win, "params": params}
    s = make_strategy(tpl, pair, tf, win, params)
    cost = dict(spread_pips=config.SPREAD_PIPS[pair], slippage_pips=config.SLIPPAGE_PIPS)

    def holdout_signal() -> Optional[np.ndarray]:
        sig = templates.get(tpl).signal_fn(s, parts[2])
        # 構造的SL/TP(価格列つきDataFrame)はシフトすると価格が無意味になるため対象外
        return sig.fillna(0).to_numpy(dtype=np.int64) if isinstance(sig, pd.Series) else None

    return staged(rec, lambda i: pips_of(engine.run_backtest(s, parts[i], **cost)),
                  holdout_signal, make_strategy("lab_replay", pair, tf, win, params),
                  parts, cost, rng.randrange(2**31))


def run_gen_trial(pair: str, tf: str, win: str, parts: List[pd.DataFrame],
                  block: gen.Block, rng: random.Random) -> Dict[str, Any]:
    rule = block.sample_rule(rng)
    rec: Dict[str, Any] = {"key": trial_key("gen", pair, tf, win, rule), "kind": "gen",
                           "tpl": "gen", "pair": pair, "tf": tf, "win": win,
                           "rule": rule, "desc": gen.describe(rule)}
    s = make_strategy("lab_replay", pair, tf, win, gen.rule_params(rule))
    cost = dict(spread_pips=config.SPREAD_PIPS[pair], slippage_pips=config.SLIPPAGE_PIPS)

    def bt(i: int) -> np.ndarray:
        sig = block.signal(rule, i)
        if sig is None or not sig.any():
            return np.array([], dtype=np.float64)
        gen.set_replay(sig)
        return pips_of(engine.run_backtest(s, parts[i], **cost))

    return staged(rec, bt, lambda: block.signal(rule, 2), s, parts, cost, rng.randrange(2**31))


def load_df(pair: str, tf: str) -> Optional[pd.DataFrame]:
    path = datafeed.OHLC_DIR / f"{pair}_{tf}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if len(df) >= 5000 else None  # データ品質ゲート: 短すぎる系列は評価しない


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-trials", type=int, default=0, help="0=無制限(動作確認用)")
    args = ap.parse_args()

    rng = random.Random(args.seed * 1000 + args.shard)
    deadline = time.time() + args.minutes * 60
    tpl_names = [t for t in templates.names() if t not in config.EXCLUDE_TEMPLATES]
    tfs, weights = list(config.TF_WEIGHTS), list(config.TF_WEIGHTS.values())
    broken: set = set()  # (tpl, tf) で例外が出た組は同一シャード内で再試行しない
    n_done = 0
    macro = gen.load_macro()
    print(f"[shard {args.shard}] macro: {sorted(macro)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not any(datafeed.OHLC_DIR.glob("*.parquet")):
        raise SystemExit(f"OHLCデータがありません: {datafeed.OHLC_DIR}")
    with open(args.out, "a", encoding="utf-8") as fout:
        while time.time() < deadline:
            pair, tf = rng.choice(config.PAIRS), rng.choices(tfs, weights)[0]
            df = load_df(pair, tf)
            if df is None:
                continue
            a, b = int(len(df) * config.SPLIT[0]), int(len(df) * config.SPLIT[1])
            parts = [df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True),
                     df.iloc[b:].reset_index(drop=True)]
            block = gen.Block(df, (a, b), engine._pip_size(pair), gen.load_refs(pair, tf, config.PAIRS, rng),
                              pair, macro, tf)
            del df
            for _ in range(BLOCK_TRIALS):
                if time.time() >= deadline or (args.max_trials and n_done >= args.max_trials):
                    break
                win = rng.choice(list(config.WINDOWS))
                tpl = "gen" if rng.random() < config.GEN_SHARE else rng.choice(tpl_names)
                if (tpl, tf) in broken:
                    continue
                try:
                    if tpl == "gen":
                        rec = run_gen_trial(pair, tf, win, parts, block, rng)
                    else:
                        rec = run_template_trial(tpl, pair, tf, win, parts, rng)
                except Exception as exc:  # noqa: BLE001 - 1試行の失敗で探索を止めない
                    if tpl != "gen":
                        broken.add((tpl, tf))
                    rec = {"tpl": tpl, "pair": pair, "tf": tf, "stage": -1,
                           "error": f"{type(exc).__name__}: {exc}"[:160]}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"[shard {args.shard}] {n_done} trials", flush=True)
            if args.max_trials and n_done >= args.max_trials:
                break
    print(f"[shard {args.shard}] done: {n_done} trials", flush=True)


if __name__ == "__main__":
    main()
