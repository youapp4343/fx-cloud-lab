"""凍結した候補(lab/watchlist.json)を、凍結後のデータだけで評価する。

毎日の実行で全期間を通しで回し、建てた時刻が freeze_at 以降のトレードだけを取り出す
(指標の助走と保有中ポジションの状態を、探索時と同じにするため)。損益は探索と同じ
時間帯別実測コスト(lab.costs)。判定は watchlist.json に書いた規則どおりに機械的に行う。

  python -m lab.forward --out out/forward.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.simplefilter("ignore", FutureWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from app.core import engine  # noqa: E402
from lab import config, costs, gen, recheck, search  # noqa: E402

WATCHLIST = Path(__file__).resolve().parent / "watchlist.json"


def forward_trades(item: Dict[str, Any], freeze_at: pd.Timestamp, macro: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    pair, tf, win = item["pair"], item["tf"], item["win"]
    df = search.load_df(pair, tf)
    if df is None:
        raise ValueError("データなし")
    cost = dict(spread_pips=config.SPREAD_PIPS[pair], slippage_pips=config.SLIPPAGE_PIPS)
    if item["tpl"] == "gen":
        rule = item["rule"]
        a, b = int(len(df) * config.SPLIT[0]), int(len(df) * config.SPLIT[1])
        block = gen.Block(df, (a, b), engine._pip_size(pair), recheck.refs_for(rule, tf), pair, macro, tf)
        missing = [c["f"] for c in rule["conds"] if c["f"] not in block.names]
        if missing:
            raise ValueError(f"特徴量が無い: {missing}")
        sig = block.signal_all(rule)
        if sig is None:
            raise ValueError("しきい値が無効")
        gen.set_replay(sig)
        s = search.make_strategy("lab_replay", pair, tf, win, gen.rule_params(rule))
    else:
        s = search.make_strategy(item["tpl"], pair, tf, win, item["params"])
    trades = engine.run_backtest(s, df.reset_index(drop=True), **cost)["trades"]
    last_bar = df["timestamp"].iloc[-1]
    # 期間末の強制決済(まだ保有中のもの)は確定損益ではないので数えない
    return [t for t in trades if t["entry_time"] >= freeze_at and not (t["exit_kind"] == "eod" and t["exit_time"] >= last_bar)]


def judge(pips: np.ndarray, rule: Dict[str, Any]) -> Dict[str, Any]:
    n = len(pips)
    out: Dict[str, Any] = {"n": n, "sum": round(float(pips.sum()), 1) if n else 0.0, "checks": []}
    if n:
        loss, gain = -pips[pips < 0].sum(), pips[pips > 0].sum()
        out["pf"] = round(float(gain / loss), 2) if loss > 0 else None
        out["mean"] = round(float(pips.mean()), 2)
    verdict = "継続"
    for cp in rule["checkpoints"]:
        if n < cp:
            break
        head = pips[:cp]
        p = float(stats.ttest_1samp(head, 0.0, alternative="greater").pvalue) if head.std(ddof=1) > 0 else 1.0
        ok = p <= rule["alpha_per_test"] and head.sum() > 0
        out["checks"].append({"n": cp, "p": round(p, 5), "sum": round(float(head.sum()), 1), "pass": bool(ok)})
        if ok:
            verdict = "合格"
            break
    if verdict != "合格" and n >= rule["checkpoints"][-1]:
        verdict = "不合格"
    out["verdict"] = verdict
    out["next_checkpoint"] = next((cp for cp in rule["checkpoints"] if n < cp), None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not WATCHLIST.exists():
        print("watchlist.json なし。前向き検証はスキップ")
        return
    wl = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    freeze_at = pd.Timestamp(wl["freeze_at"])
    macro = gen.load_macro()
    rows = []
    for item in wl["items"]:
        row = {"id": item["id"], "tpl": item["tpl"], "pair": item["pair"], "tf": item["tf"], "win": item["win"],
               "desc": item.get("desc", ""), "selection": item["selection"]}
        try:
            trades = forward_trades(item, freeze_at, macro)
            row.update(judge(costs.adjust(trades, item["pair"]), wl["rule"]))
            row["last_trade"] = str(trades[-1]["exit_time"])[:16] if trades else None
        except Exception as exc:  # noqa: BLE001 - 1候補の失敗で他を止めない
            row.update({"n": 0, "sum": 0.0, "verdict": "評価不能", "error": f"{type(exc).__name__}: {exc}"[:160]})
        rows.append(row)
        print(row["id"], row["tpl"], row["pair"], row["tf"], "n", row["n"], "sum", row["sum"], row["verdict"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "updated": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "freeze_at": wl["freeze_at"],
        "frozen_on": wl["frozen_on"], "pool": wl["pool"], "rule": wl["rule"], "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
