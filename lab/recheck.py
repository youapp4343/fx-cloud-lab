"""過去にholdoutへ到達した試行を、現在のコストモデルで評価し直す。

コストモデルを変えたとき、旧モデルで出た「生存」が新モデルでも残るかを確かめるために使う。
同じルール・同じパラメータ・同じ分割で train→confirm→holdout を回し直し、結果を
通常の試行と同じ形式で書き出す(report.py が同じkeyの旧記録を置き換える)。

  python -m lab.recheck --results site/results.jsonl --out out/recheck.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

warnings.simplefilter("ignore", FutureWarning)

import pandas as pd  # noqa: E402

from app.core import datafeed, engine  # noqa: E402
from lab import config, costs, gen, search  # noqa: E402


def refs_for(rule: Dict[str, Any], tf: str) -> Dict[str, pd.DataFrame]:
    """ルールが参照している他銘柄(ref_<PAIR>_ret_k)だけを読み込む。"""
    out: Dict[str, pd.DataFrame] = {}
    for c in rule["conds"]:
        m = re.match(r"ref_([A-Z]{6})_ret_", c["f"])
        if m and m.group(1) not in out:
            path = datafeed.OHLC_DIR / f"{m.group(1)}_{tf}.parquet"
            if path.exists():
                out[m.group(1)] = pd.read_parquet(path, columns=["timestamp", "close"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1, help="並列数。対象をkeyで振り分ける")
    args = ap.parse_args()

    todo: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("stage", 0) >= 2 and r.get("cost") != costs.MODEL and int(r["key"], 16) % args.of == args.shard:
                todo[(r["pair"], r["tf"])].append(r)
    print(f"recheck対象: {sum(len(v) for v in todo.values())}件 / {len(todo)}ブロック", flush=True)

    macro = gen.load_macro()
    rng = random.Random(args.shard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for (pair, tf), recs in sorted(todo.items()):
            df = search.load_df(pair, tf)
            if df is None:
                continue
            a, b = int(len(df) * config.SPLIT[0]), int(len(df) * config.SPLIT[1])
            parts = [df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True),
                     df.iloc[b:].reset_index(drop=True)]
            for r in recs:
                seed = rng.randrange(2**31)
                try:
                    if r["tpl"] == "gen":
                        block = gen.Block(df, (a, b), engine._pip_size(pair), refs_for(r["rule"], tf), pair, macro, tf)
                        missing = [c["f"] for c in r["rule"]["conds"] if c["f"] not in block.names]
                        if missing:
                            raise ValueError(f"特徴量が無い: {missing}")
                        new = search.eval_gen(r["rule"], pair, tf, r["win"], parts, block, seed)
                    else:
                        new = search.eval_template(r["tpl"], pair, tf, r["win"], r["params"], parts, seed)
                except Exception as exc:  # noqa: BLE001 - 1件の失敗で全体を止めない
                    new = {"tpl": r["tpl"], "pair": pair, "tf": tf, "stage": -1,
                           "error": f"recheck {type(exc).__name__}: {exc}"[:160]}
                # 旧モデルでの成績を残し、同じkeyで置き換えられるようにする
                new["key"] = r["key"]
                new["rechecked"] = True
                new["old_ho_pf"] = r.get("ho_pf")
                new["old_ho_n"] = r.get("ho_n")
                fout.write(json.dumps(new) + "\n")
                fout.flush()
                done += 1
            print(f"{pair} {tf}: {len(recs)}件", flush=True)
    print(f"recheck done: {done}", flush=True)


if __name__ == "__main__":
    main()
