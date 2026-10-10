"""前向き検証にかける候補を凍結する(一度だけ実行し、結果の lab/watchlist.json をコミットする)。

探索で上位に来た候補は、holdoutを見た時点で「数百の中から選ばれた」ものになり、同じデータでは
偶然と区別できない。そこで機械的な基準で少数を選んで中身を固定し、凍結後に生まれたデータだけで
判定し直す。選び方・凍結日・判定規則をここに書いてコミットすることが事前登録になる。

選定基準(結果を見て変えない):
  holdout到達、n>=30、平均pipsのCI95下限>0、上位3トレード除外後も黒字、プラセボp<=0.10 を満たすもののうち、
  holdoutのp値が小さい順に、同じ(銘柄・時間足・時間帯)の重複を避けて WATCH_N 件。

  python -m lab.freeze_watchlist --results site/results.jsonl --freeze-at 2026-10-11
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lab import config

OUT = Path(__file__).resolve().parent / "watchlist.json"
WATCH_N = 5
# 判定規則。前向きトレードを古い順に数え、この件数に達した時点の「先頭N件」だけで検定する。
# 毎日のぞいても判定に使うのは固定のN件なので、のぞき見による偽陽性は増えない。
CHECKPOINTS = [30, 60, 100]
FAMILY_ALPHA = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--freeze-at", required=True, help="この日時(UTC)以降に建てたトレードだけを前向き検証に使う")
    args = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} は既にある。凍結は一度だけ(作り直すなら理由を残して手で消す)")

    rows = []
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("cost") == config.COST_MODEL and r.get("stage", 0) >= 2 and "ho_p" in r
                    and r.get("ho_n", 0) >= config.HOLDOUT_MIN_N and r.get("ho_ci_lo", -1) > 0
                    and r.get("ho_sum_ex_top3", -1) > 0 and r.get("ho_placebo_p", 1.0) <= config.PLACEBO_MAX_P):
                rows.append(r)
    rows.sort(key=lambda r: r["ho_p"])
    picked, seen = [], set()
    for r in rows:
        slot = (r["pair"], r["tf"], r["win"])
        if slot in seen:
            continue
        seen.add(slot)
        picked.append(r)
        if len(picked) >= WATCH_N:
            break

    alpha = FAMILY_ALPHA / (len(picked) * len(CHECKPOINTS))
    items = []
    for i, r in enumerate(picked, 1):
        item = {k: r[k] for k in ("key", "tpl", "pair", "tf", "win") if k in r}
        item["id"] = f"W{i:02d}"
        for k in ("params", "rule", "desc"):
            if k in r:
                item[k] = r[k]
        item["selection"] = {k: r.get(k) for k in ("ho_n", "ho_pf", "ho_p", "ho_ci_lo", "ho_placebo_p", "ho_from", "ho_to")}
        items.append(item)
    out = {
        "frozen_on": time.strftime("%Y-%m-%d"),
        "freeze_at": args.freeze_at,
        "eval_model": config.COST_MODEL,
        "pool": len(rows),
        "rule": {
            "checkpoints": CHECKPOINTS,
            "alpha_per_test": round(alpha, 5),
            "pass": f"いずれかのチェックポイントで、先頭N件の片側t検定 p <= {alpha:.5f} かつ合計pips>0",
            "fail": f"{CHECKPOINTS[-1]}件に達しても合格しなければ不合格",
            "family_alpha": FAMILY_ALPHA,
            "note": "Bonferroni(候補数×チェックポイント数)。判定は固定件数の先頭N件のみで行う",
        },
        "items": items,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for it in items:
        print(it["id"], it["tpl"], it["pair"], it["tf"], it["win"], it["selection"])
    print(f"候補プール {len(rows)}件 → 凍結 {len(items)}件 / 1検定あたり alpha={alpha:.5f}")


if __name__ == "__main__":
    main()
