"""シャード結果を累積データへ統合し、スマホ向けの静的HTMLレポートを生成する。

site/results.jsonl : trainゲート通過以上の試行(累積・key重複排除)
site/stats.json    : 試行数カウンタと実行履歴(累積)
site/index.html    : レポート

多重検定: holdoutを見た全試行(累積)に対して Benjamini-Hochberg でq値を付ける。
holdoutはtrain/confirmの選抜と独立なので、補正対象はholdout到達試行のみで足りる。
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from lab import config


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bh_qvalues(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = prev
    return q


def judge(r: Dict[str, Any]) -> List[str]:
    """生存条件のうち落ちた項目を返す(空なら生存)。"""
    fails = []
    if r.get("ho_n", 0) < config.HOLDOUT_MIN_N:
        fails.append(f"n<{config.HOLDOUT_MIN_N}")
    if r.get("q", 1.0) > config.FDR_Q:
        fails.append(f"q>{config.FDR_Q}")
    if r.get("ho_ci_lo", -1.0) <= 0:
        fails.append("CI下限≤0")
    if r.get("ho_sum_ex_top3", -1.0) <= 0:
        fails.append("上位3除外で負")
    # 未計測(構造的SL/TPのテンプレート・旧記録)は判定対象外。表では「-」と出る
    if r.get("ho_placebo_p", 0.0) > config.PLACEBO_MAX_P:
        fails.append("プラセボ並み")
    return fails


def fmt_params(p: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in p.items() if k != "lot")


def row_html(r: Dict[str, Any]) -> str:
    fails = r["fails"]
    placebo = f"{r['ho_placebo_p']:.2f}" if "ho_placebo_p" in r else "-"
    old = f"<br>旧コスト(固定スプレッド)でのPF {r['old_ho_pf']:.2f}" if r.get("old_ho_pf") is not None else ""
    badge = '<span class="ok">生存</span>' if not fails else f'<span class="ng">{html.escape(" / ".join(fails))}</span>'
    return (
        "<tr>"
        f"<td><b>{html.escape(r['tpl'])}</b><br><small>{r['pair']} {r['tf']} {r['win']}</small></td>"
        f"<td>{badge}</td>"
        f"<td>{r.get('ho_n', 0)}</td>"
        f"<td>{r.get('ho_pf', 0):.2f}</td>"
        f"<td>{r.get('ho_mean', 0):.2f}<br><small>[{r.get('ho_ci_lo', 0):.2f}, {r.get('ho_ci_hi', 0):.2f}]</small></td>"
        f"<td>{r.get('ho_p_pos', 0):.3f}</td>"
        f"<td>{r.get('q', 1):.3f}</td>"
        f"<td>{placebo}</td>"
        f"<td>{r.get('ho_sum', 0):.0f}<br><small>除外後 {r.get('ho_sum_ex_top3', 0):.0f}</small></td>"
        f"<td>{r.get('ho_maxdd', 0):.0f}</td>"
        f"<td>{r['tr_pf']:.2f} / {r['cf_pf']:.2f}<br><small>n {r['tr_n']} / {r['cf_n']}</small></td>"
        f"<td><small>{html.escape(r.get('desc') or fmt_params(r['params']))}{old}</small></td>"
        "</tr>"
    )


CSS = """
:root{--bg:#fafafa;--fg:#1a1a1a;--mut:#666;--line:#ddd;--card:#fff;--ok:#0a7a3d;--ng:#a33}
@media(prefers-color-scheme:dark){:root{--bg:#121416;--fg:#e8e8e8;--mut:#9aa;--line:#2c3136;--card:#1a1d20;--ok:#4cc38a;--ng:#e5837f}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}
p,li{color:var(--mut);font-size:13px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:12px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}.kpi b{display:block;font-size:20px}
.kpi span{color:var(--mut);font-size:12px}.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th:first-child,td:first-child,td:last-child,th:last-child{text-align:left}td:last-child{white-space:normal;min-width:220px}
th{color:var(--mut);font-weight:600}small{color:var(--mut)}.ok{color:var(--ok);font-weight:700}.ng{color:var(--ng)}
"""

HEAD = ("<tr><th>戦略</th><th>判定</th><th>n</th><th>PF</th><th>平均pips [CI95]</th><th>P(&gt;0)</th>"
        "<th>q値</th><th>プラセボp</th><th>合計pips</th><th>最大DD</th><th>PF train / confirm</th><th>パラメータ</th></tr>")


def build_html(stats: Dict[str, Any], evaluated: List[Dict[str, Any]]) -> str:
    survivors = [r for r in evaluated if not r["fails"]]
    others = [r for r in evaluated if r["fails"]][:50]
    kpis = [
        ("総試行", stats["trials"]), ("train通過", stats["passed_train"]),
        ("confirm通過", stats["passed_confirm"]), ("生存", len(survivors)),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{v:,}</b><span>{k}</span></div>' for k, v in kpis)

    def table(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "<p>該当なし</p>"
        return f'<div class="scroll"><table>{HEAD}{"".join(row_html(r) for r in rows)}</table></div>'

    runs = "".join(
        f"<li>{html.escape(r['at'])} UTC — {r['trials']:,}試行, holdout到達 {r['holdout']}</li>"
        for r in reversed(stats["runs"][-10:])
    )
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FX Cloud Lab</title><style>{CSS}</style></head><body>
<h1>FX Cloud Lab</h1><p>最終更新 {html.escape(stats['updated'])} UTC</p>
<div class="kpis">{kpi_html}</div>
<h2>生存候補</h2>{table(survivors)}
<h2>holdout到達・不採用(q値順 上位50)</h2>{table(others)}
<h2>判定基準</h2><ul>
<li>時系列分割: train 60% → confirm 20% → holdout 20%(古い順)。holdoutはtrain・confirmを両方通過した試行だけが見る</li>
<li>通過条件: train n≥{config.TRAIN_MIN_N} かつ PF≥{config.TRAIN_MIN_PF} / confirm n≥{config.CONFIRM_MIN_N} かつ PF≥{config.CONFIRM_MIN_PF}</li>
<li>生存条件(holdout): n≥{config.HOLDOUT_MIN_N}、片側t検定のBH-FDR q≤{config.FDR_Q}、平均pipsのbootstrap CI95下限&gt;0、上位3トレード除外後も合計pips&gt;0、プラセボp≤{config.PLACEBO_MAX_P}</li>
<li>プラセボp: シグナルを日単位でずらした{config.PLACEBO_N}本の偽ルールと合計pipsを比較した順位。高いほど「相場の地合いに乗っただけ」</li>
<li>戦略名 gen は特徴量条件を自動合成したルール(しきい値はtrain区間の分位点で固定)。それ以外は既存テンプレートのパラメータ摂動</li>
<li>コスト: ThreeTraderの実測スプレッド(時間帯別の時間加重平均。ロールオーバーは夏UTC21時・冬UTC22時として反映)+手数料{config.COMMISSION_PIPS}pips+スリッページ{config.SLIPPAGE_PIPS}pips。
実測が無い銘柄(USDCHF・USDCAD・NZDUSD・EURGBP)は実測銘柄の時間帯別の拡大幅を足して推定。スワップ・約定拒否は未反映</li>
<li>売りポジションのSLがスプレッド拡大だけで刈られる効果は未反映(ロールオーバーをまたぐ売りは実際より良く見える)</li>
<li>表示値はすべてholdoutの値。ここの「生存」は実運用可を意味しない。次段はブローカー実ティックでの再検証とデモ運用</li>
</ul><h2>直近の実行</h2><ul>{runs}</ul></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    args.site.mkdir(parents=True, exist_ok=True)

    new = [r for p in sorted(args.in_dir.rglob("*.jsonl")) for r in load_jsonl(p)]
    stats_path = args.site / "stats.json"
    empty = {"trials": 0, "errors": 0, "passed_train": 0, "passed_confirm": 0, "runs": []}
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else dict(empty)
    if stats.get("cost_model") != config.COST_MODEL:
        # コストモデルが変わったら数え直す。旧モデルの集計は参考として残す(判定には使わない)
        legacy = {k: stats.get(k) for k in ("cost_model", "trials", "passed_train", "passed_confirm", "updated")}
        stats = {**empty, "runs": [], "cost_model": config.COST_MODEL,
                 "legacy": stats.get("legacy", []) + ([legacy] if stats.get("trials") else [])}

    results = {r["key"]: r for r in load_jsonl(args.site / "results.jsonl")}
    # 現行コストモデルの記録は、同じkeyの旧モデル記録を置き換える(再判定の結果を反映)
    is_cur = lambda r: r.get("cost") == config.COST_MODEL  # noqa: E731
    fresh = [r for r in new if "key" in r and is_cur(r)
             and (r["key"] not in results or not is_cur(results[r["key"]]))]
    for r in fresh:
        if r.get("stage", 0) >= 1:
            results[r["key"]] = r
        else:
            results.pop(r["key"], None)   # 新コストではtrainを通らなかった
    current = [r for r in results.values() if is_cur(r)]

    now = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    stats["trials"] += sum(1 for r in new if r.get("stage", 0) >= 0)
    stats["errors"] += sum(1 for r in new if r.get("stage") == -1)
    stats["passed_train"] = len(current)
    stats["passed_confirm"] = sum(1 for r in current if r["stage"] >= 2)
    stats["updated"] = now
    if new:
        stats["runs"].append({"at": now, "trials": len(new),
                              "holdout": sum(1 for r in fresh if r.get("stage", 0) >= 2)})
        stats["runs"] = stats["runs"][-200:]

    evaluated = [dict(r) for r in current if r["stage"] >= 2 and "ho_p" in r]
    for r, q in zip(evaluated, bh_qvalues([r["ho_p"] for r in evaluated])):
        r["q"] = q
        r["fails"] = judge(r)
    evaluated.sort(key=lambda r: (r["q"], -r.get("ho_pf", 0)))

    with open(args.site / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results.values():
            f.write(json.dumps(r) + "\n")
    stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (args.site / "index.html").write_text(build_html(stats, evaluated), encoding="utf-8")
    (args.site / ".nojekyll").write_text("", encoding="utf-8")
    print(f"trials={stats['trials']} passed_train={stats['passed_train']} "
          f"holdout={len(evaluated)} survivors={sum(1 for r in evaluated if not r['fails'])}")


if __name__ == "__main__":
    main()
