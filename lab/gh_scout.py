"""GitHubのFX戦略リポジトリを「検証の質」で採点して並べる(スター数は使わない)。

狙い: 見栄えの良いEAではなく、ウォークフォワード・アウトオブサンプル・コスト考慮・
統計検定・失敗の記録があるリポジトリを上に出す。あわせて、質の高いリポジトリで
使われている手法語を数え、lab/gen.py の特徴量に無いものを洗い出す材料にする。

取得は公式REST APIのみ(HTMLは読まない)。直列・ランダムsleep・全fetchログ・resume可。
README本文は scout/cache/ に置き(gitignore)、リポジトリには採点結果だけを残す。
他人のコードは取得も保存もしない。

  GH_TOKEN=$(gh auth token) python -m lab.gh_scout
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scout"
CACHE_DIR = OUT_DIR / "cache"
FETCH_LOG = CACHE_DIR / "fetch_log.jsonl"
API = "https://api.github.com"
PUSHED_SINCE = "2023-01-01"
PER_QUERY = 50
SLEEP_SEARCH = (2.5, 4.5)   # search APIは認証ありで30req/分
SLEEP_CORE = (0.4, 1.2)

QUERIES = [
    "forex walk forward", "forex out-of-sample backtest", "fx carry momentum value",
    "currency momentum backtest", "forex anomaly", "fx strategy research",
    "forex machine learning walk-forward", "mql5 walk forward", "forex intraday seasonality",
    "COT forex positioning", "forex mean reversion backtest", "fx factor investing",
    "forex backtest transaction costs", "expert advisor walk forward analysis",
    "foreign exchange trading strategy paper replication", "forex regime detection backtest",
    "currency pairs cointegration backtest", "forex order flow strategy backtest",
    "gold xauusd strategy backtest out of sample", "forex volatility breakout research",
]

# (ラベル, 正規表現, 加点)。README中に1回でも出れば加点(回数では増やさない=長文有利を避ける)
EVIDENCE = [
    ("walk_forward", r"walk[\s-]?forward", 3),
    ("out_of_sample", r"out[\s-]of[\s-]sample|\boos\b|hold[\s-]?out", 3),
    ("costs", r"transaction cost|slippage|commission|spread[s]? (?:cost|assum|includ|adjust)|after cost", 2),
    ("stat_test", r"bootstrap|monte[\s-]carlo|permutation test|p[\s-]?value|t[\s-]?stat|deflated sharpe|"
                  r"multiple (?:testing|comparison)|false discovery|white'?s reality check", 3),
    ("paper", r"arxiv\.org|ssrn\.com|doi\.org|journal of|working paper", 2),
    ("honesty", r"limitation|did not work|does not work|doesn'?t work|negative result|overfit|"
                r"data[\s-]snoop|look[\s-]?ahead|survivorship", 2),
    ("period", r"(?:19|20)\d\d\s*(?:-|–|to|until|through)\s*(?:19|20)\d\d", 1),
    ("reproducible", r"requirements\.txt|environment\.yml|pip install|notebook|\.ipynb|make test|pytest", 1),
]
HYPE = [
    ("guarantee", r"guarantee[d]?|risk[\s-]free|no[\s-]loss|never los", 3),
    ("win_rate", r"(?:9\d|100)\s?% (?:win|accura|profit)|holy grail", 3),
    ("sales", r"buy now|license key|purchase|contact (?:me|us) (?:on|via) (?:telegram|whatsapp)|"
              r"t\.me/|discount|premium version|vip", 3),
    ("martingale", r"martingale|grid recovery|doubling", 2),
]

# 手法語: lab/gen.py に同等の特徴量があるか(True=既にある)
METHODS = {
    "rsi": (r"\brsi\b", True), "moving_average_dev": (r"moving average|\bsma\b|\bema\b", True),
    "donchian_breakout": (r"donchian|channel breakout", True), "atr_volatility": (r"\batr\b|average true range", True),
    "candle_shape": (r"candlestick|pin bar|engulfing|wick", True), "lead_lag": (r"lead[\s-]lag|cross[\s-]asset", True),
    "cot": (r"\bcot\b|commitments? of traders", True), "carry_rates": (r"\bcarry\b|interest rate differential", True),
    "bollinger": (r"bollinger", False), "macd": (r"\bmacd\b", False), "adx": (r"\badx\b", False),
    "stochastic": (r"stochastic", False), "ichimoku": (r"ichimoku", False),
    "keltner_squeeze": (r"keltner|squeeze", False), "vwap": (r"\bvwap\b", False),
    "efficiency_ratio": (r"efficiency ratio|kaufman|\bkama\b", False),
    "hurst": (r"hurst|fractal dimension|variance ratio", False),
    "zscore_reversion": (r"z[\s-]?score", False), "kalman": (r"kalman", False),
    "cointegration_pairs": (r"cointegrat|pairs trading|stat(?:istical)? arb", False),
    "regime_hmm": (r"hidden markov|\bhmm\b|regime (?:switch|detect)", False),
    "garch_vol": (r"garch|realized vol|volatility forecast", False),
    "session_time": (r"asian (?:session|range)|london (?:open|session|breakout)|new york (?:open|session)|"
                     r"time[\s-]of[\s-]day|intraday seasonal", False),
    "calendar": (r"day[\s-]of[\s-]week|month[\s-]end|turn of the month|\bfix\b|wmr", False),
    "currency_strength": (r"currency strength|cross[\s-]sectional|basket", False),
    "momentum_tsmom": (r"time[\s-]series momentum|\btsmom\b|trend[\s-]following", False),
    "value_ppp": (r"\bppp\b|purchasing power|real exchange rate", False),
    "order_flow": (r"order flow|order book|microstructure|tick imbalance", False),
    "news_events": (r"\bnfp\b|non[\s-]farm|economic calendar|news event|fomc", False),
    "pivot_levels": (r"pivot point|support and resistance|round number", False),
    "autocorrelation": (r"autocorrelation|serial correlation", False),
    "entropy": (r"entropy", False),
    "ml_model": (r"random forest|xgboost|lightgbm|neural network|\blstm\b|reinforcement learning", False),
}


def _sleep(rng: tuple) -> None:
    time.sleep(random.uniform(*rng))


def api_get(path: str, accept: str = "application/vnd.github+json") -> Optional[bytes]:
    url = path if path.startswith("http") else API + path
    headers = {"Accept": accept, "User-Agent": "fx-cloud-lab-scout", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, body = 0, None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                status, body = r.status, r.read()
            break
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (403, 429):  # レート制限: 待ち時間を倍にして再試行、続くなら諦める
                time.sleep(60 * (attempt + 1))
                continue
            break
        except Exception:  # noqa: BLE001 - ネットワーク断は1件飛ばして続行
            status = -1
            break
    with open(FETCH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": url, "status": status}) + "\n")
    return body if status == 200 else None


def search_repos() -> Dict[str, Dict[str, Any]]:
    cache = CACHE_DIR / "search.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    repos: Dict[str, Dict[str, Any]] = {}
    for q in QUERIES:
        full = urllib.parse.quote(f"{q} pushed:>{PUSHED_SINCE} fork:false")
        body = api_get(f"/search/repositories?q={full}&per_page={PER_QUERY}")
        _sleep(SLEEP_SEARCH)
        if body is None:
            print(f"[search] FAILED {q}", flush=True)
            continue
        items = json.loads(body).get("items", [])
        for it in items:
            r = repos.setdefault(it["full_name"], {
                "full_name": it["full_name"], "url": it["html_url"], "description": it.get("description") or "",
                "language": it.get("language"), "stars": it.get("stargazers_count", 0),
                "pushed_at": it.get("pushed_at", "")[:10], "size_kb": it.get("size", 0),
                "license": (it.get("license") or {}).get("spdx_id"), "queries": [],
            })
            r["queries"].append(q)
        print(f"[search] {q}: {len(items)}件 (累計 {len(repos)})", flush=True)
    cache.write_text(json.dumps(repos), encoding="utf-8")
    return repos


def readme_of(full_name: str) -> str:
    cache = CACHE_DIR / "readme" / (full_name.replace("/", "__") + ".md")
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    body = api_get(f"/repos/{full_name}/readme", accept="application/vnd.github.raw+json")
    _sleep(SLEEP_CORE)
    text = body.decode("utf-8", errors="replace") if body else ""
    cache.write_text(text, encoding="utf-8")
    return text


def score(repo: Dict[str, Any], readme: str) -> Dict[str, Any]:
    text = (repo["description"] + "\n" + readme).lower()
    ev = [(k, pts) for k, pat, pts in EVIDENCE if re.search(pat, text)]
    hy = [(k, pts) for k, pat, pts in HYPE if re.search(pat, text)]
    methods = sorted(k for k, (pat, _) in METHODS.items() if re.search(pat, text))
    fx = bool(re.search(r"forex|\bfx\b|currenc|eurusd|usdjpy|xauusd|foreign exchange|mql[45]|metatrader", text))
    return {
        **repo,
        "readme_chars": len(readme),
        "evidence": [k for k, _ in ev], "hype": [k for k, _ in hy],
        "score": sum(p for _, p in ev) - sum(p for _, p in hy),
        "methods": methods, "fx_related": fx,
    }


def write_report(rows: List[Dict[str, Any]]) -> None:
    ranked = [r for r in rows if r["fx_related"] and r["readme_chars"] >= 500]
    ranked.sort(key=lambda r: (-r["score"], r["pushed_at"]), reverse=False)
    ranked.sort(key=lambda r: (r["score"], r["pushed_at"]), reverse=True)
    top = [r for r in ranked if r["score"] >= 8]
    rest = [r for r in ranked if r["score"] < 8]

    def freq(group: List[Dict[str, Any]]) -> Dict[str, float]:
        return {k: (sum(k in r["methods"] for r in group) / len(group) if group else 0.0) for k in METHODS}

    ft, fr = freq(top), freq(rest)
    lines = [
        "# GitHub スカウト結果(検証の質順)", "",
        f"生成 {time.strftime('%Y-%m-%d')} / 検索 {len(QUERIES)}クエリ / 取得 {len(rows)}件 / "
        f"FX関連かつREADME 500字以上 {len(ranked)}件 / 高スコア(8点以上) {len(top)}件", "",
        "採点はREADMEと説明文の語のみ。**コードの正しさも成績の真偽も確認していない**。"
        "スターは採点に使っていない(参考表示のみ)。", "",
        "## 上位40件", "",
        "| 点 | リポジトリ | 言語 | 最終push | ★ | 根拠 | 減点 | ライセンス |", "|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked[:40]:
        lines.append(f"| {r['score']} | [{r['full_name']}]({r['url']}) | {r['language'] or '-'} | {r['pushed_at']} | "
                     f"{r['stars']} | {', '.join(r['evidence'])} | {', '.join(r['hype']) or '-'} | {r['license'] or '-'} |")
    lines += ["", "## 手法語の出現率(高スコア群 vs それ以外)", "",
              "高スコア群で多く、`lab/gen.py` に特徴量が無いものが追加候補。", "",
              "| 手法 | 高スコア群 | それ以外 | gen.pyに |", "|---|---|---|---|"]
    for k in sorted(METHODS, key=lambda k: -ft[k]):
        lines.append(f"| {k} | {ft[k]:.0%} | {fr[k]:.0%} | {'あり' if METHODS[k][1] else '**なし**'} |")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_DIR / "repos.jsonl", "w", encoding="utf-8") as f:
        for r in ranked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"ranked={len(ranked)} top(>=8)={len(top)}")


def main() -> None:
    (CACHE_DIR / "readme").mkdir(parents=True, exist_ok=True)
    repos = search_repos()
    rows = []
    for i, (name, repo) in enumerate(sorted(repos.items()), 1):
        rows.append(score(repo, readme_of(name)))
        if i % 50 == 0:
            print(f"[readme] {i}/{len(repos)}", flush=True)
    write_report(rows)


if __name__ == "__main__":
    main()
