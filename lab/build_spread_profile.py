"""ブローカー実測のスプレッド記録から、銘柄×UTC時の平均スプレッド表を作る(ローカルで実行)。

入力: ThreeTrader のスプレッド収集CSV(ts_utc, ts_server, symbol, bid, ask, ...)。合計1GB超あるので
      1ファイルずつ・チャンク読みで集計し、全体をメモリに載せない。
出力: lab/spread_profile.json  … {pair: {"mean": [24], "median": [24], "p90": [24], "n": [24]}}

- 収集は定期ポーリングなので、同じティック(symbol, ts_server)は1回だけ数える(休場中の残骸を増やさない)
- pipの定義はバックテストに合わせる(XAUUSD=0.1、JPYクロス=0.01、他=0.0001)

  python -m lab.build_spread_profile "C:\\CodexProject\\FX\\research\\threetrader_demo\\spread"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lab import config

OUT = Path(__file__).resolve().parent / "spread_profile.json"
CHUNK = 400_000
BIN = 0.05          # ヒストグラムの刻み(pips)
MAX_PIPS = 60.0     # これ以上は最後のビンに寄せる
N_BINS = int(MAX_PIPS / BIN) + 1
MIN_TICKS = 50      # 1時間帯あたりの最低観測数
COMMISSION_PIPS = 0.4


def pip_of(pair: str) -> float:
    if pair == "XAUUSD":
        return 0.1
    return 0.01 if pair.endswith("JPY") else 0.0001


def main() -> None:
    src = Path(sys.argv[1])
    files = sorted(src.glob("tt_spread_*.csv"))
    pairs = list(config.PAIRS)
    hist = {p: np.zeros((24, N_BINS), dtype=np.int64) for p in pairs}
    total = {p: np.zeros(24) for p in pairs}

    for i, f in enumerate(files, 1):
        rows = 0
        for ch in pd.read_csv(f, usecols=["ts_utc", "ts_server", "symbol", "bid", "ask"], chunksize=CHUNK,
                              dtype={"ts_utc": "string", "ts_server": "string", "symbol": "string"}):
            ch = ch.drop_duplicates(["symbol", "ts_server"])
            ch["pair"] = ch["symbol"].str.slice(0, 6)
            ch = ch[ch["pair"].isin(pairs) & (ch["ask"] > ch["bid"])]
            if ch.empty:
                continue
            hour = ch["ts_utc"].str.slice(11, 13).astype(int).to_numpy()
            pip = ch["pair"].map(pip_of).to_numpy(dtype=float)
            sp = ((ch["ask"] - ch["bid"]).to_numpy(dtype=float)) / pip
            b = np.minimum((sp / BIN).astype(np.int64), N_BINS - 1)
            for p, idx in ch.groupby("pair").indices.items():
                np.add.at(hist[p], (hour[idx], b[idx]), 1)
                np.add.at(total[p], hour[idx], sp[idx])
            rows += len(ch)
        print(f"[{i}/{len(files)}] {f.name} ticks={rows}", flush=True)

    out = {"_meta": {"source": "ThreeTrader Raw 実測ティック(ask-bid)", "files": len(files),
                     "from": files[0].stem[-8:], "to": files[-1].stem[-8:],
                     "built": time.strftime("%Y-%m-%d"), "unit": "pips", "hours": "UTC"}}
    centers = (np.arange(N_BINS) + 0.5) * BIN
    for p in pairs:
        n = hist[p].sum(axis=1)
        ok = n >= MIN_TICKS
        if ok.sum() < 20:   # ほぼ全時間帯で観測がある銘柄だけ実測として採用
            print(f"{p}: 観測不足(有効 {int(ok.sum())}/24時間) → 推定に回す")
            continue
        safe = np.maximum(n, 1)
        cdf = np.cumsum(hist[p], axis=1) / safe[:, None]
        mean = total[p] / safe

        def col(vals: np.ndarray) -> list:
            # 観測が足りない時間(休場など)は、その銘柄の他の時間の最大値で埋める(楽観側に倒さない)
            return [round(float(vals[h] if ok[h] else vals[ok].max()), 3) for h in range(24)]

        quant = lambda x: np.array([centers[min(np.searchsorted(cdf[h], x), N_BINS - 1)] for h in range(24)])  # noqa: E731
        out[p] = {"mean": col(mean), "median": col(quant(0.5)), "p90": col(quant(0.9)),
                  "n": [int(v) for v in n], "estimated": False}
        print(f"{p}: 実測 {int(n.sum())}ティック / 補完した時間 {[h for h in range(24) if not ok[h]]}")

    # 実測が無い銘柄: 実測のあるFX銘柄から「その時間は通常より何pips広いか」の中央値を取り、
    # 仮定スプレッド水準に足す。倍率ではなく加算にするのは、通常時が0.1pips台の銘柄の倍率を
    # 0.6pips水準の銘柄に掛けると20pips超の非現実的な値になるため
    fx = [k for k in out if k not in ("_meta", "XAUUSD")]
    excess = np.median([np.array(out[k]["mean"]) - np.median(out[k]["mean"]) for k in fx], axis=0)
    for p in pairs:
        if p in out:
            continue
        base = max(config.SPREAD_PIPS[p] - COMMISSION_PIPS, 0.1)
        est = [round(float(max(base + e, 0.05)), 3) for e in excess]
        out[p] = {"mean": est, "median": est, "p90": est, "n": [0] * 24, "estimated": True}
        print(f"{p}: 推定(通常時 {base:.2f}pips + 時間帯別の拡大幅、最大 {max(est):.1f}pips)")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT, "pairs:", [k for k in out if k != "_meta"])


if __name__ == "__main__":
    main()
