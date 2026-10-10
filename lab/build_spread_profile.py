"""ブローカー実測のスプレッド記録から、銘柄×時間帯の平均スプレッド表を作る(ローカルで実行)。

入力: ThreeTrader のスプレッド収集CSV(ts_utc, ts_server, symbol, bid, ask, ...)。合計1GB超あるので
      1ファイルずつ・チャンク読みで集計し、全体をメモリに載せない。
出力: lab/spread_profile.json  … {pair: {"mean": [24], "median": [24], "p90": [24], "sec": [24]}}

- 収集側は「スプレッドが変わった時+5秒ごとの生存確認」だけを書く。行数で平均すると変化の多い
  瞬間に偏るので、各行を「次の行までの秒数」で重み付けする(上限 MAX_HOLD_S。休場や欠測の隙間を
  引き延ばさない)
- 時間軸は「米国夏時間のUTC時」に揃える。冬(米国標準時)の記録は1時間戻して同じ枠に入れる
  (ロールオーバーはNY17時 = 夏UTC21時 / 冬UTC22時)
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

from lab import config, costs

OUT = Path(__file__).resolve().parent / "spread_profile.json"
CHUNK = 400_000
BIN = 0.05          # ヒストグラムの刻み(pips)
MAX_PIPS = 60.0     # これ以上は最後のビンに寄せる
N_BINS = int(MAX_PIPS / BIN) + 1
MAX_HOLD_S = 6.0    # 1行が代表する最長秒数(収集側の生存確認間隔5秒+余裕)
MIN_SEC = 600.0     # 1時間帯あたりの最低観測秒数
COMMISSION_PIPS = config.COMMISSION_PIPS


def pip_of(pair: str) -> float:
    if pair == "XAUUSD":
        return 0.1
    return 0.01 if pair.endswith("JPY") else 0.0001


def main() -> None:
    src = Path(sys.argv[1])
    files = [f for f in sorted(src.glob("tt_spread_*.csv")) if f.stat().st_size > 100]  # 休場日の空ファイルを除く
    pairs = list(config.PAIRS)
    hist = {p: np.zeros((24, N_BINS)) for p in pairs}   # 秒数で重み付けしたヒストグラム
    total = {p: np.zeros(24) for p in pairs}             # Σ spread×秒

    for i, f in enumerate(files, 1):
        rows = 0
        for ch in pd.read_csv(f, usecols=["ts_utc", "symbol", "bid", "ask"], chunksize=CHUNK,
                              dtype={"ts_utc": "string", "symbol": "string"}):
            ch["pair"] = ch["symbol"].str.slice(0, 6)
            ch = ch[ch["pair"].isin(pairs) & (ch["ask"] > ch["bid"])]
            if ch.empty:
                continue
            ts = pd.to_datetime(ch["ts_utc"], format="mixed")
            sec = ts.astype("int64").to_numpy() / 1e9
            # 1ファイル=1日なので、夏冬の判定は先頭行の時刻で足りる(切替日の数時間の誤差は無視)
            shift = 0 if costs.is_us_dst(ts.iloc[0].to_pydatetime()) else 1
            hour = (ts.dt.hour.to_numpy() - shift) % 24
            pip = ch["pair"].map(pip_of).to_numpy(dtype=float)
            sp = ((ch["ask"] - ch["bid"]).to_numpy(dtype=float)) / pip
            b = np.minimum((sp / BIN).astype(np.int64), N_BINS - 1)
            for p, idx in ch.groupby("pair").indices.items():
                t = sec[idx]
                w = np.clip(np.diff(t, append=t[-1] + MAX_HOLD_S), 0.0, MAX_HOLD_S)
                np.add.at(hist[p], (hour[idx], b[idx]), w)
                np.add.at(total[p], hour[idx], sp[idx] * w)
            rows += len(ch)
        print(f"[{i}/{len(files)}] {f.name} rows={rows}", flush=True)

    out = {"_meta": {"source": "ThreeTrader Raw 実測(ask-bid、時間加重)", "files": len(files),
                     "from": files[0].stem[-8:], "to": files[-1].stem[-8:],
                     "built": time.strftime("%Y-%m-%d"), "unit": "pips",
                     "hours": "米国夏時間のUTC時(冬の記録は1時間戻して同じ枠に入れる)"}}
    centers = (np.arange(N_BINS) + 0.5) * BIN
    for p in pairs:
        n = hist[p].sum(axis=1)
        ok = n >= MIN_SEC
        if ok.sum() < 20:   # ほぼ全時間帯で観測がある銘柄だけ実測として採用
            print(f"{p}: 観測不足(有効 {int(ok.sum())}/24時間) → 推定に回す")
            continue
        safe = np.maximum(n, 1e-9)
        cdf = np.cumsum(hist[p], axis=1) / safe[:, None]
        mean = total[p] / safe

        def col(vals: np.ndarray) -> list:
            # 観測が足りない時間(休場など)は、その銘柄の他の時間の最大値で埋める(楽観側に倒さない)
            return [round(float(vals[h] if ok[h] else vals[ok].max()), 3) for h in range(24)]

        quant = lambda x: np.array([centers[min(np.searchsorted(cdf[h], x), N_BINS - 1)] for h in range(24)])  # noqa: E731
        out[p] = {"mean": col(mean), "median": col(quant(0.5)), "p90": col(quant(0.9)),
                  "sec": [int(v) for v in n], "estimated": False}
        print(f"{p}: 実測 {n.sum() / 3600:.0f}時間ぶん / 補完した時間帯 {[h for h in range(24) if not ok[h]]}")

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
        out[p] = {"mean": est, "median": est, "p90": est, "sec": [0] * 24, "estimated": True}
        print(f"{p}: 推定(通常時 {base:.2f}pips + 時間帯別の拡大幅、最大 {max(est):.1f}pips)")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT, "pairs:", [k for k in out if k != "_meta"])


if __name__ == "__main__":
    main()
