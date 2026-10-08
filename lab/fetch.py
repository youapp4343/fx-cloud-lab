"""Dukascopyから探索用OHLCを取得して data/ohlc/{SYMBOL}_{TF}.parquet に置く。

- H4/M30は取得せずH1/M15からリサンプルで作る
- 既存ファイルがあれば末尾から差分だけ取得する(resume / 再取得を避ける)
- 直列・ランダムsleep・全fetchをログ出力
- 連続失敗したら即停止する(強行しない)。取れた分だけで探索は続行できる
"""

from __future__ import annotations

import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from app.core import datafeed
from lab import config

OHLC_DIR = datafeed.OHLC_DIR
MAX_CONSECUTIVE_FAILURES = 3
SLEEP_RANGE = (4.0, 9.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def fetch_one(symbol: str, tf: str, today: date) -> str:
    out = OHLC_DIR / f"{symbol}_{tf}.parquet"
    old = pd.read_parquet(out) if out.exists() else None
    if old is not None and len(old):
        start = (old["timestamp"].max() - timedelta(days=2)).date()
    else:
        start = datetime.strptime(config.FETCH_START, "%Y-%m-%d").date()
    if start >= today:
        return "fresh"

    r = datafeed.download(symbol, start, today, timeframe=tf)
    raw = Path(r["path"])
    new = pd.read_parquet(raw)
    raw.unlink()
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(out, index=False)
    return f"{start}..{today} +{len(new)}行 計{len(df)}行"


def build_derived(symbol: str, tf: str) -> None:
    src_tf, rule = config.DERIVED_TFS[tf]
    src = OHLC_DIR / f"{symbol}_{src_tf}.parquet"
    if not src.exists():
        return
    base = pd.read_parquet(src).set_index("timestamp")
    out = base.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"]).reset_index()
    out.to_parquet(OHLC_DIR / f"{symbol}_{tf}.parquet", index=False)


def main() -> int:
    OHLC_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    failures = 0
    for tf in config.FETCH_TFS:
        for symbol in config.PAIRS:
            try:
                status = fetch_one(symbol, tf, today)
                log(f"FETCH {symbol} {tf} OK {status}")
                failures = 0
            except Exception as exc:  # noqa: BLE001 - 1銘柄の失敗で全体を落とさない
                failures += 1
                log(f"FETCH {symbol} {tf} FAILED {exc}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    log(f"{failures}回連続失敗のため取得を中止(遮断の可能性)。取得済み分のみで続行")
                    break
            time.sleep(random.uniform(*SLEEP_RANGE))
        else:
            continue
        break

    for symbol in config.PAIRS:
        for tf in config.DERIVED_TFS:
            build_derived(symbol, tf)

    files = sorted(p.name for p in OHLC_DIR.glob("*.parquet"))
    log(f"利用可能: {len(files)}ファイル")
    return 0 if files else 1


if __name__ == "__main__":
    sys.exit(main())
