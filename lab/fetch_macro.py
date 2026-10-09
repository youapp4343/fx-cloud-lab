"""価格以外の公開データ(COT建玉・金利)を取得して data/macro/ に置く。

出力はどちらも「その値を市場参加者が知り得た時刻」= available_at を持つ。
特徴量側は available_at <= バー時刻 の行しか使わない(公表前の値を使わない)。

  cot.parquet   : available_at, as_of, ccy, spec_net, comm_net
                  CFTC legacy COT(火曜集計→金曜公表)。祝日の公表遅延を見込み as_of+7日で利用可とする
  rates.parquet : available_at, as_of, series, value
                  FRED。日次の米国債利回りは as_of+2日、月次の3か月金利は as_of+110日で利用可とする
                  (月次系列は公表が2〜3か月遅れる。2026-10-09時点で最新が7月分だった実測に基づく)

取得は直列・ランダムsleep・全fetchログ。過去年のCOTはキャッシュし、当年分だけ取り直す。
"""

from __future__ import annotations

import io
import random
import sys
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from app.core import datafeed

MACRO_DIR = datafeed.BASE_DIR / "data" / "macro"
RAW_DIR = MACRO_DIR / "raw"
COT_START_YEAR = 2014
SLEEP_RANGE = (2.0, 5.0)
MAX_CONSECUTIVE_FAILURES = 3
UA = "fx-cloud-lab research fetch (github.com/youapp4343/fx-cloud-lab)"

COT_MARKETS = {  # 通貨コード -> (市場名の先頭。|区切りは改称前後の別名, 取引所名の一部)
    "JPY": ("JAPANESE YEN", "CHICAGO MERCANTILE"),
    "EUR": ("EURO FX", "CHICAGO MERCANTILE"),
    "GBP": ("BRITISH POUND", "CHICAGO MERCANTILE"),
    "AUD": ("AUSTRALIAN DOLLAR", "CHICAGO MERCANTILE"),
    "CAD": ("CANADIAN DOLLAR", "CHICAGO MERCANTILE"),
    "CHF": ("SWISS FRANC", "CHICAGO MERCANTILE"),
    "NZD": ("NEW ZEALAND DOLLAR|NZ DOLLAR", "CHICAGO MERCANTILE"),
    "XAU": ("GOLD", "COMMODITY EXCHANGE"),
}
COT_LAG_DAYS = 7

# series名 -> (FRED系列ID, 公表ラグ日数)
FRED_SERIES = {
    "US2Y": ("DGS2", 2),
    "US10Y": ("DGS10", 2),
    "R3M_USD": ("IR3TIB01USM156N", 110),
    "R3M_EUR": ("IR3TIB01EZM156N", 110),
    "R3M_GBP": ("IR3TIB01GBM156N", 110),
    "R3M_JPY": ("IR3TIB01JPM156N", 110),
    "R3M_AUD": ("IR3TIB01AUM156N", 110),
    "R3M_CAD": ("IR3TIB01CAM156N", 110),
    "R3M_CHF": ("IR3TIB01CHM156N", 110),
    "R3M_NZD": ("IR3TIB01NZM156N", 110),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
            log(f"FETCH {url} {r.status} {len(body)}B")
            return body
    finally:
        time.sleep(random.uniform(*SLEEP_RANGE))


def fetch_cot(this_year: int) -> pd.DataFrame:
    frames = []
    for year in range(COT_START_YEAR, this_year + 1):
        cache = RAW_DIR / f"deacot{year}.zip"
        if year == this_year or not cache.exists():  # 当年分は毎週増えるので取り直す
            cache.write_bytes(http_get(f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"))
        with zipfile.ZipFile(cache) as z:
            raw = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        raw.columns = [c.strip() for c in raw.columns]
        frames.append(raw)
    cot = pd.concat(frames, ignore_index=True)

    def col(prefix: str) -> str:
        return next(c for c in cot.columns if c.startswith(prefix))

    mkt, dt = col("Market and Exchange"), next(c for c in cot.columns if "As of Date in Form YYYY-MM-DD" in c)
    rows = []
    for ccy, (name, exch) in COT_MARKETS.items():
        m = cot[cot[mkt].str.startswith(tuple(name.split("|")), na=False) & cot[mkt].str.contains(exch, na=False)]
        s = pd.DataFrame({
            "as_of": pd.to_datetime(m[dt], errors="coerce"),
            "ccy": ccy,
            "spec_net": pd.to_numeric(m[col("Noncommercial Positions-Long")], errors="coerce")
                        - pd.to_numeric(m[col("Noncommercial Positions-Short")], errors="coerce"),
            "comm_net": pd.to_numeric(m[col("Commercial Positions-Long")], errors="coerce")
                        - pd.to_numeric(m[col("Commercial Positions-Short")], errors="coerce"),
        }).dropna().drop_duplicates("as_of", keep="last")
        log(f"COT {ccy}: {len(s)}週 {s['as_of'].min():%Y-%m-%d}..{s['as_of'].max():%Y-%m-%d}")
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out["available_at"] = out["as_of"] + pd.Timedelta(days=COT_LAG_DAYS)
    return out.sort_values(["available_at", "ccy"]).reset_index(drop=True)


def fetch_rates() -> pd.DataFrame:
    rows = []
    failures = 0
    for name, (sid, lag) in FRED_SERIES.items():
        try:
            body = http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={COT_START_YEAR}-01-01")
            df = pd.read_csv(io.BytesIO(body))
            df.columns = ["as_of", "value"]
            df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # 休場日は "." が入る
            df = df.dropna()
            df["series"] = name
            df["available_at"] = df["as_of"] + pd.Timedelta(days=lag)
            log(f"RATE {name}: {len(df)}行 ..{df['as_of'].max():%Y-%m-%d}")
            rows.append(df)
            failures = 0
        except Exception as exc:  # noqa: BLE001 - 1系列の失敗で全体を落とさない
            failures += 1
            log(f"RATE {name} FAILED {exc}")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                log("連続失敗のためFRED取得を中止(強行しない)")
                break
    if not rows:
        return pd.DataFrame(columns=["available_at", "as_of", "series", "value"])
    return pd.concat(rows, ignore_index=True).sort_values(["available_at", "series"]).reset_index(drop=True)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    try:
        cot = fetch_cot(date.today().year)
        cot.to_parquet(MACRO_DIR / "cot.parquet", index=False)
        ok += 1
    except Exception as exc:  # noqa: BLE001 - macroが取れなくても価格だけで探索は続行できる
        log(f"COT FAILED {exc}")
    rates = fetch_rates()
    if len(rates):
        rates.to_parquet(MACRO_DIR / "rates.parquet", index=False)
        ok += 1
    log(f"macro: {ok}/2 データセット更新")
    return 0


if __name__ == "__main__":
    sys.exit(main())
