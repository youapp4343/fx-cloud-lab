"""macro特徴量の公表ラグ検査(ローカルに data/ohlc と data/macro がある前提の手動テスト)。

  python -m tests.test_macro_lag
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

from lab import gen  # noqa: E402

MACRO_PREFIX = ("cot_", "us2y_", "us_curve", "rate_")


def load(pair: str) -> pd.DataFrame:
    df = pd.read_parquet(gen.datafeed.OHLC_DIR / f"{pair}_H1.parquet")
    return df[df["timestamp"] >= "2016-01-01"].reset_index(drop=True)


def main() -> None:
    macro = gen.load_macro()
    assert macro, "data/macro が空"

    for pair in ("EURUSD", "USDJPY", "XAUUSD"):
        df = load(pair)
        a, b = int(len(df) * 0.6), int(len(df) * 0.8)
        names, mat, _ = gen.build_features(df, {}, pair, macro)
        block = gen.Block(df, (a, b), 0.0001, {}, pair, macro)
        for nm in [x for x in names if x.startswith(MACRO_PREFIX)]:
            j = names.index(nm)
            cov = [round(float(np.isfinite(mat[sl, j]).mean()), 2) for sl in block.slices]
            print(f"{pair} {nm:16s} coverage {cov} {'採用' if nm in block.names else '除外'}")

    # 検査1: 時刻Tで打ち切ったmacroで作っても、T以前のバーの特徴量は変わらない
    df = load("USDJPY")
    t_cut = pd.Timestamp("2022-06-15")
    cut = {k: v[v["available_at"] <= t_cut] for k, v in macro.items()}
    n1, m1, _ = gen.build_features(df, {}, "USDJPY", macro)
    _, m2, _ = gen.build_features(df, {}, "USDJPY", cut)
    k = int((df["timestamp"] <= t_cut).sum())
    idx = [n1.index(x) for x in n1 if x.startswith(MACRO_PREFIX)]
    assert idx, "macro特徴量が1つも無い"
    assert np.allclose(m1[:k][:, idx], m2[:k][:, idx], equal_nan=True), "打ち切りで過去の特徴量が変わった=先読み"
    print("検査1 OK: 打ち切り不変")

    # 検査2: 各バーが参照するCOTの集計日は、必ずバー時刻より公表ラグ日数以上前
    g = macro["cot"][macro["cot"]["ccy"] == "JPY"].sort_values("available_at")
    j = pd.merge_asof(df[["timestamp"]], g[["available_at", "as_of"]], left_on="timestamp", right_on="available_at")
    min_age = (j["timestamp"] - j["as_of"]).min()
    assert min_age >= pd.Timedelta(days=7), min_age
    print(f"検査2 OK: COT集計日からの最小経過 {min_age}")

    # 検査3: 月次金利は観測月の初日から110日以上経ってから使われる
    r = macro["rates"]
    m = r[r["series"] == "R3M_JPY"].sort_values("available_at")
    j = pd.merge_asof(df[["timestamp"]], m[["available_at", "as_of"]], left_on="timestamp", right_on="available_at")
    min_age = (j["timestamp"] - j["as_of"]).min()
    assert min_age >= pd.Timedelta(days=110), min_age
    print(f"検査3 OK: 月次金利の観測日からの最小経過 {min_age}")


if __name__ == "__main__":
    main()
