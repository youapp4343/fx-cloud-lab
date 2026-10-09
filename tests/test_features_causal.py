"""全特徴量の先読み検査とcoverage確認(ローカルに data/ohlc がある前提の手動テスト)。

  python -m tests.test_features_causal

検査: 末尾を切り落としたデータで作り直しても、残した区間の特徴量が1つも変わらないこと。
macro・通貨強弱も同じ時刻で打ち切る(将来の公表値が過去のバーに漏れていないことの確認)。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

from lab import gen  # noqa: E402


def main() -> None:
    macro = gen.load_macro()
    for pair, tf in (("EURUSD", "H1"), ("EURUSD", "H4"), ("USDJPY", "H1"), ("XAUUSD", "H1")):
        df = pd.read_parquet(gen.datafeed.OHLC_DIR / f"{pair}_{tf}.parquet")
        df = df[df["timestamp"] >= "2016-01-01"].reset_index(drop=True)
        refs = {}
        ref_path = gen.datafeed.OHLC_DIR / f"USDJPY_{tf}.parquet"
        if pair != "USDJPY" and ref_path.exists():
            refs["USDJPY"] = pd.read_parquet(ref_path, columns=["timestamp", "close"])

        k = int(len(df) * 0.7)
        t_cut = df["timestamp"].iloc[k - 1]
        n_full, m_full, _ = gen.build_features(df, refs, pair, macro, tf)

        df_cut = df.iloc[:k].reset_index(drop=True)
        refs_cut = {n: r[r["timestamp"] <= t_cut] for n, r in refs.items()}
        macro_cut = {n: v[v["available_at"] <= t_cut] for n, v in macro.items()}
        n_cut, m_cut, _ = gen.build_features(df_cut, refs_cut, pair, macro_cut, tf)

        assert n_full == n_cut, "特徴量の並びが変わった"
        bad = [n_full[j] for j in range(len(n_full))
               if not np.allclose(m_full[:k, j], m_cut[:, j], equal_nan=True, rtol=1e-5, atol=1e-6)]
        a, b = int(len(df) * 0.6), int(len(df) * 0.8)
        block = gen.Block(df, (a, b), 0.0001, refs, pair, macro, tf)
        dropped = [n for n in n_full if n not in block.names]
        print(f"{pair} {tf}: 特徴量{len(n_full)} / 先読みNG {bad or 'なし'} / coverage除外 {dropped or 'なし'}")
        assert not bad, bad

        # 時刻の単位が混ざっても同じ結果になること(クラウドのparquetはms/us、ローカルはns)
        df_ms = df.copy()
        df_ms["timestamp"] = df_ms["timestamp"].astype("datetime64[ms]")
        macro_us = {n: v.assign(available_at=v["available_at"].astype("datetime64[us]")) for n, v in macro.items()}
        n_mix, m_mix, _ = gen.build_features(df_ms, refs, pair, macro_us, tf)
        assert n_mix == n_full and np.allclose(m_mix, m_full, equal_nan=True, rtol=1e-5, atol=1e-6), "単位混在で結果が変わった"


if __name__ == "__main__":
    main()
