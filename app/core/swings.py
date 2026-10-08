"""スイング検出とフィボナッチ水準の計算。波形パターン・フィボナッチ・ハーモニクス系戦略の基盤モジュール。

設計上の要点(確定遅延・先読みバイアス回避):
スイングハイ/ローは「前後order本より高い/低い」という定義上、発生した時点(bar_idx)では
判定できず、その後order本経過してはじめて「あれがスイングだった」と確定判断できる。
このため各スイングに confirmed_at_idx = bar_idx + order を付与し、シグナル生成側は
「バーiの時点では confirmed_at_idx <= i のスイングのみ参照してよい」という規律を守ることで
先読みバイアスを構造的に防ぐ(app/core/indicators.pyのdonchian_channelのshift(1)、
app/core/engine.pyのシグナルshift(1)+次バー始値執行と同じ思想)。
confirmed_at_idxがlen(df)を超える(=データ末尾に近すぎてまだ確定していない)スイングは、
未確定情報を返さないよう戻り値から除外する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

_SWING_COLUMNS = ["bar_idx", "price", "kind", "confirmed_at_idx"]

_RETRACEMENT_RATIOS = ("0.236", "0.382", "0.5", "0.618", "0.786")
_EXTENSION_RATIOS = ("1.272", "1.618")


def _empty_swings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_idx": pd.Series(dtype=int),
            "price": pd.Series(dtype=float),
            "kind": pd.Series(dtype=object),
            "confirmed_at_idx": pd.Series(dtype=int),
        }
    )


def detect_swings(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """スイングハイ/ローを検出する(argrelextremaベース)。

    戻り値のDataFrame列: bar_idx(int), price(float), kind(str, "H" or "L"), confirmed_at_idx(int)
    - bar_idx: そのスイング点が実際に発生したバーのインデックス(dfの位置インデックス、0始まり)
    - confirmed_at_idx = bar_idx + order: そのスイングが確定したと判断できる最初のバーインデックス。
      シグナル生成側は、バーiの時点で「confirmed_at_idx <= i」のスイングだけを参照してよい。
    - bar_idx昇順でソートして返す。
    """
    n = len(df)
    if n == 0:
        return _empty_swings()

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    # 厳密なgreater/lessだと連続同値のフラットな山/谷を取りこぼすため等号込みで検出する。
    high_idx = np.unique(argrelextrema(high, np.greater_equal, order=order)[0])
    low_idx = np.unique(argrelextrema(low, np.less_equal, order=order)[0])

    records: list[tuple[int, float, str, int]] = []
    for idx in high_idx:
        idx = int(idx)
        confirmed_at = idx + order
        if confirmed_at > n:  # データ末尾に近すぎてまだ確定していない -> 返さない
            continue
        records.append((idx, float(high[idx]), "H", confirmed_at))
    for idx in low_idx:
        idx = int(idx)
        confirmed_at = idx + order
        if confirmed_at > n:
            continue
        records.append((idx, float(low[idx]), "L", confirmed_at))

    result = pd.DataFrame(records, columns=_SWING_COLUMNS)
    # 同一(bar_idx, kind)の重複が出た場合に備えた安全策(通常argrelextremaの仕様上は発生しない)。
    result = result.drop_duplicates(subset=["bar_idx", "kind"], keep="first")
    result = result.sort_values("bar_idx", kind="stable").reset_index(drop=True)
    result["bar_idx"] = result["bar_idx"].astype(int)
    result["confirmed_at_idx"] = result["confirmed_at_idx"].astype(int)
    return result


def fib_retracement_levels(swing_start_price: float, swing_end_price: float) -> dict[str, float]:
    """swing_start→swing_endの値幅に対するリトレースメント水準(押し/戻りの価格)を返す。

    戻り値キー: "0.236","0.382","0.5","0.618","0.786"。
    level = swing_end_price - (swing_end_price - swing_start_price) * ratio という式1本で、
    上昇スイング(end>start)は end から差し引く方向、下降スイング(end<start)は
    end に足し戻す方向に自動的に符号が切り替わる(どちらも「start側に戻る」向きになる)。
    """
    diff = swing_end_price - swing_start_price
    return {ratio: swing_end_price - diff * float(ratio) for ratio in _RETRACEMENT_RATIOS}


def fib_extension_levels(
    swing_start_price: float, swing_end_price: float, retrace_price: float
) -> dict[str, float]:
    """A→B→Cの3点から、B→Aの値幅を使ったC起点のエクステンション水準を返す。

    戻り値キー: "1.272","1.618"。swing_start=A, swing_end=B, retrace_price=C として、
    level = retrace_price + (swing_end_price - swing_start_price) * ratio。
    B-Aが正(上昇スイング)ならCから上方向、負(下降スイング)なら下方向に伸びる
    (A→Bのスイングと同じ向きにさらに進む水準)。
    """
    diff = swing_end_price - swing_start_price
    return {ratio: retrace_price + diff * float(ratio) for ratio in _EXTENSION_RATIOS}


def detect_swings_zigzag(df: pd.DataFrame, atr: pd.Series, atr_multiple: float = 2.0) -> pd.DataFrame:
    """ATR×atr_multiple以上の逆行があった時点でスイング確定とみなす代替実装(ZigZag)。

    戻り値の列構造はdetect_swingsと同じ(bar_idx, price, kind, confirmed_at_idx)。
    このアルゴリズムはATRの逆行そのものが確定条件なので、confirmed_at_idxは
    「逆行を検知したバーのインデックス」(=検知した時点で確定、追加の遅延は不要)。
    """
    n = len(df)
    if n == 0:
        return _empty_swings()

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_arr = np.asarray(atr, dtype=float)

    records: list[tuple[int, float, str, int]] = []
    trend: str | None = None  # 方向未確定の間は上下両方向の反転候補を同時に監視する
    high_idx, high_px = 0, float(high[0])
    low_idx, low_px = 0, float(low[0])

    for i in range(1, n):
        if high[i] > high_px:
            high_idx, high_px = i, float(high[i])
        if low[i] < low_px:
            low_idx, low_px = i, float(low[i])

        threshold = atr_arr[i] * atr_multiple
        if np.isnan(threshold):
            continue  # ATRウォームアップ中は逆行判定不能。極値候補の更新のみ行い判定はスキップする。

        if trend != "down" and high_px - low[i] >= threshold:
            # 直近高値からthreshold以上下落 -> 直近高値をスイングハイとして確定し、以降は安値を追跡する
            records.append((high_idx, high_px, "H", i))
            trend = "down"
            low_idx, low_px = i, float(low[i])
        elif trend != "up" and high[i] - low_px >= threshold:
            # 直近安値からthreshold以上上昇 -> 直近安値をスイングローとして確定し、以降は高値を追跡する
            records.append((low_idx, low_px, "L", i))
            trend = "up"
            high_idx, high_px = i, float(high[i])

    result = pd.DataFrame(records, columns=_SWING_COLUMNS)
    result = result.sort_values("bar_idx", kind="stable").reset_index(drop=True)
    result["bar_idx"] = result["bar_idx"].astype(int)
    result["confirmed_at_idx"] = result["confirmed_at_idx"].astype(int)
    return result
