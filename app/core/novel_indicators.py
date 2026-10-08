"""新規指標ライブラリ(Phase10: 独自指標・戦略の開拓)。app/core/indicators.pyの標準指標
(SMA/EMA/RSI/ボリンジャー/ATR/ドンチャン)には無い、レジーム判定・出来高圧力・ボラティリティ
の相対水準を扱う指標を追加する。pandas Seriesを受け取る純粋関数群(indicators.pyと同じ流儀)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def variance_ratio(close: pd.Series, period: int, k: int) -> pd.Series:
    """分散比(Lo-MacKinlay型のバリアンスレシオ)。

    period本のローリング窓で「k期間リターンの分散 / (k * 1期間リターンの分散)」を計算する。
    ランダムウォークであれば理論値は1に近づく。VR>1はリターンが正の自己相関を持つ
    (トレンド・モメンタム優位)、VR<1は負の自己相関を持つ(平均回帰優位)ことを示す
    regime_adaptiveテンプレートのレジーム判定に使う指標。
    """
    ret1 = close.diff()
    retk = close.diff(k)
    var1 = ret1.rolling(period).var(ddof=0)
    vark = retk.rolling(period).var(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = vark / (k * var1)
    return vr.replace([np.inf, -np.inf], np.nan)


def _pct_rank_raw(window: np.ndarray) -> float:
    current = window[-1]
    return float((window <= current).sum()) / len(window)


def atr_percentile_rank(atr_series: pd.Series, lookback: int) -> pd.Series:
    """直近lookback本のATR分布内で、現在のATRが位置する百分位順位(0=最低,1=最高)。

    低い値(例: 0.1〜0.2以下)は「ボラティリティが自身の最近の歴史の中で低い水準にある」
    (スクイーズ、次の拡大=ブレイクアウトの予兆となりうる)ことを示す。
    vol_squeeze_breakoutテンプレートのスクイーズ判定に使う指標。
    """
    return atr_series.rolling(lookback).apply(_pct_rank_raw, raw=True)


def close_location_value(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """終値がそのバーのレンジ内のどこに位置するかを[-1, 1]で表す(Chaikin CLVと同じ定義)。

    +1に近いほど高値圏で引け(買い圧力優勢)、-1に近いほど安値圏で引け(売り圧力優勢)。
    high==lowの退化バーは0除算を避けNaNを返す。出来高と掛け合わせて累積することで
    「価格に対して出来高が伴う圧力の方向」を見るorderflow_divergenceテンプレートの基礎になる。
    """
    rng = high - low
    clv = ((close - low) - (high - close)) / rng
    return clv.where(rng != 0)
