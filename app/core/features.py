"""特徴量行列ビルダー(Phase8: MLベースの戦略発掘、Layer B)。

全ての特徴量はそのバーの終値時点までの情報のみを使う。当該バー自身のOHLC(そのバーが確定
した時点で既に分かっている値)と、過去方向にのみ依存する変換(pct_change/rolling/ewmはいずれも
現在バーとそれ以前のみを参照する)だけを使い、将来バーは一切参照しない(先読み厳禁)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.indicators import atr, bollinger_bands, rsi, sma

RETURN_LAG_BARS: list[int] = [1, 2, 3, 5, 10, 20]

_BASE_FEATURE_NAMES: list[str] = [f"return_lag_{k}" for k in RETURN_LAG_BARS] + [
    "rsi_14",
    "atr_14_ratio",
    "bb_position",
    "sma20_dev",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
]

# build_features()が返す全列名を事前に固定した順序リスト。app.core.feature_rule_template の
# feature_index はこのリストへの整数インデックスとして機能する(app.core.candle_template.PATTERN_NAMES
# と同じ設計思想: 実行時にDataFrameの列名を動的参照するのではなく固定順序にすることで、
# feature_indexをrange/step指定によるグリッドサーチ対象にできる)。
FEATURE_NAMES: list[str] = sorted(
    _BASE_FEATURE_NAMES + [f"hour_{h}" for h in range(24)] + [f"dow_{d}" for d in range(7)]
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame(列: timestamp,open,high,low,close,volume)から特徴量行列を構築する。

    含める特徴量(全て当該バー時点で計算可能、先読みなし):
    - リターンラグ: 1,2,3,5,10,20本前までの終値変化率(pct_change)
    - RSI(14)
    - ATR(14)の終値に対する比率(ボラティリティの正規化)
    - ボリンジャーバンド内位置: (close - mid) / (upper - mid)(upper==midはNaN)
    - SMA(20)からの乖離率: (close - sma20) / sma20
    - ローソク実体比率: (close-open)/(high-low)(high==lowはNaN)
    - 上ヒゲ比率・下ヒゲ比率: 同様にhigh-low基準で正規化
    - 時間帯(hour) one-hot: 24列(prefix="hour")
    - 曜日(dayofweek) one-hot: 7列(prefix="dow")

    戻り値のDataFrameはdfと同じindexを持ち、FEATURE_NAMESの全列を持つ(NaNはウォームアップ
    期間で自然に発生するのでそのまま残す。呼び出し側でdropnaする想定)。
    """
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    data: dict[str, pd.Series] = {}
    for k in RETURN_LAG_BARS:
        data[f"return_lag_{k}"] = close.pct_change(k)

    data["rsi_14"] = rsi(close, 14)
    data["atr_14_ratio"] = atr(high, low, close, 14) / close

    mid, upper, _lower = bollinger_bands(close, 20, 2.0)
    half_width = (upper - mid).where((upper - mid) != 0)  # upper==mid(std=0)はゼロ除算を避けNaNにする
    data["bb_position"] = (close - mid) / half_width

    sma20 = sma(close, 20)
    data["sma20_dev"] = (close - sma20) / sma20

    full_range = (high - low).where((high - low) != 0)  # high==lowはゼロ除算を避けNaNにする
    data["body_ratio"] = (close - open_) / full_range
    data["upper_wick_ratio"] = (high - df[["open", "close"]].max(axis=1)) / full_range
    data["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - low) / full_range

    result = pd.DataFrame(data, index=df.index)

    # categories を0-23/0-6に固定してget_dummiesすることで、そのバッチに出現しない時間帯・
    # 曜日があっても常に24列/7列を保証する(部分期間で呼ばれても列構成が変わらないようにするため)。
    hour = pd.Series(pd.Categorical(df["timestamp"].dt.hour, categories=list(range(24))), index=df.index)
    dow = pd.Series(pd.Categorical(df["timestamp"].dt.dayofweek, categories=list(range(7))), index=df.index)
    hour_dummies = pd.get_dummies(hour, prefix="hour").astype(float)
    dow_dummies = pd.get_dummies(dow, prefix="dow").astype(float)

    result = pd.concat([result, hour_dummies, dow_dummies], axis=1)
    return result[FEATURE_NAMES]


def build_target(df: pd.DataFrame, forward_bars: int = 5, mode: str = "return") -> pd.Series:
    """目的変数を構築する。

    mode="return": forward_bars本先の終値までのリターン(close.shift(-forward_bars)/close - 1)。
    mode="direction": そのリターンの符号(1/-1/0)。

    これは意図的に未来を参照する(discovery.pyのforward returnと同じ設計思想: 「このバー時点の
    特徴量が、その後forward_bars本でどうなったか」を学習するための目的変数であり、シミュレーション
    上のリアルタイム売買判断に使うわけではないため先読みバイアスではない)。
    """
    forward_return = df["close"].shift(-forward_bars) / df["close"] - 1
    if mode == "return":
        result = forward_return
    elif mode == "direction":
        result = np.sign(forward_return)
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'return' or 'direction')")
    return result.rename("target")
