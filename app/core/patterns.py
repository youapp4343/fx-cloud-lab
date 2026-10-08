"""ローソク足パターン検出器。pandas DataFrame(OHLC)を受け取りbool Seriesを返す純粋関数群。

各関数は「そのバーの終値時点で確定した情報のみ」を使う。前バー参照は df[...].shift(1) で行い、
未来のバーは一切参照しない(先読みバイアス回避)。戻り値の先頭・末尾のNaNはFalseとして埋める。
"""

from typing import Callable

import pandas as pd


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """陰線の実体を当バーの陽線実体が完全に包む強気の包み足。"""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_bearish = prev_close < prev_open
    cur_bullish = df["close"] > df["open"]
    engulf = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    result = prev_bearish & cur_bullish & engulf
    return result.fillna(False).astype(bool)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """陽線の実体を当バーの陰線実体が完全に包む弱気の包み足。"""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_bullish = prev_close > prev_open
    cur_bearish = df["close"] < df["open"]
    engulf = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    result = prev_bullish & cur_bearish & engulf
    return result.fillna(False).astype(bool)


def pin_bar_bullish(
    df: pd.DataFrame, body_ratio_max: float = 0.3, lower_wick_ratio_min: float = 2.0
) -> pd.Series:
    """実体が小さく下ヒゲが長い強気ピンバー(ハンマー系)。"""
    full_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    safe_range = full_range.where(full_range != 0)
    body_ratio = body / safe_range
    result = (body_ratio <= body_ratio_max) & (lower_wick >= lower_wick_ratio_min * body)
    return result.fillna(False).astype(bool)


def pin_bar_bearish(
    df: pd.DataFrame, body_ratio_max: float = 0.3, upper_wick_ratio_min: float = 2.0
) -> pd.Series:
    """実体が小さく上ヒゲが長い弱気ピンバー(シューティングスター系)。"""
    full_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    safe_range = full_range.where(full_range != 0)
    body_ratio = body / safe_range
    result = (body_ratio <= body_ratio_max) & (upper_wick >= upper_wick_ratio_min * body)
    return result.fillna(False).astype(bool)


def inside_bar(df: pd.DataFrame) -> pd.Series:
    """当バーが前バーのレンジ内に完全に収まる。"""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    result = (df["high"] <= prev_high) & (df["low"] >= prev_low)
    return result.fillna(False).astype(bool)


def outside_bar(df: pd.DataFrame) -> pd.Series:
    """当バーが前バーのレンジを完全に包み込む。"""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    result = (df["high"] >= prev_high) & (df["low"] <= prev_low)
    return result.fillna(False).astype(bool)


def doji(df: pd.DataFrame, body_ratio_max: float = 0.1) -> pd.Series:
    """実体が全レンジに対して極小のドージ。全レンジ0はFalse扱い(ゼロ除算回避)。"""
    full_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    safe_range = full_range.where(full_range != 0)
    body_ratio = body / safe_range
    result = body_ratio <= body_ratio_max
    return result.fillna(False).astype(bool)


def marubozu_bullish(df: pd.DataFrame, wick_ratio_max: float = 0.05) -> pd.Series:
    """上下ヒゲがほぼ無い陽線の丸坊主。"""
    full_range = df["high"] - df["low"]
    is_bullish = df["close"] > df["open"]
    upper_wick = df["high"] - df["close"]
    lower_wick = df["open"] - df["low"]
    result = (
        is_bullish
        & (upper_wick <= wick_ratio_max * full_range)
        & (lower_wick <= wick_ratio_max * full_range)
    )
    return result.fillna(False).astype(bool)


def marubozu_bearish(df: pd.DataFrame, wick_ratio_max: float = 0.05) -> pd.Series:
    """上下ヒゲがほぼ無い陰線の丸坊主。"""
    full_range = df["high"] - df["low"]
    is_bearish = df["close"] < df["open"]
    upper_wick = df["high"] - df["open"]
    lower_wick = df["close"] - df["low"]
    result = (
        is_bearish
        & (upper_wick <= wick_ratio_max * full_range)
        & (lower_wick <= wick_ratio_max * full_range)
    )
    return result.fillna(False).astype(bool)


def consecutive_up(df: pd.DataFrame, n: int = 3) -> pd.Series:
    """当バー含む直近n本が連続陽線。"""
    up = (df["close"] > df["open"]).astype(int)
    result = up.rolling(window=n).sum() == n
    return result.fillna(False).astype(bool)


def consecutive_down(df: pd.DataFrame, n: int = 3) -> pd.Series:
    """当バー含む直近n本が連続陰線。"""
    down = (df["close"] < df["open"]).astype(int)
    result = down.rolling(window=n).sum() == n
    return result.fillna(False).astype(bool)


def prior_range_breakout_up(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """当バーを含まない直近lookback本の高値を当バーcloseが上回るブレイクアウト。"""
    prior_high = df["high"].shift(1).rolling(window=lookback).max()
    result = df["close"] > prior_high
    return result.fillna(False).astype(bool)


def prior_range_breakout_down(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """当バーを含まない直近lookback本の安値を当バーcloseが下回るブレイクダウン。"""
    prior_low = df["low"].shift(1).rolling(window=lookback).min()
    result = df["close"] < prior_low
    return result.fillna(False).astype(bool)


PATTERN_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "pin_bar_bullish": pin_bar_bullish,
    "pin_bar_bearish": pin_bar_bearish,
    "inside_bar": inside_bar,
    "outside_bar": outside_bar,
    "doji": doji,
    "marubozu_bullish": marubozu_bullish,
    "marubozu_bearish": marubozu_bearish,
    "consecutive_up": consecutive_up,
    "consecutive_down": consecutive_down,
    "prior_range_breakout_up": prior_range_breakout_up,
    "prior_range_breakout_down": prior_range_breakout_down,
}
