"""EMA10押し目 + EMA50/200トレンド + RSI + MACD近似テンプレート("Easy" 系手法の機械化)。

上位のEMA50/EMA200でトレンド方向を固定し、その方向に価格がEMA10へ押し戻ってから
EMA10を再び抜き返した初動を、RSIとMACDヒストグラムの符号で確認して順張りエントリーする。

LONG(買い):
    close > EMA50 かつ close > EMA200(上昇トレンドゲート)
    かつ close が下から EMA10 を上抜け(前バー close <= 前バー EMA10 → 当バー close > EMA10)
    かつ |close - EMA10| <= near_pips(EMA10直近での抜け返しに限定。pip=JPYクオート0.01/他0.0001)
    かつ RSI(rsi_period) > rsi_long
    かつ MACDヒストグラム > 0
SHORT(売り)はすべて対称(close<EMA50 & close<EMA200 & EMA10下抜け & 近接 & RSI<rsi_short & hist<0)。

独自指標の代替(誠実性メモ):
- 元記事の proprietary な "Golden MACD(上向き/下向き)" と "Flat Trend / H4方向" 指標は
  再現不可能なため、MACDヒストグラムの符号(hist>0 / hist<0) と EMA50/EMA200トレンドゲート
  の組み合わせで代替する。よって本テンプレートは元手法の *近似* であり、成績を保証しない。
- MACDは標準パラメータ(12,26,9)固定(パラメータ化していない。まず素直な近似を優先)。
- SL/TPは engine の固定pips決済(sl_pips/tp_pips)。元手法の可変決済は簡略化した。
- pip換算は symbol が 'JPY' で終わるかの簡易判定(0.01 / 0.0001)。app/core/symbols.py の
  正典(XAU/XAG等の特例)とは一致しないが、本テンプレの対象通貨ペアでは実用上十分。
- 最良はH1想定。

先読み規律: signal_fn は shift しない生シグナル(1/-1/0)を返す。クロス検出の .shift(1) は
自分の過去バーのみ参照で先読みではない(ma_pullback/macd_cross と同流儀)。翌バー始値執行の
shift は engine.run_backtest 側が担う。EMA/RSI/MACD のウォームアップNaNは比較で False に落ち
無シグナルとなる(最終マスクは .fillna(False) してから代入)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import ema, macd, rsi
from app.core.strategy_model import Strategy


def _signal_ema10_easy(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ema_fast = max(2, int(p["ema_fast"].value))   # 既定10
    ema_mid = max(2, int(p["ema_mid"].value))     # 既定50
    ema_slow = max(2, int(p["ema_slow"].value))   # 既定200
    near_pips = float(p["near_pips"].value)
    rsi_period = max(2, int(p["rsi_period"].value))
    rsi_long = float(p["rsi_long"].value)
    rsi_short = float(p["rsi_short"].value)
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    pip = 0.01 if strategy.symbol.upper().endswith("JPY") else 0.0001
    near_dist = near_pips * pip

    e_fast = ema(df["close"], ema_fast)
    e_mid = ema(df["close"], ema_mid)
    e_slow = ema(df["close"], ema_slow)
    r = rsi(df["close"], rsi_period)
    _macd_line, _signal_line, hist = macd(df["close"])  # 標準(12,26,9)。histの符号を代替シグナルに使う

    close = df["close"]
    prev_close = close.shift(1)
    prev_fast = e_fast.shift(1)
    cross_up = (prev_close <= prev_fast) & (close > e_fast)   # 下からEMA10を上抜け
    cross_dn = (prev_close >= prev_fast) & (close < e_fast)   # 上からEMA10を下抜け
    near = (close - e_fast).abs() <= near_dist               # EMA10直近での抜け返しに限定

    long_sig = (
        (close > e_mid) & (close > e_slow) & cross_up & near & (r > rsi_long) & (hist > 0.0)
    ).fillna(False)
    short_sig = (
        (close < e_mid) & (close < e_slow) & cross_dn & near & (r < rsi_short) & (hist < 0.0)
    ).fillna(False)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "ema10_easy",
    defaults={
        "ema_fast": 10.0, "ema_mid": 50.0, "ema_slow": 200.0, "near_pips": 5.0,
        "rsi_period": 14.0, "rsi_long": 60.0, "rsi_short": 40.0, "dir_mode": 0.0,
        "sl_pips": 20.0, "tp_pips": 40.0, "lot": 0.1,
    },
    signal_fn=_signal_ema10_easy,
)
