"""ピボット + RSIダイバージェンス反転テンプレート(裁量トレーダー記事の機械化シリーズ)。

クラシック(フロアトレーダー)ピボットを *前日* のH/L/Cから算出し、日中足へブロード
キャストする。価格がレジスタンス(R1/R2/R3)を上抜けてから終値でその水準を下回り、かつ
弱気RSIダイバージェンスを伴う場面で売り(-1)、サポート(S1/S2/S3)で対称に買い(+1)を出す。

ピボット式(クラシック):
    P = (H + L + C) / 3
    R1 = 2P - L      S1 = 2P - H
    R2 = P + (R1-S1) S2 = P - (R1-S1)      # R1-S1 = 前日レンジ (H-L)
    R3 = R2 + (P-S2) S3 = S2 - (R2-P)

ダイバージェンス(スイング検出を使わない簡易版、orderflow_divergence と同流儀):
    弱気 = 価格が div_lookback 本の高値を更新(high >= rolling.max)しているのに、RSI は
           同じ窓の高値を更新していない(rsi < rsi.rolling.max)。
    強気 = 価格が div_lookback 本の安値を更新しているのに、RSI は安値を更新していない。
    RSI・rolling はすべて現在+過去バーのみで計算する因果指標(先読みなし)。

先読み規律(本テンプレートの肝):
- ピボットは *前営業日* のH/L/Cのみから作る。日次集計を groupby(暦日) で作り、shift(1) で
  1日ずらしてから各日中足へ map する。したがって当日のバーは「昨日までに確定した」日次
  H/L/C しか参照しない(当日進行中の未確定な日次値は一切入らない)。groupby は日付昇順に
  整列するため、shift(1) は常に時系列上の前営業日(週末を挟めば金曜)を指す。
- 反転判定に使う high/low/close は *当該バー自身* の確定値。翌バー始値執行の shift は
  engine.run_backtest 側(raw_signal.shift(1))が担うため、テンプレート側では執行用 shift を
  しない。
- ウォームアップNaN・初日(前日なし)のNaNは比較で False に落ち、無シグナルになる
  (最終マスクは .fillna(False) してから代入)。

近似・非実装(誠実性メモ):
- 日足境界はUTC暦日で近似する。実際のFXピボットはNYクローズ(17:00 ET)基準など複数流派が
  あり、境界の取り方で水準がずれる(近似であることに注意)。
- 記事のSL="直近スイング安値/高値"・TP="次のピボット水準"は、engine の固定pips決済に合わせ
  sl_pips/tp_pips へ簡略化した(構造的SL/TPは未実装)。
- ピボット周期は日足のみ実装。週足ピボット(週初に前週H/L/Cから算出)は変種として未実装。
- 最良は日中足(H1/H4)または日足。ダイバージェンスとピボット反転を同一バーで要求する合流
  条件のため、低頻度シグナルになりうる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import rsi
from app.core.strategy_model import Strategy


def _signal_pivot_rsi_div(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    rsi_period = max(2, int(p["rsi_period"].value))
    div_lookback = max(2, int(p["div_lookback"].value))
    use_level = min(3, max(1, int(p["use_level"].value)))  # 1=R1/S1のみ,2=+R2/S2,3=+R3/S3
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    signal = pd.Series(0, index=df.index, dtype=int)
    if len(df) == 0:
        return signal

    # --- 前日H/L/Cからクラシックピボットを算出し日中足へブロードキャスト(先読みなし) ---
    day = df["timestamp"].dt.normalize()  # UTC暦日(その日 00:00)。日足境界の近似
    agg = (
        pd.DataFrame({"high": df["high"], "low": df["low"], "close": df["close"]})
        .groupby(day)
        .agg(H=("high", "max"), L=("low", "min"), C=("close", "last"))
    )
    prev = agg.shift(1)  # 各暦日に「前営業日」の日次H/L/Cを対応(groupbyは日付昇順→shiftは前日)
    P = (prev["H"] + prev["L"] + prev["C"]) / 3.0
    rng = prev["H"] - prev["L"]  # = R1 - S1(前日レンジ)
    R1 = 2.0 * P - prev["L"]
    S1 = 2.0 * P - prev["H"]
    R2 = P + rng
    S2 = P - rng
    R3 = R2 + (P - S2)  # = P + 2*rng
    S3 = S2 - (R2 - P)  # = P - 2*rng

    # 暦日 -> 前日基準ピボット を map で各バーへ展開。use_level で有効水準数を絞る
    res_levels = [day.map(R1), day.map(R2), day.map(R3)][:use_level]
    sup_levels = [day.map(S1), day.map(S2), day.map(S3)][:use_level]

    # --- RSIダイバージェンス(因果: 現在+過去のみ) ---
    r = rsi(df["close"], rsi_period)
    price_new_high = df["high"] >= df["high"].rolling(div_lookback).max()
    price_new_low = df["low"] <= df["low"].rolling(div_lookback).min()
    rsi_not_new_high = r < r.rolling(div_lookback).max()
    rsi_not_new_low = r > r.rolling(div_lookback).min()
    bearish_div = price_new_high & rsi_not_new_high
    bullish_div = price_new_low & rsi_not_new_low

    # --- ピボット反転: 水準を突いて終値が押し戻された(有効水準のいずれか) ---
    high, low, close = df["high"], df["low"], df["close"]
    sell_reject = pd.Series(False, index=df.index)
    for rk in res_levels:  # 高値がRkを上抜け、終値はRk未満へ押し戻し
        sell_reject = sell_reject | ((high >= rk) & (close < rk))
    buy_reject = pd.Series(False, index=df.index)
    for sk in sup_levels:  # 安値がSkを下抜け、終値はSk超へ戻し
        buy_reject = buy_reject | ((low <= sk) & (close > sk))

    sell_sig = (sell_reject & bearish_div).fillna(False)
    buy_sig = (buy_reject & bullish_div).fillna(False)

    if dir_mode in (0, 1):
        signal[buy_sig] = 1
    if dir_mode in (0, 2):
        signal[sell_sig] = -1
    return signal


templates.register(
    "pivot_rsi_div",
    defaults={
        "rsi_period": 14.0, "div_lookback": 20.0, "use_level": 1.0, "dir_mode": 0.0,
        "sl_pips": 30.0, "tp_pips": 45.0, "lot": 0.1,
    },
    signal_fn=_signal_pivot_rsi_div,
)
