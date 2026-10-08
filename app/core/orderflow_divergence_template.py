"""出来高圧力ダイバージェンス逆張りテンプレート(Phase10)。

close_location_value(CLV、終値がバーレンジ内のどこにあるか[-1,1])とvolumeを掛け合わせた
「出来高加重の圧力」をpressure_period本のローリング累積で求める。価格がlookback本の中で
高値/安値を更新しているにもかかわらず、この圧力指標がそれを裏付けていない(=同じ窓の中で
圧力側は追随した新高値/新安値を作れていない)場合をダイバージェンスとみなし、逆張りシグナル
を出す。既存テンプレートに無い「出来高加重の圧力」という着眼点の新設計。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.novel_indicators import close_location_value
from app.core.strategy_model import Strategy


def _signal_orderflow_divergence(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    pressure_period = max(1, int(strategy.params["pressure_period"].value))
    lookback = max(1, int(strategy.params["lookback"].value))

    # volume列が無い、または実質ゼロ(全0/全NaN)のデータでもクラッシュ・無言の0シグナルに
    # ならないよう、その場合は全バー等ウェイト(1.0)として扱う(出来高情報が無ければCLVそのものの
    # 累積で代用するフォールバック)。app.core.datafeedはvolume欠損時に列を0.0埋めで作るため、
    # "列が無い"だけの判定では実運用パイプラインでフォールバックが到達不能だった
    # (FABLE監査で発見: 全0 volumeだとpressure≡0で新高値/新安値判定が常にFalseになり、
    # トレード0件を無言で返していた)。
    has_volume = "volume" in df.columns and bool((df["volume"].fillna(0.0) != 0.0).any())
    volume = df["volume"] if has_volume else pd.Series(1.0, index=df.index)

    clv = close_location_value(df["high"], df["low"], df["close"]).fillna(0.0)
    pressure = (clv * volume).rolling(pressure_period).sum()

    price_high = df["high"].rolling(lookback).max()
    price_low = df["low"].rolling(lookback).min()
    pressure_high = pressure.rolling(lookback).max()
    pressure_low = pressure.rolling(lookback).min()

    price_new_high = df["high"] >= price_high
    price_new_low = df["low"] <= price_low
    pressure_not_confirming_high = pressure < pressure_high
    pressure_not_confirming_low = pressure > pressure_low

    bearish_div = price_new_high & pressure_not_confirming_high
    bullish_div = price_new_low & pressure_not_confirming_low

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[bearish_div.fillna(False)] = -1
    signal[bullish_div.fillna(False)] = 1
    return signal


templates.register(
    "orderflow_divergence",
    defaults={
        "pressure_period": 20.0,
        "lookback": 20.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_orderflow_divergence,
)
