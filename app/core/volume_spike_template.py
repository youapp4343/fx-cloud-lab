"""出来高急増を伴うレンジブレイクへの追随テンプレート(volume_spike)。

app/core/vol_breakout_template.pyと同型の構造で、ボラティリティ急拡大の代理指標を
ATR比(短期/長期ATR)から出来高比(volume/volume_ma)に置き換えたもの。出来高が移動平均の
volume_ratio倍以上に急増した局面を材料発生の代理とみなし、直近レンジ(ドンチャンチャネル)の
ブレイク方向に追随する。

先読み回避: smaはバーiまでの過去windowのみ、donchian_channelはshift(1)済みで当該バーを
含まない直近レンジを使う。シグナル自体はシフトせず、engine.run_backtestのshift(1)が
次バー始値執行を保証する。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import donchian_channel, sma
from app.core.strategy_model import Strategy


def _signal_volume_spike(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """出来高急増(volume/volume_ma>=volume_ratio)かつドンチャンレンジブレイクで追随する。"""
    volume_ma_period = int(strategy.params["volume_ma_period"].value)
    volume_ratio = float(strategy.params["volume_ratio"].value)
    range_bars = int(strategy.params["range_bars"].value)

    volume_ma = sma(df["volume"], volume_ma_period)
    upper, lower = donchian_channel(df["high"], df["low"], range_bars)

    valid_ma = volume_ma.notna() & (volume_ma != 0)
    ratio = df["volume"] / volume_ma.where(valid_ma)  # 0/NaNはNaN化しゼロ除算を回避
    volume_hit = valid_ma & (ratio >= volume_ratio)

    breakout_up = volume_hit & (df["close"] > upper)
    breakout_down = volume_hit & (df["close"] < lower)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[breakout_up.fillna(False)] = 1
    signal[breakout_down.fillna(False)] = -1
    return signal


templates.register(
    "volume_spike",
    defaults={
        "volume_ma_period": 20.0,
        "volume_ratio": 2.0,
        "range_bars": 10.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_volume_spike,
)
