"""レンジ回帰スキャル(range_scalp)テンプレート。

GoGoJungle「コペルニクス・ベーシックUSDJPY版」(USDJPY M5、勝率72%/PF1.13、
単一ポジ、TP60/SL140+トレイル、最大14h保有)のロジッククラスを再現。
コンセプト「相場が大きく動いていない時にエントリー」= 低ボラ・レジームでの平均回帰逆張り。

手法:
- 低ボラ判定: ATRが直近ATR中央値×vol_factor 以下(=静かな相場)のバーのみ対象。
- 乖離: 終値がSMA(ma_period)から ATR×dev_mult 以上乖離したら回帰方向へ逆張り
  (上に乖離→売り、下に乖離→買い)。
- 退出は engine 側の tp_pips / sl_pips / trailing_pips / max_hold_bars で表現
  (広いSL・小さめTP・トレイリング・時間退出=Copernicusと同型)。

先読み回避: SMA・ATR・ATR中央値・乖離は全てバーiまでの確定情報のみ。シグナルはシフトせず、
engine.run_backtestのshift(1)が次バー始値執行を保証する(既存テンプレと同規律)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, sma
from app.core.strategy_model import Strategy


def _signal_range_scalp(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    ma_period = max(2, int(strategy.params["ma_period"].value))
    dev_mult = abs(float(strategy.params["dev_mult"].value))
    atr_period = max(2, int(strategy.params["atr_period"].value))
    vol_ma = max(5, int(strategy.params["vol_ma"].value))
    vol_factor = abs(float(strategy.params["vol_factor"].value))

    close = df["close"]
    ma = sma(close, ma_period)
    a = atr(df["high"], df["low"], close, atr_period)
    dev = close - ma

    # 低ボラ判定: 現在ATRが直近ATR中央値×factor 以下(=静かな相場)。
    atr_med = a.rolling(vol_ma).median()
    low_vol = a <= (atr_med * vol_factor)

    # direction: +1=平均回帰(乖離を逆張り, 既定) / -1=モメンタム(乖離順張り)
    direction = 1
    if "direction" in strategy.params:
        direction = 1 if float(strategy.params["direction"].value) >= 0 else -1

    threshold = dev_mult * a
    long_sig = (dev <= -threshold) & low_vol   # 下方乖離
    short_sig = (dev >= threshold) & low_vol   # 上方乖離

    # session時間フィルタ(UTC時刻)。hour_start==hour_end なら全時間許可。
    # 経済的根拠: USDJPYのMRは静かな時間帯(東京/NY午後)で機能、L/NYオープンの荒れを回避。
    hs = int(strategy.params["hour_start"].value) if "hour_start" in strategy.params else 0
    he = int(strategy.params["hour_end"].value) if "hour_end" in strategy.params else 0
    if hs != he:
        hour = df["timestamp"].dt.hour if "timestamp" in df.columns else pd.Series(df.index).dt.hour
        hour.index = df.index
        if hs < he:
            in_sess = (hour >= hs) & (hour < he)
        else:  # 日跨ぎ(例 21→6)
            in_sess = (hour >= hs) | (hour < he)
        long_sig = long_sig & in_sess
        short_sig = short_sig & in_sess

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[short_sig.fillna(False)] = -1 * direction
    signal[long_sig.fillna(False)] = 1 * direction
    return signal


templates.register(
    "range_scalp",
    defaults={
        "ma_period": 20.0,
        "dev_mult": 1.2,
        "atr_period": 14.0,
        "vol_ma": 100.0,
        "vol_factor": 1.0,
        "sl_pips": 140.0,
        "tp_pips": 60.0,
        "trailing_pips": 40.0,
        "max_hold_bars": 168.0,  # M5×168 = 14時間(Copernicusの最大保有)
        "hour_start": 0.0,  # UTC。hour_start==hour_endで全時間
        "hour_end": 0.0,
        "lot": 0.1,
    },
    signal_fn=_signal_range_scalp,
)
