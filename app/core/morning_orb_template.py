"""Morning Range Breakout(morning_orb)テンプレート。

MQL5市販EA「EURUSD Morning Range Breakout」の公開骨格を忠実再現(2026-07-16 事前登録):
- レンジ: 開始時刻(range_start_utc)から range_minutes 分のM1高値/安値。
- レンジ確定後、RangeHigh+buffer 上抜けでロング / RangeLow-buffer 下抜けでショート。
  1日1トレード(先にブレイクした側、OCO近似)。エントリー監視は cancel_minutes まで。
- SL = sl_factor × RangeWidth / TP = tp_factor × RangeWidth(レンジ幅倍率方式)を
  sl_price/tp_price 列で動的指定。
- 21:55強制決済 → engine の max_hold_bars で近似(呼び出し側で設定)。

近似(事前登録): ストップ待機注文→ブレイク検知バーの次バー始値成行(engineのshift(1))。
M1なので乖離は小さいが、実EAはRangeHigh+buffer固定価格で約定する点が異なる。

先読み回避: レンジはバーiまでの確定情報(当日レンジ窓の累積max/min)、ブレイク判定は
バーiの高値/安値/終値。engine.run_backtest の shift(1) が次バー執行を保証。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_morning_orb(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    start_h = int(strategy.params["range_start_utc"].value)
    start_m = int(strategy.params.get("range_start_min").value) if "range_start_min" in strategy.params else 0
    range_min = int(strategy.params["range_minutes"].value)
    monitor_min = int(strategy.params["monitor_minutes"].value)
    buf_pips = float(strategy.params["buffer_pips"].value)
    sl_f = float(strategy.params["sl_factor"].value)
    tp_f = float(strategy.params["tp_factor"].value)
    pip = float(strategy.params["pip_size"].value)

    ts = df["timestamp"]
    minutes_of_day = ts.dt.hour * 60 + ts.dt.minute
    day = ts.dt.normalize()
    start_abs = start_h * 60 + start_m
    # レンジ窓/監視窓(日跨ぎしない前提: start+monitorが1440以内になるよう呼び出し側が設定)
    in_range = (minutes_of_day >= start_abs) & (minutes_of_day < start_abs + range_min)
    in_monitor = (minutes_of_day >= start_abs + range_min) & (minutes_of_day < start_abs + monitor_min)

    hi = df["high"].where(in_range)
    lo = df["low"].where(in_range)
    # 当日ここまでの累積レンジ(確定バーのみ、レンジ窓外ではNaN継続でffill)
    range_hi = hi.groupby(day).cummax().groupby(day).ffill()
    range_lo = lo.groupby(day).cummin().groupby(day).ffill()
    width = range_hi - range_lo

    buf = buf_pips * pip
    up_break = in_monitor & (df["high"] > range_hi + buf) & width.notna() & (width > 0)
    dn_break = in_monitor & (df["low"] < range_lo - buf) & width.notna() & (width > 0)

    # 1日1トレード: 先に発生した側のみ、その日の以降のシグナルは無効
    brk = up_break | dn_break
    first_idx = brk.groupby(day).cumsum()  # 1以上=既にブレイク発生
    fire = brk & (first_idx == 1)
    fire_long = fire & up_break
    fire_short = fire & dn_break & ~up_break  # 同一バー両ブレイクは不利側=ショート扱いにしない(ロング優先を排し無効化)
    both_same_bar = fire & up_break & dn_break
    fire_long = fire_long & ~both_same_bar
    fire_short = (fire & dn_break) & ~up_break

    n = len(df)
    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan)
    tp_price = np.full(n, np.nan)

    li = fire_long.to_numpy()
    si = fire_short.to_numpy()
    rh = range_hi.to_numpy()
    rl = range_lo.to_numpy()
    w = width.to_numpy()

    signal[li] = 1
    signal[si] = -1
    # 参照価格=ブレイクレベル(実EAのストップ約定価格)
    ref_long = rh[li] + buf
    sl_price[li] = ref_long - sl_f * w[li]
    tp_price[li] = ref_long + tp_f * w[li]
    ref_short = rl[si] - buf
    sl_price[si] = ref_short + sl_f * w[si]
    tp_price[si] = ref_short - tp_f * w[si]

    out = pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price},
                       index=df.index)
    return out


templates.register(
    "morning_orb",
    defaults={
        "range_start_utc": 7.0,     # レンジ開始UTC時(09:00サーバーGMT+2解釈)
        "range_start_min": 0.0,
        "range_minutes": 90.0,      # 09:00-10:30
        "monitor_minutes": 715.0,   # 09:00起点で20:55まで(11h55m)
        "buffer_pips": 0.0,
        "sl_factor": 1.0,
        "tp_factor": 1.5,
        "pip_size": 0.0001,
        "sl_pips": 500.0,           # 動的sl_priceが優先されるための広い保険値
        "tp_pips": 500.0,
        "max_hold_bars": 684.0,     # 10:30エントリー→21:55 ≈ 11.4h(M1)
        "lot": 0.1,
    },
    signal_fn=_signal_morning_orb,
)
