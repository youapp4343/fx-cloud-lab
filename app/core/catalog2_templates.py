"""検証候補101-200のうち、条件が数値まで確定した手法(A判定)のテンプレート。

出典と原文条件は
  C:\\CodexProject\\FX\\research\\source_catalog_100\\claude_validation_200.md
に候補番号ごとに記録する。ここでは「原文に書いてあること」だけを実装し、
原文に無い定義はパラメータとして外に出す。推測で固定しない。

先読み規律(既存テンプレと共通):
  - 指標は確定バーまでの rolling/ewm のみ
  - signal_fn はシフトしない。翌バー始値執行の shift(1) は engine.run_backtest が行う
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.catalog_templates import _emit
from app.core.indicators import rsi
from app.core.strategy_model import Strategy


# --------------------------------------------------------------------------
# #114 RSI14 が 30/70 を「再び」抜けた瞬間に入る。
#      SL は直近安値(高値)の 10〜15pips 外、TP は RR 1.5〜2。
# --------------------------------------------------------------------------
def _sig_rsi_reentry(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(2, int(p["rsi_period"].value))
    lo = float(p["lower"].value)
    hi = float(p["upper"].value)
    look = max(2, int(p["swing_lookback"].value))
    buf = abs(float(p["sl_buffer_pips"].value)) * float(p["pip"].value)
    rr = abs(float(p["rr"].value))

    c, h, l = df["close"], df["high"], df["low"]
    r = rsi(c, n)
    # 「30以下から再び30を上抜ける」= 前バーが閾値以下、現バーが閾値超え
    long_sig = (r.shift(1) <= lo) & (r > lo)
    short_sig = (r.shift(1) >= hi) & (r < hi)

    swing_lo = l.rolling(look).min()
    swing_hi = h.rolling(look).max()
    ca = c.to_numpy()
    ls = long_sig.fillna(False).to_numpy()
    ss = short_sig.fillna(False).to_numpy()
    dist = np.where(ls, ca - (swing_lo.to_numpy() - buf),
                    np.where(ss, (swing_hi.to_numpy() + buf) - ca, np.nan))
    return _emit(len(df), ls, ss, ca, dist, rr)


templates.register("cat2_rsi_reentry_114", defaults={
    "rsi_period": 14.0, "lower": 30.0, "upper": 70.0,
    "swing_lookback": 10.0, "sl_buffer_pips": 12.0, "pip": 0.01, "rr": 1.75,
    "sl_pips": 20.0, "tp_pips": 35.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_rsi_reentry)


# --------------------------------------------------------------------------
# #116 / #118 RSI14 の閾値超えで逆張りし、固定幅で決済する。
#      #116 は USDJPY M15 で TP=SL=50銭。#118 は M5/M15 で TP10〜15 / SL10。
#      どちらも「閾値を超えている間」ではなく閾値到達で入る素直な逆張り。
# --------------------------------------------------------------------------
def _sig_rsi_fixed(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(2, int(p["rsi_period"].value))
    lo = float(p["lower"].value)
    hi = float(p["upper"].value)
    c = df["close"]
    r = rsi(c, n)
    # 閾値に入った最初のバーだけを拾う(閾値内に居続ける間は連打しない)
    in_lo = r <= lo
    in_hi = r >= hi
    long_sig = in_lo & ~in_lo.shift(1).fillna(False)
    short_sig = in_hi & ~in_hi.shift(1).fillna(False)
    return _emit(len(df), long_sig.fillna(False).to_numpy(),
                 short_sig.fillna(False).to_numpy(), c.to_numpy(), None, 1.0)


templates.register("cat2_rsi_fixed_116", defaults={
    "rsi_period": 14.0, "lower": 30.0, "upper": 70.0,
    "sl_pips": 50.0, "tp_pips": 50.0, "max_hold_bars": 192.0, "lot": 0.1,
}, signal_fn=_sig_rsi_fixed)

templates.register("cat2_rsi_fixed_118", defaults={
    "rsi_period": 14.0, "lower": 30.0, "upper": 70.0,
    "sl_pips": 10.0, "tp_pips": 12.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_rsi_fixed)


# --------------------------------------------------------------------------
# #144 東京時間の高安に OCO を置き、17時(日本時間)に仕掛ける。
#      利食い +100pips / 損切り -50pips、金曜は除外。
#      ★データは UTC。日本時間 = UTC+9 なので、東京 9:00-15:00 JST は
#        UTC 0:00-6:00。仕掛けの 17:00 JST は UTC 8:00。
#        夏時間の無い日本時間で定義されているので、UTC固定でずれない。
# --------------------------------------------------------------------------
def _sig_tokyo_oco(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    box_start = int(p["box_start_utc"].value)      # 東京の箱を作り始める時刻(UTC)
    box_end = int(p["box_end_utc"].value)          # 箱を締める時刻(UTC)
    fire = int(p["fire_utc"].value)                # 仕掛ける時刻(UTC)
    skip_friday = bool(int(p["skip_friday"].value))

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour.to_numpy()
    day = ts.dt.dayofweek.to_numpy()               # 月=0 … 金=4
    date = ts.dt.date.to_numpy()

    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)

    in_box = (hour >= box_start) & (hour < box_end)
    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    # 日ごとに箱を作り、fire 時刻の最初のバーで、箱の外に出ている方向へ入る。
    # OCO の厳密な再現(先に触れた側だけ約定)はバー内の順序が要るのでできない。
    # ここでは「fire時点で箱の上(下)に居る側へ入る」に落とし、その旨を結果に明記する。
    box_hi = {}
    box_lo = {}
    for i in range(n):
        d = date[i]
        if in_box[i]:
            box_hi[d] = max(box_hi.get(d, -np.inf), h[i])
            box_lo[d] = min(box_lo.get(d, np.inf), l[i])
    fired = set()
    for i in range(n):
        if hour[i] != fire:
            continue
        d = date[i]
        if d in fired or d not in box_hi:
            continue
        if skip_friday and day[i] == 4:
            continue
        fired.add(d)
        if c[i] > box_hi[d]:
            long_sig[i] = True
        elif c[i] < box_lo[d]:
            short_sig[i] = True

    return _emit(n, long_sig, short_sig, c, None, 1.0)


templates.register("cat2_tokyo_oco_144", defaults={
    "box_start_utc": 0.0, "box_end_utc": 6.0, "fire_utc": 8.0, "skip_friday": 1.0,
    "sl_pips": 50.0, "tp_pips": 100.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_tokyo_oco)
