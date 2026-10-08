"""検証候補101-200のうち B判定(一部のみ機械化可能)の手法テンプレート。

★重要な前提: 151-200 は MQL5 の robots.txt が /*/code/download/*/ と
  /*/code/viewcode/* を Disallow しているためソースを取得していない。
  したがってこれらは「配布EAの再現」ではなく、
  **ページの説明文に書かれた形の手法**を実装したものである。
  閾値・期間の既定値は原文に無いのでパラメータとして外に出し、グリッドで振る。
  結果は必ずこの前提つきで解釈すること。

先読み規律: 指標は確定バーまでのみ。執行の shift(1) は engine.run_backtest が行う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.catalog_templates import _emit
from app.core.indicators import atr, ema
from app.core.strategy_model import Strategy


def _day_ohlc(date: np.ndarray, o: np.ndarray, c: np.ndarray) -> tuple[dict, dict, list]:
    """日ごとの始値・終値。日足を別途読まずに実行足から作る。"""
    day_open: dict = {}
    day_close: dict = {}
    for i in range(len(date)):
        d = date[i]
        if d not in day_open:
            day_open[d] = o[i]
        day_close[d] = c[i]
    return day_open, day_close, sorted(day_open)


# --------------------------------------------------------------------------
# #187 Turnaround Tuesday: 月曜が陽線なら火曜に売り、陰線なら買い。
# --------------------------------------------------------------------------
def _sig_turnaround(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    fire_h = int(p["fire_utc"].value)
    ref_dow = int(p["ref_dow"].value)
    trade_dow = int(p["trade_dow"].value)
    reverse = bool(int(p["reverse"].value))

    ts = pd.to_datetime(df["timestamp"])
    dow = ts.dt.dayofweek.to_numpy()
    hour = ts.dt.hour.to_numpy()
    date = ts.dt.date.to_numpy()
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    day_open, day_close, days = _day_ohlc(date, o, c)
    idx = {d: k for k, d in enumerate(days)}
    dow_of = {d: pd.Timestamp(d).dayofweek for d in days}

    longs = np.zeros(n, dtype=bool)
    shorts = np.zeros(n, dtype=bool)
    fired = set()
    for i in range(n):
        if dow[i] != trade_dow or hour[i] != fire_h:
            continue
        d = date[i]
        if d in fired:
            continue
        k = idx.get(d)
        if k is None:
            continue
        prev = None
        for j in range(k - 1, max(-1, k - 6), -1):      # 祝日で飛ぶので遡って探す
            if dow_of[days[j]] == ref_dow:
                prev = days[j]
                break
        if prev is None:
            continue
        fired.add(d)
        bull = day_close[prev] > day_open[prev]
        want_long = (not bull) if reverse else bull
        if want_long:
            longs[i] = True
        else:
            shorts[i] = True
    return _emit(n, longs, shorts, c, None, 1.0)


templates.register("cat2_turnaround_187", defaults={
    "fire_utc": 8.0, "ref_dow": 0.0, "trade_dow": 1.0, "reverse": 1.0,
    "sl_pips": 40.0, "tp_pips": 40.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_turnaround)


# --------------------------------------------------------------------------
# #180 Weekly Day Reversal: 曜日別の反転/継続。#187 を一般化して曜日を振る。
# --------------------------------------------------------------------------
templates.register("cat2_weekday_rev_180", defaults={
    "fire_utc": 8.0, "ref_dow": 0.0, "trade_dow": 2.0, "reverse": 1.0,
    "sl_pips": 40.0, "tp_pips": 40.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_turnaround)


# --------------------------------------------------------------------------
# #197 Dominance: 前日が陽線か陰線かで当日の方向を決め、MA確認をつけて1日1回。
# --------------------------------------------------------------------------
def _sig_dominance(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    fire_h = int(p["fire_utc"].value)
    ma_len = max(2, int(p["ma_period"].value))
    use_ma = bool(int(p["use_ma"].value))

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour.to_numpy()
    date = ts.dt.date.to_numpy()
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    ma = pd.Series(c).rolling(ma_len).mean().to_numpy()
    n = len(df)
    day_open, day_close, days = _day_ohlc(date, o, c)
    idx = {d: k for k, d in enumerate(days)}

    longs = np.zeros(n, dtype=bool)
    shorts = np.zeros(n, dtype=bool)
    fired = set()
    for i in range(n):
        if hour[i] != fire_h:
            continue
        d = date[i]
        if d in fired:
            continue
        k = idx.get(d)
        if not k:
            continue
        fired.add(d)
        prev = days[k - 1]
        if use_ma and not np.isfinite(ma[i]):
            continue
        bull = day_close[prev] > day_open[prev]
        if bull and (not use_ma or c[i] > ma[i]):
            longs[i] = True
        elif (not bull) and (not use_ma or c[i] < ma[i]):
            shorts[i] = True
    return _emit(n, longs, shorts, c, None, 1.0)


templates.register("cat2_dominance_197", defaults={
    "fire_utc": 8.0, "ma_period": 50.0, "use_ma": 1.0,
    "sl_pips": 40.0, "tp_pips": 60.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_dominance)


# --------------------------------------------------------------------------
# #184 Inside Bar: 直前の足が親バーの内側に収まった後、親バーの高安を抜けた方向へ。
#      ★原典は逆指値(pending)だが engine は成行のみ。
#        「親バーの高安を終値で抜けたバー」で入る形に落としてある。
#        逆指値なら抜けた瞬間に約定するので、こちらの方が入口が遅く不利側。
# --------------------------------------------------------------------------
def _sig_inside_bar(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    atr_min = abs(float(p["atr_min_ratio"].value))
    h, l, c = df["high"], df["low"], df["close"]
    a = atr(h, l, c, 14)

    mother_h, mother_l = h.shift(2), l.shift(2)
    inside = (h.shift(1) <= mother_h) & (l.shift(1) >= mother_l)
    big = (mother_h - mother_l) >= atr_min * a
    longs = inside & big & (c > mother_h)
    shorts = inside & big & (c < mother_l)
    return _emit(len(df), longs.fillna(False).to_numpy(), shorts.fillna(False).to_numpy(),
                 c.to_numpy(), None, 1.0)


templates.register("cat2_inside_bar_184", defaults={
    "atr_min_ratio": 0.5,
    "sl_pips": 25.0, "tp_pips": 40.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_inside_bar)


# --------------------------------------------------------------------------
# #196 FVG: 3本足で窓が開いた方向へ。EMAトレンドで方向を絞る。
# --------------------------------------------------------------------------
def _sig_fvg(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ema_len = max(2, int(p["ema_period"].value))
    gap_atr = abs(float(p["gap_min_atr"].value))
    h, l, c = df["high"], df["low"], df["close"]
    a = atr(h, l, c, 14)
    e = ema(c, ema_len)

    gap_up = l - h.shift(2)          # 2本前の高値より現在の安値が上 = 上向きの窓
    gap_dn = l.shift(2) - h
    longs = (gap_up >= gap_atr * a) & (c > e)
    shorts = (gap_dn >= gap_atr * a) & (c < e)
    return _emit(len(df), longs.fillna(False).to_numpy(), shorts.fillna(False).to_numpy(),
                 c.to_numpy(), None, 1.0)


templates.register("cat2_fvg_196", defaults={
    "ema_period": 50.0, "gap_min_atr": 0.1,
    "sl_pips": 25.0, "tp_pips": 40.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_fvg)


# --------------------------------------------------------------------------
# #161 GoldLondonBreakout: アジア時間(UTC 0〜box_end)のレンジを確定し、
#      ロンドン開始でレンジ外に居る側へ。SLはATR基準、TPはRR倍。
# --------------------------------------------------------------------------
def _sig_london_break(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    box_end = int(p["box_end_utc"].value)
    fire_h = int(p["fire_utc"].value)
    atr_mult = abs(float(p["atr_mult"].value))
    rr = abs(float(p["rr"].value))

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour.to_numpy()
    date = ts.dt.date.to_numpy()
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    a = atr(df["high"], df["low"], df["close"], 14).to_numpy()
    n = len(df)

    in_box = (hour >= 0) & (hour < box_end)
    box_hi: dict = {}
    box_lo: dict = {}
    for i in np.flatnonzero(in_box):
        d = date[i]
        box_hi[d] = max(box_hi.get(d, -np.inf), h[i])
        box_lo[d] = min(box_lo.get(d, np.inf), l[i])

    longs = np.zeros(n, dtype=bool)
    shorts = np.zeros(n, dtype=bool)
    fired = set()
    for i in range(n):
        if hour[i] != fire_h:
            continue
        d = date[i]
        if d in fired or d not in box_hi:
            continue
        fired.add(d)
        if c[i] > box_hi[d]:
            longs[i] = True
        elif c[i] < box_lo[d]:
            shorts[i] = True
    return _emit(n, longs, shorts, c, atr_mult * a, rr)


templates.register("cat2_london_break_161", defaults={
    "box_end_utc": 6.0, "fire_utc": 7.0, "atr_mult": 1.5, "rr": 2.0,
    "sl_pips": 30.0, "tp_pips": 60.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_london_break)


# --------------------------------------------------------------------------
# #181 Simple EMA Cross: 2本のEMAのクロスだけ。最も単純な対照群。
# --------------------------------------------------------------------------
def _sig_ema_cross(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    f = ema(df["close"], max(2, int(p["fast"].value)))
    s = ema(df["close"], max(3, int(p["slow"].value)))
    up = (f > s) & (f.shift(1) <= s.shift(1))
    dn = (f < s) & (f.shift(1) >= s.shift(1))
    return _emit(len(df), up.fillna(False).to_numpy(), dn.fillna(False).to_numpy(),
                 df["close"].to_numpy(), None, 1.0)


templates.register("cat2_ema_cross_181", defaults={
    "fast": 12.0, "slow": 26.0,
    "sl_pips": 30.0, "tp_pips": 45.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_ema_cross)
