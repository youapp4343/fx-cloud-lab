"""Gold Breakout Fusion EA v1 (gold_break_fusion) テンプレート。

XAUUSD ブレイクアウト3モジュールを module_id で切替え、各々を独立検証する(仕様Phase1)。
全レンジ・バッファ・SLは確定D1のATR20[1]で正規化(固定pips不使用)。
  Module 1 Rolling: M15直近32本の高安ブレイク(時間非依存)
  Module 2 AsiaLondon: アジア(00-07UTC)レンジを07-11UTCにブレイク(M5確定足)
  Module 3 Compression: ER<=0.25・幅・上下接触>=2・重複>=0.5で本物のボックス認定→ブレイク

共通: EffectiveBuffer=max(0.03*ATRd, 2*spread)、BodyRatio>=閾値、終値がブレイク方向端30/25%以内、
構造SL=2候補の近い方(=高い方が買いSL)、TP=1.5R、6h時間退出(engine max_hold_bars)、
同方向1日1回、スプレッドゲート(spread<=0.10R)、最小SL>=10*spread、最大SL<=0.5*ATRd。

前提: df に atr_d1 列(確定前日D1 ATR20、先読み無し)が必要。pip=0.1。side: 0=両/1=買/-1=売。
執行はengine shift(1)で次足。近似(Phase1): CurrentSpread=定数(assumed_spread)、
MedianSpread60ゲート・リスクマネージャ(日次/週次)・SL後再エントリ禁止/クールダウンは
Phase3/実ティックで実装。ここは各モジュールの純OOS期待値の判定に集中。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy

PIP = 0.1  # XAUUSD


def _body_ratio(o, h, l, c):
    rng = h - l
    return np.where(rng > 0, np.abs(c - o) / rng, 0.0)


def _finalize(n, sig, sl, tp, side_filter):
    if side_filter == 1:
        mask = sig != 1
        sig = sig.copy(); sig[mask] = 0
    elif side_filter == -1:
        mask = sig != -1
        sig = sig.copy(); sig[mask] = 0
    sl = np.where(sig != 0, sl, np.nan)
    tp = np.where(sig != 0, tp, np.nan)
    return sig, sl, tp


def _sl_tp(close, sig, sl_struct, tp_r):
    """構造SL価格(sl_struct)からR=|close-sl|、TP=close+sig*tp_r*R。"""
    R = np.abs(close - sl_struct)
    tp = close + sig * tp_r * R
    return sl_struct, tp, R


def _gate_ok(R, spread_price, atr, sig):
    """スプレッドゲート(spread<=0.10R) + 最小SL(>=10*spread) + 最大SL(<=0.5ATRd)。"""
    if sig == 0 or not np.isfinite(R) or R <= 0:
        return False
    if spread_price > 0.10 * R:
        return False
    if R < 10 * spread_price:
        return False
    if R > 0.50 * atr:
        return False
    return True


def _mod_rolling(o, h, l, c, ts, atr, spread_price, p):
    lookback = int(p["lookback_bars"].value)
    minw, maxw = float(p["min_width_atr"].value), float(p["max_width_atr"].value)
    minbody = float(p["min_body_ratio"].value)
    buf_atr = float(p["break_buffer_atr"].value)
    stopcap = float(p["stop_cap_atr"].value)
    tp_r = float(p["tp_r"].value)
    s = pd.Series(h)
    rh = s.rolling(lookback).max().shift(1).to_numpy()
    rl = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    body = _body_ratio(o, h, l, c)
    n = len(c)
    sig = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    day = ts.dt.floor("D").to_numpy()
    done = {}  # (day, dir) -> True
    for i in range(n):
        if np.isnan(rh[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        width = rh[i] - rl[i]
        if not (minw * atr[i] <= width <= maxw * atr[i]):
            continue
        eff = max(buf_atr * atr[i], 2 * spread_price)
        mid = (rh[i] + rl[i]) / 2.0
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        # 買い
        if c[i] > rh[i] + eff and body[i] >= minbody and (c[i] - l[i]) / rng >= 0.70:
            if not done.get((day[i], 1)):
                sl_s = max(mid - 0.05 * atr[i], c[i] - stopcap * atr[i])
                _, tpx, R = _sl_tp(c[i], 1, sl_s, tp_r)
                if _gate_ok(R, spread_price, atr[i], 1):
                    sig[i] = 1; sl[i] = sl_s; tp[i] = tpx; done[(day[i], 1)] = True
                    continue
        # 売り
        if c[i] < rl[i] - eff and body[i] >= minbody and (h[i] - c[i]) / rng >= 0.70:
            if not done.get((day[i], -1)):
                sl_s = min(mid + 0.05 * atr[i], c[i] + stopcap * atr[i])
                _, tpx, R = _sl_tp(c[i], -1, sl_s, tp_r)
                if _gate_ok(R, spread_price, atr[i], -1):
                    sig[i] = -1; sl[i] = sl_s; tp[i] = tpx; done[(day[i], -1)] = True
    return sig, sl, tp


def _mod_asia_london(o, h, l, c, ts, atr, spread_price, p):
    minw, maxw = float(p["min_width_atr"].value), float(p["max_width_atr"].value)
    minbody = float(p["min_body_ratio"].value)
    buf_atr = float(p["break_buffer_atr"].value)
    stopcap = float(p["stop_cap_atr"].value)
    tp_r = float(p["tp_r"].value)
    hour = ts.dt.hour.to_numpy()
    day = ts.dt.floor("D").to_numpy()
    body = _body_ratio(o, h, l, c)
    n = len(c)
    sig = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    # 各日のアジアレンジ(00-07UTC)を逐次構築
    cur_day = None; a_hi = -np.inf; a_lo = np.inf; done_dir = set()
    for i in range(n):
        if cur_day is None or day[i] != cur_day:
            cur_day = day[i]; a_hi = -np.inf; a_lo = np.inf; done_dir = set()
        hr = hour[i]
        if hr < 7:  # アジア時間: レンジ更新のみ、エントリー無し
            a_hi = max(a_hi, h[i]); a_lo = min(a_lo, l[i]); continue
        if not (7 <= hr < 11):
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or not np.isfinite(a_hi):
            continue
        width = a_hi - a_lo
        if not (minw * atr[i] <= width <= maxw * atr[i]):
            continue
        eff = max(buf_atr * atr[i], 2 * spread_price)
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        if 1 not in done_dir and c[i] > a_hi + eff and body[i] >= minbody and (c[i] - l[i]) / rng >= 0.70:
            sl_s = max(a_lo - 0.02 * atr[i], c[i] - stopcap * atr[i])
            _, tpx, R = _sl_tp(c[i], 1, sl_s, tp_r)
            if _gate_ok(R, spread_price, atr[i], 1):
                sig[i] = 1; sl[i] = sl_s; tp[i] = tpx; done_dir.add(1); continue
        if -1 not in done_dir and c[i] < a_lo - eff and body[i] >= minbody and (h[i] - c[i]) / rng >= 0.70:
            sl_s = min(a_hi + 0.02 * atr[i], c[i] + stopcap * atr[i])
            _, tpx, R = _sl_tp(c[i], -1, sl_s, tp_r)
            if _gate_ok(R, spread_price, atr[i], -1):
                sig[i] = -1; sl[i] = sl_s; tp[i] = tpx; done_dir.add(-1)
    return sig, sl, tp


def _mod_compression(o, h, l, c, ts, atr, spread_price, p):
    lb = int(p["lookback_bars"].value)
    minw, maxw = float(p["min_width_atr"].value), float(p["max_width_atr"].value)
    maxer = float(p["max_er"].value)
    min_touch = int(p["min_touches"].value)
    tol = float(p["touch_tolerance"].value)
    min_overlap = float(p["min_overlap_ratio"].value)
    minbody = float(p["min_body_ratio"].value)
    min_tr_exp = float(p["min_tr_expansion"].value)
    buf_atr = float(p["break_buffer_atr"].value)
    stopcap = float(p["stop_cap_atr"].value)
    tp_r = float(p["tp_r"].value)
    tr = (pd.Series(h) - pd.Series(l)).to_numpy()  # 簡易TR(High-Low)
    tr_med20 = pd.Series(tr).rolling(20).median().shift(1).to_numpy()
    body = _body_ratio(o, h, l, c)
    day = ts.dt.floor("D").to_numpy()
    n = len(c)
    sig = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    done = {}
    for i in range(lb, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(tr_med20[i]):
            continue
        wl = i - lb
        bh = h[wl:i].max(); bl = l[wl:i].min()  # 直近lb本(現在足除く)
        bw = bh - bl
        if bw <= 0 or not (minw * atr[i] <= bw <= maxw * atr[i]):
            continue
        # ER(lb)
        seg = c[wl:i]
        denom = np.abs(np.diff(seg)).sum()
        if denom <= 0:
            continue
        er = abs(seg[-1] - seg[0]) / denom
        if er > maxer:
            continue
        # 上下接触>=2、接触は2本以上離す
        up_t = 0; last_up = -10; lo_t = 0; last_lo = -10
        for j in range(wl, i):
            if h[j] >= bh - tol * bw and j - last_up >= 2:
                up_t += 1; last_up = j
            if l[j] <= bl + tol * bw and j - last_lo >= 2:
                lo_t += 1; last_lo = j
        if up_t < min_touch or lo_t < min_touch:
            continue
        # 重複率(前半 vs 後半)
        half = lb // 2
        fh = h[wl:wl + half].max(); fl = l[wl:wl + half].min()
        sh = h[wl + half:i].max(); sl2 = l[wl + half:i].min()
        ov = max(0.0, min(fh, sh) - max(fl, sl2))
        denom2 = min(fh - fl, sh - sl2)
        if denom2 <= 0 or ov / denom2 < min_overlap:
            continue
        eff = max(buf_atr * atr[i], 2 * spread_price)
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        # ブレイク(TR拡大+実体+終値端25%)
        if c[i] > bh + eff and body[i] >= minbody and tr[i] >= tr_med20[i] * min_tr_exp and (c[i] - l[i]) / rng >= 0.75:
            if not done.get((day[i], 1)):
                sl_s = max(bl - 0.02 * atr[i], c[i] - stopcap * atr[i])
                _, tpx, R = _sl_tp(c[i], 1, sl_s, tp_r)
                if _gate_ok(R, spread_price, atr[i], 1):
                    sig[i] = 1; sl[i] = sl_s; tp[i] = tpx; done[(day[i], 1)] = True; continue
        if c[i] < bl - eff and body[i] >= minbody and tr[i] >= tr_med20[i] * min_tr_exp and (h[i] - c[i]) / rng >= 0.75:
            if not done.get((day[i], -1)):
                sl_s = min(bh + 0.02 * atr[i], c[i] + stopcap * atr[i])
                _, tpx, R = _sl_tp(c[i], -1, sl_s, tp_r)
                if _gate_ok(R, spread_price, atr[i], -1):
                    sig[i] = -1; sl[i] = sl_s; tp[i] = tpx; done[(day[i], -1)] = True
    return sig, sl, tp


def _signal_gold_break_fusion(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    module = int(p["module_id"].value)
    side_filter = int(p["side"].value) if "side" in p else 0
    spread_pips = float(p["assumed_spread"].value) if "assumed_spread" in p else 3.0
    spread_price = spread_pips * PIP

    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"])
    atr = df["atr_d1"].to_numpy(float)

    if module == 1:
        sig, sl, tp = _mod_rolling(o, h, l, c, ts, atr, spread_price, p)
    elif module == 2:
        sig, sl, tp = _mod_asia_london(o, h, l, c, ts, atr, spread_price, p)
    else:
        sig, sl, tp = _mod_compression(o, h, l, c, ts, atr, spread_price, p)

    sig, sl, tp = _finalize(len(c), sig, sl, tp, side_filter)
    return pd.DataFrame({"signal": sig, "sl_price": sl, "tp_price": tp}, index=df.index)


templates.register(
    "gold_break_fusion",
    defaults={
        "module_id": 1.0,
        "side": 0.0,
        "assumed_spread": 3.0,      # pips(pip=0.1 → $0.30)
        "break_buffer_atr": 0.03,
        "tp_r": 1.5,
        "min_body_ratio": 0.55,
        "lookback_bars": 32.0,      # Rolling既定(Compressionは16, AsiaLondonは未使用)
        "min_width_atr": 0.20,
        "max_width_atr": 1.20,
        "stop_cap_atr": 0.35,
        # Compression専用
        "max_er": 0.25,
        "min_touches": 2.0,
        "touch_tolerance": 0.08,
        "min_overlap_ratio": 0.50,
        "min_tr_expansion": 1.20,
        # engine退出
        "max_hold_bars": 24.0,      # M15×24=6h(M5モジュールは72に上書き)
        "sl_pips": 30.0,            # フォールバック
        "tp_pips": 45.0,
        "lot": 0.1,
    },
    signal_fn=_signal_gold_break_fusion,
)
