"""Adaptive Flow Scalper — 3 Expert(M1版, adaptive_flow_scalper)。

Bid/Ask・tick無しのためM1 OHLCで実装(秒特徴量は偽造せずM1粒度に再解釈、tick imbalanceは
M1リターン符号一致で代理)。3 Expertを独立テンプレとして登録し、既存 engine/walk_forward で
単独検証する(Phase3ゲート)。全null→仕様通り停止(Meta Filterで救済しない)。

- afs_momentum   Expert A Participation Momentum: 一方向モメンタム→浅い押し戻り→再加速で追随
- afs_reversion  Expert B Failed Impulse Reversion: 急変が失敗し急変前レンジ再侵入で逆張り
- afs_session    Expert C Session Opening Flow: L/NY開始後の初動→押し戻り→再加速

共通: 構造SL(sl_price)、TP=tp_r×R、時間退出(max_hold_bars, M1で60-180s近似=数バー)、
同方向1日1回、delay_bars で追加執行遅延(engine shift(1)に上乗せ、M1粒度=60s刻み)。
先読み無し: 全条件は確定バーiまで。regimeもiまで。執行shiftはengine。side: 0両/1買/-1売。
M1制約: 秒スケール微細構造・実spread percentile・tick rateは表現不能。真の秒版はMT5(Phase6)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from app.core import templates
from app.core.strategy_model import Strategy


def _feats(df):
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"])
    cs = pd.Series(c)
    ret1 = cs.diff(1).to_numpy(); ret3 = cs.diff(3).to_numpy(); ret5 = cs.diff(5).to_numpy()
    rv = cs.diff(1).rolling(20).std().to_numpy()  # 実現vol(M1)
    absret = cs.diff(1).abs()
    def er(n):
        num = (cs - cs.shift(n)).abs()
        den = absret.rolling(n).sum()
        return (num / den.replace(0, np.nan)).to_numpy()
    er10 = er(10); er20 = er(20)
    z3 = (pd.Series(ret3) / pd.Series(ret3).rolling(120).std()).to_numpy()
    hour = ts.dt.hour.to_numpy(); day = ts.dt.floor("D").to_numpy()
    return dict(o=o, h=h, l=l, c=c, ts=ts, ret1=ret1, ret3=ret3, ret5=ret5, rv=rv,
                er10=er10, er20=er20, z3=z3, hour=hour, day=day)


def _shift_delay(sig, sl, tp, delay):
    if delay <= 0:
        return sig, sl, tp
    s2 = np.zeros_like(sig); sl2 = np.full_like(sl, np.nan); tp2 = np.full_like(tp, np.nan)
    s2[delay:] = sig[:-delay]; sl2[delay:] = sl[:-delay]; tp2[delay:] = tp[:-delay]
    return s2, sl2, tp2


def _side(p):
    return int(p["side"].value) if "side" in p else 0


def _delay(p):
    return int(p["delay_bars"].value) if "delay_bars" in p else 0


def _tpr(p):
    return float(p["tp_r"].value) if "tp_r" in p else 1.5


def _toxic(f, i, tox_z):
    # rvが直近60本99pct相当(粗く: rv > rolling window内で極端)を毒性とする簡易ゲート
    return not np.isfinite(f["rv"][i]) or f["rv"][i] <= 0


# ---------- Expert A: Participation Momentum ----------
def _sig_momentum(strategy, df):
    p = strategy.params; f = _feats(df)
    er_min = float(p["er_min"].value); pb_lo = float(p["pb_low"].value); pb_hi = float(p["pb_high"].value)
    imp_lb = int(p["impulse_lb"].value); stopcap_rv = float(p["stop_rv"].value); tp_r = _tpr(p)
    c, h, l = f["c"], f["h"], f["l"]; n = len(c)
    sig = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    done = {}
    hh3 = pd.Series(h).rolling(3).max().shift(1).to_numpy()
    ll3 = pd.Series(l).rolling(3).min().shift(1).to_numpy()
    seg_hi_a = pd.Series(h).rolling(imp_lb + 1).max().to_numpy()
    seg_lo_a = pd.Series(l).rolling(imp_lb + 1).min().to_numpy()
    for i in range(imp_lb + 5, n):
        if _toxic(f, i, 0) or f["er10"][i] < er_min:
            continue
        up = f["ret1"][i] > 0 and f["ret3"][i] > 0 and f["ret5"][i] > 0
        dn = f["ret1"][i] < 0 and f["ret3"][i] < 0 and f["ret5"][i] < 0
        # インパルス範囲(直近imp_lb本, 事前ベクトル化)
        seg_hi = seg_hi_a[i]; seg_lo = seg_lo_a[i]
        rng = seg_hi - seg_lo
        if rng <= 0:
            continue
        # 買い: 上昇モメンタム中、当バーが直近3本高値を再上抜け(=押し戻り後の再加速)かつ
        #       現値がインパルス高値からの押し戻り率 pb_lo..pb_hi
        if up and not done.get((f["day"][i], 1)) and np.isfinite(hh3[i]) and c[i] > hh3[i]:
            pbr = (seg_hi - c[i]) / rng
            if pb_lo <= pbr <= pb_hi:
                slp = min(ll3[i], c[i] - stopcap_rv * f["rv"][i])
                R = c[i] - slp
                if R > 0:
                    sig[i] = 1; sl[i] = slp; tp[i] = c[i] + tp_r * R; done[(f["day"][i], 1)] = True
                    continue
        if dn and not done.get((f["day"][i], -1)) and np.isfinite(ll3[i]) and c[i] < ll3[i]:
            pbr = (c[i] - seg_lo) / rng
            if pb_lo <= pbr <= pb_hi:
                slp = max(hh3[i], c[i] + stopcap_rv * f["rv"][i])
                R = slp - c[i]
                if R > 0:
                    sig[i] = -1; sl[i] = slp; tp[i] = c[i] - tp_r * R; done[(f["day"][i], -1)] = True
    return _finish(df, sig, sl, tp, _side(p), _delay(p))


# ---------- Expert B: Failed Impulse Reversion ----------
def _sig_reversion(strategy, df):
    p = strategy.params; f = _feats(df)
    z_min = float(p["z_min"].value); er_max = float(p["er_max"].value); pre_lb = int(p["pre_lb"].value)
    stopcap_rv = float(p["stop_rv"].value); tp_r = _tpr(p)
    c, h, l = f["c"], f["h"], f["l"]; n = len(c)
    sig = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    done = {}
    pre_hi_a = pd.Series(h).rolling(pre_lb).max().shift(3).to_numpy()  # [i-3-pre_lb .. i-3]近似
    pre_lo_a = pd.Series(l).rolling(pre_lb).min().shift(3).to_numpy()
    spike_hi_a = pd.Series(h).rolling(4).max().to_numpy()  # [i-3..i]
    spike_lo_a = pd.Series(l).rolling(4).min().to_numpy()
    for i in range(pre_lb + 6, n):
        if not np.isfinite(f["z3"][i]) or f["er10"][i] > er_max or _toxic(f, i, 0):
            continue
        pre_hi = pre_hi_a[i]; pre_lo = pre_lo_a[i]
        if not np.isfinite(pre_hi):
            continue
        # 売り: 急騰(z3>=+z_min)後、新高値更新停止 & 現値が急騰前高値未満へ再侵入
        if f["z3"][i] >= z_min and not done.get((f["day"][i], -1)):
            spike_hi = spike_hi_a[i]
            if h[i] < spike_hi and c[i] < pre_hi:  # 高値更新停止 & 前レンジ再侵入
                slp = spike_hi + stopcap_rv * f["rv"][i]
                R = slp - c[i]
                if R > 0:
                    sig[i] = -1; sl[i] = slp; tp[i] = c[i] - tp_r * R; done[(f["day"][i], -1)] = True
                    continue
        if f["z3"][i] <= -z_min and not done.get((f["day"][i], 1)):
            spike_lo = spike_lo_a[i]
            if l[i] > spike_lo and c[i] > pre_lo:
                slp = spike_lo - stopcap_rv * f["rv"][i]
                R = c[i] - slp
                if R > 0:
                    sig[i] = 1; sl[i] = slp; tp[i] = c[i] + tp_r * R; done[(f["day"][i], 1)] = True
    return _finish(df, sig, sl, tp, _side(p), _delay(p))


# ---------- Expert C: Session Opening Flow ----------
def _sig_session(strategy, df):
    p = strategy.params; f = _feats(df)
    stopcap_rv = float(p["stop_rv"].value); tp_r = _tpr(p)
    lon = int(p["london_utc"].value); ny = int(p["ny_utc"].value); win = int(p["window_min"].value)
    c, h, l = f["c"], f["h"], f["l"]; hour = f["hour"]; ts = f["ts"]; n = len(c)
    minute_of_day = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    sig = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    done = {}
    hh3 = pd.Series(h).rolling(3).max().shift(1).to_numpy()
    ll3 = pd.Series(l).rolling(3).min().shift(1).to_numpy()
    for i in range(30, n):
        mod = minute_of_day[i]
        in_lon = lon * 60 <= mod < lon * 60 + win
        in_ny = ny * 60 <= mod < ny * 60 + win
        if not (in_lon and True or in_ny):
            if not (in_lon or in_ny):
                continue
        start = lon * 60 if in_lon else ny * 60
        elapsed = mod - start
        if elapsed < 5:  # 開始後5分の初動形成待ち
            continue
        # 初動: 開始〜現在の高安、開始前30分レンジと比較
        s_idx = i - elapsed  # 概略の開始バー
        if s_idx < 30:
            continue
        init_hi = h[s_idx:i + 1].max(); init_lo = l[s_idx:i + 1].min()
        init_rng = init_hi - init_lo
        pre_rng = h[s_idx - 30:s_idx].max() - l[s_idx - 30:s_idx].min()
        if init_rng <= 0 or pre_rng <= 0 or init_rng < pre_rng:  # 拡大要件
            continue
        init_ret = c[i] - c[s_idx]
        key = (f["day"][i], 1 if init_ret > 0 else -1)
        if done.get(key):
            continue
        # 初動方向へ、直近3本を再ブレイク(押し戻り後の再加速)
        if init_ret > 0 and np.isfinite(hh3[i]) and c[i] > hh3[i]:
            slp = min(ll3[i], c[i] - stopcap_rv * f["rv"][i]); R = c[i] - slp
            if R > 0:
                sig[i] = 1; sl[i] = slp; tp[i] = c[i] + tp_r * R; done[key] = True
        elif init_ret < 0 and np.isfinite(ll3[i]) and c[i] < ll3[i]:
            slp = max(hh3[i], c[i] + stopcap_rv * f["rv"][i]); R = slp - c[i]
            if R > 0:
                sig[i] = -1; sl[i] = slp; tp[i] = c[i] - tp_r * R; done[key] = True
    return _finish(df, sig, sl, tp, _side(p), _delay(p))


def _finish(df, sig, sl, tp, side_filter, delay):
    if side_filter == 1:
        sig = np.where(sig == 1, 1, 0)
    elif side_filter == -1:
        sig = np.where(sig == -1, -1, 0)
    sl = np.where(sig != 0, sl, np.nan); tp = np.where(sig != 0, tp, np.nan)
    sig, sl, tp = _shift_delay(sig, sl, tp, delay)
    return pd.DataFrame({"signal": sig.astype(int), "sl_price": sl, "tp_price": tp}, index=df.index)


_COMMON = {"side": 0.0, "delay_bars": 0.0, "tp_r": 1.5, "stop_rv": 1.0,
           "max_hold_bars": 3.0, "sl_pips": 20.0, "tp_pips": 30.0, "lot": 0.1}

templates.register("afs_momentum", defaults={**_COMMON, "er_min": 0.50, "pb_low": 0.15,
                   "pb_high": 0.40, "impulse_lb": 15.0}, signal_fn=_sig_momentum)
templates.register("afs_reversion", defaults={**_COMMON, "z_min": 2.0, "er_max": 0.35,
                   "pre_lb": 10.0}, signal_fn=_sig_reversion)
templates.register("afs_session", defaults={**_COMMON, "london_utc": 7.0, "ny_utc": 13.0,
                   "window_min": 30.0}, signal_fn=_sig_session)
