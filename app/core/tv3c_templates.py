# -*- coding: utf-8 -*-
"""TradingView catalog 201-300 の第3バッチ。

「B判定の残りは描画オブジェクト依存」と一度書いたが、それは検証不足だった。
描画呼び出しが少ないものを実際に読んだところ、条件は確定していた。
ここはその回収分。残りの本数についても「描画依存」と断定せず未確認とする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, rsi, sma
from app.core.strategy_model import Strategy
from app.core.tv3_templates import (_EXIT_DEFAULTS, _atr_exit, _cooldown,
                                    _cross_up, _pivot_high, _pivot_low)


# ==========================================================================
# 231 Uptrick: Liquid Reversal Bands
#     Zスコアで幅を決めるバンドの外→内クロス。素の逆張り。
# ==========================================================================
def _sig_liquid_bands(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    c = df["close"]
    fair = ema(c, max(2, int(p["fair_len"].value)))
    zl = max(10, int(p["z_len"].value))
    w = (c - fair).abs()
    zw = ((w - w.rolling(zl).mean()) / w.rolling(zl).std()).fillna(0)
    base = w.rolling(zl).mean() * (1 + zw.clip(-3, 3) * 0.5)
    sm = max(1, int(p["smooth_len"].value))
    ub = ema(fair + base * float(p["upper_mult"].value), sm)
    lb = ema(fair - base * float(p["lower_mult"].value), sm)
    buy = (c.shift(1) <= lb.shift(1)) & (c > lb)
    sell = (c.shift(1) >= ub.shift(1)) & (c < ub)
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_231_liquid_bands", defaults={
    "fair_len": 50.0, "z_len": 100.0, "smooth_len": 18.0, "upper_mult": 2.4,
    "lower_mult": 2.4, "cooldown": 0.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_liquid_bands)


# ==========================================================================
# 262 Volume Sampled Supertrend [BackQuant]
#     時間でなく「累積出来高が閾値に達するたび」に新しいサンプルを作り、
#     そのサンプル系列で Supertrend を走らせる。出来高足Supertrend。
#     dukascopy の volume はティック数なので「約定回数サンプリング」になる。
# ==========================================================================
def _sig_vol_sampled_st(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"].astype(float)
    n = len(df)
    thr = float(v.median()) * float(p["vol_per_sample"].value)
    hh, ll, cc, vv = h.to_numpy(), l.to_numpy(), c.to_numpy(), v.to_numpy()
    hi = lo = np.nan
    acc = 0.0
    s_h, s_l, s_c = [], [], []
    idx = np.full(n, -1)
    for i in range(n):
        hi = hh[i] if not np.isfinite(hi) else max(hi, hh[i])
        lo = ll[i] if not np.isfinite(lo) else min(lo, ll[i])
        acc += vv[i] if np.isfinite(vv[i]) else 0.0
        if thr > 0 and acc >= thr:
            s_h.append(hi)
            s_l.append(lo)
            s_c.append(cc[i])
            idx[i] = len(s_c) - 1
            hi = lo = np.nan
            acc = 0.0
    if len(s_c) < 30:
        return _atr_exit(df, p, np.zeros(n, bool), np.zeros(n, bool))

    sh, sl2, sc = np.array(s_h), np.array(s_l), np.array(s_c)
    m = len(sc)
    prev = np.concatenate([[sc[0]], sc[:-1]])
    tr = np.maximum(sh - sl2, np.maximum(np.abs(sh - prev), np.abs(sl2 - prev)))
    na_ = max(2, int(p["atr_len"].value))
    a = pd.Series(tr).ewm(alpha=1.0 / na_, adjust=False, min_periods=na_).mean().to_numpy()
    mid = (sh + sl2) / 2
    f = float(p["factor"].value)
    d = np.ones(m)
    ub = lb = np.nan
    for j in range(m):
        if not np.isfinite(a[j]):
            continue
        u, lo2 = mid[j] + f * a[j], mid[j] - f * a[j]
        lb = lo2 if (not np.isfinite(lb) or lo2 > lb or sc[j - 1] < lb) else lb
        ub = u if (not np.isfinite(ub) or u < ub or sc[j - 1] > ub) else ub
        d[j] = (-1 if sc[j] > ub else 1) if d[j - 1] == 1 else (1 if sc[j] < lb else -1)
    long_s = np.zeros(m, bool)
    short_s = np.zeros(m, bool)
    long_s[1:] = (d[1:] == -1) & (d[:-1] == 1)
    short_s[1:] = (d[1:] == 1) & (d[:-1] == -1)

    buy = np.zeros(n, bool)
    sell = np.zeros(n, bool)
    ok = idx >= 0
    buy[ok] = long_s[idx[ok]]
    sell[ok] = short_s[idx[ok]]
    return _atr_exit(df, p, buy, sell)


templates.register("tv3_262_vol_sampled_st", defaults={
    "vol_per_sample": 10.0, "atr_len": 10.0, "factor": 3.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_vol_sampled_st)


# ==========================================================================
# 233 SuperTrend Weighted by Divergence
#     ダイバージェンスが出ているとATR倍率を divSensitivity 分だけ縮め、
#     反転を早める。レジーム反転でエントリー。
# ==========================================================================
def _sig_st_divweighted(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    a = atr(h, l, c, max(1, int(p["atr_len"].value))).to_numpy()
    r = rsi(c, 14)
    pv = max(1, int(p["pivot_len"].value))
    ph, pl = _pivot_high(h, pv, pv), _pivot_low(l, pv, pv)
    ph_v, pl_v = ph.ffill(), pl.ffill()
    phr = r.shift(pv).where(ph.notna()).ffill()
    plr = r.shift(pv).where(pl.notna()).ffill()
    bear_div = ph.notna() & (ph_v > ph_v.shift(1).where(ph.notna()).ffill()) \
        & (phr < phr.shift(1).where(ph.notna()).ffill())
    bull_div = pl.notna() & (pl_v < pl_v.shift(1).where(pl.notna()).ffill()) \
        & (plr > plr.shift(1).where(pl.notna()).ffill())
    div = (bear_div | bull_div).fillna(False).to_numpy()

    base = float(p["base_factor"].value)
    sens = float(p["div_sensitivity"].value)
    mid = ((h + l) / 2).to_numpy()
    cl = c.to_numpy()
    n = len(df)
    reg = np.ones(n)
    tl = ts = np.nan
    for i in range(n):
        if not np.isfinite(a[i]):
            continue
        f = base * (1 - sens) if div[i] else base
        bt, bb = mid[i] + f * a[i], mid[i] - f * a[i]
        prev = reg[i - 1] if i else 1
        if prev == 1:
            tl = bb if not np.isfinite(tl) else max(bb, tl)
            ts = bt
            reg[i] = -1 if cl[i] < tl else 1
        else:
            ts = bt if not np.isfinite(ts) else min(bt, ts)
            tl = bb
            reg[i] = 1 if cl[i] > ts else -1
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    up[1:] = (reg[1:] == 1) & (reg[:-1] != 1)
    dn[1:] = (reg[1:] == -1) & (reg[:-1] != -1)
    return _atr_exit(df, p, up, dn)


templates.register("tv3_233_st_divweighted", defaults={
    "atr_len": 10.0, "base_factor": 3.0, "pivot_len": 2.0, "div_sensitivity": 0.5,
    **_EXIT_DEFAULTS}, signal_fn=_sig_st_divweighted)


# ==========================================================================
# 246 Reversal Confirmation
#     直近 trendLookback 本の trendStrength 割合が同色 + ATR倍以上動いた後の
#     反対色ローソク。翌足で押し返しと EMA3/5 の向きが揃えば confirm。
# ==========================================================================
def _sig_rev_confirm(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    lb = max(3, int(p["trend_lookback"].value))
    need = lb * float(p["trend_strength"].value)
    gcnt = (c > o).astype(int).shift(1).rolling(lb).sum()
    rcnt = (c < o).astype(int).shift(1).rolling(lb).sum()
    a = atr(h, l, c, 14)
    move = (c.shift(1) - c.shift(lb + 1)).abs()
    big = move > a * float(p["min_move_atr"].value)
    sig_up = (gcnt >= need) & big
    sig_dn = (rcnt >= need) & big

    bull_rev = sig_dn & (c > o)
    bear_rev = sig_up & (c < o)
    e3, e5 = ema(c, 3), ema(c, 5)
    if float(p["require_confirm"].value) > 0.5:
        buy = bull_rev.shift(1).fillna(False) & (c > c.shift(1)) & (e3 > e5)
        sell = bear_rev.shift(1).fillna(False) & (c < c.shift(1)) & (e3 < e5)
    else:
        buy, sell = bull_rev, bear_rev
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


for _nm, _rc in (("tv3_246_rev_confirm", 1.0), ("tv3_246_rev_candle", 0.0)):
    templates.register(_nm, defaults={
        "trend_lookback": 7.0, "trend_strength": 0.7, "min_move_atr": 2.0,
        "require_confirm": _rc, **_EXIT_DEFAULTS}, signal_fn=_sig_rev_confirm)
