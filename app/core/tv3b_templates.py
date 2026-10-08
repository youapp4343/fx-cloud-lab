# -*- coding: utf-8 -*-
"""TradingView catalog 201-300 の第2バッチ。

a04の自動仕分けでB判定(要精読)に落ちていたが、実際に読んだら条件が完全に確定
していたもの。落ちた理由はシグナル変数名が buySignal 形式でなく
`isLongSignal` `rawBullish` `should_show_buy` 等で、名前の正規表現に当たらなかっただけ。

tv3_templates.py と同じ規律(原文どおり、ロールオーバー帯除外、shiftはengine側)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, rsi, sma
from app.core.strategy_model import Strategy
from app.core.tv3_templates import _EXIT_DEFAULTS, _atr_exit, _cooldown, _cross_dn, _cross_up


def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def _vwma(c: pd.Series, v: pd.Series, n: int) -> pd.Series:
    return (c * v).rolling(n).sum() / v.rolling(n).sum()


# ==========================================================================
# 252 Wick Reversal Indicator (Secrets of a Pivot Boss)
#     267と同族だが Wick_Multiplier が 3.5、かつ第2条件に H-L>=SMA50 が付く。
#     原文は `A and B or C and D and E` で、Pineの優先順位は and > or。そのまま実装。
# ==========================================================================
def _sig_wick_pivotboss(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    wm, bp = float(p["wick_mult"].value), float(p["body_pct"].value)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    avg = sma(rng, 50)
    lng = (((c > o) & ((o - l) >= (c - o) * wm) & ((h - c) <= rng * bp))
           | ((c == o) & (c != h) & (rng >= (h - c) * wm)
              & ((h - c) <= rng * bp) & (rng >= avg)))
    sht = (((c < o) & ((h - o) >= (o - c) * wm) & ((c - l) <= rng * bp))
           | ((c == o) & (c != l) & (rng >= (c - l) * wm)
              & ((c - l) <= rng * bp) & (rng >= avg)))
    return _atr_exit(df, p, lng.fillna(False).to_numpy(), sht.fillna(False).to_numpy())


templates.register("tv3_252_wick_pivotboss", defaults={
    "wick_mult": 3.5, "body_pct": 0.25, **_EXIT_DEFAULTS},
    signal_fn=_sig_wick_pivotboss)


# ==========================================================================
# 243 Linda Raschke 5 SMA Reversal
#     5SMAの上に minBarsExtension 本以上いた直後に、初めて下抜けた足で買う。
#     ※原文どおり「伸びきった後の反対側への抜け」を順張りでなく逆張りで取る形。
# ==========================================================================
def _sig_raschke5(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(1, int(p["sma_len"].value))
    need = max(1, int(p["min_bars"].value))
    c = df["close"]
    m = sma(c, n)
    above, below = (c > m), (c < m)

    def _streak(mask: pd.Series) -> pd.Series:
        g = (~mask).cumsum()
        return mask.groupby(g).cumsum()

    ca, cb = _streak(above), _streak(below)
    buy = (ca.shift(1) >= need) & (c < m) & (ca == 0)
    sell = (cb.shift(1) >= need) & (c > m) & (cb == 0)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_243_raschke5", defaults={
    "sma_len": 5.0, "min_bars": 7.0, **_EXIT_DEFAULTS}, signal_fn=_sig_raschke5)


# ==========================================================================
# 271 Live Breakout Zones (No Repaint)
#     直近rangeLen本の高値を終値が抜けたら買い。素のドンチアンブレイク。
# ==========================================================================
def _sig_donchian_break(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(2, int(p["range_len"].value))
    c = df["close"]
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    buy = c > hi.shift(1)
    sell = c < lo.shift(1)
    buy &= ~buy.shift(1).fillna(False)
    sell &= ~sell.shift(1).fillna(False)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_271_live_breakout", defaults={
    "range_len": 20.0, **_EXIT_DEFAULTS}, signal_fn=_sig_donchian_break)


# ==========================================================================
# 260 SD Bands Filtered Signals
#     ★「バンド幅が sideways_threshold 未満」= レンジ中だけ逆張り。
#     原文の閾値1.5は価格の絶対値で、銘柄依存。pipに直してパラメータに出す。
# ==========================================================================
def _sig_sd_bands(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(1, int(p["length"].value))
    mult = float(p["mult"].value)
    c, h, l = df["close"], df["high"], df["low"]
    m = sma(c, n)
    sd = c.rolling(n).std()
    ub, lb = m + sd * mult, m - sd * mult
    # 原文の絶対値閾値は銘柄依存なので、自分の分位で「狭い」を定義する
    width = ub - lb
    sideways = width < width.rolling(max(50, n * 5)).quantile(float(p["narrow_q"].value))
    buy = sideways & (c > lb) & (l <= lb)
    sell = sideways & (c < ub) & (h >= ub)
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_260_sd_bands", defaults={
    "length": 20.0, "mult": 2.0, "narrow_q": 0.3, "cooldown": 5.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_sd_bands)


# ==========================================================================
# 264 Entry Buy/Sell with Adjustable EMA-WMA Difference (Brian Le)
#     ★RSIそのものに EMA9 / WMA45 をかけ、そのクロスで入る(価格のMAではない)。
# ==========================================================================
def _sig_rsi_ema_wma(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    r = rsi(df["close"], max(2, int(p["rsi_len"].value)))
    e = ema(r, max(2, int(p["ema_len"].value)))
    w = _wma(r, max(2, int(p["wma_len"].value)))
    up_e, dn_e = _cross_up(r, e), _cross_dn(r, e)
    up_w, dn_w = _cross_up(r, w), _cross_dn(r, w)
    diff_ok = (e - w).abs() >= float(p["min_diff"].value)

    mode = int(round(float(p["mode"].value)))   # 0=special(両方クロス) 1=final(EMAのみ+条件)
    if mode == 0:
        buy, sell = up_e & up_w, dn_e & dn_w
    else:
        buy = up_e & (r < w) & (r > float(p["rsi_buy"].value))
        sell = dn_e & (r > w) & (r < float(p["rsi_sell"].value))
    buy &= diff_ok
    sell &= diff_ok
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


for _nm, _md in (("tv3_264_rsi_emawma_special", 0.0), ("tv3_264_rsi_emawma_final", 1.0)):
    templates.register(_nm, defaults={
        "rsi_len": 14.0, "ema_len": 9.0, "wma_len": 45.0, "min_diff": 5.0,
        "rsi_buy": 40.0, "rsi_sell": 60.0, "mode": _md, **_EXIT_DEFAULTS},
        signal_fn=_sig_rsi_ema_wma)


# ==========================================================================
# 268 TRENDSYNC BUY/SELL BY SIMPLY_DANTE-FX
#     ★原文にそのまま実装すると、売り条件が uptrend を参照している
#       (sellSignal = uptrend and higherHigh ...)。作者の取り違えと思われるが、
#       「原文に書いてあることだけを実装する」規律に従いそのまま移す。
# ==========================================================================
def _sig_trendsync(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    m = sma(c, max(2, int(p["sma_len"].value)))
    a = atr(h, l, c, max(2, int(p["atr_len"].value)))
    r = rsi(c, max(2, int(p["rsi_len"].value)))
    thr = float(p["rsi_thr"].value)

    higher_high = h > h.shift(1).rolling(3).max()
    higher_low = l > l.shift(1).rolling(3).min()
    lower_high = h < h.shift(1).rolling(3).max()
    lower_low = l < l.shift(1).rolling(3).min()
    up, dn = c > m, c < m
    rng_ok = (h - l) > a * float(p["range_mult"].value)
    mom_up, mom_dn = r > thr, r < (100 - thr)

    buy_s = up & higher_low & rng_ok & mom_up
    sell_s = up & higher_high & rng_ok & mom_dn        # ← 原文どおり uptrend
    buy_d = dn & lower_low & rng_ok & mom_up
    sell_d = dn & lower_high & rng_ok & mom_dn

    buy = (buy_s & ~buy_s.shift(1).fillna(False)) | (buy_d & ~buy_d.shift(1).fillna(False))
    sell = (sell_s & ~sell_s.shift(1).fillna(False)) | (sell_d & ~sell_d.shift(1).fillna(False))
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_268_trendsync", defaults={
    "sma_len": 200.0, "atr_len": 14.0, "rsi_len": 14.0, "rsi_thr": 50.0,
    "range_mult": 1.2, **_EXIT_DEFAULTS}, signal_fn=_sig_trendsync)


# ==========================================================================
# 216 Reversal Zones with Signals
#     RSIが極値のときのピボットを水平線(ゾーン)として記憶し、そこを抜けたら入る。
# ==========================================================================
def _sig_reversal_zones(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    from app.core.tv3_templates import _pivot_high, _pivot_low
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    r = rsi(c, 14)
    ph, pl = _pivot_high(h, 5, 5), _pivot_low(l, 5, 5)
    zone_bot = pl.where(pl.notna() & (r < float(p["rsi_os"].value))).ffill()
    zone_top = ph.where(ph.notna() & (r > float(p["rsi_ob"].value))).ffill()
    buy = _cross_up(c, zone_bot) & (c > zone_bot) & zone_bot.notna()
    sell = _cross_dn(c, zone_top) & (c < zone_top) & zone_top.notna()
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_216_reversal_zones", defaults={
    "rsi_ob": 65.0, "rsi_os": 35.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_reversal_zones)


# ==========================================================================
# 298 Pymander's EZ Breakout Finder
#     直近高値抜け + 実体が平均のmomentumMult倍以上 + VWMAトレンドバイアス。
#     meanReversion=1 にすると「VWMAの下での上抜けだけ」に反転する。
# ==========================================================================
def _sig_ez_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    n = max(1, int(p["lookback"].value))
    prev_hi = h.shift(1).rolling(n).max()
    prev_lo = l.shift(1).rolling(n).min()
    body = (c - o).abs()
    mom = body > sma(body, n) * float(p["momentum_mult"].value)
    vw = _vwma(c, v, max(1, int(p["vwma_len"].value)))

    mr = float(p["mean_reversion"].value) > 0.5
    if float(p["use_trend"].value) > 0.5:
        bull_t = (c < vw) if mr else (c > vw)
        bear_t = (c > vw) if mr else (c < vw)
    else:
        bull_t = bear_t = pd.Series(True, index=df.index)
    use_wick = float(p["use_wick"].value) > 0.5
    src_hi = h if use_wick else c
    src_lo = l if use_wick else c

    buy = (src_hi > prev_hi) & mom & bull_t
    sell = (src_lo < prev_lo) & mom & bear_t
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


for _nm, _mr in (("tv3_298_ez_breakout", 0.0), ("tv3_298_ez_meanrev", 1.0)):
    templates.register(_nm, defaults={
        "lookback": 20.0, "momentum_mult": 1.5, "use_trend": 1.0, "use_wick": 1.0,
        "vwma_len": 10.0, "mean_reversion": _mr, "cooldown": 5.0, **_EXIT_DEFAULTS},
        signal_fn=_sig_ez_breakout)


# ==========================================================================
# 276 SuperTrend Ensemble [MiesOnCharts]
#     ATR倍率を kMin..kMax まで振った複数のSupertrendを走らせ、
#     方向の得票率が voteThr を超えたら入る。
# ==========================================================================
def _sig_st_ensemble(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    kmin, kmax, kstep = (float(p["k_min"].value), float(p["k_max"].value),
                         float(p["k_step"].value))
    a = atr(df["high"], df["low"], df["close"], max(1, int(p["atr_len"].value))).to_numpy()
    mid = ((df["high"] + df["low"]) / 2).to_numpy()
    cl = df["close"].to_numpy()
    ks = np.arange(kmin, kmax + 1e-9, kstep)
    n = len(cl)
    votes = np.zeros(n)
    for k in ks:
        s = np.nan
        d = 1
        col = np.zeros(n)
        for i in range(n):
            if not np.isfinite(a[i]):
                col[i] = d
                continue
            off = k * a[i]
            if not np.isfinite(s):
                s, d = mid[i] - off, 1
            elif d == 1:
                s = max(s, mid[i] - off)
                if cl[i] < s:
                    d, s = -1, mid[i] + off
            else:
                s = min(s, mid[i] + off)
                if cl[i] > s:
                    d, s = 1, mid[i] - off
            col[i] = d
        votes += col
    share = votes / len(ks)
    thr = float(p["vote_thr"].value)
    up = pd.Series((share >= thr) & (np.roll(share, 1) < thr), index=df.index)
    dn = pd.Series((share <= -thr) & (np.roll(share, 1) > -thr), index=df.index)
    up.iloc[0] = dn.iloc[0] = False
    return _atr_exit(df, p, up.to_numpy(), dn.to_numpy())


templates.register("tv3_276_st_ensemble", defaults={
    "k_min": 1.25, "k_max": 4.0, "k_step": 0.5, "atr_len": 14.0, "vote_thr": 0.85,
    **{k: v for k, v in _EXIT_DEFAULTS.items() if k != "atr_period"},
    "atr_period": 14.0}, signal_fn=_sig_st_ensemble)


# ==========================================================================
# 232 SUPERTREND RIBBON
#     ATR倍率を baseF から stepF 刻みで8本作り、強気バンド数(bcnt)で多数決。
#     bcnt>=confirm が confirmBars 本続いたら強気に確定、bcnt<=(8-confirm) で弱気。
#     間はトレンドを保持する(ヒステリシス)。
# ==========================================================================
def _st_dirs(df: pd.DataFrame, atr_p: int, factors) -> np.ndarray:
    """複数倍率のSupertrend方向を (n, len(factors)) で返す。1=強気, -1=弱気。

    numbaでJIT化(純Python版と完全一致を確認済み。8本リボンで実測637倍)。
    """
    from app.core.tv3_fast import _st_dirs_core
    a = atr(df["high"], df["low"], df["close"], max(1, atr_p)).to_numpy()
    mid = ((df["high"] + df["low"]) / 2).to_numpy()
    return _st_dirs_core(mid, df["close"].to_numpy(), a,
                         np.asarray(factors, dtype=float))


def _sig_st_ribbon(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    base, step = float(p["base_factor"].value), float(p["step_factor"].value)
    factors = [base + i * step for i in range(8)]
    dirs = _st_dirs(df, int(p["atr_period_st"].value), factors)
    bcnt = (dirs == 1).sum(axis=1)

    confirm = int(p["confirm"].value)
    up_thr, dn_thr = confirm, 8 - confirm
    need = max(1, int(p["confirm_bars"].value))
    n = len(df)
    master = np.zeros(n, dtype=int)
    up_s = dn_s = 0
    m = 0
    for i in range(n):
        up_s = up_s + 1 if bcnt[i] >= up_thr else 0
        dn_s = dn_s + 1 if bcnt[i] <= dn_thr else 0
        if m == 0:
            m = 1 if bcnt[i] >= 4 else -1
        if up_s >= need:
            m = 1
        if dn_s >= need:
            m = -1
        master[i] = m
    flip_up = (master == 1) & (np.roll(master, 1) != 1)
    flip_dn = (master == -1) & (np.roll(master, 1) != -1)
    flip_up[0] = flip_dn[0] = False
    return _atr_exit(df, p, flip_up, flip_dn)


templates.register("tv3_232_st_ribbon", defaults={
    "atr_period_st": 10.0, "base_factor": 1.0, "step_factor": 0.4,
    "confirm": 6.0, "confirm_bars": 1.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_st_ribbon)


# ==========================================================================
# 290 SuperTrendy
#     ATR倍率をボラ regime(atr/SMA50(atr) を 0.75〜1.40 にクリップ)で伸縮させ、
#     さらに効率比(ER)が低い(=チョッピー)ほど確認本数を増やす。
# ==========================================================================
def _sig_supertrendy(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    n_atr = max(1, int(p["atr_len"].value))
    a = atr(h, l, c, n_atr)
    vol_ratio = (a / sma(a, 50)).fillna(1.0)
    vol_regime = vol_ratio.clip(0.75, 1.40).to_numpy()

    er_len = max(3, int(p["er_len"].value))
    net = (c - c.shift(er_len)).abs()
    path = c.diff().abs().rolling(er_len).sum()
    er = (net / path).where(path > 0, 0.0)
    er_s = ema(er, 3).fillna(0.0).to_numpy()

    base = float(p["base_factor"].value)
    conf0 = int(p["confirm_bars"].value)
    av = a.to_numpy()
    mid = ((h + l) / 2).to_numpy()
    cl = c.to_numpy()
    n = len(df)
    ub = lb = np.nan
    tdir = 1
    bull = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(av[i]):
            bull[i] = tdir == 1
            continue
        off = base * vol_regime[i] * av[i]
        u_b, l_b = mid[i] + off, mid[i] - off
        ub = u_b if (not np.isfinite(ub) or u_b < ub or cl[i - 1] > ub) else ub
        lb = l_b if (not np.isfinite(lb) or l_b > lb or cl[i - 1] < lb) else lb
        tdir = (-1 if cl[i] < lb else 1) if tdir == 1 else (1 if cl[i] > ub else -1)
        bull[i] = tdir == 1

    # 確認本数はERで可変(原文どおり)
    ad_conf = np.where(er_s < 0.25, max(conf0 + 2, 4),
                       np.where(er_s < 0.45, conf0 + 1,
                                np.where(er_s < 0.65, conf0, max(conf0 - 1, 1))))
    locked = bull[0]
    streak = 0
    up = np.zeros(n, dtype=bool)
    dn = np.zeros(n, dtype=bool)
    for i in range(n):
        streak = 0 if bull[i] == locked else streak + 1
        if streak >= ad_conf[i]:
            if bull[i]:
                up[i] = True
            else:
                dn[i] = True
            locked = bull[i]
            streak = 0
    return _atr_exit(df, p, up, dn)


templates.register("tv3_290_supertrendy", defaults={
    "atr_len": 14.0, "base_factor": 3.5, "er_len": 10.0, "confirm_bars": 2.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_supertrendy)


# ==========================================================================
# 226 Adaptive Entropy Trend [QuantAlgo]
#     対数リターンのシャノンエントロピーで EMA の平滑係数を可変にし、
#     ATRバンド(トレンド強度で伸縮)の内側バンド抜けでトレンド確定。
# ==========================================================================
def _sig_entropy_trend(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(5, int(p["lookback"].value))
    fast_m = float(p["fast_mult"].value)
    c = df["close"]
    a = atr(df["high"], df["low"], c, n).to_numpy()

    r = np.log(c / c.shift(1)).to_numpy()
    cl = c.to_numpy()
    N = len(cl)
    nbin = 8
    ent = np.full(N, 0.5)
    max_ent = np.log2(nbin)
    for i in range(n, N):
        w = r[i - n + 1:i + 1]
        w = w[np.isfinite(w)]
        if len(w) < 3 or w.max() == w.min():
            continue
        hist, _ = np.histogram(w, bins=nbin)
        pr = hist / hist.sum()
        pr = pr[pr > 0]
        ent[i] = float(-(pr * np.log2(pr)).sum()) / max_ent

    alpha = 2.0 / (n * (0.3 + ent * 1.4) + 1.0)
    aema = np.full(N, np.nan)
    prev = cl[0]
    for i in range(N):
        prev = prev + alpha[i] * (cl[i] - prev)
        aema[i] = prev
    strength = 1.0 - ent
    bw = a * fast_m * (0.5 + strength)
    inner_u, inner_l = aema + bw, aema - bw

    tdir = np.zeros(N, dtype=int)
    d = 0
    for i in range(N):
        if np.isfinite(inner_u[i]) and cl[i] > inner_u[i]:
            d = 1
        elif np.isfinite(inner_l[i]) and cl[i] < inner_l[i]:
            d = -1
        tdir[i] = d
    up = (tdir == 1) & (np.roll(tdir, 1) != 1)
    dn = (tdir == -1) & (np.roll(tdir, 1) != -1)
    up[0] = dn[0] = False
    return _atr_exit(df, p, up, dn)


templates.register("tv3_226_entropy_trend", defaults={
    "lookback": 25.0, "fast_mult": 1.8, **_EXIT_DEFAULTS},
    signal_fn=_sig_entropy_trend)


# ==========================================================================
# 207 Bollinger Bands Weighted Alert System (BBWAS)
#     ★BBの外帯を(出来高を伴って)ブレイクした後 lookback_bars 本以内に、
#       1σ帯(または中心)へ戻るクロスで**逆方向**に入る。ブレイク失敗の取り込み。
# ==========================================================================
def _sig_bbwas(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    from app.core.indicators import bollinger_bands
    c, v = df["close"], df["volume"].astype(float)
    n = max(5, int(p["bb_len"].value))
    ub, basis, lb = bollinger_bands(c, n, float(p["bb_mult"].value))
    dev = (ub - basis) / max(float(p["bb_mult"].value), 1e-9)
    ub_va, lb_va = basis + dev, basis - dev

    vma = sma(v, n)
    vol_ok = (v >= vma + v.rolling(n).std() * float(p["vol_dev_mult"].value)
              if float(p["consider_volume"].value) > 0.5
              else pd.Series(True, index=df.index))
    brk_up = (c > ub.shift(1)) & vol_ok
    brk_dn = (c < lb.shift(1)) & vol_ok

    look = max(1, int(p["lookback_bars"].value))

    def _since(mask: pd.Series) -> pd.Series:
        g = mask.cumsum()
        return (~mask).groupby(g).cumsum().where(g > 0)

    s_up, s_dn = _since(brk_up.fillna(False)), _since(brk_dn.fillna(False))
    # 戻り先は 1σ帯 か 中心線(原文の early_exit 相当をパラメータ化)
    tgt_s = ub_va if float(p["revert_to_1sigma"].value) > 0.5 else basis
    tgt_l = lb_va if float(p["revert_to_1sigma"].value) > 0.5 else basis
    sell = (s_up <= look) & _cross_dn(c, tgt_s.shift(1))
    buy = (s_dn <= look) & _cross_up(c, tgt_l.shift(1))
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_207_bbwas", defaults={
    "bb_len": 20.0, "bb_mult": 2.0, "consider_volume": 1.0, "vol_dev_mult": 1.0,
    "lookback_bars": 10.0, "revert_to_1sigma": 1.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_bbwas)


# ==========================================================================
# 255 COMBO - EMA / LRI / SuperTrend / HMA Stack
#     HMA 3本の整列 + 線形回帰 + EMA を Supertrend の反転に重ねる。
#     highConf は反転時に全部そろっている場合。
# ==========================================================================
def _sig_combo_stack(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    from app.core.tv3_templates import _supertrend_dir
    c = df["close"]

    def _hma(s, n):
        n = max(2, int(n))
        return _wma(2 * _wma(s, max(1, n // 2)) - _wma(s, n),
                    max(1, int(round(np.sqrt(n)))))

    scalp = float(p["scalping"].value) > 0.5
    hf, hm_, hs = (10, 20, 100) if scalp else (20, 50, 100)
    h_f, h_m, h_s = _hma(c, hf), _hma(c, hm_), _hma(c, hs)
    hma_bull = (h_s < h_m) & (h_s < h_f) & (h_f > h_m)
    hma_bear = (h_s > h_m) & (h_s > h_f) & (h_m > h_f)

    e = ema(c, max(2, int(p["ema_len"].value)))
    lri = _linreg255(c, max(2, int(p["lri_len"].value)))
    d = _supertrend_dir(df, max(1, int(p["st_atr"].value)), float(p["st_mult"].value))
    flip_up = (d == -1) & (d.shift(1) == 1)
    flip_dn = (d == 1) & (d.shift(1) == -1)

    if float(p["high_conf"].value) > 0.5:
        buy = flip_up & hma_bull & (c > e) & (c > lri)
        sell = flip_dn & hma_bear & (c < e) & (c < lri)
    else:
        buy, sell = flip_up, flip_dn
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


def _linreg255(s: pd.Series, n: int) -> pd.Series:
    n = max(2, int(n))
    x = np.arange(n, dtype=float)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()

    def _f(y):
        ym = y.mean()
        return ym + (((x - xm) * (y - ym)).sum() / den) * (n - 1 - xm)

    return s.rolling(n).apply(_f, raw=True)


for _nm, _hc in (("tv3_255_combo_stack", 0.0), ("tv3_255_combo_highconf", 1.0)):
    templates.register(_nm, defaults={
        "scalping": 0.0, "ema_len": 200.0, "lri_len": 20.0, "st_atr": 10.0,
        "st_mult": 3.0, "high_conf": _hc, **_EXIT_DEFAULTS},
        signal_fn=_sig_combo_stack)


# ==========================================================================
# 283 Reversal Engine M15 - Practical Early/Confirm
#     下落しきった位置(EMA50から乖離・レンジ下端)で、下ヒゲ + Stochクロス +
#     出来高、かつ EMA200 が下向きでないときに反転買い。early は条件を緩めた版。
# ==========================================================================
def _sig_reversal_engine(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    from app.core.indicators import atr as _atr, rsi as _rsi, stochastic
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    a = _atr(h, l, c, 14)
    r = _rsi(c, 14)
    k, d_ = stochastic(h, l, c, 14, 3, 3)

    lookb = max(5, int(p["zone_lookback"].value))
    bottom_zone = c <= (l.rolling(lookb).min() + (h.rolling(lookb).max()
                                                  - l.rolling(lookb).min()) * 0.25)
    bottom_now = l <= l.shift(1).rolling(lookb).min()
    overext_dn = c < e50 - a * float(p["overext_atr"].value)
    trend_ok = e200 >= e200.shift(1)
    rng = (h - l).replace(0, np.nan)
    wick_ok = (np.minimum(o, c) - l) / rng >= float(p["wick_ratio_min"].value)
    bull_confirm = (c > c.shift(1)) & (c > o)
    early_rev = c > l.shift(1)
    osc_ok = (r < float(p["rsi_max"].value)) & (k < float(p["stoch_max"].value)) \
        & _cross_up(k, d_)
    vol_ok = v > sma(v, 20) * float(p["vol_mult"].value)

    early = float(p["early_mode"].value) > 0.5
    if early:
        buy = trend_ok & bottom_zone & overext_dn & early_rev & osc_ok & vol_ok
    else:
        buy = (trend_ok & bottom_zone & bottom_now & overext_dn & bull_confirm
               & early_rev & wick_ok & osc_ok & vol_ok)
    # 原文は買い(反転)専用。売りはエグジット用シグナルなのでエントリーにしない
    none = np.zeros(len(df), dtype=bool)
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd), none)


for _nm, _em in (("tv3_283_reversal_engine", 0.0), ("tv3_283_reversal_early", 1.0)):
    templates.register(_nm, defaults={
        "zone_lookback": 20.0, "overext_atr": 1.5, "wick_ratio_min": 0.25,
        "rsi_max": 40.0, "stoch_max": 30.0, "vol_mult": 1.0, "early_mode": _em,
        "cooldown": 10.0, **_EXIT_DEFAULTS}, signal_fn=_sig_reversal_engine)
