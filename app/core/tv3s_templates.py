# -*- coding: utf-8 -*-
"""TradingView catalog 201-300 の strategy 5本を FX に移植する。

第1部の監査で、strategy 6本の取引リストは全て非FX(BTC/ETH/IWM/MNQ)だった。
「TradingViewでこう出た」は追認する意味がないが、**ロジック自体はFXで試せる**。
むしろ実際にバックテスト結果が公開されている群なので、こちらの土俵で
独立に検定する価値は他の指標より高い。

286 Scalp Signal Bot は除外(sweep/clustering/session が多層で、
既定値の組み合わせによって条件が変わる。確定しない)。

原文の決済も可能な限り再現する:
  270 グリッド指値 → 反対側のグリッド水準で利確、フラクタル±ATRで損切り
  275 レンジ中点 → スロー側の高安でストップ、cloud_percent 反転で手仕舞い
  281 ATR等幅の TP/SL
  282 価格比率の TP/SL
  291 ロング専用。linreg 上抜けで手仕舞い
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.catalog_templates import _emit
from app.core.indicators import atr, ema, sma
from app.core.osc_extra import linreg_slope  # noqa: F401  (存在確認用)
from app.core.strategy_model import Strategy
from app.core.tv3_templates import _EXIT_DEFAULTS, _ROLLOVER_HOURS, _pivot_high, _pivot_low


def _wma(s: pd.Series, n: int) -> pd.Series:
    n = max(1, int(n))
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def _linreg(s: pd.Series, n: int) -> pd.Series:
    """Pine の ta.linreg(src, len, 0) = 直近n本の回帰直線の現在値。"""
    n = max(2, int(n))
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def _f(y):
        ym = y.mean()
        slope = ((x - xm) * (y - ym)).sum() / denom
        return ym + slope * (n - 1 - xm)

    return s.rolling(n).apply(_f, raw=True)


def _emit_struct(df: pd.DataFrame, longs, shorts, sl_long, tp_long,
                 sl_short, tp_short) -> pd.DataFrame:
    """構造的なSL/TPをそのまま渡す(ATR倍への丸めをしない)。

    ロールオーバー帯 UTC20-23 は tv3_templates と同じ理由で落とす。
    """
    hour = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
    ok = ~np.isin(hour, _ROLLOVER_HOURS)
    n = len(df)
    L = np.asarray(longs, dtype=bool) & ok
    S = np.asarray(shorts, dtype=bool) & ok
    c = df["close"].to_numpy()

    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    slL, tpL = np.asarray(sl_long, float), np.asarray(tp_long, float)
    slS, tpS = np.asarray(sl_short, float), np.asarray(tp_short, float)
    # SL/TPが価格として整合しないシグナルは捨てる(ロングでSL>=価格 等)
    L &= np.isfinite(slL) & np.isfinite(tpL) & (slL < c) & (tpL > c)
    S &= np.isfinite(slS) & np.isfinite(tpS) & (slS > c) & (tpS < c)
    sig[L] = 1
    sig[S] = -1
    slp[L], tpp[L] = slL[L], tpL[L]
    slp[S], tpp[S] = slS[S], tpS[S]
    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp})


# ==========================================================================
# 270 Adaptive Fractal Grid Scalping Strategy   (原著: BYBIT BTCUSDT.P 15m)
#     SMAの上下でバイアスを決め、フラクタル±ATRのグリッド水準に指値。
#     利確は反対側のグリッド、損切りはフラクタル±ATR*trail。
#     ATRが volatilityThreshold を超えている時だけ発注する。
#     ※原文は指値(limit)。当方のエンジンは成行なので「その水準に触れた足で入る」
#       に置き換える。触れた時点で約定は保証されるので執行は原文より不利にならない。
# ==========================================================================
def _sig_fractal_grid(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    a = atr(h, l, c, max(2, int(p["atr_len"].value)))
    m = sma(c, max(2, int(p["sma_len"].value)))
    fh = _pivot_high(h, 2, 2).ffill()
    fl = _pivot_low(l, 2, 2).ffill()

    grid_hi = fh + a * float(p["grid_mult_high"].value)
    grid_lo = fl - a * float(p["grid_mult_low"].value)
    trail = float(p["trail_mult"].value)
    # 原文の volatilityThreshold は価格の絶対値。ATRの自己分位に置き換える
    vol_ok = a > a.rolling(200).quantile(float(p["vol_q"].value))

    bull, bear = (c > m), (c < m)
    buy = bull & vol_ok & fl.notna() & (l <= grid_lo)
    sell = bear & vol_ok & fh.notna() & (h >= grid_hi)
    return _emit_struct(df, buy.fillna(False), sell.fillna(False),
                        (fl - a * trail), grid_hi,
                        (fh + a * trail), grid_lo)


templates.register("tv3s_270_fractal_grid", defaults={
    "atr_len": 14.0, "sma_len": 50.0, "grid_mult_high": 1.0, "grid_mult_low": 1.0,
    "trail_mult": 1.5, "vol_q": 0.5, "max_hold_bars": 96.0, "lot": 0.1,
    "sl_pips": 30.0, "tp_pips": 60.0}, signal_fn=_sig_fractal_grid)


# ==========================================================================
# 275 Breakout Scalper   (原著: AMEX IWM 5m, README PF1.24)
#     速いレンジ中点と遅いレンジ中点の差をATRで正規化した cloud_percent。
#     EMAが速い中点の上 かつ cloud_percent が閾値超えで、遅い高値に逆指値。
# ==========================================================================
def _sig_breakout_scalper(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    fw = max(2, int(p["fast_window"].value))
    sw = max(2, int(p["slow_window"].value))
    h, l, c = df["high"], df["low"], df["close"]
    f_lo, f_hi = l.rolling(fw).min(), h.rolling(fw).max()
    s_lo, s_hi = l.rolling(sw).min(), h.rolling(sw).max()
    f_mid, s_mid = (f_lo + f_hi) / 2, (s_lo + s_hi) / 2
    inst = ema(c, max(1, int(p["instant_period"].value)))
    cp = (f_mid - s_mid) / atr(h, l, c, sw)
    thr = float(p["cloud_min_percent"].value)

    buy = (inst > f_mid) & (cp > thr) & (h >= s_hi)     # 逆指値 slow_high に触れた足
    sell = (inst < f_mid) & (cp < -thr) & (l <= s_lo)
    sl_l = np.maximum(s_lo, f_lo)
    sl_s = np.minimum(s_hi, f_hi)
    # 原文の利確は cloud_percent の符号反転(価格目標が無い)。
    # 当方は価格でしか決済できないので、SLまでの距離のRR倍をTPに置く
    rr = float(p["rr"].value)
    d_l = (c - sl_l)
    d_s = (sl_s - c)
    return _emit_struct(df, buy.fillna(False), sell.fillna(False),
                        sl_l, c + rr * d_l, sl_s, c - rr * d_s)


templates.register("tv3s_275_breakout_scalper", defaults={
    "fast_window": 10.0, "slow_window": 40.0, "instant_period": 3.0,
    "cloud_min_percent": 0.2, "rr": 2.0, "max_hold_bars": 96.0, "lot": 0.1,
    "sl_pips": 30.0, "tp_pips": 60.0}, signal_fn=_sig_breakout_scalper)


# ==========================================================================
# 281 ChopFlow ATR Scalp Strategy (OBV EMA)   (原著: MNQ 5m, 3ヶ月2万取引)
#     Choppiness Index が閾値未満(=トレンド寄り)で、OBVがそのEMAの上/下。
#     決済はATR等幅の TP/SL。
#     ※ OBVは volume を使う。dukascopyの volume はティック数なので、
#       「約定回数の符号付き累積」という代理になる。原著の出来高とは別物。
# ==========================================================================
def _sig_chopflow(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"].astype(float)
    n_atr = max(2, int(p["atr_len"].value))
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1.0 / n_atr, adjust=False, min_periods=n_atr).mean()

    n_ch = max(2, int(p["chop_len"].value))
    rng = h.rolling(n_ch).max() - l.rolling(n_ch).min()
    chop = 100 * np.log10(tr.rolling(n_ch).sum() / rng) / np.log10(n_ch)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_e = ema(obv, max(2, int(p["obv_ema_len"].value)))

    thr = float(p["chop_thresh"].value)
    trend = chop < thr
    buy = trend & (obv > obv_e)
    sell = trend & (obv < obv_e)
    # 原文どおり連続点灯するので、立ち上がりに絞る(毎バー発注は非現実的)
    buy &= ~buy.shift(1).fillna(False)
    sell &= ~sell.shift(1).fillna(False)

    d = (a * float(p["atr_mult"].value))
    return _emit_struct(df, buy.fillna(False), sell.fillna(False),
                        c - d, c + d, c + d, c - d)


templates.register("tv3s_281_chopflow", defaults={
    "atr_len": 14.0, "chop_len": 14.0, "chop_thresh": 50.0, "obv_ema_len": 20.0,
    "atr_mult": 1.5, "max_hold_bars": 96.0, "lot": 0.1,
    "sl_pips": 30.0, "tp_pips": 60.0}, signal_fn=_sig_chopflow)


# ==========================================================================
# 282 Swing/Scalper HULL + T3 avg   (原著: BINANCE ETHUSDT 3h, README PF1.27)
#     HullMA と T3 の平均が上向きに転じたらロング、下向きでショート。
#     決済は価格に対する比率の TP/SL。
# ==========================================================================
def _t3(s: pd.Series, n: int, vf: float = 0.7) -> pd.Series:
    e1 = ema(s, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    e4 = ema(e3, n)
    e5 = ema(e4, n)
    e6 = ema(e5, n)
    c1 = -(vf ** 3)
    c2 = 3 * vf ** 2 + 3 * vf ** 3
    c3 = -6 * vf ** 2 - 3 * vf - 3 * vf ** 3
    c4 = 1 + 3 * vf + vf ** 3 + 3 * vf ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _sig_hull_t3(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(4, int(p["ma_len"].value))
    c = df["close"]
    hull = _wma(2 * _wma(c, max(1, n // 2)) - _wma(c, n), max(1, int(round(np.sqrt(n)))))
    avg = (hull + _t3(c, n, float(p["v_factor"].value))) / 2
    up = (avg > avg.shift(1)) & (avg.shift(1) <= avg.shift(2))
    dn = (avg < avg.shift(1)) & (avg.shift(1) >= avg.shift(2))
    tp_r, sl_r = float(p["tp_pct"].value) / 100.0, float(p["sl_pct"].value) / 100.0
    return _emit_struct(df, up.fillna(False), dn.fillna(False),
                        c * (1 - sl_r), c * (1 + tp_r),
                        c * (1 + sl_r), c * (1 - tp_r))


templates.register("tv3s_282_hull_t3", defaults={
    "ma_len": 20.0, "v_factor": 0.7, "tp_pct": 2.0, "sl_pct": 1.0,
    "max_hold_bars": 96.0, "lot": 0.1, "sl_pips": 30.0, "tp_pips": 60.0},
    signal_fn=_sig_hull_t3)


# ==========================================================================
# 291 Linear Channel - Scalp Strategy 15M   (原著: BINANCE BTCUSDT 15m, PF5.88)
#     ★ロング専用。linregを-2%下げたバンドを終値が下回り、かつHMA400より上。
#     手仕舞いは終値がlinregを上抜けたとき(ショートは無い)。
#     README のPF5.88は18取引のみ。原著の期間もBTCの上昇局面。
#
#     ★★FXでは既定値のまま一度も発火しない。close が linreg(55) を下回る量の実測
#     (2023-01〜2026-07、-2%到達回数):
#         BTCUSD M15  925回  (p1 = -1.83%)
#         XAUUSD M15   40回  (p1 = -0.89%)
#         USDJPY M15    0回  (3年半の最大乖離が -1.92%)
#         EURUSD M15    0回  (同 -1.16%)
#         GBPJPY M15    0回  (同 -1.84%)
#     -2% という定義パラメータが暗号資産のボラ水準に合わせて決まっており、
#     FXのボラでは到達し得ない。これは移植ミスではなく **戦略がFXに適用できない**
#     ということ。閾値を下げれば発火するが、それは別の戦略になるので変えない。
# ==========================================================================
def _sig_linear_channel(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(2, int(p["length"].value))
    c = df["close"]
    lr = _linreg(c, n)
    band = lr * (float(p["band_pct"].value) / 100.0 + 1.0)
    hn = max(2, int(p["hma_len"].value))
    hma = _wma(2 * _wma(c, max(1, hn // 2)) - _wma(c, hn),
               max(1, int(round(np.sqrt(hn)))))

    buy = (c < band) & (hma < c)
    buy &= ~buy.shift(1).fillna(False)
    # 手仕舞いは linreg 上抜け。価格目標として linreg をTPに置く
    sl_r = float(p["sl_pct"].value) / 100.0
    none = pd.Series(False, index=df.index)
    return _emit_struct(df, buy.fillna(False), none,
                        c * (1 - sl_r), lr, c * 0, c * 0)


templates.register("tv3s_291_linear_channel", defaults={
    "length": 55.0, "band_pct": -2.0, "hma_len": 400.0, "sl_pct": 2.0,
    "max_hold_bars": 200.0, "lot": 0.1, "sl_pips": 30.0, "tp_pips": 60.0},
    signal_fn=_sig_linear_channel)
