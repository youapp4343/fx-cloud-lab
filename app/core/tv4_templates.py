# -*- coding: utf-8 -*-
"""TradingView catalog 301-400(FX向け)のうち、条件が確定したものの移植。

出所・原文は
  research/tradingview_signals_fx_fourth100_original_capture_20260814/<no>_<name>/
スライスは research/tv_fourth100/slices/<no>.txt
原文と移植の対応表は research/tv_fourth100/対応表.md

規律(tv3系と共通):
  - 原文に書いてあることだけを実装する。無い値は defaults に出して推測固定しない
  - Pineの ta.crossover(a,b) は「前バーで a<=b、現バーで a>b」
  - signal_fn はシフトしない。翌バー執行の shift(1) は engine 側
  - ロールオーバー帯 UTC20-23 のエントリーは落とす(実測でコスト過小評価が判明)

★未来参照により移植しないもの(手順3で判定):
  304 / 329 / 351 / 385  request.security(..., 上位足, lookahead_on) をオフセット無しで使用
  377 / 379              上位足の平均足OHLC 4本すべてを lookahead_on で取得
  → バックテストで知り得ない情報を使うため、検定しても意味がない
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, rsi, sma, stochastic
from app.core.strategy_model import Strategy
from app.core.tv3_templates import (_EXIT_DEFAULTS, _atr_exit, _cooldown,
                                    _cross_dn, _cross_up, _pivot_high, _pivot_low)


# ==========================================================================
# 302 EURUSD Sniper
#     EMA10/20のクロス + EMA50によるトレンド方向の一致。
# ==========================================================================
def _sig_302(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    c = df["close"]
    f = ema(c, max(1, int(p["fast"].value)))
    s = ema(c, max(2, int(p["slow"].value)))
    t = ema(c, max(2, int(p["trend"].value)))
    buy = _cross_up(f, s) & (c > t)
    sell = _cross_dn(f, s) & (c < t)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv4_302_eurusd_sniper", defaults={
    "fast": 10.0, "slow": 20.0, "trend": 50.0, **_EXIT_DEFAULTS}, signal_fn=_sig_302)


# ==========================================================================
# 303 EURUSD M15 Sniper Arrows
#     EMA200の上/下 かつ 直近20本の高値(安値)を更新。素のブレイクアウト。
# ==========================================================================
def _sig_303(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    c, h, l = df["close"], df["high"], df["low"]
    e = ema(c, max(2, int(p["ema_len"].value)))
    n = max(2, int(p["break_len"].value))
    hh = h.rolling(n).max().shift(1)
    ll = l.rolling(n).min().shift(1)
    buy = (c > e) & (c > hh)
    sell = (c < e) & (c < ll)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv4_303_m15_sniper", defaults={
    "ema_len": 200.0, "break_len": 20.0, **_EXIT_DEFAULTS}, signal_fn=_sig_303)


# ==========================================================================
# 307 Forex Trend Master Follower
#     EMA9/21クロス + RSIが逆側の極値でないこと。
# ==========================================================================
def _sig_307(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    c = df["close"]
    f = ema(c, max(1, int(p["fast"].value)))
    s = ema(c, max(2, int(p["slow"].value)))
    r = rsi(c, max(2, int(p["rsi_len"].value)))
    buy = _cross_up(f, s) & (r < float(p["rsi_ob"].value))
    sell = _cross_dn(f, s) & (r > float(p["rsi_os"].value))
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv4_307_trend_master", defaults={
    "fast": 9.0, "slow": 21.0, "rsi_len": 14.0, "rsi_ob": 70.0, "rsi_os": 30.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_307)


# ==========================================================================
# 336 Enhanced Forex Indicator
#     EMA50/200の両方を上抜けている状態 かつ 典型価格の上昇 + 包み足。
#     ★原文の bullishEngulfing は typicalPrice のクロスで定義されている
#       (通常の包み足の定義とは違うが、原文どおり実装する)
# ==========================================================================
def _sig_336(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    es = ema(c, max(2, int(p["ema_short"].value)))
    el = ema(c, max(2, int(p["ema_long"].value)))
    tp = (h + l + c) / 3.0
    bull = _cross_up(tp, tp.shift(1)) & (c > o) & (o.shift(1) > c.shift(1))
    bear = _cross_dn(tp, tp.shift(1)) & (c < o) & (o.shift(1) < c.shift(1))
    buy = (c > es) & (c > el) & bull
    sell = (c < es) & (c < el) & bear
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv4_336_enhanced_fx", defaults={
    "ema_short": 50.0, "ema_long": 200.0, **_EXIT_DEFAULTS}, signal_fn=_sig_336)


# ==========================================================================
# 354 Forex Multi-Factor Indicator
#     ★原文は buy_signal しか定義していない。しかも ta.crossunder(slow, fast)
#       = 遅いMAが速いMAを下抜ける = 実質ゴールデンクロス。RSI・出来高の入力は
#       宣言されているだけで条件に使われていない。原文どおり買いのみ実装する。
# ==========================================================================
def _sig_354(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    c = df["close"]
    f = sma(c, max(1, int(p["fast"].value)))
    s = sma(c, max(2, int(p["slow"].value)))
    buy = _cross_dn(s, f)          # 原文どおり crossunder(slow, fast)
    none = np.zeros(len(df), dtype=bool)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), none)


templates.register("tv4_354_multifactor", defaults={
    "fast": 10.0, "slow": 30.0, **_EXIT_DEFAULTS}, signal_fn=_sig_354)


# ==========================================================================
# 360 Ralph Indicator - ZaraTrust Smart Money
#     直近ピボット高値を終値が上抜け / ピボット安値を下抜け。
# ==========================================================================
def _sig_360(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    n = max(1, int(p["pivot_len"].value))
    res = _pivot_high(h, n, n).ffill()
    sup = _pivot_low(l, n, n).ffill()
    buy = _cross_up(c, res)
    sell = _cross_dn(c, sup)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv4_360_ralph_smc", defaults={
    "pivot_len": 5.0, **_EXIT_DEFAULTS}, signal_fn=_sig_360)


# ==========================================================================
# 372 HINA CVD Pro
#     ★出来高を高値安値内の位置で買い/売りに按分し、その差(デルタ)のZスコア。
#       Zが負に振れた(売られた)のに終値が高値寄り(close_pos>0.70)なら買い。
#       = 「売り圧が出たのに price が耐えた」の吸収パターン。
#     volume は dukascopy/ThreeTrader ともティック数なので、
#     「約定回数の按分」という代理になる(原著の出来高とは別物)。
# ==========================================================================
def _sig_372(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    tr = tr.replace(0, np.nan)
    up = c >= o
    buy_eff = v * np.where(up, (c - l) / tr, (h - o) / tr)
    sell_eff = v * np.where(up, (h - c) / tr, (c - l) / tr)
    delta = pd.Series(buy_eff - sell_eff, index=df.index)
    n = max(5, int(p["z_len"].value))
    z = (delta - sma(delta, n)) / delta.rolling(n).std()
    rng = (h - l).replace(0, np.nan)
    pos = (c - l) / rng
    thr = float(p["z_thres"].value)
    buy = (z < -thr) & (pos > float(p["pos_hi"].value))
    sell = (z > thr) & (pos < float(p["pos_lo"].value))
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv4_372_hina_cvd", defaults={
    "z_len": 50.0, "z_thres": 2.0, "pos_hi": 0.70, "pos_lo": 0.30,
    **_EXIT_DEFAULTS}, signal_fn=_sig_372)


# ==========================================================================
# 346 PCA Projection Pulse
#     価格変化と出来高をそれぞれZ化し、第1主成分方向へ射影。
#     閾値を初めて超えた足(立ち上がり)で入り、クールダウンとポジション状態で抑制。
# ==========================================================================
def _sig_346(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    c, v = df["close"], df["volume"].astype(float)
    n = max(10, int(p["z_len"].value))
    rc = c.diff()
    zc = (rc - sma(rc, n)) / rc.rolling(n).std()
    zv = (v - sma(v, n)) / v.rolling(n).std()
    # 2変量の主成分方向(共分散から)。原文の ex/ey/eMag に相当
    cov = zc.rolling(n).cov(zv)
    var_c = zc.rolling(n).var()
    ex, ey = var_c, cov
    mag = np.sqrt(ex ** 2 + ey ** 2).replace(0, np.nan)
    proj = zc * (ex / mag) + zv * (ey / mag)
    thr = float(p["thr"].value)
    up = proj >= thr
    dn = proj <= -thr
    buy = up & ~up.shift(1).fillna(False)
    sell = dn & ~dn.shift(1).fillna(False)
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv4_346_pca_pulse", defaults={
    "z_len": 50.0, "thr": 2.0, "cooldown": 10.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_346)


# ==========================================================================
# 320 EURUSD 15m
#     ボリンジャー系の内外バンドに対し、押し目戻り(DR)とブレイク(BO)の2系統。
#     トレンド・ADX・出来高・値幅余地(ATR基準)・確認足・クールダウン・
#     アーミング(1回撃ったら反対側に振れるまで再武装しない)で絞る。
# ==========================================================================
def _sig_320(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    from app.core.osc_extra import adx
    p = strategy.params
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].astype(float)
    n = max(5, int(p["bb_len"].value))
    basis = sma(c, n)
    sd = c.rolling(n).std()
    inner_u, inner_l = basis + float(p["inner_mult"].value) * sd, \
        basis - float(p["inner_mult"].value) * sd
    outer_u, outer_l = basis + float(p["outer_mult"].value) * sd, \
        basis - float(p["outer_mult"].value) * sd
    a = atr(h, l, c, 14)
    e = ema(c, max(2, int(p["trend_len"].value)))
    bull, bear = c > e, c < e
    adx_ok = adx(h, l, c, 14) >= float(p["adx_min"].value)
    vol_ok = v > sma(v, 20) * float(p["vol_mult"].value)
    room = float(p["min_room_atr"].value)
    room_l = ((outer_u - c) / a) >= room
    room_s = ((c - outer_l) / a) >= room
    conf_l, conf_s = c > o, c < o

    depth = float(p["dr_depth_atr"].value)
    dr_l = bull & adx_ok & vol_ok & room_l & conf_l & (l <= basis - depth * a) & (c >= basis)
    dr_s = bear & adx_ok & vol_ok & room_s & conf_s & (h >= basis + depth * a) & (c <= basis)
    bo_l = bull & adx_ok & vol_ok & room_l & conf_l & _cross_up(c, inner_u)
    bo_s = bear & adx_ok & vol_ok & room_s & conf_s & _cross_dn(c, inner_l)

    buy = (dr_l | (bo_l & ~dr_l)).fillna(False).to_numpy()
    sell = (dr_s | (bo_s & ~dr_s)).fillna(False).to_numpy()
    cd = int(p["cool_bars"].value)
    return _atr_exit(df, p, _cooldown(buy, cd), _cooldown(sell, cd))


templates.register("tv4_320_eurusd_15m", defaults={
    "bb_len": 20.0, "inner_mult": 1.5, "outer_mult": 2.5, "trend_len": 50.0,
    "adx_min": 20.0, "vol_mult": 1.0, "min_room_atr": 1.0, "dr_depth_atr": 0.5,
    "cool_bars": 10.0, **_EXIT_DEFAULTS}, signal_fn=_sig_320)
