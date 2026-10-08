# -*- coding: utf-8 -*-
"""TradingView catalog 201-300 の A判定31本のうち、条件が確定したものの移植。

出所と原文は
  research/tradingview_signals_third100_original_capture_20260814/<no>_<name>/01_Pineソース原文.pine
依存チェーンのスライスは
  research/tv_third100/slices/<no>.txt

移植の規律:
  - 原文に書いてある条件だけを実装する。書いていない値は defaults に出して推測で固定しない
  - Pineの `ta.crossover(a,b)` は「前バーで a<=b、現バーで a>b」。shift(1)で厳密に再現する
  - signal_fn はシフトしない。翌バー執行の shift(1) は engine.run_backtest 側
  - `barstate.isconfirmed` は確定バーのみを見る当方の実装では常に真なので落とす
  - `volume` は dukascopy の **ティック数**。原文の出来高フィルタは活動量の代理として実装する
    (買い出来高/売り出来高の分離が要るものは移植不可としてこのファイルに入れない)

移植不可(理由を明記):
  210 / 211 Bot Webhook v8.4 [SOL]/[ETH]
      30秒足・45秒足を含む21本のタイムフレーム別条件のOR。分足未満のデータが無く、
      入力243個で総当りも不能。
  242 Pressure Reversal Engine [JOAT]
      ボリュームプロファイルの買い出来高/売り出来高の分離(bidPressure/askPressure)が必須。
      ティック数しか無いので原理的に再現できない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.catalog_templates import _emit
from app.core.indicators import atr, bollinger_bands, ema, macd, rsi, sma, stochastic
from app.core.osc_extra import adx, dmi, linreg_slope
from app.core.strategy_model import Strategy


# ==========================================================================
# 共通ヘルパ
# ==========================================================================
def _cross_up(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """Pine の ta.crossover。前バーで a<=b かつ 現バーで a>b。"""
    prev_a, prev_b = a.shift(1), (b.shift(1) if isinstance(b, pd.Series) else b)
    return (prev_a <= prev_b) & (a > b)


def _cross_dn(a: pd.Series, b: pd.Series | float) -> pd.Series:
    prev_a, prev_b = a.shift(1), (b.shift(1) if isinstance(b, pd.Series) else b)
    return (prev_a >= prev_b) & (a < b)


def _supertrend_dir(df: pd.DataFrame, period: int, mult: float) -> pd.Series:
    """Pine の ta.supertrend が返す direction(-1=上昇トレンド, 1=下降トレンド)。

    Pineの符号は直感と逆。原文の `direction == -1` が強気なので、そのまま合わせる。
    帯のクランプが逐次依存でベクトル化できないため numba でJIT化してある
    (純Python版と完全一致を確認済み。実測533倍)。
    """
    from app.core.tv3_fast import _st_dir_core
    h, l, c = df["high"], df["low"], df["close"]
    a = atr(h, l, c, period)
    mid = ((h + l) / 2.0).to_numpy()
    return pd.Series(_st_dir_core(mid, c.to_numpy(), a.to_numpy(), float(mult)),
                     index=df.index)


def _pivot_high(h: pd.Series, left: int, right: int) -> pd.Series:
    """ta.pivothigh。右側 right 本ぶん遅れて確定する(先読みにならない)。"""
    w = h.rolling(left + right + 1, center=False).max()
    is_p = (h.shift(right) == w) & (h.shift(right).notna())
    return h.shift(right).where(is_p)


def _pivot_low(l: pd.Series, left: int, right: int) -> pd.Series:
    w = l.rolling(left + right + 1, center=False).min()
    is_p = (l.shift(right) == w) & (l.shift(right).notna())
    return l.shift(right).where(is_p)


def _cooldown(sig: np.ndarray, bars: int) -> np.ndarray:
    """直前のシグナルから bars 本空いていないものを落とす。原文のクールダウン再現。"""
    if bars <= 0:
        return sig
    out = np.zeros(len(sig), dtype=bool)
    last = -10 ** 9
    for i in np.flatnonzero(sig):
        if i - last > bars:
            out[i] = True
            last = i
    return out


# dukascopy の bid 由来データはロールオーバー帯で印字が中値から沈む。
# 実測(data/spread_profile_all.csv, 10銘柄×134万本)での自分の中央値に対する倍率:
#   UTC20 median 1.01 / p90 1.19 / p99 1.96   ← 裾が既に膨らむ
#   UTC21 median 3.17 / p90 6.50 / p99 9.85
#   UTC22 median 1.87 / p90 5.34 / p99 8.47
#   UTC23 median 1.19 / p90 1.41 / p99 1.80
# 他の時間は median 0.99〜1.02 で平坦。過去に「生存」した6件は全部この帯だったので
# 無条件で落とす。20時は従来の除外帯に入っていなかったが、p99が2倍近いので含める。
_ROLLOVER_HOURS = (20, 21, 22, 23)


def _atr_exit(df: pd.DataFrame, p, longs: np.ndarray, shorts: np.ndarray) -> pd.DataFrame:
    """原文が決済を定義していないものは ATR 倍で統一する(推測でなくパラメータ化)。

    あわせてロールオーバー帯のシグナルを落とす。backtest_grid.py 側には
    この除外が無いので、テンプレ側で確実に効かせる。
    """
    hour = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
    ok = ~np.isin(hour, _ROLLOVER_HOURS)
    longs = np.asarray(longs, dtype=bool) & ok
    shorts = np.asarray(shorts, dtype=bool) & ok

    a = atr(df["high"], df["low"], df["close"], max(2, int(p["atr_period"].value)))
    dist = (a * float(p["atr_mult"].value)).to_numpy()
    return _emit(len(df), longs, shorts, df["close"].to_numpy(), dist,
                 float(p["rr"].value))


_EXIT_DEFAULTS = {"atr_period": 14.0, "atr_mult": 1.5, "rr": 2.0,
                  "sl_pips": 30.0, "tp_pips": 60.0, "max_hold_bars": 96.0, "lot": 0.1}


# ==========================================================================
# 256 Buy/Sell Indicator  — SMA9/21クロス + RSIが逆側の極値でないこと
# 253 Price Movement + Buy/Sell Volume Dashboard — EMA9/21クロス(+任意で出来高)
# 224 Universal Signal Backtester [LuxAlgo] — 既定は EMA9/21クロス + ATRフィルタ
# ==========================================================================
def _sig_ma_cross(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    fast = max(1, int(p["fast"].value))
    slow = max(2, int(p["slow"].value))
    c = df["close"]
    use_ema = float(p["use_ema"].value) > 0.5
    mf = ema(c, fast) if use_ema else sma(c, fast)
    ms = ema(c, slow) if use_ema else sma(c, slow)
    up, dn = _cross_up(mf, ms), _cross_dn(mf, ms)

    # 256: RSIが反対側の極値に達していないこと
    if float(p["use_rsi_gate"].value) > 0.5:
        r = rsi(c, max(2, int(p["rsi_period"].value)))
        up &= r < float(p["rsi_ob"].value)
        dn &= r > float(p["rsi_os"].value)
    # 253: 出来高確認
    if float(p["use_volume"].value) > 0.5 and "volume" in df:
        v = df["volume"]
        strong = v > sma(v, max(2, int(p["vol_period"].value))) * float(p["vol_mult"].value)
        up &= strong
        dn &= strong
    # 224: ATRが自身の移動平均より上(ボラが出ている時だけ)
    if float(p["use_atr_filter"].value) > 0.5:
        a = atr(df["high"], df["low"], c, 14)
        ok = a > sma(a, max(2, int(p["atr_filter_len"].value)))
        up &= ok
        dn &= ok

    return _atr_exit(df, p, up.fillna(False).to_numpy(), dn.fillna(False).to_numpy())


for _nm, _d in (
    ("tv3_256_ma_cross_rsi", {"fast": 9.0, "slow": 21.0, "use_ema": 0.0,
                              "use_rsi_gate": 1.0, "rsi_period": 14.0,
                              "rsi_ob": 70.0, "rsi_os": 30.0,
                              "use_volume": 0.0, "use_atr_filter": 0.0}),
    ("tv3_253_ema_cross_vol", {"fast": 9.0, "slow": 21.0, "use_ema": 1.0,
                               "use_rsi_gate": 0.0, "rsi_period": 14.0,
                               "rsi_ob": 70.0, "rsi_os": 30.0,
                               "use_volume": 1.0, "use_atr_filter": 0.0}),
    ("tv3_224_lux_universal", {"fast": 9.0, "slow": 21.0, "use_ema": 1.0,
                               "use_rsi_gate": 0.0, "rsi_period": 14.0,
                               "rsi_ob": 70.0, "rsi_os": 30.0,
                               "use_volume": 0.0, "use_atr_filter": 1.0}),
):
    templates.register(_nm, defaults={**_d, "vol_period": 20.0, "vol_mult": 1.0,
                                      "atr_filter_len": 14.0, **_EXIT_DEFAULTS},
                       signal_fn=_sig_ma_cross)


# ==========================================================================
# 209 Volume Flow Indicator Signals / iSolani
#     VFI のゼロラインクロス。原文は smoothVFI=true, vcoef=3 固定。
# ==========================================================================
def _sig_vfi(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(2, int(p["length"].value))
    coef = float(p["coef"].value)
    typ = (df["high"] + df["low"] + df["close"]) / 3.0
    inter = np.log(typ) - np.log(typ.shift(1))
    vinter = inter.rolling(30).std()
    cutoff = coef * vinter * df["close"]
    v = df["volume"].astype(float)
    vave = sma(v, n)
    vmax = vave * 3.0
    vc = np.minimum(v, vmax)
    mf = typ - typ.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0))
    vfi = sma(pd.Series(vcp, index=df.index).rolling(n).sum() / vave, 3)
    up = (vfi > 0) & (vfi.shift(1) <= 0)
    dn = (vfi < 0) & (vfi.shift(1) >= 0)
    return _atr_exit(df, p, up.fillna(False).to_numpy(), dn.fillna(False).to_numpy())


templates.register("tv3_209_vfi_zero", defaults={"length": 50.0, "coef": 4.0,
                                                 **_EXIT_DEFAULTS},
                   signal_fn=_sig_vfi)


# ==========================================================================
# 234 Supertrend - Volume Confirmed Signal
#     方向転換の「後」confirmWindow本以内で、出来高が平均以上、実体が方向どおり。
#     原文は転換バー自体を除外する(not dirChanged_toLong)。
# ==========================================================================
def _sig_st_volconf(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    d = _supertrend_dir(df, max(1, int(p["atr_period"].value)), float(p["factor"].value))
    win = max(2, int(p["confirm_window"].value))
    to_long = (d == -1) & (d.shift(1) == 1)
    to_short = (d == 1) & (d.shift(1) == -1)
    # 転換から win 本以内(転換バー自身は除く)
    in_long = to_long.shift(1).fillna(False)
    in_short = to_short.shift(1).fillna(False)
    for k in range(2, win + 1):
        in_long |= to_long.shift(k).fillna(False)
        in_short |= to_short.shift(k).fillna(False)
    in_long &= (d == -1)
    in_short &= (d == 1)

    v = df["volume"].astype(float)
    hv = v >= sma(v, max(2, int(p["vol_lookback"].value))) * float(p["vol_mult"].value)
    bull = df["close"] >= df["open"]
    up = in_long & bull & hv & ~to_long
    dn = in_short & ~bull & hv & ~to_short
    return _atr_exit(df, p, up.fillna(False).to_numpy(), dn.fillna(False).to_numpy())


templates.register("tv3_234_st_volconf", defaults={
    "atr_period": 3.0, "factor": 2.0, "vol_lookback": 20.0, "vol_mult": 1.0,
    "confirm_window": 2.0, **{k: v for k, v in _EXIT_DEFAULTS.items() if k != "atr_period"},
    "exit_atr_period": 14.0}, signal_fn=_sig_st_volconf)


# ==========================================================================
# 229 / 230 SuperTrend LONG(SHORT) Only + EMA/ADX フィルタ
#     原文は片方向のみ。filterMode の既定は "NONE"。
# ==========================================================================
def _sig_st_filtered(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    d = _supertrend_dir(df, max(1, int(p["atr_len"].value)), float(p["mult"].value))
    c = df["close"]
    e = ema(c, max(2, int(p["ema_len"].value)))
    a = adx(df["high"], df["low"], c, max(2, int(p["adx_len"].value)))
    mode = int(round(float(p["filter_mode"].value)))   # 0=NONE 1=EMA 2=ADX 3=EMA+ADX
    ema_on, adx_on = mode in (1, 3), mode in (2, 3)
    adx_pass = (a >= float(p["adx_min"].value)) if adx_on else pd.Series(True, index=df.index)

    side = int(round(float(p["side"].value)))          # 1=long only, -1=short only
    if side > 0:
        flip = (d == -1) & (d.shift(1) == 1)
        gate = (c > e) if ema_on else pd.Series(True, index=df.index)
        up = flip & gate & adx_pass
        dn = pd.Series(False, index=df.index)
    else:
        flip = (d == 1) & (d.shift(1) == -1)
        gate = (c < e) if ema_on else pd.Series(True, index=df.index)
        dn = flip & gate & adx_pass
        up = pd.Series(False, index=df.index)
    return _atr_exit(df, p, up.fillna(False).to_numpy(), dn.fillna(False).to_numpy())


for _nm, _side in (("tv3_229_st_long_only", 1.0), ("tv3_230_st_short_only", -1.0)):
    templates.register(_nm, defaults={
        "atr_len": 10.0, "mult": 3.0, "filter_mode": 0.0, "ema_len": 200.0,
        "adx_len": 14.0, "adx_min": 18.0, "side": _side, **_EXIT_DEFAULTS},
        signal_fn=_sig_st_filtered)


# ==========================================================================
# 267 Wick Reversal - Gavi
#     下ヒゲが実体の wickMult 倍以上、かつ上side の残りが全幅の bodyPct 以内。
# ==========================================================================
def _sig_wick_rev(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    wm = float(p["wick_mult"].value)
    bp = float(p["body_pct"].value)
    n = max(2, int(p["avg_len"].value))
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    rng_avg = (atr(h, l, c, n) if float(p["use_atr"].value) > 0.5 else sma(rng, n))
    eq = (c - o).abs() <= (rng.rolling(200).median() * 1e-4).fillna(0) + 1e-12

    long1 = (c > o) & ((o - l) >= (c - o) * wm) & ((h - c) <= rng * bp)
    long2 = (c < o) & ((c - l) >= (o - c) * wm) & ((h - c) <= rng * bp)
    long3 = eq & (c != h) & (rng >= (h - c) * wm) & ((h - c) <= rng * bp)
    long4 = (o == h) & (c == h) & (rng >= rng_avg)
    short1 = (c < o) & ((h - o) >= (o - c) * wm) & ((c - l) <= rng * bp)
    short2 = (c > o) & ((h - c) >= (c - o) * wm) & ((c - l) <= rng * bp)
    short3 = eq & (c != l) & (rng >= (c - l) * wm) & ((c - l) <= rng * bp)
    short4 = (o == l) & (c == l) & (rng >= rng_avg)

    up = long1 | long2 | long3 | long4
    dn = short1 | short2 | short3 | short4
    return _atr_exit(df, p, up.fillna(False).to_numpy(), dn.fillna(False).to_numpy())


templates.register("tv3_267_wick_reversal", defaults={
    "wick_mult": 2.5, "body_pct": 0.25, "avg_len": 50.0, "use_atr": 0.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_wick_rev)


# ==========================================================================
# 244 10 in 1 MAs + Dashboard + Buy/Sell v6
#     多段トレンドゲート + 6項目スコア>=5。原文は全部ハードコード(入力0個)。
# ==========================================================================
def _sig_10in1(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    ma1, ma3 = ema(c, 200), ema(c, 50)
    ma4, ma5 = ema(c, 1500), ema(c, 250)
    ma6, ma7 = ema(c, 21), ema(c, 42)
    a = atr(h, l, c, 14)
    r = rsi(c, 14)
    vavg = sma(v, 21)
    ml, sl, hist = macd(c, 12, 26, 9)

    trend_gate = (c > ma4) & (c > ma5) & (c > ma1) & (ma1 > ma1.shift(5)) \
        & (c > ma3) & (ma3 > ma3.shift(3))
    score = (_cross_up(ma6, ma7).astype(int)
             + ((ml > sl) & (hist > hist.shift(1))).astype(int)
             + ((r > 50) & (r < 72)).astype(int)
             + (v > vavg * 1.3).astype(int)
             + ((c > o) & (c - o > a * 0.4)).astype(int)
             + ((c > c.shift(1)) & (c.shift(1) > c.shift(2))).astype(int))
    buy = trend_gate & (score >= float(p["min_score"].value))

    hard_sell = (c < ma1) | (c < ma5)
    soft = (_cross_dn(ma6, ma7).astype(int) * 2
            + ((ml < sl) & (hist < hist.shift(1))).astype(int)
            + (r > 78).astype(int)
            + ((v > vavg * 1.5) & (c < o) & (c < c.shift(1))).astype(int))
    sell = hard_sell | (soft >= 3)
    # 原文は plotshape で立ち上がりだけ出す
    buy = buy & ~buy.shift(1).fillna(False)
    sell = sell & ~sell.shift(1).fillna(False)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_244_10in1", defaults={"min_score": 5.0, **_EXIT_DEFAULTS},
                   signal_fn=_sig_10in1)


# ==========================================================================
# 257 Quant Reversal Index [AlgoPoint]
#     Hurst<0.5(平均回帰レジーム)のときだけ、OU帯の 0/100 クロスで入る。
# ==========================================================================
def _sig_quant_reversal(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    n = max(20, int(p["quant_len"].value))
    mult = float(p["ou_mult"].value)
    hthr = float(p["hurst"].value)
    src = df["close"]

    var1 = (src - src.shift(2)).rolling(n).var()
    var2 = (src - src.shift(8)).rolling(n).var()
    hurst = (np.log(var2) - np.log(var1)) / (2 * (np.log(8) - np.log(2)))
    mr = hurst < hthr

    dp = src.diff()
    pp = src.shift(1)
    cov = ((dp - sma(dp, n)) * (pp - sma(pp, n))).rolling(n).mean()
    lam = cov / pp.rolling(n).var()
    hl = (-np.log(2) / lam).where(lam < 0)
    hl_len = hl.round().clip(2, 500).fillna(20).astype(int)

    # 半減期は可変長。実装は上位いくつかの固定長で近似せず、逐次で正確に出す
    s = src.to_numpy()
    n_all = len(s)
    mu = np.full(n_all, np.nan)
    sg = np.full(n_all, np.nan)
    hl_a = hl_len.to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(s)])
    csq = np.concatenate([[0.0], np.cumsum(s * s)])
    for i in range(n_all):
        w = int(hl_a[i])
        if i + 1 < w or w < 2:
            continue
        a0, a1 = i + 1 - w, i + 1
        m = (csum[a1] - csum[a0]) / w
        q = (csq[a1] - csq[a0]) / w - m * m
        mu[i] = m
        sg[i] = np.sqrt(max(q, 0.0)) * np.sqrt(w / (w - 1))
    mu_s = pd.Series(mu, index=df.index)
    sg_s = pd.Series(sg, index=df.index)
    up_b, lo_b = mu_s + mult * sg_s, mu_s - mult * sg_s
    qi = ((src - lo_b) / (up_b - lo_b)) * 100.0

    buy = _cross_up(qi, 0.0) & mr
    sell = _cross_dn(qi, 100.0) & mr
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_257_quant_reversal", defaults={
    "quant_len": 100.0, "ou_mult": 2.0, "hurst": 0.5, **_EXIT_DEFAULTS},
    signal_fn=_sig_quant_reversal)


# ==========================================================================
# 287 IDM + IFVG Sniper Entry (VASWANI)
#     EMAトレンド + 直近スイングの誘導(IDM)またはヒゲ拒否 + FVGゾーンタッチ。
# ==========================================================================
def _sig_idm_ifvg(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ef = ema(c, max(2, int(p["ema_fast"].value)))
    es = ema(c, max(2, int(p["ema_slow"].value)))
    sw = max(2, int(p["swing_len"].value))

    bear_tr = (c < ef) & (ef < es) & (ef < ef.shift(1))
    bull_tr = (c > ef) & (ef > es) & (ef > ef.shift(1))
    rec_hi = h.shift(1).rolling(sw).max()
    rec_lo = l.shift(1).rolling(sw).min()
    idm_sell = (h > rec_hi) & (c < rec_hi)
    idm_buy = (l < rec_lo) & (c > rec_lo)

    # FVG: 3本前高値 < 現安値(強気ギャップ)/ 3本前安値 > 現高値(弱気ギャップ)
    bull_gap = l > h.shift(2)
    bear_gap = h < l.shift(2)
    bull_top = l.where(bull_gap).ffill()
    bull_bot = h.shift(2).where(bull_gap).ffill()
    bear_top = l.shift(2).where(bear_gap).ffill()
    bear_bot = h.where(bear_gap).ffill()

    sell_touch = (h >= bear_bot) & (l <= bear_top)
    buy_touch = (l <= bull_top) & (h >= bull_bot)

    body = (c - o).abs()
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    bear_rej = (c < o) & (uw > body * 0.7)
    bull_rej = (c > o) & (lw > body * 0.7)
    sell_weak = (c < bear_bot) | (c < ef)
    buy_strong = (c > bull_top) | (c > ef)

    sell = bear_tr & sell_touch & sell_weak & (idm_sell | bear_rej)
    buy = bull_tr & buy_touch & buy_strong & (idm_buy | bull_rej)
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_287_idm_ifvg", defaults={
    "ema_fast": 50.0, "ema_slow": 200.0, "swing_len": 10.0, "cooldown": 15.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_idm_ifvg)


# ==========================================================================
# 247 Reversal Radar (ConfluenceJP)
#     ダイバージェンス / 反転足 / BB+RSI の3手掛かりのうち needConfs 個以上。
# ==========================================================================
def _sig_reversal_radar(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    r = rsi(c, max(2, int(p["rsi_len"].value)))
    ob, os_ = float(p["rsi_ob"].value), float(p["rsi_os"].value)
    up_b, mid_b, lo_b = bollinger_bands(c, max(5, int(p["bb_len"].value)),
                                        float(p["bb_mult"].value))
    ma = sma(c, max(2, int(p["ma_len"].value)))

    body = (c - o).abs()
    rng = h - l
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    small = body <= rng * 0.35
    hammer = small & (lw >= rng * 0.5) & (c > l)
    star = small & (uw >= rng * 0.5) & (c < h)
    bull_eng = (c > o) & (o <= c.shift(1)) & (o.shift(1) > c.shift(1)) & (c >= o.shift(1))
    bear_eng = (c < o) & (o >= c.shift(1)) & (o.shift(1) < c.shift(1)) & (c <= o.shift(1))

    lb, rb = max(1, int(p["pivot_left"].value)), max(1, int(p["pivot_right"].value))
    ph, pl = _pivot_high(h, lb, rb), _pivot_low(l, lb, rb)
    ph_rsi = r.shift(rb).where(ph.notna())
    pl_rsi = r.shift(rb).where(pl.notna())
    ph_v, ph_prev = ph.ffill(), ph.ffill().shift(1).where(ph.notna()).ffill()
    pl_v, pl_prev = pl.ffill(), pl.ffill().shift(1).where(pl.notna()).ffill()
    phr, phr_prev = ph_rsi.ffill(), ph_rsi.ffill().shift(1).where(ph.notna()).ffill()
    plr, plr_prev = pl_rsi.ffill(), pl_rsi.ffill().shift(1).where(pl.notna()).ffill()
    bear_div = ph.notna() & (ph_v > ph_prev) & (phr < phr_prev)
    bull_div = pl.notna() & (pl_v < pl_prev) & (plr > plr_prev)

    need = float(p["need_confs"].value)
    bull_n = (bull_div.astype(int) + (hammer | bull_eng).astype(int)
              + ((l <= lo_b) | ((r.shift(1) < os_) & (r > os_))).astype(int))
    bear_n = (bear_div.astype(int) + (star | bear_eng).astype(int)
              + ((h >= up_b) | ((r.shift(1) > ob) & (r < ob))).astype(int))

    if float(p["use_trend"].value) > 0.5:
        bull_ok = (ma < ma.shift(1)) | ((c < ma) & ((ma - c) / ma > 0.01))
        bear_ok = (ma > ma.shift(1)) | ((c > ma) & ((c - ma) / ma > 0.01))
    else:
        bull_ok = bear_ok = pd.Series(True, index=df.index)

    buy = bull_ok & (bull_n >= need)
    sell = bear_ok & (bear_n >= need)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_247_reversal_radar", defaults={
    "rsi_len": 14.0, "rsi_ob": 70.0, "rsi_os": 30.0, "bb_len": 20.0, "bb_mult": 2.0,
    "ma_len": 50.0, "use_trend": 1.0, "pivot_left": 3.0, "pivot_right": 3.0,
    "need_confs": 2.0, **_EXIT_DEFAULTS}, signal_fn=_sig_reversal_radar)


# ==========================================================================
# 250 Possible Reversal Zone Detector
#     EMAチャネル外から内側へ戻るクロス + 任意フィルタ。原文の既定は出来高のみON。
# ==========================================================================
def _sig_rev_zone(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    n = max(1, int(p["length"].value))
    basis = ema(c, n)
    dev = float(p["mult"].value) * c.rolling(n).std()
    up_b, lo_b = basis + dev, basis - dev

    base_long = _cross_up(c, lo_b) & (l.shift(1) < lo_b.shift(1))
    base_short = _cross_dn(c, up_b) & (h.shift(1) > up_b.shift(1))

    if float(p["use_rsi"].value) > 0.5:
        r = rsi(c, max(2, int(p["rsi_len"].value)))
        base_long &= r <= float(p["rsi_os"].value)
        base_short &= r >= float(p["rsi_ob"].value)
    if float(p["use_vol"].value) > 0.5:
        ok = v >= sma(v, max(2, int(p["vol_len"].value))) * float(p["vol_mult"].value)
        base_long &= ok
        base_short &= ok
    if float(p["use_pinbar"].value) > 0.5:
        body = (c - o).abs()
        uw = h - np.maximum(c, o)
        lw = np.minimum(c, o) - l
        base_long &= (lw > body * 2) & (uw < body)
        base_short &= (uw > body * 2) & (lw < body)

    return _atr_exit(df, p, base_long.fillna(False).to_numpy(),
                     base_short.fillna(False).to_numpy())


templates.register("tv3_250_rev_zone", defaults={
    "length": 20.0, "mult": 1.5, "use_rsi": 0.0, "rsi_len": 10.0,
    "rsi_ob": 55.0, "rsi_os": 35.0, "use_vol": 1.0, "vol_len": 15.0,
    "vol_mult": 1.0, "use_pinbar": 0.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_rev_zone)


# ==========================================================================
# 251 Institutional Early Entry OB Signals
#     BOS(構造ブレイク)で作られたオーダーブロックへの再タッチ + EMA/RSI/Stoch。
# ==========================================================================
def _sig_ob_early(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    ef = ema(c, max(2, int(p["ema_fast"].value)))
    es = ema(c, max(2, int(p["ema_slow"].value)))
    r = rsi(c, max(2, int(p["rsi_len"].value)))
    k, d_ = stochastic(h, l, c, max(2, int(p["stoch_len"].value)),
                       max(1, int(p["smooth_k"].value)), max(1, int(p["smooth_d"].value)))

    sl_ = max(2, int(p["structure_len"].value))
    bull_bos = c > h.shift(1).rolling(sl_).max()
    bear_bos = c < l.shift(1).rolling(sl_).min()

    last_bear_hi = h.where(c < o).ffill()
    last_bear_lo = l.where(c < o).ffill()
    last_bull_hi = h.where(c > o).ffill()
    last_bull_lo = l.where(c > o).ffill()
    bull_ob_hi = last_bear_hi.where(bull_bos).ffill()
    bull_ob_lo = last_bear_lo.where(bull_bos).ffill()
    bear_ob_hi = last_bull_hi.where(bear_bos).ffill()
    bear_ob_lo = last_bull_lo.where(bear_bos).ffill()

    valid = int(p["ob_valid_bars"].value)
    since_bull = (~bull_bos).cumsum() - (~bull_bos).cumsum().where(bull_bos).ffill()
    since_bear = (~bear_bos).cumsum() - (~bear_bos).cumsum().where(bear_bos).ffill()
    bull_active = (since_bull <= valid) & (since_bull < since_bear.fillna(1e9))
    bear_active = (since_bear <= valid) & (since_bear < since_bull.fillna(1e9))

    in_bull = bull_active & (l <= bull_ob_hi) & (h >= bull_ob_lo)
    in_bear = bear_active & (h >= bear_ob_lo) & (l <= bear_ob_hi)
    bull_mid = (bull_ob_hi + bull_ob_lo) / 2
    bear_mid = (bear_ob_hi + bear_ob_lo) / 2

    mode = int(round(float(p["mode"].value)))   # 0=Aggressive 1=Balanced 2=Conservative
    if mode == 0:
        ema_bull, ema_bear = c > ef, c < ef
    elif mode == 1:
        ema_bull, ema_bear = (ef > es) & (c > ef), (ef < es) & (c < ef)
    else:
        ema_bull = (ef > es) & (c > ef) & (ef > ef.shift(1))
        ema_bear = (ef < es) & (c < ef) & (ef < ef.shift(1))

    vol_ok = (v > sma(v, max(2, int(p["vol_len"].value))) * float(p["vol_mult"].value)
              if float(p["use_volume"].value) > 0.5 else pd.Series(True, index=df.index))
    stoch_bull = ((k > d_) | (k > 50)) if float(p["use_stoch"].value) > 0.5 \
        else pd.Series(True, index=df.index)
    stoch_bear = ((k < d_) | (k < 50)) if float(p["use_stoch"].value) > 0.5 \
        else pd.Series(True, index=df.index)

    buy = in_bull & ((c > o) | (c > bull_mid)) & ema_bull & (r > float(p["rsi_buy"].value)) \
        & stoch_bull & vol_ok
    sell = in_bear & ((c < o) | (c < bear_mid)) & ema_bear & (r < float(p["rsi_sell"].value)) \
        & stoch_bear & vol_ok
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_251_ob_early", defaults={
    "ema_fast": 21.0, "ema_slow": 50.0, "rsi_len": 14.0, "rsi_buy": 50.0,
    "rsi_sell": 50.0, "stoch_len": 14.0, "smooth_k": 3.0, "smooth_d": 3.0,
    "structure_len": 10.0, "ob_valid_bars": 20.0, "mode": 1.0,
    "use_volume": 1.0, "vol_len": 20.0, "vol_mult": 1.0, "use_stoch": 1.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_ob_early)


# ==========================================================================
# 272 / 280 GerardMali Smart Buy Sell(MultiConfluence / PRO v3)
#     7ゲート全通過 + ボーナススコア>=min + 4(5)トリガーのいずれか + クールダウン。
#     280は280固有のS/R条件が加わるが、ゲートとトリガーの骨格は共通なので
#     bonus_mode で分岐させる。
# ==========================================================================
def _sig_gerard(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    e1, e2 = ema(c, 9), ema(c, 21)
    e3, e4 = ema(c, 50), ema(c, 200)
    r = rsi(c, 14)
    ml, sg, hist = macd(c, 12, 26, 9)
    k, d_ = stochastic(h, l, c, 14, 3, 3)
    a = adx(h, l, c, 14)
    dip, dim = dmi(h, l, c, 14)
    adx_min = float(p["adx_min"].value)
    ext = float(p["ext_pct"].value)

    e1_rise, e2_rise = e1 > e1.shift(1), e2 > e2.shift(1)
    e1_fall, e2_fall = e1 < e1.shift(1), e2 < e2.shift(1)
    upper_close = c >= (l + (h - l) * 0.45)
    lower_close = c <= (h - (h - l) * 0.45)
    dist200 = (c - e4).abs() / e4 * 100
    over_up = (c > e4) & (dist200 > ext)
    over_dn = (c < e4) & (dist200 > ext)
    bear_div = (c > c.shift(14)) & (r < r.shift(14)) & (r > 55)
    bull_div = (c < c.shift(14)) & (r > r.shift(14)) & (r < 45)

    buy_gates = ((a > adx_min) & e1_rise & e2_rise & (r > 38) & (r < 72)
                 & (hist > hist.shift(1)) & (hist > 0) & upper_close & ~over_up & ~bear_div)
    sell_gates = ((a > adx_min) & e1_fall & e2_fall & (r < 62) & (r > 28)
                  & (hist < hist.shift(1)) & (hist < 0) & lower_close & ~over_dn & ~bull_div)

    bull_bonus = ((e1 > e2).astype(int) + (e2 > e3).astype(int) + (c > e4).astype(int)
                  + (r > r.shift(3)).astype(int)
                  + (_cross_up(k, d_) & (k.shift(1) < 30)).astype(int)
                  + (dip > dim).astype(int) + _cross_up(r, 50.0).astype(int))
    bear_bonus = ((e1 < e2).astype(int) + (e2 < e3).astype(int) + (c < e4).astype(int)
                  + (r < r.shift(3)).astype(int)
                  + (_cross_dn(k, d_) & (k.shift(1) > 70)).astype(int)
                  + (dim > dip).astype(int) + _cross_dn(r, 50.0).astype(int))

    buy_trig = (_cross_up(ml, sg) & (hist.shift(1) < 0)) | (_cross_up(k, d_) & (k.shift(1) < 35)) \
        | (_cross_up(r, 50.0) & (r.shift(2) < 44)) | (_cross_up(e1, e2) & e2_rise & (a > 22))
    sell_trig = (_cross_dn(ml, sg) & (hist.shift(1) > 0)) | (_cross_dn(k, d_) & (k.shift(1) > 65)) \
        | (_cross_dn(r, 50.0) & (r.shift(2) > 56)) | (_cross_dn(e1, e2) & e2_fall & (a > 22))

    mb = float(p["min_bonus"].value)
    buy = buy_gates & (bull_bonus >= mb) & buy_trig
    sell = sell_gates & (bear_bonus >= mb) & sell_trig
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


for _nm, _mb in (("tv3_272_gerard_confluence", 4.0), ("tv3_280_gerard_pro", 5.0)):
    templates.register(_nm, defaults={
        "adx_min": 20.0, "ext_pct": 3.0, "min_bonus": _mb, "cooldown": 10.0,
        **_EXIT_DEFAULTS}, signal_fn=_sig_gerard)


# ==========================================================================
# 219 Breakout Odds [Rider Algo]
#     BB幅圧縮 / ATR圧縮 / 相対出来高 / 端までの距離 / 水準テスト の加重スコアが
#     閾値超え + 方向判定 + EMAトレンド許可 + 直近足の確認 + クールダウン。
# ==========================================================================
def _sig_breakout_odds(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    lb = max(10, int(p["lookback"].value))
    up_b, mid_b, lo_b = bollinger_bands(c, max(5, int(p["bb_len"].value)),
                                        float(p["bb_mult"].value))
    bbw = (up_b - lo_b) / mid_b
    bb_sc = 100.0 * (1.0 - (bbw - bbw.rolling(lb).min())
                     / (bbw.rolling(lb).max() - bbw.rolling(lb).min()))
    a = atr(h, l, c, max(5, int(p["atr_len"].value)))
    atr_sc = 100.0 * (1.0 - (a - a.rolling(lb).min())
                      / (a.rolling(lb).max() - a.rolling(lb).min()))
    vr = v / sma(v, max(5, int(p["vol_len"].value)))
    vol_sc = (vr * 50.0).clip(0, 100)
    hi, lo = h.rolling(lb).max(), l.rolling(lb).min()
    pos = (c - lo) / (hi - lo)
    dist_sc = 100.0 * (1.0 - (pos - 0.5).abs() * 2)
    tol = float(p["sr_tol"].value) / 100.0
    tests = (((h >= hi * (1 - tol)).rolling(lb).sum()
              + (l <= lo * (1 + tol)).rolling(lb).sum()) * 10).clip(0, 100)

    wsum = (float(p["w_bb"].value) + float(p["w_atr"].value) + float(p["w_vol"].value)
            + float(p["w_dist"].value) + float(p["w_tests"].value))
    comp = (bb_sc * float(p["w_bb"].value) + atr_sc * float(p["w_atr"].value)
            + vol_sc * float(p["w_vol"].value) + dist_sc * float(p["w_dist"].value)
            + tests * float(p["w_tests"].value)) / wsum

    mom = linreg_slope(c, max(3, int(p["momentum_len"].value)))
    body_bias = (c - o).rolling(max(3, int(p["candle_bias_len"].value))).mean()
    is_bull = (pos > 0.5) & (mom > 0) & (body_bias > 0)
    is_bear = (pos < 0.5) & (mom < 0) & (body_bias < 0)

    e = ema(c, max(50, int(p["trend_ema"].value)))
    tr_bull = (c > e) if float(p["use_trend"].value) > 0.5 else pd.Series(True, index=df.index)
    tr_bear = (c < e) if float(p["use_trend"].value) > 0.5 else pd.Series(True, index=df.index)

    thr = float(p["threshold"].value)
    fired = _cross_up(comp, thr) | (comp >= thr)
    buy = fired & is_bull & tr_bull & (c > o)
    sell = fired & is_bear & tr_bear & (c < o)
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_219_breakout_odds", defaults={
    "lookback": 50.0, "threshold": 70.0, "bb_len": 20.0, "bb_mult": 2.0,
    "atr_len": 14.0, "vol_len": 20.0, "sr_tol": 0.5, "w_bb": 25.0, "w_atr": 20.0,
    "w_vol": 25.0, "w_dist": 15.0, "w_tests": 15.0, "momentum_len": 10.0,
    "use_trend": 1.0, "trend_ema": 200.0, "candle_bias_len": 10.0, "cooldown": 10.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_breakout_odds)


# ==========================================================================
# 205 Smart Signal - BUY SELL with Targets-VJ
#     tradeMode で3系統。既定は "Short Term"。
# ==========================================================================
def _sig_smart_vj(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    ef = ema(c, max(2, int(p["ema_fast"].value)))
    es = ema(c, max(2, int(p["ema_slow"].value)))
    tr = ema(c, max(2, int(p["trend_len"].value)))
    r = rsi(c, 14)
    ml, sg, _ = macd(c, 12, 26, 9)
    k, d_ = stochastic(h, l, c, 14, 3, 3)
    rng = (h - l).replace(0, np.nan)
    body_pct = (c - o).abs() / rng
    min_body = float(p["min_body"].value)
    vol_spike = v > sma(v, 20) * float(p["vol_mult"].value)
    above, below = c > tr, c < tr

    bull_c = (c > o) & (body_pct >= min_body)
    bear_c = (c < o) & (body_pct >= min_body)
    ema_up, ema_dn = _cross_up(ef, es), _cross_dn(ef, es)
    macd_up, macd_dn = _cross_up(ml, sg), _cross_dn(ml, sg)
    st_up = _cross_up(k, d_) & (k < 20)
    st_dn = _cross_dn(k, d_) & (k > 80)
    rsi_rec = _cross_up(r, float(p["rsi_os"].value))
    rsi_fall = _cross_dn(r, float(p["rsi_ob"].value))

    mode = int(round(float(p["trade_mode"].value)))   # 0=Scalping 1=Short 2=Long
    if mode == 0:
        buy = bull_c & (ema_up | rsi_rec | st_up) & vol_spike & above
        sell = bear_c & (ema_dn | rsi_fall | st_dn) & vol_spike & below
    elif mode == 1:
        buy = (ema_up | macd_up) & (r < 60) & bull_c & vol_spike & above
        sell = (ema_dn | macd_dn) & (r > 40) & bear_c & vol_spike & below
    else:
        buy = ema_up & macd_up & (r > 40) & (r < 65) & above & vol_spike
        sell = ema_dn & macd_dn & (r < 60) & (r > 35) & below & vol_spike
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_205_smart_vj", defaults={
    "ema_fast": 9.0, "ema_slow": 21.0, "trend_len": 50.0, "trade_mode": 1.0,
    "min_body": 0.3, "vol_mult": 1.2, "rsi_ob": 70.0, "rsi_os": 30.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_smart_vj)


# ==========================================================================
# 239 APEX Trend & Signal Engine [Viprasol]
#     レジーム判定でトレンド追随(TF)/平均回帰(MR)を切替。一目の雲で確認。
# ==========================================================================
def _sig_apex(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    base = ema(c, max(2, int(p["baseline_len"].value)))
    a = adx(h, l, c, 14)
    dip, dim = dmi(h, l, c, 14)
    trending = a >= float(p["adx_trend"].value)

    n = max(5, int(p["ext_len"].value))
    exb, exs = sma(c, n), c.rolling(n).std()
    u1, l1 = exb + 1.5 * exs, exb - 1.5 * exs

    ten = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kij = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa = ((ten + kij) / 2).shift(26)
    sb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    above_cloud = c > np.maximum(sa, sb)
    below_cloud = c < np.minimum(sa, sb)

    hi, lo = h.rolling(n).max(), l.rolling(n).min()
    eq = (hi + lo) / 2
    disc, prem = c < eq, c > eq
    use_loc = float(p["use_location"].value) > 0.5
    loc_l = disc if use_loc else pd.Series(True, index=df.index)
    loc_s = prem if use_loc else pd.Series(True, index=df.index)

    mode = int(round(float(p["mode"].value)))   # 0=Auto 1=TF 2=MR
    tf_on = (trending if mode == 0 else (mode == 1))
    mr_on = (~trending if mode == 0 else (mode == 2))
    if isinstance(tf_on, bool):
        tf_on = pd.Series(tf_on, index=df.index)
    if isinstance(mr_on, bool):
        mr_on = pd.Series(mr_on, index=df.index)

    tf_long = tf_on & _cross_up(c, base) & (base > base.shift(1)) & (dip > dim) & trending & loc_l
    tf_short = tf_on & _cross_dn(c, base) & (base < base.shift(1)) & (dim > dip) & trending & loc_s
    mr_long = mr_on & ~trending & (l <= l1) & _cross_up(c, l1) & loc_l
    mr_short = mr_on & ~trending & (h >= u1) & _cross_dn(c, u1) & loc_s

    buy = (tf_long | mr_long)
    sell = (tf_short | mr_short)
    if float(p["use_cloud"].value) > 0.5:
        buy &= above_cloud
        sell &= below_cloud
    cd = int(p["sig_gap"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_239_apex", defaults={
    "baseline_len": 20.0, "adx_trend": 20.0, "ext_len": 20.0, "mode": 0.0,
    "use_location": 1.0, "use_cloud": 1.0, "sig_gap": 5.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_apex)


# ==========================================================================
# 206 Buy-Sell with Liquidity Breakout PRO
#     直近ピボットの水平線をATRバッファ込みで初めて抜けた足。同じ水準は1回だけ。
# ==========================================================================
def _sig_liq_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"].astype(float)
    lb, rb = max(1, int(p["left_len"].value)), max(1, int(p["right_len"].value))
    ph = _pivot_high(h, lb, rb).ffill()
    pl = _pivot_low(l, lb, rb).ffill()
    a = atr(h, l, c, max(2, int(p["atr_len"].value)))
    buf = a * float(p["atr_buf_mult"].value) if float(p["use_atr_buf"].value) > 0.5 else 0.0

    vol_ok = (v > sma(v, max(2, int(p["vol_len"].value))) * float(p["vol_mult"].value)
              if float(p["use_vol"].value) > 0.5 else pd.Series(True, index=df.index))
    bull_raw = (c > ph + buf) & (c.shift(1) <= ph.shift(1) + (buf.shift(1) if
                                                             isinstance(buf, pd.Series) else 0))
    bear_raw = (c < pl - buf) & (c.shift(1) >= pl.shift(1) - (buf.shift(1) if
                                                             isinstance(buf, pd.Series) else 0))
    # 同じ水準を2回叩かない(原文の brokenHighLevel != lastMajorHigh)
    fired_hi = ph.where(bull_raw).ffill()
    fired_lo = pl.where(bear_raw).ffill()
    buy = bull_raw & ((fired_hi.shift(1).isna()) | (fired_hi.shift(1) != ph))
    sell = bear_raw & ((fired_lo.shift(1).isna()) | (fired_lo.shift(1) != pl))
    buy &= vol_ok
    sell &= vol_ok
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_206_liq_breakout", defaults={
    "left_len": 10.0, "right_len": 10.0, "atr_len": 14.0, "use_atr_buf": 1.0,
    "atr_buf_mult": 0.25, "use_vol": 1.0, "vol_len": 20.0, "vol_mult": 1.2,
    **_EXIT_DEFAULTS}, signal_fn=_sig_liq_breakout)


# ==========================================================================
# 289 Zero to Hero + Buy/Sell Labels
#     6項目のスコアを 0-100 に丸めて閾値。原文の shortSignal は
#     「bearHeroScore <= heroShortThresh」で、名前と向きが逆に見えるがそのまま実装する。
# ==========================================================================
def _sig_zero_to_hero(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"].astype(float)
    ef = ema(c, max(2, int(p["ema_fast"].value)))
    es = ema(c, max(2, int(p["ema_slow"].value)))
    r = rsi(c, max(2, int(p["rsi_len"].value)))
    a = atr(h, l, c, max(2, int(p["atr_len"].value)))
    atr_pct = a / c * 100
    vr = v / sma(v, max(2, int(p["vol_len"].value)))

    ema_align = np.where((c > ef) & (ef > es), 60, np.where(c > ef, 30, 0))
    slope = linreg_slope(c, 20)
    slope_sc = np.where(slope > 0, 20, np.where(slope < 0, -20, 0))
    rsi_sc = np.where((r > 50) & (r < 70), 20, np.where((r >= 70) | (r <= 30), 10, 10))
    vol_sc = np.where(vr >= float(p["min_vol_ratio"].value), 20, 0)
    volty_sc = np.where((atr_pct >= float(p["min_atr_pct"].value))
                        & (atr_pct <= float(p["max_atr_pct"].value)), 20, 0)
    hh = (h > h.shift(1)) & (l > l.shift(1))
    ll = (h < h.shift(1)) & (l < l.shift(1))
    struct = np.where(hh, 20, np.where(ll, -20, 0))

    hero = np.clip(ema_align + np.maximum(slope_sc, 0) + rsi_sc + vol_sc
                   + volty_sc + np.maximum(struct, 0), 0, 100)
    bear = np.clip((100 - ema_align) + np.maximum(-slope_sc, 0) + (20 - rsi_sc)
                   + vol_sc + volty_sc + np.maximum(-struct, 0), 0, 100)
    buy = pd.Series(hero >= float(p["long_thresh"].value), index=df.index)
    sell = pd.Series(bear <= float(p["short_thresh"].value), index=df.index)
    # 連続点灯を立ち上がりだけに絞る(原文はラベル。売買として使うため)
    buy &= ~buy.shift(1).fillna(False)
    sell &= ~sell.shift(1).fillna(False)
    return _atr_exit(df, p, buy.to_numpy(), sell.to_numpy())


templates.register("tv3_289_zero_to_hero", defaults={
    "long_thresh": 80.0, "short_thresh": 20.0, "ema_fast": 20.0, "ema_slow": 50.0,
    "rsi_len": 14.0, "vol_len": 20.0, "min_vol_ratio": 1.2, "max_atr_pct": 5.0,
    "min_atr_pct": 0.5, "atr_len": 14.0, **{k: v for k, v in _EXIT_DEFAULTS.items()
                                            if k != "atr_period"},
    "atr_period": 14.0}, signal_fn=_sig_zero_to_hero)


# ==========================================================================
# 297 [ALPHAX] Bollinger/Keltner Volatility
#     BB外帯タッチ + スコア + RSIの傾き + スクイーズ中は出さない。
# ==========================================================================
def _sig_alphax_bbkc(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    n = max(5, int(p["bb_len"].value))
    ub, mb, lb_ = bollinger_bands(c, n, float(p["bb_mult"].value))
    a = atr(h, l, c, n)
    kc_u, kc_l = mb + float(p["kc_mult"].value) * a, mb - float(p["kc_mult"].value) * a
    sq_on = (ub < kc_u) & (lb_ > kc_l)          # スクイーズ中
    r = rsi(c, max(2, int(p["rsi_len"].value)))

    body = (c - o).abs()
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    bull_rej = lw > body
    bear_rej = uw > body
    vol_ok = v > sma(v, 20) * float(p["vol_mult"].value)

    bull_touch = l <= lb_ * 1.001
    bear_touch = h >= ub * 0.999
    score_b = ((r < 35).astype(int) + bull_rej.astype(int) + vol_ok.astype(int)
               + (c > o).astype(int) + (c > c.shift(1)).astype(int))
    score_s = ((r > 65).astype(int) + bear_rej.astype(int) + vol_ok.astype(int)
               + (c < o).astype(int) + (c < c.shift(1)).astype(int))
    mc = float(p["min_conf"].value)

    buy = bull_touch & (score_b >= mc) & ((c > o) | bull_rej) & (r < 45) & vol_ok & ~sq_on
    sell = bear_touch & (score_s >= mc) & ((c < o) | bear_rej) & (r > 55) & vol_ok & ~sq_on
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_297_alphax_bbkc", defaults={
    "bb_len": 20.0, "bb_mult": 2.0, "kc_mult": 1.5, "rsi_len": 14.0,
    "vol_mult": 1.0, "min_conf": 3.0, "cooldown": 10.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_alphax_bbkc)


# ==========================================================================
# 201 / 279 [ALPHAX] Reversal(RSI Divergence + WPR)/ Impulse Bands
#     RSIダイバージェンス or トレンド反転 + Williams%R の速度 + 信頼度スコア。
#     mode=0 が 201(ダイバージェンス起点)、mode=1 が 279(トレンド反転起点)。
# ==========================================================================
def _sig_alphax_wpr(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    r = rsi(c, max(2, int(p["rsi_len"].value)))
    nf, ns = max(2, int(p["wpr_fast"].value)), max(2, int(p["wpr_slow"].value))
    wf = -100 * (h.rolling(nf).max() - c) / (h.rolling(nf).max() - l.rolling(nf).min())
    ws = -100 * (h.rolling(ns).max() - c) / (h.rolling(ns).max() - l.rolling(ns).min())
    wvel = wf - wf.shift(1)
    vr = v / sma(v, 20)
    vol_ok = vr > float(p["vol_mult"].value)

    lb, rb = 3, 3
    ph, pl = _pivot_high(h, lb, rb), _pivot_low(l, lb, rb)
    ph_v = ph.ffill()
    pl_v = pl.ffill()
    phr = r.shift(rb).where(ph.notna()).ffill()
    plr = r.shift(rb).where(pl.notna()).ffill()
    bear_div = ph.notna() & (ph_v > ph_v.shift(1).where(ph.notna()).ffill()) \
        & (phr < phr.shift(1).where(ph.notna()).ffill())
    bull_div = pl.notna() & (pl_v < pl_v.shift(1).where(pl.notna()).ffill()) \
        & (plr > plr.shift(1).where(pl.notna()).ffill())

    body_ratio = (c - o).abs() / (h - l).replace(0, np.nan)
    conf_b = (np.where(wf < -80, 8, np.where(wf < -70, 5, 0))
              + np.where(wvel > 3, 7, 0)
              + np.where((c > o) & (body_ratio > 0.5), 5, np.where(c > o, 2, 0))
              + np.where(vr > 1.5, 5, np.where(vr > 1.0, 2, 0))
              + np.where(r < 35, 5, np.where(r < 45, 3, 0))) * 3.0
    conf_s = (np.where(wf > -20, 8, np.where(wf > -30, 5, 0))
              + np.where(wvel < -3, 7, 0)
              + np.where((c < o) & (body_ratio > 0.5), 5, np.where(c < o, 2, 0))
              + np.where(vr > 1.5, 5, np.where(vr > 1.0, 2, 0))
              + np.where(r > 65, 5, np.where(r > 55, 3, 0))) * 3.0
    mc = float(p["min_conf"].value)

    mode = int(round(float(p["mode"].value)))
    if mode == 0:   # 201: ダイバージェンス起点
        trig_b, trig_s = bull_div, bear_div
    else:           # 279: トレンド反転起点
        e = ema(c, max(2, int(p["trend_len"].value)))
        trig_b = _cross_up(c, e)
        trig_s = _cross_dn(c, e)

    buy = trig_b & (_cross_up(wf, ws) | (wvel > 2)) & (conf_b >= mc) & vol_ok
    sell = trig_s & (_cross_dn(wf, ws) | (wvel < -2)) & (conf_s >= mc) & vol_ok
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


for _nm, _mode in (("tv3_201_alphax_rsidiv_wpr", 0.0), ("tv3_279_alphax_impulse", 1.0)):
    templates.register(_nm, defaults={
        "rsi_len": 14.0, "wpr_fast": 14.0, "wpr_slow": 50.0, "vol_mult": 1.0,
        "min_conf": 50.0, "mode": _mode, "trend_len": 20.0, "cooldown": 10.0,
        **_EXIT_DEFAULTS}, signal_fn=_sig_alphax_wpr)


# ==========================================================================
# 237 [ALPHAX] Edge - Support/Resistance / Breakout Entry
#     合流数>=minConfluence + トレンド許可 + 強い実体 + 行き過ぎ除外 + 構造あり。
# ==========================================================================
def _sig_alphax_edge(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    a = adx(h, l, c, 14)
    dip, dim = dmi(h, l, c, 14)
    r = rsi(c, 14)
    av = atr(h, l, c, 14)
    ef = ema(c, max(2, int(p["ema_fast"].value)))
    es = ema(c, max(2, int(p["ema_slow"].value)))
    lb = max(5, int(p["sr_lookback"].value))
    res1 = h.shift(1).rolling(lb).max()
    sup1 = l.shift(1).rolling(lb).min()
    zone = float(p["sr_zone"].value)

    at_sup = (c - sup1).abs() < av * zone
    at_res = (c - res1).abs() < av * zone
    brk_up = (c > res1) & (c.shift(1) <= res1.shift(1))
    brk_dn = (c < sup1) & (c.shift(1) >= sup1.shift(1))
    trend_dir_up = (ef > es) & (c > ef)
    trend_dir_dn = (ef < es) & (c < ef)

    rng = (h - l).replace(0, np.nan)
    strong = ((c - o).abs() / rng) > 0.35
    over = (c - ef).abs() / av > 3.5
    fake_b = brk_up & (r > 70)
    fake_s = brk_dn & (r < 30)
    trend_filter = a >= float(p["adx_min"].value)

    cf_b = ((r > 50).astype(int) + (dip > dim).astype(int) + (c > ef).astype(int)
            + (ef > es).astype(int) + (v > sma(v, 20)).astype(int)
            + at_sup.astype(int) + brk_up.astype(int))
    cf_s = ((r < 50).astype(int) + (dim > dip).astype(int) + (c < ef).astype(int)
            + (ef < es).astype(int) + (v > sma(v, 20)).astype(int)
            + at_res.astype(int) + brk_dn.astype(int))
    mc = float(p["min_confluence"].value)

    buy = ((cf_b >= mc) & (c > o) & trend_filter & strong & ~fake_b & ~over
           & (at_sup | brk_up | trend_dir_up))
    sell = ((cf_s >= mc) & (c < o) & trend_filter & strong & ~fake_s & ~over
            & (at_res | brk_dn | trend_dir_dn))
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_237_alphax_edge", defaults={
    "ema_fast": 21.0, "ema_slow": 50.0, "sr_lookback": 20.0, "sr_zone": 0.5,
    "adx_min": 20.0, "min_confluence": 4.0, "cooldown": 10.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_alphax_edge)


# ==========================================================================
# 212 Adaptive Buy Sell Signal [AvantCoin]
#     合流スコア>=min かつ 逆側<3、複数フィルタ全通過、クールダウン。
# ==========================================================================
def _sig_avantcoin(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    ef, es = ema(c, 9), ema(c, 21)
    e50, e200 = ema(c, 50), ema(c, 200)
    r = rsi(c, 14)
    ml, sg, hist = macd(c, 12, 26, 9)
    k, d_ = stochastic(h, l, c, 14, 3, 3)
    a = adx(h, l, c, 14)
    av = atr(h, l, c, 14)

    bull = ((ef > es).astype(int) + (c > e50).astype(int) + (c > e200).astype(int)
            + (r > 50).astype(int) + (ml > sg).astype(int) + (hist > 0).astype(int)
            + (k > d_).astype(int) + (v > sma(v, 20)).astype(int))
    bear = ((ef < es).astype(int) + (c < e50).astype(int) + (c < e200).astype(int)
            + (r < 50).astype(int) + (ml < sg).astype(int) + (hist < 0).astype(int)
            + (k < d_).astype(int) + (v > sma(v, 20)).astype(int))

    strong_trend = a >= float(p["adx_min"].value)
    hi_vol = av > sma(av, 50) * float(p["max_vol_mult"].value)
    bull_candle, bear_candle = c > o, c < o
    brk_up = c > h.shift(1).rolling(10).max()
    brk_dn = c < l.shift(1).rolling(10).min()

    mc = float(p["min_confluence"].value)
    buy = ((bull >= mc) & (bear < 3) & strong_trend & bull_candle & brk_up & ~hi_vol)
    sell = ((bear >= mc) & (bull < 3) & strong_trend & bear_candle & brk_dn & ~hi_vol)
    cd = int(p["cooldown"].value)
    return _atr_exit(df, p, _cooldown(buy.fillna(False).to_numpy(), cd),
                     _cooldown(sell.fillna(False).to_numpy(), cd))


templates.register("tv3_212_avantcoin", defaults={
    "min_confluence": 5.0, "adx_min": 20.0, "max_vol_mult": 2.0, "cooldown": 10.0,
    **_EXIT_DEFAULTS}, signal_fn=_sig_avantcoin)


# ==========================================================================
# 213 Fair Value Range Breakout [by Oberlunar]
#     直近のFVGゾーンに触れて、方向が一致していれば入る(バリューエリア確認は省略不可
#     なので、レンジ中央からの位置で代用せず「未実装」として使わない)。
#     ここでは原文の中核である「FVG方向 + ゾーンタッチ」だけを実装する。
# ==========================================================================
def _sig_fvg_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    bull_gap = l > h.shift(2)
    bear_gap = h < l.shift(2)
    top_b = l.where(bull_gap).ffill()
    bot_b = h.shift(2).where(bull_gap).ffill()
    top_s = l.shift(2).where(bear_gap).ffill()
    bot_s = h.where(bear_gap).ffill()

    age = int(p["max_age"].value)

    def _first_touch(gap: pd.Series, top: pd.Series, bot: pd.Series,
                     side: int) -> np.ndarray:
        """原文の fvgUsed を再現する。1つのゾーンにつき最初の1回だけ点灯させる。

        ffill したゾーンに価格が居座ると毎バー条件を満たしてしまうので、
        ここを外すとシグナルが桁違いに増える(実測 24% のバーで点灯した)。
        """
        g = gap.to_numpy()
        tp, bt = top.to_numpy(), bot.to_numpy()
        hh, ll, cc = h.to_numpy(), l.to_numpy(), c.to_numpy()
        out = np.zeros(len(g), dtype=bool)
        born = -10 ** 9
        used = True
        for i in range(len(g)):
            if g[i]:
                born, used = i, False       # 新しいゾーンができたら未使用に戻す
                continue
            if used or i - born > age or not np.isfinite(tp[i]):
                continue
            if side > 0:
                if ll[i] <= tp[i] and hh[i] >= bt[i] and cc[i] > bt[i]:
                    out[i], used = True, True
            else:
                if hh[i] >= bt[i] and ll[i] <= tp[i] and cc[i] < tp[i]:
                    out[i], used = True, True
        return out

    buy = _first_touch(bull_gap.fillna(False), top_b, bot_b, 1)
    sell = _first_touch(bear_gap.fillna(False), top_s, bot_s, -1)
    return _atr_exit(df, p, buy, sell)


templates.register("tv3_213_fvg_breakout", defaults={
    "max_age": 20.0, **_EXIT_DEFAULTS}, signal_fn=_sig_fvg_breakout)
