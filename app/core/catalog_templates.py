"""検証候補100件(CLAUDE_VALIDATION_CATALOG)のnote手法を機械化したテンプレート群。

出典と原文条件は
  C:\\CodexProject\\FX\\research\\source_catalog_100\\claude_validation_results.md
に候補番号ごとに記録してある。ここでは「原文に書いてあること」だけを実装し、
原文に無い定義(閾値・遡及本数など)はパラメータとして外に出す。推測で固定しない。

先読み規律(既存テンプレと共通):
  - 全ての指標は確定バーまでの rolling/ewm のみ
  - signal_fn はシフトしない。翌バー始値執行の shift(1) は engine.run_backtest が行う
  - 上位足を参照するものは、確定した上位足バーのみを使う
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, rsi, sma
from app.core.strategy_model import Strategy


# --------------------------------------------------------------------------
# 共通ヘルパ
# --------------------------------------------------------------------------
def _stoch(df: pd.DataFrame, k: int, d: int, slowing: int) -> tuple[pd.Series, pd.Series]:
    """MT4/MT5と同じ定義のストキャスティクス(%K, %D)を返す。"""
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    raw = 100.0 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    kline = raw.rolling(slowing).mean() if slowing > 1 else raw
    dline = kline.rolling(d).mean()
    return kline, dline


def _rci(close: pd.Series, period: int) -> pd.Series:
    """RCI(順位相関指数)。時点順位と価格順位のスピアマン相関×100。

    ★ベクトル化必須。素のPythonループで書くと M5×3年(22万本)×52期間×18パラメータ×3区間
    で計算量が爆発し、スイープが十数時間ハングする(実際に踏んだ)。
    sliding_window_view で窓行列を作り、argsort を軸方向に一括適用する。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    v = close.to_numpy(dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    if n < period or period < 2:
        return pd.Series(out, index=close.index)
    w = sliding_window_view(v, period)[:, ::-1]      # 各行: 直近が先頭
    order = np.argsort(-w, axis=1, kind="stable")    # 価格の高い順
    p_rank = np.empty_like(order)
    np.put_along_axis(p_rank, order, np.arange(1, period + 1), axis=1)
    t_rank = np.arange(1, period + 1)                # 時点順位(直近が1位)
    d2 = ((t_rank - p_rank) ** 2).sum(axis=1)
    out[period - 1:] = (1.0 - 6.0 * d2 / (period * (period * period - 1))) * 100.0
    return pd.Series(out, index=close.index)


def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    m = tp.rolling(period).mean()
    md = (tp - m).abs().rolling(period).mean()
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df["high"], df["low"], df["close"], period)
    pdi = 100 * pd.Series(plus, index=df.index).rolling(period).mean() / tr.replace(0, np.nan)
    mdi = 100 * pd.Series(minus, index=df.index).rolling(period).mean() / tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(period).mean()


def _htf_flag(df: pd.DataFrame, mult: int, flag_on_htf) -> np.ndarray:
    """上位足で計算した真偽値を、確定した次の実行足バーから有効にして展開する。"""
    n = len(df)
    idx = np.arange(n) // mult
    g = df.groupby(idx).agg(open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last"))
    v = np.asarray(flag_on_htf(g), dtype=bool)
    out = np.zeros(n, dtype=bool)
    for k in range(len(g)):
        s = (k + 1) * mult
        if s >= n:
            break
        out[s:min((k + 2) * mult, n)] = v[k]
    return out


def _emit(n: int, longs: np.ndarray, shorts: np.ndarray,
          close: np.ndarray, dist: np.ndarray | None, rr: float) -> pd.DataFrame:
    """signal と(距離が与えられていれば)構造SL/TPを組み立てる。"""
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    if dist is not None:
        ok = np.isfinite(dist) & (dist > 0)
        longs = longs & ok
        shorts = shorts & ok
    sig[longs] = 1
    sig[shorts] = -1
    if dist is not None:
        slp[longs] = close[longs] - dist[longs]
        tpp[longs] = close[longs] + rr * dist[longs]
        slp[shorts] = close[shorts] + dist[shorts]
        tpp[shorts] = close[shorts] - rr * dist[shorts]
    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp})


# --------------------------------------------------------------------------
# #1 / #4 一目均衡表 三役好転・逆転
# --------------------------------------------------------------------------
def _sig_ichimoku_sanyaku(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    t_len = max(2, int(p["tenkan"].value))
    k_len = max(2, int(p["kijun"].value))
    s_len = max(2, int(p["senkou_b"].value))
    mode = int(p["mode"].value)              # 0=三役 / 1=転換基準クロス+雲
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))
    h, l, c = df["high"], df["low"], df["close"]

    tenkan = (h.rolling(t_len).max() + l.rolling(t_len).min()) / 2
    kijun = (h.rolling(k_len).max() + l.rolling(k_len).min()) / 2
    # 先行スパンは k_len 先にプロットされる → 現在位置で見える雲は k_len 前の値
    span_a = ((tenkan + kijun) / 2).shift(k_len)
    span_b = ((h.rolling(s_len).max() + l.rolling(s_len).min()) / 2).shift(k_len)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    # 遅行線: 現在の終値を k_len 前に置く → 「遅行線>ローソク足」= 現終値 > k_len前の終値
    chikou_above = c > c.shift(k_len)
    chikou_below = c < c.shift(k_len)

    cross_up = (tenkan.shift(1) <= kijun.shift(1)) & (tenkan > kijun)
    cross_dn = (tenkan.shift(1) >= kijun.shift(1)) & (tenkan < kijun)

    if mode == 0:      # 三役好転/逆転(#4)
        long_sig = (tenkan > kijun) & chikou_above & (c > cloud_top)
        short_sig = (tenkan < kijun) & chikou_below & (c < cloud_bot)
        # 条件が揃った初回バーのみ発火
        long_sig = long_sig & ~long_sig.shift(1).fillna(False)
        short_sig = short_sig & ~short_sig.shift(1).fillna(False)
    else:              # 転換線/基準線クロス + 雲フィルタ(#1)
        long_sig = cross_up & (c > cloud_top)
        short_sig = cross_dn & (c < cloud_bot)

    a = atr(h, l, c, 14)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(),
                 short_sig.fillna(False).to_numpy(), c.to_numpy(), dist, rr)


for _nm, _mode in (("cat_ichimoku_sanyaku", 0.0), ("cat_ichimoku_cross", 1.0)):
    templates.register(_nm, defaults={
        "tenkan": 9.0, "kijun": 26.0, "senkou_b": 52.0, "mode": _mode,
        "atr_mult": 2.0, "rr": 1.5,
        "sl_pips": 30.0, "tp_pips": 45.0, "max_hold_bars": 96.0, "lot": 0.1,
    }, signal_fn=_sig_ichimoku_sanyaku)


# --------------------------------------------------------------------------
# #6 BB(±0.7σ/±1.65σ) + MACD(9,17,9) トレンド中期の押し目
# --------------------------------------------------------------------------
def _sig_bb_macd_trend(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ma_len = max(2, int(p["ma_period"].value))
    inner = abs(float(p["inner_sigma"].value))
    outer = abs(float(p["outer_sigma"].value))
    fast = max(2, int(p["macd_fast"].value))
    slow = max(3, int(p["macd_slow"].value))
    hold_n = max(1, int(p["trend_bars"].value))
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    c = df["close"]
    mid = sma(c, ma_len)
    sd = c.rolling(ma_len).std(ddof=0)
    up_in, dn_in = mid + inner * sd, mid - inner * sd
    up_out, dn_out = mid + outer * sd, mid - outer * sd
    macd = ema(c, fast) - ema(c, slow)

    # トレンド中期: MACDが0の同じ側に hold_n 本連続、かつ外側σを終値が抜けた実績
    macd_pos = (macd > 0).rolling(hold_n).sum() == hold_n
    macd_neg = (macd < 0).rolling(hold_n).sum() == hold_n
    broke_up = (c > up_out).rolling(hold_n).max().astype(bool)
    broke_dn = (c < dn_out).rolling(hold_n).max().astype(bool)

    # エントリー: 内側σ または 中央線まで押した(安値が到達)
    touch_up = (df["low"] <= up_in) | (df["low"] <= mid)
    touch_dn = (df["high"] >= dn_in) | (df["high"] >= mid)

    long_sig = (macd_pos & broke_up & touch_up).fillna(False)
    short_sig = (macd_neg & broke_dn & touch_dn).fillna(False)
    long_sig = long_sig & ~long_sig.shift(1).fillna(False)
    short_sig = short_sig & ~short_sig.shift(1).fillna(False)

    a = atr(df["high"], df["low"], c, 14)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.to_numpy(), short_sig.to_numpy(), c.to_numpy(), dist, rr)


templates.register("cat_bb_macd_trend", defaults={
    "ma_period": 20.0, "inner_sigma": 0.7, "outer_sigma": 1.65,
    "macd_fast": 9.0, "macd_slow": 17.0, "trend_bars": 9.0,
    "atr_mult": 1.5, "rr": 1.0,
    "sl_pips": 20.0, "tp_pips": 20.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_bb_macd_trend)


# --------------------------------------------------------------------------
# #8 BB±3σの大突破 → 次の足で決済
# --------------------------------------------------------------------------
def _sig_bb3_next_bar(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ma_len = max(2, int(p["ma_period"].value))
    sig_mult = abs(float(p["sigma"].value))
    excess_atr = abs(float(p["excess_atr"].value))   # 「大きく」を ATR比で定義(原文に数値なし)
    c, h, l = df["close"], df["high"], df["low"]
    mid = sma(c, ma_len)
    sd = c.rolling(ma_len).std(ddof=0)
    up, dn = mid + sig_mult * sd, mid - sig_mult * sd
    a = atr(h, l, c, 14)

    over_up = (c - up) >= excess_atr * a
    over_dn = (dn - c) >= excess_atr * a
    return _emit(len(df), over_dn.fillna(False).to_numpy(), over_up.fillna(False).to_numpy(),
                 c.to_numpy(), None, 1.0)


templates.register("cat_bb3_next_bar", defaults={
    "ma_period": 20.0, "sigma": 3.0, "excess_atr": 0.3,
    "sl_pips": 100.0, "tp_pips": 100.0,   # 実質は max_hold_bars=1 の時間退出で決済する
    "max_hold_bars": 1.0, "lot": 0.1,
}, signal_fn=_sig_bb3_next_bar)


# --------------------------------------------------------------------------
# #12 ファスト/スローストキャスの2段階トリガー
# --------------------------------------------------------------------------
def _sig_stoch_two_stage(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    k = max(2, int(p["k_period"].value))
    d = max(1, int(p["d_period"].value))
    slowing = max(2, int(p["slowing"].value))
    f_lo, f_hi = float(p["fast_low"].value), float(p["fast_high"].value)
    s_lo, s_hi = float(p["slow_low"].value), float(p["slow_high"].value)
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    fast_k, _ = _stoch(df, k, d, 1)
    slow_k, _ = _stoch(df, k, d, slowing)
    pre_long = (fast_k < f_lo) & (slow_k < s_lo)
    pre_short = (fast_k > f_hi) & (slow_k > s_hi)
    long_sig = pre_long.shift(1).fillna(False) & (slow_k.shift(1) <= s_lo) & (slow_k > s_lo)
    short_sig = pre_short.shift(1).fillna(False) & (slow_k.shift(1) >= s_hi) & (slow_k < s_hi)

    a = atr(df["high"], df["low"], df["close"], 14)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 df["close"].to_numpy(), dist, rr)


templates.register("cat_stoch_two_stage", defaults={
    "k_period": 14.0, "d_period": 3.0, "slowing": 3.0,
    "fast_low": 10.0, "fast_high": 90.0, "slow_low": 20.0, "slow_high": 80.0,
    "atr_mult": 1.5, "rr": 1.5,
    "sl_pips": 20.0, "tp_pips": 30.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_stoch_two_stage)


# --------------------------------------------------------------------------
# #16 MACDゼロ下GC + 陽線2本 + 20EMAフィルタ
# --------------------------------------------------------------------------
def _sig_macd_gc_two_candles(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    fast = max(2, int(p["macd_fast"].value))
    slow = max(3, int(p["macd_slow"].value))
    sig_len = max(1, int(p["macd_signal"].value))
    within = max(1, int(p["within_bars"].value))     # クロスから何本以内を有効とするか
    ema_len = max(2, int(p["ema_filter"].value))
    use_ema = int(p["use_ema_filter"].value)
    sl_lb = max(1, int(p["sl_lookback"].value))
    rr = abs(float(p["rr"].value))

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    macd = ema(c, fast) - ema(c, slow)
    sigl = ema(macd, sig_len)
    gc = (macd.shift(1) <= sigl.shift(1)) & (macd > sigl) & (macd < 0)
    dc = (macd.shift(1) >= sigl.shift(1)) & (macd < sigl) & (macd > 0)
    gc_recent = gc.rolling(within).max().astype(bool)
    dc_recent = dc.rolling(within).max().astype(bool)

    bull2 = (c > o) & (c.shift(1) > o.shift(1))
    bear2 = (c < o) & (c.shift(1) < o.shift(1))
    e = ema(c, ema_len)
    long_ok = (c > e) if use_ema else True
    short_ok = (c < e) if use_ema else True

    long_sig = (gc_recent & bull2 & long_ok).fillna(False)
    short_sig = (dc_recent & bear2 & short_ok).fillna(False)
    long_sig = long_sig & ~long_sig.shift(1).fillna(False)
    short_sig = short_sig & ~short_sig.shift(1).fillna(False)

    # SL=直近安値/高値、TP=RR倍
    ll = l.rolling(sl_lb).min()
    hh = h.rolling(sl_lb).max()
    ca = c.to_numpy()
    dist = np.where(long_sig.to_numpy(), ca - ll.to_numpy(),
                    np.where(short_sig.to_numpy(), hh.to_numpy() - ca, np.nan))
    return _emit(len(df), long_sig.to_numpy(), short_sig.to_numpy(), ca, dist, rr)


templates.register("cat_macd_gc_two_candles", defaults={
    "macd_fast": 12.0, "macd_slow": 26.0, "macd_signal": 9.0,
    "within_bars": 5.0, "ema_filter": 20.0, "use_ema_filter": 1.0,
    "sl_lookback": 5.0, "rr": 1.0,
    "sl_pips": 15.0, "tp_pips": 15.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_macd_gc_two_candles)


# --------------------------------------------------------------------------
# #18 ファストストキャスのクロス + 陽線/陰線 + 時間帯限定
# --------------------------------------------------------------------------
def _sig_stoch_fast_cross(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    k = max(2, int(p["k_period"].value))
    d = max(1, int(p["d_period"].value))
    hs, he = int(p["hour_start"].value), int(p["hour_end"].value)
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    kk, dd = _stoch(df, k, d, 1)
    o, c = df["open"], df["close"]
    # 原文: 2本前で%K<=%D、1本前で%K>%D、1本前が陽線 → 1本前=確定バーなので当バーで判定
    long_sig = (kk.shift(1) <= dd.shift(1)) & (kk > dd) & (c > o)
    short_sig = (kk.shift(1) >= dd.shift(1)) & (kk < dd) & (c < o)
    if hs != he and "timestamp" in df.columns:
        hr = pd.to_datetime(df["timestamp"]).dt.hour
        in_sess = (hr >= hs) & (hr < he) if hs < he else ((hr >= hs) | (hr < he))
        long_sig &= in_sess
        short_sig &= in_sess

    a = atr(df["high"], df["low"], c, 14)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


templates.register("cat_stoch_fast_cross", defaults={
    "k_period": 5.0, "d_period": 3.0, "hour_start": 7.0, "hour_end": 18.0,
    "atr_mult": 1.5, "rr": 1.5,
    "sl_pips": 20.0, "tp_pips": 30.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_stoch_fast_cross)


# --------------------------------------------------------------------------
# #27 EMA5/14/40 整列 + ストキャスクロス + ATR(SL3倍/TP2倍) + 200EMAフィルタ
# --------------------------------------------------------------------------
def _sig_ema3_stoch_atr(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    e1 = max(2, int(p["ema_fast"].value))
    e2 = max(3, int(p["ema_mid"].value))
    e3 = max(4, int(p["ema_slow"].value))
    k = max(2, int(p["k_period"].value))
    d = max(1, int(p["d_period"].value))
    slowing = max(1, int(p["slowing"].value))
    filt = max(10, int(p["ema_filter"].value))
    use_filter = int(p["use_filter"].value)
    sl_atr = abs(float(p["sl_atr"].value))
    tp_atr = abs(float(p["tp_atr"].value))

    c = df["close"]
    a1, a2, a3 = ema(c, e1), ema(c, e2), ema(c, e3)
    kk, dd = _stoch(df, k, d, slowing)
    up_order = (a1 > a2) & (a2 > a3)
    dn_order = (a1 < a2) & (a2 < a3)
    gc = (kk.shift(1) <= dd.shift(1)) & (kk > dd)
    dc = (kk.shift(1) >= dd.shift(1)) & (kk < dd)

    long_sig = up_order & gc
    short_sig = dn_order & dc
    if use_filter:
        f = ema(c, filt)
        long_sig &= (c > f)
        short_sig &= (c < f)

    a = atr(df["high"], df["low"], c, 14).to_numpy()
    ca = c.to_numpy()
    ls = long_sig.fillna(False).to_numpy()
    ss = short_sig.fillna(False).to_numpy()
    n = len(df)
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    ok = np.isfinite(a) & (a > 0)
    ls &= ok
    ss &= ok
    sig[ls] = 1
    sig[ss] = -1
    slp[ls] = ca[ls] - sl_atr * a[ls]
    tpp[ls] = ca[ls] + tp_atr * a[ls]
    slp[ss] = ca[ss] + sl_atr * a[ss]
    tpp[ss] = ca[ss] - tp_atr * a[ss]
    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


templates.register("cat_ema3_stoch_atr", defaults={
    "ema_fast": 5.0, "ema_mid": 14.0, "ema_slow": 40.0,
    "k_period": 5.0, "d_period": 3.0, "slowing": 3.0,
    "ema_filter": 200.0, "use_filter": 1.0,
    "sl_atr": 3.0, "tp_atr": 2.0,
    "sl_pips": 20.0, "tp_pips": 13.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_ema3_stoch_atr)


# --------------------------------------------------------------------------
# #28 CCI ±150 折り返し
# --------------------------------------------------------------------------
def _sig_cci_turn(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    period = max(2, int(p["cci_period"].value))
    lvl = abs(float(p["level"].value))
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))
    cc = _cci(df, period)
    long_sig = (cc.shift(1) <= -lvl) & (cc > -lvl)
    short_sig = (cc.shift(1) >= lvl) & (cc < lvl)
    a = atr(df["high"], df["low"], df["close"], 14)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 df["close"].to_numpy(), dist, rr)


templates.register("cat_cci_turn", defaults={
    "cci_period": 14.0, "level": 150.0, "atr_mult": 1.5, "rr": 1.5,
    "sl_pips": 20.0, "tp_pips": 30.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_cci_turn)


# --------------------------------------------------------------------------
# #32 上位足EMA配列 + 下位足EMAタッチ + RCI9/26/52
# --------------------------------------------------------------------------
def _sig_ema_rci(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    mult = max(2, int(p["htf_mult"].value))
    e1 = max(2, int(p["ema1"].value))
    e2 = max(3, int(p["ema2"].value))
    e3 = max(4, int(p["ema3"].value))
    r_s = max(3, int(p["rci_short"].value))
    r_m = max(3, int(p["rci_mid"].value))
    lo = float(p["rci_low"].value)
    rr = abs(float(p["rr"].value))
    sl_lb = max(1, int(p["sl_lookback"].value))

    c = df["close"]

    def htf_up(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        a1, a2, a3 = ema(gc, e1), ema(gc, e2), ema(gc, e3)
        return ((gc > a3) & (a1 > a2) & (a2 > a3)).fillna(False).to_numpy()

    def htf_dn(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        a1, a2, a3 = ema(gc, e1), ema(gc, e2), ema(gc, e3)
        return ((gc < a3) & (a1 < a2) & (a2 < a3)).fillna(False).to_numpy()

    up = _htf_flag(df, mult, htf_up)
    dn = _htf_flag(df, mult, htf_dn)

    a1, a2 = ema(c, e1), ema(c, e2)
    touch_up = (df["low"] <= a1) | (df["low"] <= a2)
    touch_dn = (df["high"] >= a1) | (df["high"] >= a2)

    rs, rm = _rci(c, r_s), _rci(c, r_m)
    rci_long = (rs.shift(1) <= lo) & (rs > rs.shift(1)) & (rm > 0)
    rci_short = (rs.shift(1) >= -lo) & (rs < rs.shift(1)) & (rm < 0)

    long_sig = (pd.Series(up, index=df.index) & touch_up & rci_long).fillna(False)
    short_sig = (pd.Series(dn, index=df.index) & touch_dn & rci_short).fillna(False)

    ll = df["low"].rolling(sl_lb).min().to_numpy()
    hh = df["high"].rolling(sl_lb).max().to_numpy()
    ca = c.to_numpy()
    dist = np.where(long_sig.to_numpy(), ca - ll, np.where(short_sig.to_numpy(), hh - ca, np.nan))
    return _emit(len(df), long_sig.to_numpy(), short_sig.to_numpy(), ca, dist, rr)


templates.register("cat_ema_rci", defaults={
    "htf_mult": 4.0, "ema1": 20.0, "ema2": 50.0, "ema3": 200.0,
    "rci_short": 9.0, "rci_mid": 26.0, "rci_low": -80.0,
    "sl_lookback": 5.0, "rr": 2.0,
    "sl_pips": 20.0, "tp_pips": 40.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_ema_rci)


# --------------------------------------------------------------------------
# #35 MA80方向 + RCI52張り付き + RCI10転換 + CCI14ゼロ抜け
# --------------------------------------------------------------------------
def _sig_ma_rci_cci(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ma_len = max(5, int(p["ma_period"].value))
    slope_lb = max(1, int(p["slope_lb"].value))
    r_long = max(5, int(p["rci_long"].value))
    r_short = max(3, int(p["rci_short"].value))
    stick = float(p["stick_level"].value)
    cci_len = max(2, int(p["cci_period"].value))
    rr = abs(float(p["rr"].value))
    sl_lb = max(1, int(p["sl_lookback"].value))

    c = df["close"]
    m = sma(c, ma_len)
    up_trend = m > m.shift(slope_lb)
    dn_trend = m < m.shift(slope_lb)
    rl, rs = _rci(c, r_long), _rci(c, r_short)
    cc = _cci(df, cci_len)

    long_sig = (up_trend & (rl >= stick) & (rs.shift(1) < 0) & (rs > rs.shift(1))
                & (cc.shift(1) < 0) & (cc > 0))
    short_sig = (dn_trend & (rl <= -stick) & (rs.shift(1) > 0) & (rs < rs.shift(1))
                 & (cc.shift(1) > 0) & (cc < 0))

    ll = df["low"].rolling(sl_lb).min().to_numpy()
    hh = df["high"].rolling(sl_lb).max().to_numpy()
    ca = c.to_numpy()
    ls = long_sig.fillna(False).to_numpy()
    ss = short_sig.fillna(False).to_numpy()
    dist = np.where(ls, ca - ll, np.where(ss, hh - ca, np.nan))
    return _emit(len(df), ls, ss, ca, dist, rr)


templates.register("cat_ma_rci_cci", defaults={
    "ma_period": 80.0, "slope_lb": 3.0, "rci_long": 52.0, "rci_short": 10.0,
    "stick_level": 50.0, "cci_period": 14.0, "sl_lookback": 5.0, "rr": 2.0,
    "sl_pips": 20.0, "tp_pips": 40.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_ma_rci_cci)


# --------------------------------------------------------------------------
# #36 ドンチャン20/10(タートル) / #38 ドンチャン20 + ATRフィルタ
# --------------------------------------------------------------------------
def _sig_donchian(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    entry_n = max(2, int(p["entry_n"].value))
    exit_n = max(2, int(p["exit_n"].value))
    atr_len = max(2, int(p["atr_period"].value))
    sl_atr = abs(float(p["sl_atr"].value))
    use_atr_filter = int(p["use_atr_filter"].value)
    atr_ratio = abs(float(p["atr_filter_ratio"].value))

    h, l, c = df["high"], df["low"], df["close"]
    up = h.rolling(entry_n).max().shift(1)
    dn = l.rolling(entry_n).min().shift(1)
    ex_up = h.rolling(exit_n).max().shift(1)
    ex_dn = l.rolling(exit_n).min().shift(1)

    long_sig = c > up
    short_sig = c < dn
    if use_atr_filter:
        a = atr(h, l, c, atr_len)
        ok = a >= a.rolling(20).mean() * atr_ratio
        long_sig &= ok
        short_sig &= ok
    # 逆ブレイクでの退出はドテンで表現(反対シグナルを出す)
    long_sig = long_sig | (c > ex_up) & False      # 明示: 退出はengineのSL/TPとmax_holdで扱う
    a = atr(h, l, c, atr_len)
    dist = (sl_atr * a).to_numpy()
    rr = abs(float(p["rr"].value))
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


templates.register("cat_donchian_turtle", defaults={
    "entry_n": 20.0, "exit_n": 10.0, "atr_period": 20.0, "sl_atr": 2.0,
    "use_atr_filter": 0.0, "atr_filter_ratio": 0.8, "rr": 2.0,
    "sl_pips": 40.0, "tp_pips": 80.0, "max_hold_bars": 240.0, "lot": 0.1,
}, signal_fn=_sig_donchian)

templates.register("cat_donchian_atr", defaults={
    "entry_n": 20.0, "exit_n": 10.0, "atr_period": 14.0, "sl_atr": 2.0,
    "use_atr_filter": 1.0, "atr_filter_ratio": 0.8, "rr": 2.0,
    "sl_pips": 40.0, "tp_pips": 80.0, "max_hold_bars": 240.0, "lot": 0.1,
}, signal_fn=_sig_donchian)


# --------------------------------------------------------------------------
# #41 ピボット(前日高安終値) 逆張り / 順張り
# --------------------------------------------------------------------------
def _sig_pivot(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    mode = int(p["mode"].value)          # 0=逆張り(R/S反発) 1=順張り(PP反発)
    level = max(1, int(p["level"].value))
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    ts = pd.to_datetime(df["timestamp"])
    day = ts.dt.normalize()
    g = df.groupby(day).agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    pp = (g["h"] + g["l"] + g["c"]) / 3.0
    r1 = 2 * pp - g["l"]
    s1 = 2 * pp - g["h"]
    r2 = pp + (g["h"] - g["l"])
    s2 = pp - (g["h"] - g["l"])
    prev = pd.DataFrame({"pp": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2}).shift(1)
    m = day.map(prev["pp"]), day.map(prev["r1"]), day.map(prev["s1"]), \
        day.map(prev["r2"]), day.map(prev["s2"])
    PP, R1, S1, R2, S2 = (x.to_numpy() for x in m)

    h, l, c, o = (df[x].to_numpy() for x in ("high", "low", "close", "open"))
    res = R1 if level == 1 else R2
    sup = S1 if level == 1 else S2
    if mode == 0:
        # レジ/サポに触れて反転足で確定
        long_sig = (l <= sup) & (c > o)
        short_sig = (h >= res) & (c < o)
    else:
        long_sig = (c > PP) & (l <= PP) & (c > o)
        short_sig = (c < PP) & (h >= PP) & (c < o)

    a = atr(df["high"], df["low"], df["close"], 14).to_numpy()
    dist = atr_mult * a
    return _emit(len(df), np.nan_to_num(long_sig, nan=0).astype(bool),
                 np.nan_to_num(short_sig, nan=0).astype(bool), c, dist, rr)


for _nm, _md in (("cat_pivot_reversal", 0.0), ("cat_pivot_follow", 1.0)):
    templates.register(_nm, defaults={
        "mode": _md, "level": 1.0, "atr_mult": 1.5, "rr": 1.5,
        "sl_pips": 20.0, "tp_pips": 30.0, "max_hold_bars": 48.0, "lot": 0.1,
    }, signal_fn=_sig_pivot)


# --------------------------------------------------------------------------
# #48 / #50 包み足(高安圏 / パーフェクトオーダー + 高安ブレイク)
# --------------------------------------------------------------------------
def _sig_engulf_zone(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    zone_lb = max(2, int(p["zone_lookback"].value))
    mode = int(p["mode"].value)          # 0=高安圏の包み足 1=EMA配列方向のみ
    e1 = max(2, int(p["ema_fast"].value))
    e2 = max(3, int(p["ema_mid"].value))
    e3 = max(4, int(p["ema_slow"].value))
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bull = (c.shift(1) < o.shift(1)) & (c > o) & (o <= c.shift(1)) & (c >= o.shift(1))
    bear = (c.shift(1) > o.shift(1)) & (c < o) & (o >= c.shift(1)) & (c <= o.shift(1))

    if mode == 0:
        near_low = l <= l.rolling(zone_lb).min().shift(1) * 1.0 + 0.0
        near_low = l <= l.rolling(zone_lb).min()
        near_high = h >= h.rolling(zone_lb).max()
        long_sig = bull & near_low
        short_sig = bear & near_high
    else:
        a1, a2, a3 = ema(c, e1), ema(c, e2), ema(c, e3)
        long_sig = bull & (a1 > a2) & (a2 > a3)
        short_sig = bear & (a1 < a2) & (a2 < a3)

    a = atr(h, l, c, 14)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


for _nm, _md in (("cat_engulf_zone", 0.0), ("cat_engulf_po", 1.0)):
    templates.register(_nm, defaults={
        "zone_lookback": 20.0, "mode": _md,
        "ema_fast": 20.0, "ema_mid": 50.0, "ema_slow": 100.0,
        "atr_mult": 1.5, "rr": 1.5,
        "sl_pips": 20.0, "tp_pips": 30.0, "max_hold_bars": 48.0, "lot": 0.1,
    }, signal_fn=_sig_engulf_zone)


# --------------------------------------------------------------------------
# #30 ATRスクイーズ → 価格ブレイク + ATRブレイク
# --------------------------------------------------------------------------
def _sig_atr_box_break(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    box_n = max(3, int(p["box_bars"].value))
    atr_len = max(2, int(p["atr_period"].value))
    sq_pct = abs(float(p["squeeze_pct"].value))      # ATR百分位がこれ以下でスクイーズ
    atr_break = abs(float(p["atr_break"].value))     # ATRが直前平均の何倍で「ATR突破」
    sl_atr = abs(float(p["sl_atr"].value))
    rr = abs(float(p["rr"].value))

    h, l, c = df["high"], df["low"], df["close"]
    a = atr(h, l, c, atr_len)
    rank = a.rolling(100).rank(pct=True)
    squeezed = (rank.shift(1) <= sq_pct)
    up = h.rolling(box_n).max().shift(1)
    dn = l.rolling(box_n).min().shift(1)
    atr_up = a > a.shift(1).rolling(box_n).mean() * atr_break

    long_sig = squeezed & (c > up) & atr_up
    short_sig = squeezed & (c < dn) & atr_up
    dist = (sl_atr * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


templates.register("cat_atr_box_break", defaults={
    "box_bars": 30.0, "atr_period": 14.0, "squeeze_pct": 0.3, "atr_break": 1.2,
    "sl_atr": 2.0, "rr": 1.5,
    "sl_pips": 30.0, "tp_pips": 45.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_atr_box_break)


# --------------------------------------------------------------------------
# #31 CCI±100 + ADX25 + 実体の大きいローソク足(マーチンは実装しない)
# --------------------------------------------------------------------------
def _sig_cci_adx_pa(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    cci_len = max(2, int(p["cci_period"].value))
    cci_lvl = abs(float(p["cci_level"].value))
    adx_len = max(2, int(p["adx_period"].value))
    adx_min = abs(float(p["adx_min"].value))
    body_atr = abs(float(p["body_atr"].value))
    mult = max(2, int(p["htf_mult"].value))
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = atr(h, l, c, 14)
    cc = _cci(df, cci_len)
    ax = _adx(df, adx_len)
    body = (c - o).abs()
    big_body = body >= body_atr * a

    def up_htf(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        return (ema(gc, 20) > ema(gc, 50)).fillna(False).to_numpy()

    def dn_htf(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        return (ema(gc, 20) < ema(gc, 50)).fillna(False).to_numpy()

    up = pd.Series(_htf_flag(df, mult, up_htf), index=df.index)
    dn = pd.Series(_htf_flag(df, mult, dn_htf), index=df.index)

    long_sig = up & (cc > cci_lvl) & (ax >= adx_min) & big_body & (c > o)
    short_sig = dn & (cc < -cci_lvl) & (ax >= adx_min) & big_body & (c < o)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


templates.register("cat_cci_adx_pa", defaults={
    "cci_period": 14.0, "cci_level": 100.0, "adx_period": 14.0, "adx_min": 25.0,
    "body_atr": 1.0, "htf_mult": 4.0, "atr_mult": 1.5, "rr": 1.5,
    "sl_pips": 20.0, "tp_pips": 30.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_cci_adx_pa)


# --------------------------------------------------------------------------
# #17 EMA20/80 の MTF 押し目(H4環境 → H1ゾーン → 実行足で反転足)
# --------------------------------------------------------------------------
def _sig_ema2080_mtf(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    mult = max(2, int(p["htf_mult"].value))
    e_fast = max(2, int(p["ema_fast"].value))
    e_slow = max(3, int(p["ema_slow"].value))
    rr = abs(float(p["rr"].value))
    buf_atr = abs(float(p["sl_buffer_atr"].value))

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    def up_htf(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        return (ema(gc, e_fast) > ema(gc, e_slow)).fillna(False).to_numpy()

    def dn_htf(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        return (ema(gc, e_fast) < ema(gc, e_slow)).fillna(False).to_numpy()

    up = pd.Series(_htf_flag(df, mult, up_htf), index=df.index)
    dn = pd.Series(_htf_flag(df, mult, dn_htf), index=df.index)

    ef, es = ema(c, e_fast), ema(c, e_slow)
    zone_lo = pd.concat([ef, es], axis=1).min(axis=1)
    zone_hi = pd.concat([ef, es], axis=1).max(axis=1)
    in_zone_up = l <= zone_hi
    in_zone_dn = h >= zone_lo

    bull = (c.shift(1) < o.shift(1)) & (c > o) & (o <= c.shift(1)) & (c >= o.shift(1))
    bear = (c.shift(1) > o.shift(1)) & (c < o) & (o >= c.shift(1)) & (c <= o.shift(1))

    long_sig = up & in_zone_up & bull
    short_sig = dn & in_zone_dn & bear

    a = atr(h, l, c, 14).to_numpy()
    ca = c.to_numpy()
    ls = long_sig.fillna(False).to_numpy()
    ss = short_sig.fillna(False).to_numpy()
    dist = np.where(ls, ca - (zone_lo.to_numpy() - buf_atr * a),
                    np.where(ss, (zone_hi.to_numpy() + buf_atr * a) - ca, np.nan))
    return _emit(len(df), ls, ss, ca, dist, rr)


templates.register("cat_ema2080_mtf", defaults={
    "htf_mult": 4.0, "ema_fast": 20.0, "ema_slow": 80.0,
    "sl_buffer_atr": 0.5, "rr": 2.0,
    "sl_pips": 25.0, "tp_pips": 50.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_ema2080_mtf)


# --------------------------------------------------------------------------
# #13 ストキャス + BB±2σ の合流
# --------------------------------------------------------------------------
def _sig_stoch_bb(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ma_len = max(2, int(p["ma_period"].value))
    sig_mult = abs(float(p["sigma"].value))
    k = max(2, int(p["k_period"].value))
    d = max(1, int(p["d_period"].value))
    slowing = max(1, int(p["slowing"].value))
    lo, hi = float(p["low"].value), float(p["high"].value)
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    c = df["close"]
    mid = sma(c, ma_len)
    sd = c.rolling(ma_len).std(ddof=0)
    up, dn = mid + sig_mult * sd, mid - sig_mult * sd
    kk, dd = _stoch(df, k, d, slowing)
    gc = (kk.shift(1) <= dd.shift(1)) & (kk > dd)
    dc = (kk.shift(1) >= dd.shift(1)) & (kk < dd)

    long_sig = (df["low"] <= dn) & (kk <= lo) & gc
    short_sig = (df["high"] >= up) & (kk >= hi) & dc
    a = atr(df["high"], df["low"], c, 14)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


templates.register("cat_stoch_bb", defaults={
    "ma_period": 20.0, "sigma": 2.0, "k_period": 14.0, "d_period": 3.0, "slowing": 3.0,
    "low": 20.0, "high": 80.0, "atr_mult": 1.5, "rr": 1.5,
    "sl_pips": 20.0, "tp_pips": 30.0, "max_hold_bars": 48.0, "lot": 0.1,
}, signal_fn=_sig_stoch_bb)


# --------------------------------------------------------------------------
# #9 BBレンジ反発(ミドル横ばい時に±σを抜けて戻る)
# --------------------------------------------------------------------------
def _sig_bb_range_bounce(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ma_len = max(2, int(p["ma_period"].value))
    sig_mult = abs(float(p["sigma"].value))
    flat_atr = abs(float(p["flat_atr"].value))     # ミドルの傾きがATRのこの割合以下なら横ばい
    rr = abs(float(p["rr"].value))
    atr_mult = abs(float(p["atr_mult"].value))

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    mid = sma(c, ma_len)
    sd = c.rolling(ma_len).std(ddof=0)
    up, dn = mid + sig_mult * sd, mid - sig_mult * sd
    a = atr(h, l, c, 14)
    flat = (mid - mid.shift(5)).abs() <= flat_atr * a

    broke_up = (c.shift(1) > up.shift(1))
    broke_dn = (c.shift(1) < dn.shift(1))
    back_dn = broke_up & (c < o)          # 抜けた次の足で戻る気配(陰線)
    back_up = broke_dn & (c > o)

    long_sig = flat & back_up
    short_sig = flat & back_dn
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


templates.register("cat_bb_range_bounce", defaults={
    "ma_period": 20.0, "sigma": 2.0, "flat_atr": 0.5, "atr_mult": 1.5, "rr": 1.0,
    "sl_pips": 20.0, "tp_pips": 20.0, "max_hold_bars": 24.0, "lot": 0.1,
}, signal_fn=_sig_bb_range_bounce)


# --------------------------------------------------------------------------
# #46 EMA10/25/45/75 の抜け(S/Rラインは裁量のため条件から外す)
# --------------------------------------------------------------------------
def _sig_ema4_break(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    e1 = max(2, int(p["ema1"].value))
    e2 = max(3, int(p["ema2"].value))
    e4 = max(5, int(p["ema4"].value))
    atr_mult = abs(float(p["atr_mult"].value))
    rr = abs(float(p["rr"].value))
    c, h, l = df["close"], df["high"], df["low"]
    a1, a2, a4 = ema(c, e1), ema(c, e2), ema(c, e4)
    # 10EMAと25EMAが75EMAを上抜けた後、ローソク足全体(安値も)が10EMAの上
    above = (a1 > a4) & (a2 > a4) & (l > a1)
    below = (a1 < a4) & (a2 < a4) & (h < a1)
    long_sig = above & ~above.shift(1).fillna(False)
    short_sig = below & ~below.shift(1).fillna(False)
    a = atr(h, l, c, 20)
    dist = (atr_mult * a).to_numpy()
    return _emit(len(df), long_sig.fillna(False).to_numpy(), short_sig.fillna(False).to_numpy(),
                 c.to_numpy(), dist, rr)


templates.register("cat_ema4_break", defaults={
    "ema1": 10.0, "ema2": 25.0, "ema4": 75.0, "atr_mult": 2.0, "rr": 1.5,
    "sl_pips": 30.0, "tp_pips": 45.0, "max_hold_bars": 96.0, "lot": 0.1,
}, signal_fn=_sig_ema4_break)


# --------------------------------------------------------------------------
# #3 上位足EMA20/75 の順張り環境 → 実行足でEMAへの押し/戻りを反発足で取る
# --------------------------------------------------------------------------
def _sig_ema_pullback_mtf(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    mult = max(2, int(p["htf_mult"].value))
    hf = max(2, int(p["htf_fast"].value))
    hs = max(3, int(p["htf_slow"].value))
    e_pull = max(2, int(p["ema_pull"].value))
    e_turn = max(2, int(p["ema_turn"].value))
    buf = abs(float(p["sl_buffer_atr"].value))
    rr = abs(float(p["rr"].value))

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    def htf_up(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        f, s = ema(gc, hf), ema(gc, hs)
        # 原文の「両者が上向き」「価格は速いEMAの上」まで含める
        return ((f > s) & (f > f.shift(1)) & (s > s.shift(1)) & (gc > f)).fillna(False).to_numpy()

    def htf_dn(g: pd.DataFrame) -> np.ndarray:
        gc = g["close"]
        f, s = ema(gc, hf), ema(gc, hs)
        return ((f < s) & (f < f.shift(1)) & (s < s.shift(1)) & (gc < f)).fillna(False).to_numpy()

    up = pd.Series(_htf_flag(df, mult, htf_up), index=df.index)
    dn = pd.Series(_htf_flag(df, mult, htf_dn), index=df.index)

    ep, et = ema(c, e_pull), ema(c, e_turn)
    # 押し = 実行足の安値が押し目EMAまで到達したこと
    pulled_up = l <= ep
    pulled_dn = h >= ep
    # 反発足 = 陽線(陰線)確定 かつ 速いEMAがその足で向きを変えた
    turn_up = (c > o) & (et > et.shift(1)) & (et.shift(1) <= et.shift(2))
    turn_dn = (c < o) & (et < et.shift(1)) & (et.shift(1) >= et.shift(2))

    long_sig = up & pulled_up & turn_up
    short_sig = dn & pulled_dn & turn_dn

    a = atr(h, l, c, 14).to_numpy()
    ca = c.to_numpy()
    ls = long_sig.fillna(False).to_numpy()
    ss = short_sig.fillna(False).to_numpy()
    # SL は原文どおり「エントリー足の安値(高値)の少し外」
    dist = np.where(ls, ca - (l.to_numpy() - buf * a),
                    np.where(ss, (h.to_numpy() + buf * a) - ca, np.nan))
    return _emit(len(df), ls, ss, ca, dist, rr)


templates.register("cat_ema_pullback_mtf", defaults={
    "htf_mult": 5.0, "htf_fast": 20.0, "htf_slow": 75.0,
    "ema_pull": 20.0, "ema_turn": 5.0, "sl_buffer_atr": 0.3, "rr": 1.5,
    "sl_pips": 8.0, "tp_pips": 12.0, "max_hold_bars": 60.0, "lot": 0.1,
}, signal_fn=_sig_ema_pullback_mtf)
