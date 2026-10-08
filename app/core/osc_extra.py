"""追加テクニカル指標ライブラリ(indicators.pyと直交する系統)。pandas Seriesを受け取る純粋関数群。"""

from typing import Tuple

import numpy as np
import pandas as pd

from app.core.indicators import atr, ema, rsi, sma


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """DMI(方向性指数)。Wilderの平滑化による(+DI, -DI)を返す。ADXの土台。"""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr_smooth = atr(high, low, close, period)
    plus_dm_smooth = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = (100.0 * plus_dm_smooth / tr_smooth).where(tr_smooth != 0)
    minus_di = (100.0 * minus_dm_smooth / tr_smooth).where(tr_smooth != 0)
    return plus_di, minus_di


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX(平均方向性指数)。+DI/-DIの乖離度(DX)をさらにWilder平滑化したトレンド強度。"""
    plus_di, minus_di = dmi(high, low, close, period)
    denom = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / denom).where(denom != 0)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def aroon_oscillator(high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
    """Aroon Oscillator。直近period+1本中の高値/安値到達位置の差(Aroon Up - Aroon Down)。"""
    up = 100.0 * high.rolling(period + 1).apply(lambda x: x.argmax(), raw=True) / period
    down = 100.0 * low.rolling(period + 1).apply(lambda x: x.argmin(), raw=True) / period
    return up - down


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Vortex Indicator。上昇/下降の運動量比を表す(VI+, VI-)を返す。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    tr_sum = tr.rolling(period).sum()
    vi_plus = (vm_plus.rolling(period).sum() / tr_sum).where(tr_sum != 0)
    vi_minus = (vm_minus.rolling(period).sum() / tr_sum).where(tr_sum != 0)
    return vi_plus, vi_minus


def efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman効率比(KER)。正味変化/経路長の絶対値和。0(レンジ)〜1(強トレンド)。"""
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (change / volatility).where(volatility != 0)


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Choppiness Index。100に近いほどレンジ相場、0に近いほど強トレンド。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr_sum = tr.rolling(period).sum()
    rng = high.rolling(period).max() - low.rolling(period).min()
    ratio = (tr_sum / rng).where(rng != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * np.log10(ratio) / np.log10(period)


def hurst_exponent(close: pd.Series, window: int = 100) -> pd.Series:
    """Hurst指数(ラグ別分散のログ回帰による簡易推定)。0.5=ランダムウォーク、>0.5=トレンド性、<0.5=平均回帰性。

    各時点で直近window本のみを使うローリング推定(未来データ不使用)。
    """
    max_lag = max(int(window // 5), 2)
    lags = np.arange(2, max_lag + 1)

    def _hurst(x: np.ndarray) -> float:
        taus = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        valid = taus > 0
        if valid.sum() < 2:
            return np.nan
        slope = np.polyfit(np.log(lags[valid]), np.log(taus[valid]), 1)[0]
        return slope

    return close.rolling(window=window).apply(_hurst, raw=True)


def schaff_trend_cycle(close: pd.Series, fast: int = 23, slow: int = 50, cycle: int = 10) -> pd.Series:
    """Schaff Trend Cycle。MACDを二重ストキャスティクス+再帰平滑化したサイクル系オシレーター。0-100。

    再帰平滑(factor=0.5)のため経路依存であり、逐次ループで計算する。
    """
    macd_line = (ema(close, fast) - ema(close, slow)).to_numpy()
    ll1 = pd.Series(macd_line, index=close.index).rolling(cycle).min().to_numpy()
    hh1 = pd.Series(macd_line, index=close.index).rolling(cycle).max().to_numpy()
    n = len(macd_line)

    f1 = np.full(n, np.nan)
    pf = np.full(n, np.nan)
    for i in range(n):
        if hh1[i] != hh1[i] or ll1[i] != ll1[i]:
            continue
        rng = hh1[i] - ll1[i]
        f1[i] = (macd_line[i] - ll1[i]) / rng * 100.0 if rng > 0 else (f1[i - 1] if i > 0 else np.nan)
        prev_pf = pf[i - 1] if i > 0 else np.nan
        pf[i] = f1[i] if prev_pf != prev_pf else prev_pf + 0.5 * (f1[i] - prev_pf)

    pf_series = pd.Series(pf, index=close.index)
    ll2 = pf_series.rolling(cycle).min().to_numpy()
    hh2 = pf_series.rolling(cycle).max().to_numpy()

    f2 = np.full(n, np.nan)
    stc = np.full(n, np.nan)
    for i in range(n):
        if hh2[i] != hh2[i] or ll2[i] != ll2[i]:
            continue
        rng2 = hh2[i] - ll2[i]
        f2[i] = (pf[i] - ll2[i]) / rng2 * 100.0 if rng2 > 0 else (f2[i - 1] if i > 0 else np.nan)
        prev_stc = stc[i - 1] if i > 0 else np.nan
        stc[i] = f2[i] if prev_stc != prev_stc else prev_stc + 0.5 * (f2[i] - prev_stc)

    return pd.Series(stc, index=close.index)


def wavetrend(high: pd.Series, low: pd.Series, close: pd.Series, n1: int = 10, n2: int = 21) -> Tuple[pd.Series, pd.Series]:
    """WaveTrend Oscillator(LazyBear版)。(wt1, wt2=wt1の4期間SMA)を返す。"""
    ap = (high + low + close) / 3.0
    esa = ema(ap, n1)
    d = ema((ap - esa).abs(), n1)
    ci = ((ap - esa) / (0.015 * d)).where(d != 0)
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)
    return wt1, wt2


def inverse_fisher_rsi(close: pd.Series, rsi_period: int = 14) -> pd.Series:
    """Inverse Fisher Transform of RSI(Ehlers)。RSIを9期間WMA平滑後に非線形圧縮し-1〜+1へ。"""
    v1 = 0.1 * (rsi(close, rsi_period) - 50.0)
    weights = np.arange(1, 10, dtype=float)
    wma = v1.rolling(9).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return (np.exp(2.0 * wma) - 1.0) / (np.exp(2.0 * wma) + 1.0)


def laguerre_rsi(close: pd.Series, gamma: float = 0.5) -> pd.Series:
    """Laguerre RSI(Ehlers)。4段ラゲールフィルタによる低ラグRSI。0〜1。経路依存のためループ計算。"""
    p = close.to_numpy(dtype=float)
    n = len(p)
    l0 = l1 = l2 = l3 = 0.0
    out = np.full(n, np.nan)
    for i in range(n):
        l0_prev, l1_prev, l2_prev, l3_prev = l0, l1, l2, l3
        l0 = (1 - gamma) * p[i] + gamma * l0_prev
        l1 = -gamma * l0 + l0_prev + gamma * l1_prev
        l2 = -gamma * l1 + l1_prev + gamma * l2_prev
        l3 = -gamma * l2 + l2_prev + gamma * l3_prev

        cu = 0.0
        cd = 0.0
        if l0 >= l1:
            cu += l0 - l1
        else:
            cd += l1 - l0
        if l1 >= l2:
            cu += l1 - l2
        else:
            cd += l2 - l1
        if l2 >= l3:
            cu += l2 - l3
        else:
            cd += l3 - l2

        denom = cu + cd
        out[i] = cu / denom if denom != 0 else 0.5
    return pd.Series(out, index=close.index)


def dpo(close: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator。(period/2+1)本前の終値から現在のSMAを引きトレンド成分を除去。"""
    shift = period // 2 + 1
    return close.shift(shift) - sma(close, period)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume。終値の上下方向に応じて出来高を加減算する累積出来高指標。"""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def cvd(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """累積ボリュームデルタ(CVD)近似。ティック内訳がないため実体方向+終値位置から買い/売り優勢を推定し累積。"""
    rng = high - low
    body_ratio = ((close - open_) / rng).where(rng != 0, 0.0)
    clv_ratio = ((2.0 * close - high - low) / rng).where(rng != 0, 0.0)
    delta = volume * (body_ratio + clv_ratio) / 2.0
    return delta.cumsum()


def klinger_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    fast: int = 34, slow: int = 55,
) -> pd.Series:
    """Klinger Volume Oscillator。hlc3のトレンド反転を考慮した出来高フォース(vf)のEMA差分。

    トレンド継続/反転の判定(cm)が経路依存のため逐次ループで計算する。
    """
    hlc3 = ((high + low + close) / 3.0).to_numpy()
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    dm = h - l
    n = len(h)

    trend = np.zeros(n)
    cm = np.zeros(n)
    trend[0] = 1.0
    cm[0] = dm[0]
    for i in range(1, n):
        if hlc3[i] > hlc3[i - 1]:
            trend[i] = 1.0
        elif hlc3[i] < hlc3[i - 1]:
            trend[i] = -1.0
        else:
            trend[i] = trend[i - 1]
        cm[i] = (cm[i - 1] + dm[i]) if trend[i] == trend[i - 1] else (dm[i - 1] + dm[i])

    with np.errstate(divide="ignore", invalid="ignore"):
        dm_over_cm = np.where(cm != 0, dm / cm, 0.0)
    vf = volume.to_numpy(dtype=float) * np.abs(2.0 * dm_over_cm - 1.0) * trend * 100.0
    vf_series = pd.Series(vf, index=high.index)
    return ema(vf_series, fast) - ema(vf_series, slow)


def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Force Index(Elder)。価格変化×出来高をEMA平滑した値動きの勢い指標。"""
    return ema(close.diff() * volume, period)


def eom(high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Ease of Movement(Arms)。値幅の変動を出来高で正規化し平滑化(値が動きやすいかを測る)。"""
    distance = (high + low) / 2.0 - (high.shift(1) + low.shift(1)) / 2.0
    rng = high - low
    box_ratio = (volume / rng).where(rng != 0)
    emv = (distance / box_ratio).where(box_ratio != 0)
    return sma(emv, period)


def zscore(close: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score。直近period本の平均からの乖離を標準偏差単位で表す平均回帰指標。"""
    mean = sma(close, period)
    std = close.rolling(period).std(ddof=0)
    return ((close - mean) / std).where(std != 0)


def rci(close: pd.Series, period: int = 9) -> pd.Series:
    """RCI(順位相関指数)。日付順位と価格順位のスピアマン順位相関。-100〜+100。"""
    date_rank = np.arange(period, 0, -1).astype(float)  # 最新バー=1位 ... 最古バー=period位
    denom = period * (period**2 - 1)

    def _rci(x: np.ndarray) -> float:
        price_rank = pd.Series(x).rank(ascending=False).to_numpy()  # 高値=1位
        d2 = np.sum((date_rank - price_rank) ** 2)
        return (1.0 - 6.0 * d2 / denom) * 100.0

    return close.rolling(window=period).apply(_rci, raw=True)


def linreg_slope(close: pd.Series, period: int = 20) -> pd.Series:
    """線形回帰スロープ。直近period本の最小二乗直線の傾き(1本あたりの変化量)。"""
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        return np.dot(x - x_mean, y - y.mean()) / ss_xx

    return close.rolling(window=period).apply(_slope, raw=True)


def r_squared(close: pd.Series, period: int = 20) -> pd.Series:
    """決定係数R²。直近period本を線形回帰した際の当てはまりの良さ。0〜1、高いほど直線的トレンド。"""
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    def _r2(y: np.ndarray) -> float:
        y_mean = y.mean()
        ss_yy = np.sum((y - y_mean) ** 2)
        if ss_yy == 0:
            return np.nan
        slope = np.dot(x - x_mean, y - y_mean) / ss_xx
        return (slope**2 * ss_xx) / ss_yy

    return close.rolling(window=period).apply(_r2, raw=True)


def bop(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Balance of Power。1本の実体方向をその値幅で正規化(-1〜+1)。"""
    rng = high - low
    return ((close - open_) / rng).where(rng != 0)


def elder_bull_power(high: pd.Series, close: pd.Series, period: int = 13) -> pd.Series:
    """Elder's Bull Power。高値がEMA(close)をどれだけ上回るか(買い圧力)。"""
    return high - ema(close, period)


def elder_bear_power(low: pd.Series, close: pd.Series, period: int = 13) -> pd.Series:
    """Elder's Bear Power。安値がEMA(close)をどれだけ下回るか(売り圧力)。"""
    return low - ema(close, period)


def clv(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Close Location Value。レンジ内での終値位置(-1=安値〜+1=高値)。"""
    rng = high - low
    return (((close - low) - (high - close)) / rng).where(rng != 0)


def qstick(open_: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    """Qstick。(終値-始値)のperiod期間SMA。陽線/陰線の連続性・勢いを測る。"""
    return sma(close - open_, period)


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change。period本前からの変化率(%)。"""
    prev = close.shift(period)
    return (100.0 * (close - prev) / prev).where(prev != 0)


def rvi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    """Relative Vigor Index。4点加重平均(1:2:2:1)した実体/値幅比をperiod期間で平滑化。"""
    num = (
        (close - open_)
        + 2 * (close.shift(1) - open_.shift(1))
        + 2 * (close.shift(2) - open_.shift(2))
        + (close.shift(3) - open_.shift(3))
    ) / 6.0
    den = (
        (high - low)
        + 2 * (high.shift(1) - low.shift(1))
        + 2 * (high.shift(2) - low.shift(2))
        + (high.shift(3) - low.shift(3))
    ) / 6.0
    num_sum = num.rolling(period).sum()
    den_sum = den.rolling(period).sum()
    return (num_sum / den_sum).where(den_sum != 0)


def kst(
    close: pd.Series,
    roc_periods: Tuple[int, int, int, int] = (10, 15, 20, 30),
    sma_periods: Tuple[int, int, int, int] = (10, 10, 10, 15),
) -> pd.Series:
    """Know Sure Thing(Pring)。4本の異なる期間のROCをそれぞれ平滑化し加重(1:2:3:4)合成したモメンタム。"""
    weights = (1.0, 2.0, 3.0, 4.0)
    total = pd.Series(0.0, index=close.index)
    for rp, sp, w in zip(roc_periods, sma_periods, weights):
        total = total + w * sma(roc(close, rp), sp)
    return total


def trix(close: pd.Series, period: int = 15) -> pd.Series:
    """TRIX。3重EMA平滑化系列の変化率(%)。ノイズを抑えたモメンタム。"""
    triple_ema = ema(ema(ema(close, period), period), period)
    prev = triple_ema.shift(1)
    return (100.0 * (triple_ema - prev) / prev).where(prev != 0)


def ppo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Percentage Price Oscillator。MACDを百分率化し価格水準に依存しない比較を可能にする。"""
    slow_ema = ema(close, slow)
    return (100.0 * (ema(close, fast) - slow_ema) / slow_ema).where(slow_ema != 0)
