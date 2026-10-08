"""テクニカル指標ライブラリ。pandas Seriesを受け取る純粋関数群。"""

from typing import Tuple

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """単純移動平均。"""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移動平均。"""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI。Wilderの平滑化(alpha=1/period)を使用。"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        result = 100.0 - (100.0 / (1.0 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)  # 損失ゼロ(一定上昇)はRSI=100
    result = result.mask((avg_loss == 0) & (avg_gain == 0), 50.0)  # 変動なし(横ばい)はRSI=50
    return result


def bollinger_bands(
    series: pd.Series, period: int = 20, sigma: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ボリンジャーバンド。(中央, 上, 下)を返す。"""
    middle = sma(series, period)
    std = series.rolling(window=period).std(ddof=0)
    upper = middle + sigma * std
    lower = middle - sigma * std
    return middle, upper, lower


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range。Wilderの平滑化(alpha=1/period)を使用。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def donchian_channel(
    high: pd.Series, low: pd.Series, period: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """ドンチャンチャネル。(直近period本の最高値, 最安値)を返す。

    当該バーを除外するため shift(1) してからrollingする(先読みバイアス回避)。
    """
    upper = high.shift(1).rolling(window=period).max()
    lower = low.shift(1).rolling(window=period).min()
    return upper, lower


def macd(
    series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD。(macdライン, シグナルライン, ヒストグラム)を返す。標準的なEMA差分定義。"""
    macd_line = ema(series, fast_period) - ema(series, slow_period)
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3, slowing: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """ストキャスティクス(スロー)。(%K, %D)を返す。

    %K = SMA(生%K, slowing)、%D = SMA(%K, d_period)。MT4/MT5のiStochastic
    (MODE_SMA, STO_LOWHIGH)と同じ定義。レンジ0の退化バーはNaN。
    """
    lowest = low.rolling(window=k_period).min()
    highest = high.rolling(window=k_period).max()
    rng = highest - lowest
    raw_k = (100.0 * (close - lowest) / rng).where(rng != 0)
    k = raw_k.rolling(window=slowing).mean()
    d = k.rolling(window=d_period).mean()
    return k, d


def envelopes(
    series: pd.Series, period: int = 20, deviation_pct: float = 0.1
) -> Tuple[pd.Series, pd.Series]:
    """エンベロープ。SMA±deviation_pct%の(上バンド, 下バンド)を返す(MT4/MT5 iEnvelopesと同じ定義)。"""
    mid = sma(series, period)
    upper = mid * (1.0 + deviation_pct / 100.0)
    lower = mid * (1.0 - deviation_pct / 100.0)
    return upper, lower


def parabolic_sar(
    high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2
) -> pd.Series:
    """パラボリックSAR(Wilderの標準定義、MT4/MT5 iSARと同じパラメータ規約)。

    経路依存(トレンド状態・EP・AFを引き継ぐ)のため逐次ループで計算する。
    各バーのSAR値はそのバーの高安確定前に決まっている(前バーまでの情報から算出される)
    ため、当該バーの高安との比較にそのまま使ってよい(先読みなし)。
    """
    n = len(high)
    sar = pd.Series(index=high.index, dtype=float)
    if n < 2:
        return sar

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    out = sar.to_numpy()

    uptrend = True
    af = step
    ep = h[0]
    cur = l[0]
    out[0] = cur

    for i in range(1, n):
        cur = cur + af * (ep - cur)
        if uptrend:
            # SARは直近2本の安値より上に来てはならない(Wilderの規約)
            cur = min(cur, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < cur:
                uptrend = False
                cur = ep
                ep = l[i]
                af = step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, max_step)
        else:
            cur = max(cur, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > cur:
                uptrend = True
                cur = ep
                ep = h[i]
                af = step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + step, max_step)
        out[i] = cur

    return pd.Series(out, index=high.index)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Commodity Channel Index(MT4/MT5 iCCIと同じLambert定義、定数0.015)。"""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: (abs(x - x.mean())).mean(), raw=True)
    return ((tp - sma_tp) / (0.015 * mad)).where(mad != 0)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R。-100(売られすぎ)〜0(買われすぎ)。レンジ0はNaN。"""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    rng = hh - ll
    return (-100.0 * (hh - close) / rng).where(rng != 0)


def demarker(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """DeMarker。0〜1(0.3以下売られすぎ/0.7以上買われすぎ、MT4/MT5 iDeMarkerと同定義)。"""
    demax = (high - high.shift(1)).clip(lower=0.0)
    demin = (low.shift(1) - low).clip(lower=0.0)
    sma_max = demax.rolling(period).mean()
    sma_min = demin.rolling(period).mean()
    denom = sma_max + sma_min
    return (sma_max / denom).where(denom != 0)


def momentum(close: pd.Series, period: int = 14) -> pd.Series:
    """Momentum(MT4定義: close/close[period]*100、100が中立)。"""
    prev = close.shift(period)
    return (close / prev * 100.0).where(prev != 0)


def stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3) -> pd.Series:
    """Stochastic RSI(TradingView定義)。RSIのレンジ内位置を0-100で返す(K平滑のみ)。"""
    r = rsi(series, rsi_period)
    lo = r.rolling(stoch_period).min()
    hi = r.rolling(stoch_period).max()
    rng = hi - lo
    raw = (100.0 * (r - lo) / rng).where(rng != 0)
    return raw.rolling(k_smooth).mean()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index(出来高加重RSI)。volume全0なら全NaN。"""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    up = mf.where(tp > tp.shift(1), 0.0)
    dn = mf.where(tp < tp.shift(1), 0.0)
    pos = up.rolling(period).sum()
    neg = dn.rolling(period).sum()
    denom = pos + neg
    return (100.0 * pos / denom).where(denom != 0)


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                        p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Ultimate Oscillator(Larry Williams、4:2:1加重)。0-100。"""
    pc = close.shift(1)
    bp = close - pd.concat([low, pc], axis=1).min(axis=1)
    tr = pd.concat([high, pc], axis=1).max(axis=1) - pd.concat([low, pc], axis=1).min(axis=1)
    def _avg(p):
        s_tr = tr.rolling(p).sum()
        return (bp.rolling(p).sum() / s_tr).where(s_tr != 0)
    return 100.0 * (4 * _avg(p1) + 2 * _avg(p2) + _avg(p3)) / 7.0


def cmo(series: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator。-100〜+100。"""
    d = series.diff()
    up = d.clip(lower=0.0).rolling(period).sum()
    dn = (-d.clip(upper=0.0)).rolling(period).sum()
    denom = up + dn
    return (100.0 * (up - dn) / denom).where(denom != 0)


def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
    """True Strength Index(二重EMA平滑モメンタム)。-100〜+100。"""
    m = series.diff()
    num = m.ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
    den = m.abs().ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
    return (100.0 * num / den).where(den != 0)


def fisher_transform(high: pd.Series, low: pd.Series, period: int = 9) -> pd.Series:
    """Fisher Transform(Ehlers)。中値のレンジ位置をフィッシャー変換、経路依存平滑のためループ。"""
    mid = (high + low) / 2.0
    lo = mid.rolling(period).min()
    hi = mid.rolling(period).max()
    rng = (hi - lo).to_numpy()
    pos = ((mid - lo) / (hi - lo)).where((hi - lo) != 0).to_numpy()
    n = len(mid)
    x = 0.0
    out = [float("nan")] * n
    fish = 0.0
    for i in range(n):
        if rng[i] != rng[i] or pos[i] != pos[i]:  # NaN
            continue
        x = 0.66 * (pos[i] - 0.5) + 0.67 * x
        x = max(min(x, 0.999), -0.999)
        import math
        fish = 0.5 * math.log((1 + x) / (1 - x)) + 0.5 * fish
        out[i] = fish
    return pd.Series(out, index=high.index)


def connors_rsi(series: pd.Series, rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100) -> pd.Series:
    """Connors RSI(短期平均回帰の定番)。RSI(3)+連騰RSI(2)+ROC1のPercentRankの平均。"""
    r1 = rsi(series, rsi_period)
    d = series.diff()
    streak = pd.Series(0.0, index=series.index)
    s = 0.0
    dv = d.to_numpy()
    sv = streak.to_numpy()
    for i in range(1, len(series)):
        if dv[i] > 0:
            s = s + 1 if s > 0 else 1.0
        elif dv[i] < 0:
            s = s - 1 if s < 0 else -1.0
        else:
            s = 0.0
        sv[i] = s
    streak = pd.Series(sv, index=series.index)
    r2 = rsi(streak, streak_period)
    roc1 = series.pct_change()
    rank = roc1.rolling(rank_period).apply(lambda w: (w[:-1] < w[-1]).mean() * 100.0, raw=True)
    return (r1 + r2 + rank) / 3.0


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series,
                       volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow。-1〜+1。volume全0なら全NaN。"""
    rng = high - low
    clv = (((close - low) - (high - close)) / rng).where(rng != 0)
    num = (clv * volume).rolling(period).sum()
    den = volume.rolling(period).sum()
    return (num / den).where(den != 0)


def tdi(
    series: pd.Series,
    rsi_period: int = 13,
    price_period: int = 2,
    signal_period: int = 7,
    band_period: int = 34,
    band_mult: float = 1.618,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Traders Dynamic Index。(price_line, signal_line, market_base, upper, lower) を返す。

    RSIを平滑したprice_line(緑)とsignal_line(赤)のクロス、およびRSIのボリンジャー
    バンド(market_base=中央線, upper/lower=±band_mult*σ)で構成する定番の合議系指標。
    全て確定バーのRSIから rolling で計算する純粋関数のため、配布版MT4指標にある
    「過去線が後から書き換わる(リペイント)」問題は構造的に存在しない(記事#7の注意点への対策)。
    """
    r = rsi(series, rsi_period)
    price_line = r.rolling(window=price_period).mean()
    signal_line = r.rolling(window=signal_period).mean()
    market_base = r.rolling(window=band_period).mean()
    band_std = r.rolling(window=band_period).std(ddof=0)
    upper = market_base + band_mult * band_std
    lower = market_base - band_mult * band_std
    return price_line, signal_line, market_base, upper, lower
