"""200EMA大乖離×ローソク足パターン(ema_dev_candle)テンプレート。

X投稿「200EMAとローソク足が大きく乖離してる時に"出る足"で急騰を当てる」の検証
(2026-07-16 事前登録)。投稿は足種を非公開のため、強反転/強モメンタム代表3種
(ピンバー・包み足・丸坊主)×乖離条件の相互作用クラスとして検定する。

定義(標準的、事前登録):
- 乖離条件: |close - EMA200| >= dev_atr_min × ATR14
- pattern 1=ピンバー: 逆ヒゲ >= 2×実体 かつ 逆ヒゲ >= 0.6×レンジ(下ヒゲ=強気/上ヒゲ=弱気)
- pattern 2=包み足: 実体が前バー実体を包含し色が反転
- pattern 3=丸坊主: 実体 >= 0.8×レンジ(陽=強気/陰=弱気)
- 側一致: EMAより下に乖離→強気足のみ / 上に乖離→弱気足のみ(fade方向)
- direction 1=fade(乖離の回帰=急騰当て解釈) / -1=follow(乖離方向へ順張り)
- 出口: SL=atr_sl_mult×ATR / TP=atr_tp_mult×ATR(sl_price/tp_price列)+max_hold_bars

先読み回避: EMA/ATR/足判定は全てバーi確定情報。engineのshift(1)が次バー始値執行を保証。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema
from app.core.strategy_model import Strategy


def _signal_ema_dev_candle(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    ema_p = max(2, int(strategy.params["ema_period"].value))
    atr_p = max(2, int(strategy.params["atr_period"].value))
    k = abs(float(strategy.params["dev_atr_min"].value))
    pattern = int(strategy.params["pattern"].value)
    direction = 1 if float(strategy.params["direction"].value) >= 0 else -1
    sl_m = abs(float(strategy.params["atr_sl_mult"].value))
    tp_m = abs(float(strategy.params["atr_tp_mult"].value))

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    e = ema(df["close"], ema_p).to_numpy()
    a = atr(df["high"], df["low"], df["close"], atr_p).to_numpy()

    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    up_wick = h - np.maximum(o, c)
    dn_wick = np.minimum(o, c) - l

    if pattern == 1:  # ピンバー
        bull = (dn_wick >= 2 * body) & (dn_wick >= 0.6 * rng)
        bear = (up_wick >= 2 * body) & (up_wick >= 0.6 * rng)
    elif pattern == 2:  # 包み足
        po, pc = np.roll(o, 1), np.roll(c, 1)
        bull = (c > o) & (pc < po) & (c >= po) & (o <= pc)
        bear = (c < o) & (pc > po) & (c <= po) & (o >= pc)
        bull[0] = bear[0] = False
    elif pattern == 3:  # 丸坊主
        bull = (c > o) & (body >= 0.8 * rng)
        bear = (c < o) & (body >= 0.8 * rng)
    else:  # 4/5: 連続逆色足(>=2本)→打ち消し大足(X投稿の画像仕様)
        red = c < o
        green = c > o
        # 直近で「前2本が陰線」(階段状の下げ)→当バー陽線が前バー高値を上抜き
        r1 = np.roll(red, 1); r2 = np.roll(red, 2)
        g1 = np.roll(green, 1); g2 = np.roll(green, 2)
        ph = np.roll(h, 1); pl = np.roll(l, 1)
        b1 = np.roll(body, 1); b2 = np.roll(body, 2)
        bull = green & r1 & r2 & (c > ph)
        bear = red & g1 & g2 & (c < pl)
        if pattern == 5:  # 強: 実体が直近2本の実体合計以上
            bull &= body >= (b1 + b2)
            bear &= body >= (b1 + b2)
        bull[:2] = bear[:2] = False

    dev = c - e
    below = dev <= -k * a  # 下方乖離
    above = dev >= k * a   # 上方乖離
    valid = np.isfinite(e) & np.isfinite(a) & (a > 0)

    # fade: 下方乖離×強気足→買い / 上方乖離×弱気足→売り。followはミラー。
    long_sig = below & bull & valid
    short_sig = above & bear & valid
    if direction == -1:  # follow: 上方乖離×強気足→買い / 下方乖離×弱気足→売り
        long_sig = above & bull & valid
        short_sig = below & bear & valid

    side = int(strategy.params["side"].value) if "side" in strategy.params else 0
    if side > 0:
        short_sig[:] = False
    elif side < 0:
        long_sig[:] = False

    n = len(df)
    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan)
    tp_price = np.full(n, np.nan)
    signal[long_sig] = 1
    signal[short_sig] = -1
    sl_price[long_sig] = c[long_sig] - sl_m * a[long_sig]
    tp_price[long_sig] = c[long_sig] + tp_m * a[long_sig]
    sl_price[short_sig] = c[short_sig] + sl_m * a[short_sig]
    tp_price[short_sig] = c[short_sig] - tp_m * a[short_sig]

    return pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price},
                        index=df.index)


templates.register(
    "ema_dev_candle",
    defaults={
        "ema_period": 200.0,
        "atr_period": 14.0,
        "dev_atr_min": 2.0,
        "pattern": 1.0,      # 1=ピンバー 2=包み足 3=丸坊主
        "direction": 1.0,    # 1=fade / -1=follow
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 2.0,
        "side": 0.0,         # 0=両方向 1=買いのみ -1=売りのみ

        "sl_pips": 500.0,    # 動的price優先のための保険
        "tp_pips": 500.0,
        "max_hold_bars": 96.0,
        "lot": 0.1,
    },
    signal_fn=_signal_ema_dev_candle,
)
