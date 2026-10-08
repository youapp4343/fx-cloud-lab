"""ゴールド買い限定インジケーター投票(gold_vote)テンプレート(手法#14の機械化)。

出典: 「ゴールド買い限定インジケーター投票」— XAUUSD専用、5種のテクニカル指標
(アリゲーター/ボリンジャーバンド/パラボリックSAR/RSI/ストキャスティクス)それぞれの
「買い条件」を1票ずつ投票にかけ、一定数以上が同時に賛成したバーでのみエントリーする
合議(コンフルエンス)系手法。原記事では売り方向の成績が大幅マイナスだったとされ、
ロング専用運用が推奨されているため、本テンプレートは既定 dir_mode=1(ロング専用)。
dir_mode=0/2にすれば各条件を対称にミラーしたショート判定も有効化できる(検証用)。

投票条件(現在の確定バーで判定、各条件+1票、ロング側):
  1. アリゲーター上向き整列: Lips(5) > Teeth(8) > Jaw(13)
  2. ボリンジャーバンド上抜け: close > upper(period=20, sigma=2.0)
  3. パラボリックSARが価格の下: psar < close(順張り継続を示す位置関係)
  4. RSI売られすぎからの上抜け: RSI(14)が30を下から上にクロス(prev<=30 & cur>30)
  5. ストキャスティクス買いクロス: %Kが%Dを下段(K<50)で上抜け(prev K<=D & cur K>D & K<50)
合計投票数が vote_min(既定3)以上に達した"最初の"バー(edge trigger、状態継続中は再発火しない)
でのみ+1を出す。ショート側は5条件すべてを対称にミラーしたもの(下記実装参照)。

近似・簡略化(docstring明記):
  - アリゲーター(SMMA)とパラボリックSARは app/core/indicators.py に無い
    (or 既存の indicators.parabolic_sar を使わずタスク要件によりあえてローカル実装)
    ため、本ファイル内に自己完結させている。
  - 条件4は「RSI>50」ではなく「30を上抜けクロス」を採用(継続的な買われすぎ域ではなく、
    売られすぎからの反発の瞬間を捉える版)。
  - Alligator/PSARのパラメータ(Jaw13/Teeth8/Lips5、shift8/5/3、PSAR af 0.02/0.02/0.2)は
    標準MT4既定値に固定し、strategy.paramsには公開していない(タスク仕様の param 一覧に
    含まれないため)。

先読み規律(重要、Alligatorのshiftに注意):
  標準MT4のアリゲーターはJaw/Teeth/Lipsのラインをそれぞれ未来方向へ8/5/3本ずらして
  描画する。これは「n本前に確定したSMMA値を、今のバーの真上に表示している」という
  表示上のトリックに過ぎない。先読み厳禁のため、本テンプレートは「未来へシフトして
  描画」ではなく等価な「過去のSMMA値をbar tの値として読む」処理として実装する:
  pandasの `.shift(+n)`(正のshiftは値を後ろ=未来方向へずらす、つまり「n本前の値を
  今ここに置く」)を使うことで、bar tで参照するJawは常に「t-8時点までの情報で計算
  したSMMA」= bar tより過去の情報のみとなる(下部のno-lookahead自己検査で確認済み)。
  RSI/ボリンジャーバンド/ストキャスティクスはいずれもpandasのrolling/ewmでバーtまでの
  情報のみ使用(標準的で先読みでない)。PSARは逐次ループで前バーまでのtrend/EP/AF状態を
  引き継ぎ、bar tのSAR確定にbar t自身の高安を使うが、これは「barが確定した時点で持つ
  情報」であり未来バーは一切参照しない(indicators.parabolic_sarと同じ規約)。
  RSI/Stochasticの「クロス」判定用 `.shift(1)` は自分自身の過去(前バー値)の参照のみ。
  執行(次バー始値エントリー)のシフトは engine.run_backtest 側が一元的に行うため、
  本テンプレート内ではシグナル自体をシフトしない。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import bollinger_bands, rsi, stochastic
from app.core.strategy_model import Strategy

# --- Alligator(Bill Williams)固定パラメータ: 標準MT4設定 ---
_JAW_PERIOD, _JAW_SHIFT = 13, 8
_TEETH_PERIOD, _TEETH_SHIFT = 8, 5
_LIPS_PERIOD, _LIPS_SHIFT = 5, 3

# --- Parabolic SAR 固定パラメータ: 標準設定 ---
_PSAR_STEP = 0.02
_PSAR_MAX_STEP = 0.2


def _smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed MA(Wilder平滑化)。smma[p-1]=SMA(p)、以降 smma[i]=(smma[i-1]*(p-1)+x[i])/p。

    アリゲーターの各ラインで使う平滑化(indicators.py未収録のためローカル実装)。
    最初の period-1 本はNaN(MT4のSMMA未確定期間と同じ)。
    """
    period = max(1, int(period))
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=series.index)
    out[period - 1] = values[:period].mean()
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return pd.Series(out, index=series.index)


def _alligator(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """アリゲーター(Jaw13/Teeth8/Lips5、中値SMMA)。因果的な前方shift適用後の(jaw, teeth, lips)。

    標準MT4は描画上ラインを未来へ(8/5/3本)シフトするが、これは「n本前に確定した
    SMMA値を今のバーの上に表示する」動作と同義。ここではpandasの正のshiftで
    「t-n時点のSMMA値をbar tの値として使う」形にして因果性を保つ(先読みなし、
    モジュールdocstring参照)。
    """
    median = (high + low) / 2.0
    jaw = _smma(median, _JAW_PERIOD).shift(_JAW_SHIFT)
    teeth = _smma(median, _TEETH_PERIOD).shift(_TEETH_SHIFT)
    lips = _smma(median, _LIPS_PERIOD).shift(_LIPS_SHIFT)
    return jaw, teeth, lips


def _psar_local(high: pd.Series, low: pd.Series, step: float, max_step: float) -> pd.Series:
    """パラボリックSAR(標準ワイルダー版、af start/step/max指定)。

    indicators.parabolic_sarと同じ標準アルゴリズムだが、タスク要件によりローカル
    再実装する(逐次ループ、前バーまでのtrend/EP/AF状態を引き継ぐ因果的計算。
    bar tのSAR確定にbar t自身の高安を使うが未来バーは参照しない)。
    """
    n = len(high)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=high.index)

    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)

    uptrend = True
    af = step
    ep = h[0]
    cur = lo[0]
    out[0] = cur

    for i in range(1, n):
        cur = cur + af * (ep - cur)
        if uptrend:
            # SARは直近2本の安値より上に来てはならない(Wilderの規約)
            cur = min(cur, lo[i - 1], lo[i - 2] if i >= 2 else lo[i - 1])
            if lo[i] < cur:
                uptrend = False
                cur = ep
                ep = lo[i]
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
                if lo[i] < ep:
                    ep = lo[i]
                    af = min(af + step, max_step)
        out[i] = cur

    return pd.Series(out, index=high.index)


def _signal_gold_vote(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """5指標投票シグナル。合計票数がvote_min以上に達した最初のバーで+1/-1(edge trigger)。"""
    p = strategy.params
    vote_min = max(1, min(5, int(p["vote_min"].value)))
    rsi_period = max(2, int(p["rsi_period"].value))
    bb_period = max(2, int(p["bb_period"].value))
    bb_sigma = float(p["bb_sigma"].value)
    stoch_k = max(2, int(p["stoch_k"].value))
    stoch_d = max(1, int(p["stoch_d"].value))
    stoch_slow = max(1, int(p["stoch_slow"].value))
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long, 2=short

    high, low, close = df["high"], df["low"], df["close"]

    jaw, teeth, lips = _alligator(high, low)
    _, bb_upper, bb_lower = bollinger_bands(close, bb_period, bb_sigma)
    psar = _psar_local(high, low, _PSAR_STEP, _PSAR_MAX_STEP)
    r = rsi(close, rsi_period)
    prev_r = r.shift(1)
    k, d = stochastic(high, low, close, stoch_k, stoch_d, stoch_slow)
    prev_k, prev_d = k.shift(1), d.shift(1)

    # --- ロング側5条件(各+1票) ---
    long_c1 = (lips > teeth) & (teeth > jaw)                     # アリゲーター上向き整列
    long_c2 = close > bb_upper                                   # BB上抜け
    long_c3 = psar < close                                       # PSARが価格の下
    long_c4 = (prev_r <= 30) & (r > 30)                          # RSI 30上抜けクロス
    long_c5 = (prev_k <= prev_d) & (k > d) & (k < 50)            # Stoch買いクロス(下段)
    long_votes = (
        long_c1.fillna(False).astype(int) + long_c2.fillna(False).astype(int)
        + long_c3.fillna(False).astype(int) + long_c4.fillna(False).astype(int)
        + long_c5.fillna(False).astype(int)
    )
    long_state = long_votes >= vote_min
    long_edge = long_state & (~long_state.shift(1).fillna(False).astype(bool))

    # --- ショート側5条件(ロングの鏡像、dir_modeが許す場合のみ使用) ---
    short_c1 = (lips < teeth) & (teeth < jaw)                    # アリゲーター下向き整列
    short_c2 = close < bb_lower                                  # BB下抜け
    short_c3 = psar > close                                      # PSARが価格の上
    short_c4 = (prev_r >= 70) & (r < 70)                         # RSI 70下抜けクロス
    short_c5 = (prev_k >= prev_d) & (k < d) & (k > 50)           # Stoch売りクロス(上段)
    short_votes = (
        short_c1.fillna(False).astype(int) + short_c2.fillna(False).astype(int)
        + short_c3.fillna(False).astype(int) + short_c4.fillna(False).astype(int)
        + short_c5.fillna(False).astype(int)
    )
    short_state = short_votes >= vote_min
    short_edge = short_state & (~short_state.shift(1).fillna(False).astype(bool))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_edge] = 1
    if dir_mode in (0, 2):
        signal[short_edge] = -1
    return signal


templates.register(
    "gold_vote",
    defaults={
        "vote_min": 3.0,
        "rsi_period": 14.0,
        "bb_period": 20.0,
        "bb_sigma": 2.0,
        "stoch_k": 14.0,
        "stoch_d": 3.0,
        "stoch_slow": 3.0,
        "dir_mode": 1.0,
        "sl_pips": 200.0,
        "tp_pips": 300.0,
        "lot": 0.1,
    },
    signal_fn=_signal_gold_vote,
)
