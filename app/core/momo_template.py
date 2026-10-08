"""Momo(モメンタム・リバーサル)テンプレート — Kathy Lien の5分足 "Momo" 手法の機械化。

出典: Kathy Lien "Day Trading and Swing Trading the Currency Market" の5分足モメンタム戦略。
EMA20 と MACD(12,26,9) を併用し、価格がEMA20の反対側から自分側へ抜け、かつMACDが
ゼロラインを同方向にクロスした「モメンタムの反転」を捉える順張り手法。

機械化した核:
- LONG: 直近まで価格がEMA20の下(prev close < EMA20)かつMACDヒストグラムが直近で負
  だった状態から、当バーで close がEMA20を上抜け、かつMACDライン(macd_line)が直近
  `macd_lookback` 本以内にゼロを上抜けている。
- SHORT: 上記の完全なミラー(EMA20の上・hist>0・close下抜け・MACDゼロ下抜け)。

近似の注記: 原法の「EMA20から+10pip抜けた指値でエントリー」は、pip幅の指値執行を
バックテスト基盤で厳密再現できないため *確定した確認バーでの成行* で近似する(engine側の
raw_signal.shift(1) により実際の約定は確認バーの翌バー始値)。この近似はエントリーを
やや不利側(遅い・水準を追う)に倒すため、+10pip指値より楽観的にはならない。また
原法の「2本目のバーで手仕舞い/トレール」等の裁量決済は捨て、決済はengineの sl/tp に委ねる。

先読み規律: signal_fn はshiftしない生シグナル(1/-1/0)を返す。翌バー始値執行のshiftは
engine.run_backtest 側が担う。EMAクロス検出とMACDゼロクロス検出の `.shift(1)`、および
「直近N本以内」を測る rolling は *当バーと自分の過去バーのみ* を参照するため先読みではない
(ma_pullback / tdi_cross と同じ流儀)。ウォームアップのNaNは比較でFalseに落ち無シグナルとなる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import ema, macd
from app.core.strategy_model import Strategy


def _within(mask: pd.Series, lookback: int) -> pd.Series:
    """maskが直近lookback本(当バー含む)に1度でもTrueだったかを返す。

    当バーで終わる後方windowのみを見るため先読みは無い。min_periods=1でウォームアップ中も
    部分windowで評価する(指標側のNaNにより実質的なシグナルは指標が揃うまで出ない)。
    """
    return mask.astype(float).rolling(lookback, min_periods=1).sum() > 0


def _signal_momo(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ema_period = max(2, int(p["ema_period"].value))
    macd_fast = max(1, int(p["macd_fast"].value))
    macd_slow = max(2, int(p["macd_slow"].value))
    macd_signal = max(1, int(p["macd_signal"].value))
    macd_lookback = max(1, int(p["macd_lookback"].value))
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    close = df["close"]
    line = ema(close, ema_period)
    macd_line, _macd_sig, hist = macd(close, macd_fast, macd_slow, macd_signal)

    # EMA20クロス検出(自分の過去バー参照): 当バーでEMAを跨いだ最初の点のみTrue。
    # (close > line) 側の cross_up は「prev close < EMA20 → 当バー上抜け」を内包する。
    pc, pl = close.shift(1), line.shift(1)
    cross_up_ema = (close > line) & (pc <= pl)
    cross_dn_ema = (close < line) & (pc >= pl)

    # MACDラインのゼロクロス検出(自分の過去バー参照)。
    ml_prev = macd_line.shift(1)
    zero_up = (macd_line > 0.0) & (ml_prev <= 0.0)
    zero_dn = (macd_line < 0.0) & (ml_prev >= 0.0)

    long_raw = (
        cross_up_ema
        & _within(hist < 0.0, macd_lookback)     # 直近まで下向きモメンタムだった
        & _within(zero_up, macd_lookback)        # MACDが直近でゼロを上抜けた
    ).fillna(False)
    short_raw = (
        cross_dn_ema
        & _within(hist > 0.0, macd_lookback)
        & _within(zero_dn, macd_lookback)
    ).fillna(False)

    # 再アーム: 条件が連続バーで張り付いても、離れて再成立した最初のバーのみ発火。
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "momo",
    defaults={
        "ema_period": 20.0,
        "macd_fast": 12.0, "macd_slow": 26.0, "macd_signal": 9.0,
        "macd_lookback": 5.0, "dir_mode": 0.0,
        "sl_pips": 20.0, "tp_pips": 20.0, "lot": 0.1,
    },
    signal_fn=_signal_momo,
)
