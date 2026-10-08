"""1分足エンベロープ逆張り(envelope_fade)テンプレート。

ぶせなブログ公開手法の機械化(2026-07-15 事前登録):
- M1 close の 20EMA からの乖離率(%)がゾーン帯 [zone_lo, zone_hi) に到達したら
  EMA方向へ逆張り(下に乖離→買い、上に乖離→売り)。
- 反転確認(サイトの「1〜2ティック反転」のM1バー近似): 買いは陽線確定、売りは陰線確定。
  confirm=0 で無効化(ロバスト性チェック用)。
- 再エントリー禁止ルール: 同一ゾーンで一度発火したら、価格が20EMAへ戻る
  (乖離符号がリセット)まで同方向の再シグナルを出さない(サイトの「EMA接触でリセット」)。
  ※外側ゾーン進行での再エントリー(パターンB)は外側ゾーンの別バックテストが担当する近似。
- TP/SL は engine 側 tp_pips/sl_pips(ゾーン別 2/3/4/5/6 pips、1:1)。

先読み回避: EMA・乖離・ローソク色は全てバーi確定情報のみ。engine.run_backtest の
shift(1) が「確認足の次バー始値でエントリー」を保証(サイトの手順と同型)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _first_per_epoch(cand: np.ndarray, reset: np.ndarray) -> np.ndarray:
    """EMAリセット区間(epoch)ごとに最初の候補だけ残す。"""
    out = np.zeros(len(cand), dtype=bool)
    idx = np.flatnonzero(cand)
    if len(idx) == 0:
        return out
    epoch = np.cumsum(reset)
    ep = epoch[idx]
    keep = np.concatenate([[True], ep[1:] != ep[:-1]])
    out[idx[keep]] = True
    return out


def _signal_envelope_fade(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    ma_period = max(2, int(strategy.params["ma_period"].value))
    zone_lo = abs(float(strategy.params["zone_lo_pct"].value))
    zone_hi = abs(float(strategy.params["zone_hi_pct"].value))
    confirm = int(strategy.params["confirm"].value) if "confirm" in strategy.params else 1
    side = int(strategy.params["side"].value) if "side" in strategy.params else 0

    close = df["close"]
    op = df["open"]
    ema = close.ewm(span=ma_period, adjust=False).mean()
    dev_pct = ((close - ema) / ema * 100.0).to_numpy()

    in_long_zone = (dev_pct <= -zone_lo) & (dev_pct > -zone_hi)
    in_short_zone = (dev_pct >= zone_lo) & (dev_pct < zone_hi)
    if confirm:
        bull = (close > op).to_numpy()
        bear = (close < op).to_numpy()
        in_long_zone &= bull
        in_short_zone &= bear

    fire_long = _first_per_epoch(in_long_zone, dev_pct >= 0)
    fire_short = _first_per_epoch(in_short_zone, dev_pct <= 0)

    signal = pd.Series(0, index=df.index, dtype=int)
    if side >= 0:
        signal[fire_long] = 1
    if side <= 0:
        signal[fire_short] = -1
    return signal


templates.register(
    "envelope_fade",
    defaults={
        "ma_period": 20.0,
        "zone_lo_pct": 0.10,
        "zone_hi_pct": 0.15,
        "confirm": 1.0,
        "side": 0.0,       # 0=両方向, 1=買いのみ, -1=売りのみ
        "sl_pips": 2.0,
        "tp_pips": 2.0,
        "max_hold_bars": 60.0,  # M1×60=1時間の安全弁(サイトは数秒〜数分)
        "lot": 0.1,
    },
    signal_fn=_signal_envelope_fade,
)
