"""フロー系スキャル(flow_scalp)テンプレート。

背景(2026-08-11):
Adaptive Flow Scalper の検証(project_adaptive_flow_scalper_null)で、失敗した急変への
逆張りに **gross +0.18 pips の実在エッジ** が確認されたが、M1・往復コスト約1.6pips に
負けて収益化できなかった。

ここで問い直すのは保有時間である。方向的エッジは保有時間におおむね比例して伸びる一方、
コストは1トレードあたり固定なので、同じエッジでも保有を伸ばせばコスト比は改善する
(統計的有意性=t値はホライズンに不変だが、収益性=コスト比は改善する。この2つは別物)。
またUSDJPYの実測往復コストは0.8pipsで、EURJPY/GBPJPYの3.0-3.8pipsとは別物である
(過去のスキャル探索はJPYクロスの高コストに引きずられていた)。

よってM5/M15で15〜60分保有の帯を、フロー3特徴で検証する。

特徴(全て確定バーのみ):
  imbalance : 直近lookback本の「符号付き出来高」比率。
              sum(sign(close-open) * volume) / sum(volume) ∈ [-1, 1]
              dukascopyのvolumeはティック数なので、売買方向別の出来高ではなく
              「値動きの向き×活況度」の代理指標である点に注意。
  run_len   : 同方向に連続した本数
  range_exp : 当バーのレンジ / ATR。急変の度合い

シグナル:
  mode >= 0 (fade)  : |imbalance| >= imb_thr かつ range_exp >= exp_thr で **逆方向**
                      (急変後の行き過ぎ戻り。afsで実在が確認された側)
  mode <  0 (follow): 同条件で **同方向**(フロー継続)
  run_min を満たす連続本数も条件に加える(0で無効)

決済は engine 側の sl_pips / tp_pips / max_hold_bars に委ねる。
先読み回避: 全ての量はバーiまでの確定値のみ。engine.run_backtest が shift(1) するため
実際の執行は次バー始値になる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _run_length(sign: pd.Series) -> pd.Series:
    """同符号が連続している本数(当バーを含む)。符号0は連続を切る。"""
    s = sign.to_numpy()
    out = np.zeros(len(s), dtype=float)
    cur = 0.0
    prev = 0.0
    for i in range(len(s)):
        if s[i] == 0:
            cur = 0.0
        elif s[i] == prev:
            cur += 1.0
        else:
            cur = 1.0
        out[i] = cur
        prev = s[i]
    return pd.Series(out, index=sign.index)


def _signal_flow_scalp(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    lookback = max(2, int(p["lookback"].value))
    imb_thr = abs(float(p["imb_thr"].value))
    exp_thr = abs(float(p["exp_thr"].value))
    run_min = max(0, int(p["run_min"].value))
    atr_period = max(2, int(p["atr_period"].value))
    mode = 1 if float(p["mode"].value) >= 0 else -1   # +1=fade(逆張り) / -1=follow(順張り)

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    sign = np.sign(c - o)
    signed_vol = sign * vol
    imbalance = signed_vol.rolling(lookback).sum() / vol.rolling(lookback).sum().replace(0, np.nan)

    a = atr(h, l, c, atr_period)
    range_exp = (h - l) / a.replace(0, np.nan)

    runs = _run_length(pd.Series(sign, index=df.index))

    hot = (range_exp >= exp_thr) & (runs >= run_min if run_min > 0 else True)
    up_flow = (imbalance >= imb_thr) & hot
    dn_flow = (imbalance <= -imb_thr) & hot

    signal = pd.Series(0, index=df.index, dtype=int)
    # fade: 上向きフロー→売り / follow: 上向きフロー→買い
    signal[up_flow.fillna(False)] = -1 * mode
    signal[dn_flow.fillna(False)] = 1 * mode
    return signal


templates.register(
    "flow_scalp",
    defaults={
        "lookback": 6.0,       # M5なら30分、M15なら90分ぶんのフロー
        "imb_thr": 0.5,        # 符号付き出来高比率のしきい値
        "exp_thr": 1.5,        # レンジ/ATR のしきい値(急変判定)
        "run_min": 0.0,        # 連続本数の下限(0で無効)
        "atr_period": 14.0,
        "mode": 1.0,           # +1=fade / -1=follow
        "sl_pips": 12.0,
        "tp_pips": 12.0,
        "max_hold_bars": 6.0,  # M5なら30分、M15なら90分で時間退出
        "lot": 0.1,
    },
    signal_fn=_signal_flow_scalp,
)


# --- 変種 --------------------------------------------------------------------
# グリッドが振るのは sl_pips/tp_pips/atr_period/exp_thr の4つだけなので、
# 方向モードと保有時間は別テンプレートとして登録し、探索の対象に含める。
# (同じ探索空間内で並べることで、多重比較の補正対象にも自動的に含まれる)

templates.register(
    "flow_scalp_follow",
    defaults={
        "lookback": 6.0, "imb_thr": 0.5, "exp_thr": 1.5, "run_min": 0.0,
        "atr_period": 14.0,
        "mode": -1.0,            # フロー継続(順張り)
        "sl_pips": 12.0, "tp_pips": 12.0, "max_hold_bars": 6.0, "lot": 0.1,
    },
    signal_fn=_signal_flow_scalp,
)

templates.register(
    "flow_scalp_hold",
    defaults={
        "lookback": 6.0, "imb_thr": 0.5, "exp_thr": 1.5, "run_min": 0.0,
        "atr_period": 14.0,
        "mode": 1.0,             # fade
        "sl_pips": 20.0, "tp_pips": 20.0,
        "max_hold_bars": 12.0,   # M5で60分保有(コスト比が更に改善する帯)
        "lot": 0.1,
    },
    signal_fn=_signal_flow_scalp,
)
