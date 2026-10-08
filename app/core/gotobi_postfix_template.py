# -*- coding: utf-8 -*-
"""ゴトー日 仲値"後"ショート(実稼働中の戦略)のPython実装。

既存の `gotobi_nakane` テンプレは **JST7時エントリーの仲値"前"ロング**で、
これは過去に却下済みの変種([[project_gotobi_nakane_rejected]])。
実運用しているのは **09:55 JST エントリー → 10:25 JST 決済のショート**で、
これまで MQL(`mql/GotobiFixShort_JPY.mq5`)にしか存在しなかった。

ThreeTrader の実ティックで検定するために Python 側にも起こす。
MQL原文の入力そのまま:
    InpEntryJstHour = 9   / InpEntryJstMin = 55
    InpExitJstHour  = 10  / InpExitJstMin  = 25   (30分保有)
    InpSlPips = 0 / InpTpPips = 0                 (時間決済のみ)
    InpMaxSpreadPips = 3.0                        (★仲値スプレッド上限。核心の関門)

ゴトー日の定義も原文どおり: 5,10,15,20,25,月末。土日は前営業日(金)に繰り上げ。

★データはUTC。JST = UTC + 9。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _is_gotobi(ts: pd.Series) -> np.ndarray:
    """JSTの日付でゴトー日か判定する。土日は前営業日へ繰り上げ。"""
    d = pd.DataFrame({"y": ts.dt.year, "m": ts.dt.month, "d": ts.dt.day,
                      "dow": ts.dt.dayofweek})
    # 月末日
    last = ts.dt.to_period("M").dt.to_timestamp("M").dt.day
    base = d.d.isin([5, 10, 15, 20, 25]) | (d.d == last)

    # 繰り上げ: 元のゴトー日が土(5)なら金(-1)、日(6)なら金(-2)に移る
    out = base.to_numpy().copy()
    for shift, dow in ((1, 5), (2, 6)):
        cand = ts + pd.Timedelta(days=shift)
        cl = cand.dt.to_period("M").dt.to_timestamp("M").dt.day
        moved = (cand.dt.day.isin([5, 10, 15, 20, 25]) | (cand.dt.day == cl)) \
            & (cand.dt.dayofweek == dow) & (ts.dt.dayofweek == 4)
        out |= moved.to_numpy()
    return out


def _sig_gotobi_postfix(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ts = pd.to_datetime(df["timestamp"])
    jst = ts + pd.Timedelta(hours=9)

    eh, em = int(p["entry_jst_hour"].value), int(p["entry_jst_min"].value)
    xh, xm = int(p["exit_jst_hour"].value), int(p["exit_jst_min"].value)
    ent_min = eh * 60 + em
    exit_min = xh * 60 + xm
    hold = max(1, exit_min - ent_min)     # 分。バー幅で本数に直す

    mod = jst.dt.hour * 60 + jst.dt.minute
    # バー幅を推定して「エントリー時刻を含むバー」を取る
    step = int(pd.Series(ts).diff().dt.total_seconds().median() // 60) or 5
    at_entry = (mod >= ent_min) & (mod < ent_min + step)
    gotobi = _is_gotobi(jst)

    sig = np.zeros(len(df), dtype=int)
    hit = at_entry.to_numpy() & gotobi

    # ★仲値スプレッド上限。原文の核心の関門。実測 sp_med で判定できる
    maxsp = float(p["max_spread_pips"].value)
    if maxsp > 0 and "sp_med" in df.columns:
        hit &= (df["sp_med"].to_numpy() <= maxsp) | ~np.isfinite(df["sp_med"].to_numpy())

    sig[hit] = -1                          # ショート
    # SL/TPは原文で0(時間決済のみ)。保有本数をバー数で渡す
    n_bar = max(1, int(round(hold / step)))
    return pd.DataFrame({"signal": sig,
                         "sl_price": np.nan, "tp_price": np.nan,
                         "hold_bars": n_bar}, index=df.index)


templates.register("gotobi_postfix_short", defaults={
    "entry_jst_hour": 9.0, "entry_jst_min": 55.0,
    "exit_jst_hour": 10.0, "exit_jst_min": 25.0,
    "max_spread_pips": 3.0,
    # 時間決済のみ。SL/TPは広く置いて実質無効化する
    "sl_pips": 500.0, "tp_pips": 500.0, "max_hold_bars": 6.0, "lot": 0.1,
}, signal_fn=_sig_gotobi_postfix)

# スプレッド関門を外した版(関門の効きを測るための対照)
templates.register("gotobi_postfix_nogate", defaults={
    "entry_jst_hour": 9.0, "entry_jst_min": 55.0,
    "exit_jst_hour": 10.0, "exit_jst_min": 25.0,
    "max_spread_pips": 0.0,
    "sl_pips": 500.0, "tp_pips": 500.0, "max_hold_bars": 6.0, "lot": 0.1,
}, signal_fn=_sig_gotobi_postfix)
