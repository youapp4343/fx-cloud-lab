"""曜日・時間帯の季節性アノマリーテンプレート(Phase9)。

指定した曜日+時間帯の窓に入った瞬間にのみ指定方向へエントリーする。窓を外れても
既存ポジションは自動決済しない(trade_hoursフィルタのような「新規エントリーのみ制限」
とは異なり、時間窓そのものをエントリー条件として扱う設計のため)。決済はsl_pips/tp_pips/
max_hold_bars等、既存のrun_backtestの通常の決済ロジックにそのまま委ねる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _entering_mask(hit: pd.Series) -> pd.Series:
    """直前バーがhit=Falseで当バーがhit=Trueの「立ち上がり」だけをTrueにする。

    bool dtypeのSeriesに対してpandasの.shift(1)はNaN境界のためobject dtypeに
    昇格し、続く.fillna(False)がFutureWarning(silent downcasting)を出すため、
    numpy配列で同等のロジックを組む(先頭バーはhit[-1]相当をFalse扱い)。
    """
    hit_arr = hit.to_numpy()
    prev_arr = np.concatenate(([False], hit_arr[:-1]))
    return pd.Series(hit_arr & ~prev_arr, index=hit.index)


def _signal_seasonal(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    weekday = int(strategy.params["weekday"].value) % 7
    hour_start = int(strategy.params["hour_start"].value) % 24
    # hour_endは%24でラップしない: 24は「その日の終端まで」を表す正当な値であり、
    # ラップすると[0,24)(=1日中、seasonal_scanのhour_bucket_size=24が返しうる値)が
    # (hour>=0)&(hour<0)=常に偽という空集合になってしまう不具合があった(FABLE監査で発見)。
    hour_end = max(0, min(24, int(strategy.params["hour_end"].value)))
    direction = 1 if strategy.params["direction"].value >= 0 else -1

    ts = df["timestamp"]
    weekday_match = ts.dt.dayofweek == weekday
    if hour_start <= hour_end:
        hour_match = (ts.dt.hour >= hour_start) & (ts.dt.hour < hour_end)
        hit = weekday_match & hour_match
    else:
        # 日またぎ(例: weekday=月, hour_start=22, hour_end=2 →「月22:00〜火02:00」の
        # 1つの連続窓)。weekday条件を「hour>=start」「hour<end」それぞれに独立でANDすると
        # 「月22:00〜24:00」と「月00:00〜02:00」という同一曜日内の無関係な2窓になってしまう
        # 不具合があった(FABLE監査で発見)。正しくは窓の後半を翌曜日に割り当てる。
        next_weekday = (weekday + 1) % 7
        part1 = weekday_match & (ts.dt.hour >= hour_start)
        part2 = (ts.dt.dayofweek == next_weekday) & (ts.dt.hour < hour_end)
        hit = part1 | part2

    # 窓の立ち上がりバーのみシグナルを出す(毎バー出すと同じ窓内でドテンを繰り返してしまうため)。
    entering = _entering_mask(hit)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[entering] = direction
    return signal


templates.register(
    "seasonal",
    defaults={
        "weekday": 0.0,
        "hour_start": 8.0,
        "hour_end": 16.0,
        "direction": 1.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_seasonal,
)
