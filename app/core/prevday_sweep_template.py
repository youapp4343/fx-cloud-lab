"""前日安値リクイディティ・スイープ後の買い テンプレート(手法#13)。

出典: 「前日安値を一瞬下抜けてから戻す(=下のストップ/逆指値注文を刈ってから反発する)」
値動きを狙う順張り的逆張り(スイープ&リクレイム)。前日高値側のショートはミラーで実装
するが、元記事は「買いにはエッジがあるが売りは負ける」としており、既定はdir_mode=1
(ロングのみ)。曜日(火曜)・時間帯(東京-ロンドン)フィルタはこのテンプレートの責務外で、
検証スクリプト側でengineのtrade_hoursフィルタとして適用する前提(このファイルでは
曜日・時刻の判定は一切行わない)。

前日高安の計算(先読み厳禁):
- timestamp(UTC, tz-naive)をUTC暦日でグルーピングし、日足(D1)のlow/highに集約する
  (`groupby(day)` はpandasの単純集約であり、価格データの未来参照なし)。
- その日足シリーズを1行shift(1)する。これは「暦日で1日前」ではなく「データが存在する
  直前の日」を指す(週末・祝日でバーが無い日はそもそも行が無いため、月曜は自動的に
  直前金曜の値を参照する=週末スキップが副産物として得られる)。
- shift済みの日足を各イントラデイ足へ日付キーでブロードキャストする。ある日Dの
  1本目のバーであっても、参照する値は「日Dの集約(まだ未完成/存在しない)」ではなく
  「日D-1の完成済み集約」であるため、当日データの部分的な混入は構造的に起きない。

近似(docstring明記、必読):
- D1のバケットはUTC暦日(タスク指定通り)。ブローカーのサーバー日足境界(例: NY17:00
  クローズ基準)とは一致しない可能性がある近似。
- 「Re-arm per day」は「1暦日につきロング1回・ショート1回まで」として実装した
  (条件が同日内で再度成立しても2回目以降は発火しない)。前バーとの比較によるエッジ
  トリガではなく、「その日まだ発火していないか」を見る日次フラグで判定する。
- pip定義は symbol.upper().endswith("JPY") の簡易版(タスク指定通り。app/core/symbols.py
  の厳密版とは別、ema10_easy_template.pyと同じ簡易パターン)。

先読み規律: 前日高安はshift(1)済みの日足集約のみを参照するため構造的に前日以前の
確定データのみを用いる(上記の通り)。当バーの判定は当バーのlow/high/closeと日次フラグ
(自分の過去状態)のみを使用し、未来のバーを一切参照しない。執行shiftはengine側
(raw_signal.shift(1))が行うため、テンプレート内での追加シフトは行わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_prevday_sweep(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    buffer_pips = float(p["buffer_pips"].value)
    dir_mode = int(p["dir_mode"].value)  # 0=both,1=long,2=short

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    pip = 0.01 if strategy.symbol.upper().endswith("JPY") else 0.0001
    buffer = buffer_pips * pip

    day = df["timestamp"].dt.floor("D")  # UTC暦日キー(バー開始時刻ベース)

    daily = df.groupby(day).agg(low=("low", "min"), high=("high", "max"))
    prevday_low_by_day = daily["low"].shift(1)   # 直前に存在する日の安値(先読み無し)
    prevday_high_by_day = daily["high"].shift(1)  # 直前に存在する日の高値

    prevday_low = day.map(prevday_low_by_day).to_numpy(dtype=float)
    prevday_high = day.map(prevday_high_by_day).to_numpy(dtype=float)

    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    day_arr = day.to_numpy()

    allow_long = dir_mode in (0, 1)
    allow_short = dir_mode in (0, 2)

    sig = np.zeros(n, dtype=int)

    fired_long_today = False
    fired_short_today = False
    cur_day = None

    for t in range(n):
        if day_arr[t] != cur_day:  # 日替わりで発火フラグをリセット(=再アーム)
            cur_day = day_arr[t]
            fired_long_today = False
            fired_short_today = False

        pl = prevday_low[t]
        if allow_long and not fired_long_today and not np.isnan(pl):
            if low[t] < pl - buffer and close[t] > pl:  # 下抜け(スイープ)+終値で奪還
                sig[t] = 1
                fired_long_today = True

        ph = prevday_high[t]
        if allow_short and not fired_short_today and not np.isnan(ph):
            if sig[t] == 0 and high[t] > ph + buffer and close[t] < ph:  # 上抜け+終値で奪還
                sig[t] = -1
                fired_short_today = True

    return pd.Series(sig, index=df.index, dtype=int)


templates.register(
    "prevday_sweep",
    defaults={
        "buffer_pips": 3.0,
        "dir_mode": 1.0,
        "sl_pips": 30.0,
        "tp_pips": 45.0,
        "lot": 0.1,
    },
    signal_fn=_signal_prevday_sweep,
)
