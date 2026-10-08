"""セッション時間帯アンカーのオープニングレンジブレイクテンプレート(Phase10)。

指定した時間帯(session_hour、UTC基準)以降にその日最初に到達したバーをセッション開始と
みなし、そこからrange_bars本で作られる値幅(オープニングレンジ)を記録する。レンジ形成完了後
active_bars本の間にレンジを上/下に終値で抜けたら、その方向に追随する。時間帯アンカーという
点でseasonalテンプレートと親戚だが、固定の曜日×時間窓ではなく「レンジ形成→ブレイク追随」と
いう値幅ベースの構造を持つ、seasonalとbreakoutのハイブリッドにあたる新設計。

先読み回避についての設計メモ: セッションの状態(いつ開始したか、レンジがいくつか、既に
ブレイク済みか)は日付境界をまたいで前のバーの結果に依存する経路依存(path-dependent)な
ロジックである。これをpandas groupby等でベクトル化しようとすると、日付境界や週末ギャップ
(金曜終値の翌バーが月曜になる等)でオフバイワンのような先読みバグを混入しやすい。そのため
本テンプレートに限り、engine.pyのバーループと同じ「時系列順の単純な1パス走査」を採用する。
1パスの逐次走査であれば「バーiの時点でi以前(i自身を含む、確定済みの高値・安値・終値)の
情報しか参照していない」ことがループを読むだけで自明であり、監査が容易になる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_session_range_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    session_hour = int(strategy.params["session_hour"].value) % 24
    range_bars = max(1, int(strategy.params["range_bars"].value))
    active_bars = max(1, int(strategy.params["active_bars"].value))

    hour_arr = df["timestamp"].dt.hour.to_numpy()
    date_arr = df["timestamp"].dt.date.to_numpy()
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    signal = np.zeros(n, dtype=int)

    current_session_date = None  # 直近にセッション開始とみなした日付(datetime.date)
    session_start_i = -1
    session_high = -np.inf
    session_low = np.inf
    broken = False  # 当セッションで既にレンジを抜けたか(1セッション1回のみシグナル)
    prev_date = None  # ギャップ検出用(週末等でデータが欠けている日数を見る)
    blocked_date = None  # ギャップ直後と判定しアンカー禁止にした日付(その日いっぱい継続)

    for i in range(n):
        d = date_arr[i]
        if d != prev_date:
            # 週末等で2日以上データが飛んだ直後の日(例: 金曜21時台の最終バーの次が
            # 日曜21〜23時台のブローカー再開バー)は、session_hour以降最初のバーという条件
            # だけで判定すると薄商いの市場再開直後を「セッション開始」として丸1日誤アンカー
            # し続けてしまう(FABLE監査で発見: 実データでsession_hour=8時指定でも日曜21-23時台
            # 以降の全バーがアンカー対象になり、全セッションの96%でシグナルが出ていた=
            # フォワードで再現しない薄商いのマイクロストラクチャだった)。日付が変わった瞬間
            # だけギャップを判定してblocked_dateに記録し、その日はアンカーを見送って
            # 翌日以降の通常ギャップでのsession_hour到達を待つ。
            if prev_date is not None and (d - prev_date).days >= 2:
                blocked_date = d
            prev_date = d
        if d != current_session_date and hour_arr[i] >= session_hour and d != blocked_date:
            # その日でsession_hour以降に到達した最初のバー = 新セッション開始。
            # session_hourがバー境界と厳密一致しなくても(例: H4でsession_hour=8だが
            # 8時ちょうどのバーが無い)、条件を満たす最初のバーを開始点とするため
            # どの時間足でも機能する。
            current_session_date = d
            session_start_i = i
            session_high = high[i]
            session_low = low[i]
            broken = False
            continue
        if session_start_i < 0:
            continue  # まだ一度もセッションが開始していない(データ先頭がsession_hour未満)
        bars_since_start = i - session_start_i
        if bars_since_start < range_bars:
            # オープニングレンジ形成中: high/lowを拡張するだけでシグナルは出さない
            session_high = max(session_high, high[i])
            session_low = min(session_low, low[i])
            continue
        if bars_since_start >= range_bars + active_bars:
            continue  # アクティブ期間終了、このセッションではもうシグナルを出さない
        if not broken:
            if close[i] > session_high:
                signal[i] = 1
                broken = True
            elif close[i] < session_low:
                signal[i] = -1
                broken = True

    return pd.Series(signal, index=df.index, dtype=int)


templates.register(
    "session_range_breakout",
    defaults={
        "session_hour": 8.0,
        "range_bars": 4.0,
        "active_bars": 20.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_session_range_breakout,
)
