"""月末ロンドンフィキシング・アノマリーテンプレート(Wave5 案C)。

ロンドンフィキシング(LDN Fix = ロンドン現地16:00のWM/Refinitivベンチマークレート決定)
に月末の機関投資家リバランス・ヘッジ調整フローが集中し、「fixへ向かう一方向の値動き」と
「fix通過後の巻き戻し」がアノマリーとして知られる。本テンプレートはその2仮説を実装する。

方向は事前仮説固定(スキャン却下): 月末イベントは年12回しかなく、OOSはペアあたり
36サンプル程度。この標本数で方向・時間窓をスキャンすると多重比較の偽陽性製造機になる
ため、文献仮説の2モードのみを固定実装する:
- into_fix (mode=0): fix前の参照区間(ロンドン13:00→15:00)の確定方向に順張り。
  シグナルはfixの2バー前(H1なら14:00開始バー)で確定し、engineのshift(1)により
  fix直前(15:00開始バー)の始値でエントリー。決済はsl/tp/max_hold_bars(fix後の決済)。
- post_fix_reversal (mode=1、本命): fixへ向かう参照区間(13:00→16:00)の確定方向の
  「逆」を取る。シグナルはfixちょうどに閉まるバー(H1なら15:00開始バー)で確定し、
  engineのshift(1)によりfix直後(16:00開始バー)の始値でエントリー。

時刻の扱い(DST対応): timestampはtz-naive UTC(プロジェクト規約)。fixはロンドン現地
16:00で、GMT期はUTC16:00 / BST期はUTC15:00と1時間ずれるため、zoneinfoベースの
`ts.dt.tz_localize("UTC").dt.tz_convert("Europe/London")` で現地壁時計に変換して判定する
(pytz不使用)。WindowsはOS付属のtzデータベースが無いためPyPIのtzdataパッケージが必須
(本リポジトリの実行環境でtzdata導入済み・Europe/LondonのDST境界動作を確認済み)。

月末営業日の判定は `ts.normalize() + BMonthEnd(0) == ts.normalize()`(pandasの純カレンダー
演算、価格データの未来参照なし)。days_before=k でその k 営業日前(T-k)を対象日にできる。
祝日(元日・クリスマス等)が月末営業日に当たる場合はpandas営業日(土日除外のみ)では
検出できないが、その日はそもそもバーが存在しない→シグナルバーの壁時計完全一致に失敗
→その月は自動スキップ、で整合する(許容仕様)。

方向決定と損益測定の起点分離(既知バグパターン①対策): 方向決定区間は「参照アンカー
バーの始値 → シグナルバーの終値」で完結し、エントリーはengine側のshift(1)により
翌バー始値で行われる。方向を決めた値動きそのものが損益に混入しない(event_tagging.pyで
実測・修正された定数バイアスの再発防止。テンプレート内での執行用shiftは禁止=engineが
shiftするため、ここで追加shiftすると二重shiftになる)。

バー境界の注意: シグナルバーは壁時計の完全一致で特定する(H1: post=ロンドン15:00開始
/into=14:00開始、M15: 15:45/15:30開始)。H4以上はバー境界がfixに一致しないため
シグナルは出ない(仕様)。fix直後バーが欠損している場合はengineの規約(shift後の次バー
始値執行)により、次に存在するバーの始値で執行される(稀なケース、保守側ではない点に注意)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, BMonthEnd

from app.core import templates
from app.core.strategy_model import Strategy

LONDON_TZ = "Europe/London"
LONDON_FIX_HOUR = 16  # LDN Fix = ロンドン現地16:00


def _is_target_business_day(day: pd.Timestamp, days_before: int) -> bool:
    """dayが「月末営業日のdays_before営業日前」かを純カレンダー演算で判定する。

    days_before=0 のとき `day + BMonthEnd(0) == day`(プラン正典の式)と等価。
    土日は `BDay(0)` が翌営業日へ丸めるため不一致になり自動的に除外される。
    カレンダー演算のみで価格データを参照しないため先読みは構造的に存在しない。
    """
    if day + BDay(0) != day:  # 営業日(月-金)でない
        return False
    return day + BDay(days_before) == day + BMonthEnd(0)


def _signal_month_end_fix(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    post_fix_reversal = strategy.params["mode"].value >= 0.5  # 0=into_fix / 1=post_fix_reversal
    days_before = max(0, int(strategy.params["days_before"].value))
    ref_hours = float(strategy.params["ref_hours"].value)

    signal = pd.Series(0, index=df.index, dtype=int)
    n = len(df)
    if n < 2 or ref_hours <= 0:
        return signal

    ts = df["timestamp"]
    bar_period = ts.diff().median()  # 中央値: 週末ギャップ(2〜48時間の外れ値)に頑健なバー幅推定
    if pd.isna(bar_period) or bar_period <= pd.Timedelta(0):
        return signal

    # ロンドン現地壁時計(tz変換後にnaive化)。DSTで壁時計が重複するのは10月最終日曜の
    # 深夜01-02時のみで、fix参照窓(平日午後)とは重ならないため、窓内の壁時計は時系列順に単調。
    # なお対象バー(ロンドン午後)はロンドン日付とUTC日付が常に一致する時間帯だが、
    # fixはロンドンのイベントなのでロンドン現地カレンダーを正とする。
    london = ts.dt.tz_localize("UTC").dt.tz_convert(LONDON_TZ).dt.tz_localize(None)
    london_day = london.dt.normalize()

    target_days = [
        d for d in london_day.unique() if _is_target_business_day(pd.Timestamp(d), days_before)
    ]
    on_target_day = london_day.isin(target_days)

    # シグナルバーの特定: post_fix_reversalは「fixちょうどに閉まるバー」(開始=fix−1バー幅)、
    # into_fixは「その1本前」(開始=fix−2バー幅。終値確定がfix−1バー幅時点 → engineのshift(1)で
    # 翌バー=fix直前バーの始値執行)。壁時計の完全一致で特定するため、祝日等でこのバーが
    # 欠損している月は自動的に無シグナル(=その月スキップ)。
    fix_dt = london_day + pd.Timedelta(hours=LONDON_FIX_HOUR)
    offset = bar_period if post_fix_reversal else 2 * bar_period
    is_signal_bar = (on_target_day & (london == fix_dt - offset)).to_numpy()

    london_arr = london.to_numpy()
    ref_start_arr = (fix_dt - pd.to_timedelta(ref_hours, unit="h")).to_numpy()
    open_arr = df["open"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    signal_arr = np.zeros(n, dtype=int)
    # 月1回ガード: 壁時計完全一致により構造的にも月1シグナルだが、重複timestamp等の
    # 異常データでも「月1回まで」を保証する(最初に現れたバーを採用)。
    signaled_months: set = set()

    for i in np.flatnonzero(is_signal_bar):
        ref_start = ref_start_arr[i]
        if london_arr[i] < ref_start:
            continue  # ref_hoursが小さすぎてシグナルバー自身が参照窓外(パラメータ縮退、安全側でスキップ)
        bar_ts = pd.Timestamp(london_arr[i])
        month_key = (bar_ts.year, bar_ts.month)
        if month_key in signaled_months:
            continue
        # 参照アンカー探索: 参照窓[fix−ref_hours, シグナルバー]内の最初のバーまで後退走査。
        # i以前の確定済みバーのみを参照する逐次走査であり、先読みが無いことはループを読むだけで
        # 自明(session_range_template.pyの1パス流儀)。前日以前のバーは壁時計の絶対時刻が
        # ref_start(対象日の午後)より必ず小さいため、日付境界の混入も構造的に起きない。
        j = i
        while j - 1 >= 0 and london_arr[j - 1] >= ref_start:
            j -= 1
        direction = close_arr[i] - open_arr[j]
        if direction == 0:
            continue  # 参照区間が完全フラットで方向未確定ならエントリーしない
        trend = 1 if direction > 0 else -1
        signal_arr[i] = -trend if post_fix_reversal else trend
        signaled_months.add(month_key)

    return pd.Series(signal_arr, index=df.index, dtype=int)


templates.register(
    "month_end_fix",
    defaults={
        "mode": 1.0,  # 0=into_fix / 1=post_fix_reversal(本命)
        "days_before": 0.0,  # 0=月末営業日当日、1=T-1(前営業日)
        "ref_hours": 3.0,  # 参照区間の長さ(時間)。fix−ref_hoursが参照開始(既定: ロンドン13:00)
        "sl_pips": 30.0,
        "tp_pips": 30.0,
        "max_hold_bars": 16.0,  # engineが自動認識(H1で約16時間=fix後の時間決済)
        "lot": 0.1,
    },
    signal_fn=_signal_month_end_fix,
)
