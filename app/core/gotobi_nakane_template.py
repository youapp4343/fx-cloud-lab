"""ゴトー日仲値(gotobi_nakane)テンプレート(裁量トレーダー記事 手法#1の機械化)。

ゴトー日(五十日 = 毎月5,10,15,20,25日 + 月末日)は本邦輸入企業の対外決済が集中し、
東京仲値(09:55 JST)に向けて対円の外貨買い(=JPYクロス上昇)フローが出やすいアノマリー。
scan_gotobi_nakane.py の総当たり探索で FDR + IS/OOS + 非ゴトー日Welch を通過した仕様
「JPYクロスを東京朝07:00 JSTに買い、2〜3時間保有」を、engine.run_backtest で執行・MQL出力
できる固定仕様テンプレートとして実装する。方向は事前仮説で買い固定(scan却下: OOSでの
方向探索は多重比較の偽陽性源。month_end_fix と同じ方針)。

ゴトー日判定(機械的・先読みなし): 各月の {5,10,15,20,25} + 月末日を名目対象日とし、
土日なら直前営業日へ前倒し(実需決済が前営業日に寄るため)。祝日はバー欠損で自動スキップ。
日本は UTC+9 固定(DSTなし)なので壁時計 = timestamp + 9h で得られ、zoneinfo不要
(month_end_fix のロンドンDST処理より単純)。ゴトー日集合は純カレンダー演算のみで価格を見ない。

先読み・執行境界(既知バグ①対策): signal_fn は shift しない生シグナルを返す。エントリー時刻
(07:00 JST)ちょうどのバーの1本前(H1: 06:00 JST、M15: 06:45 JST)をシグナルバーとして
壁時計完全一致で特定し、engine の shift(1) により entry_hour_jst 開始バーの始値で執行される。
方向を決める材料は日付(カレンダー)のみで価格の値動きを含まないため、方向決定と損益測定の
起点分離は構造的に保証される(month_end_fix と同じ流儀)。決済は engine の sl/tp/max_hold_bars。
"""

from __future__ import annotations

import calendar as _cal

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from app.core import templates
from app.core.strategy_model import Strategy

JST_OFFSET = pd.Timedelta(hours=9)  # 日本標準時 = UTC+9 固定(DSTなし)


def _gotobi_day_set(jst_days: np.ndarray) -> set:
    """出現するJST日付から、ゴトー日(土日前倒し済み)の集合を純カレンダーで構築する。"""
    months = {(int(pd.Timestamp(d).year), int(pd.Timestamp(d).month)) for d in jst_days}
    result: set = set()
    for (y, m) in months:
        last_day = _cal.monthrange(y, m)[1]
        nominal = [5, 10, 15, 20, 25]
        if last_day not in nominal:
            nominal.append(last_day)
        for day in nominal:
            d = pd.Timestamp(year=y, month=m, day=day)
            while d.weekday() >= 5:      # 土(5)/日(6) → 直前営業日へ前倒し
                d -= pd.Timedelta(days=1)
            result.add(d.normalize())
    return result


def _signal_gotobi_nakane(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    entry_hour_jst = int(p["entry_hour_jst"].value)   # 既定7(東京朝07:00 JST)
    day_mode = int(p["day_mode"].value)               # 0=ゴトー日(本命)/1=非ゴトー日(プラセボ対照)

    signal = pd.Series(0, index=df.index, dtype=int)
    n = len(df)
    if n < 2:
        return signal

    ts = df["timestamp"]
    bar_period = ts.diff().median()  # 中央値: 週末ギャップに頑健なバー幅推定
    if pd.isna(bar_period) or bar_period <= pd.Timedelta(0):
        return signal

    jst = ts + JST_OFFSET
    jst_day = jst.dt.normalize()
    gset = _gotobi_day_set(jst_day.unique().to_numpy())

    # エントリー時刻(07:00 JST)ちょうどのバーの1本前をシグナルバーとする
    # (engineのshift(1)で07:00 JST開始バーの始値執行)。壁時計完全一致で特定するため、
    # 祝日等でこのバーが欠損する日は自動的に無シグナル(=その日スキップ)。
    entry_dt = jst_day + pd.Timedelta(hours=entry_hour_jst)
    signal_wallclock = entry_dt - bar_period

    jst_day_norm = jst_day
    in_gotobi = jst_day_norm.isin(gset)
    day_mask = in_gotobi if day_mode == 0 else (~in_gotobi & _is_business_day(jst_day_norm))
    is_signal_bar = ((jst == signal_wallclock) & day_mask).to_numpy()

    jst_day_arr = jst_day.to_numpy()
    signal_arr = np.zeros(n, dtype=int)
    signaled_days: set = set()  # 1日1シグナルガード(壁時計一致で構造的にも1本だが異常データでも保証)
    for i in np.flatnonzero(is_signal_bar):
        day_key = pd.Timestamp(jst_day_arr[i]).normalize()
        if day_key in signaled_days:
            continue
        signal_arr[i] = 1  # 買い固定(JPYクロス上昇の事前仮説)
        signaled_days.add(day_key)

    return pd.Series(signal_arr, index=df.index, dtype=int)


def _is_business_day(days: pd.Series) -> pd.Series:
    """月〜金を True(プラセボの非ゴトー日を平日に限定するため)。純カレンダー。"""
    return days.dt.weekday < 5


templates.register(
    "gotobi_nakane",
    defaults={
        "entry_hour_jst": 7.0,   # 東京朝07:00 JST エントリー
        "day_mode": 0.0,         # 0=ゴトー日 / 1=非ゴトー日(プラセボ)
        "sl_pips": 500.0,        # 既定は時間決済(max_hold_bars)を binding にするため広く設定
        "tp_pips": 500.0,
        "max_hold_bars": 2.0,    # H1で2h保有(=scanのhold2h)。M15検証時は8に設定
        "lot": 0.1,
    },
    signal_fn=_signal_gotobi_nakane,
)
