"""戦略A: アジアレンジ収縮 → ロンドンブレイク(ERトレンドレジーム限定, ATR正規化) テンプレート。

ChatGPT設計のスペックをそのまま機械化したもの。「アジア時間の値幅が普段より縮小している
(=様子見・レンジ形成)日に限り、かつ直近のH1相当効率比(ER)がトレンド地合いを示している
ときだけ、ロンドン勢の参入でアジアレンジをATR正規化した幅だけ抜けた方向へ順張りする」
という設計思想の忠実な機械化。レンジ収縮×ERトレンド×24hモメンタムの3フィルタを重ねた
選択的エントリーのため、他テンプレートよりトレード頻度は低くなる想定(仕様通り)。

時間帯の定義(UTC、tz-naiveがプロジェクト規約):
- アジアレンジ形成窓 = 00:00〜06:59 UTC(hourが0..6)。この窓のバーのhigh/lowから
  AsianHigh/AsianLow/AsianRange(=High-Low)をUTC日付ごとに算出し、06:59時点
  (=07:00バー到達時点)で確定(ロック)する。
- トレード判定 = 07:00 UTC以降の確定バーのみ(それ以前は常にシグナルなし)。

DailyATR(日足ATR20相当のM15近似、近似1): スペックが許容した2案のうち、b.「M15を
UTC日付でgroupbyして日足相当(high=当日最高値/low=当日最安値/close=当日最終M15足の
終値)を再構成し、そのATR20を前日値としてbroadcast」を採用した。a.案(atr()にperiod=
20*96=1920を渡してM15粒度へ直接適用)も試したが、M15バー単位のtrue rangeを
Wilder平滑するだけでは「1本のM15バー内の値幅」の平均に収束してしまい、日中の
方向性を伴った値幅の積み上げ(=本来の日足ATR、EURUSDで概ね60〜90pips)を再現できず
数pips〜10pips程度に過小推定される(実測: a案は平均約9.6pips、b案は平均約66.6pips)。
これによりRangeRatio(=AsianRange/DailyATR)が常時1を大きく超え0.40の上限を割ることが
無くなり、レンジフィルタが恒久的に不成立になる不具合が生じたため、数値として妥当な
b案を採用した。b案は「当日分のバー(まだ確定していない当日の日足高安)」を一切使わず、
必ず前営業日までの完全に確定した日足からATR20を計算し.shift(1)してから当日全バーへ
broadcastするため、日足ATRは常に前日確定値のまま当日中は更新されない
(=「日足ATRは前日まで」の要件を文字通り満たす)。daily_atr_periodパラメータ(既定1920)は
「M15本数換算で20日分」という定義を保持し、内部で days=round(daily_atr_period/96)へ
変換して日足ATR期間として使う。

RangeRatio = AsianRange / DailyATR。0.15 <= RangeRatio <= 0.40 の日のみエントリー対象
(レンジが狭すぎ/広すぎる日は見送り)。

トレンドレジーム(近似2): ER48(H1相当48本のKaufman効率比)を、M15足では
period = 48*4 = 192 に換算した efficiency_ratio(close, 192) で近似する。
ER >= er_trend(既定0.25)の場合のみ「トレンド地合い」と判定しブレイクを許可し、
それ未満は完全に無取引(このテンプレートはブレイク専用でレンジ逆張りは行わない)。

エントリー条件(すべて確定バーのみで判定。次バー始値執行のshift(1)はengine側が担う):
- 買い: ER>=er_trend & RangeRatio条件 & close > AsianHigh + brk_atr*DailyATR
  & 24hリターン(close/close.shift(96)-1)> 0
- 売り: ER>=er_trend & RangeRatio条件 & close < AsianLow - brk_atr*DailyATR
  & 24hリターン < 0 (買いの鏡像)
- 1通貨1日1回(UTC日付が変わるたびに再アーム): その日最初に条件を満たしたバーのみを
  採用し、同日内でそれ以降にどちらかの条件が満たされても無視する。dir_modeで禁止された
  方向の条件は「1日1回」判定の対象自体から除外する(禁止方向側の条件が先に成立しても
  許可された方向の枠を消費しない)。

先読み回避:
- AsianHigh/AsianLowは当日00:00〜06:59のバーのみから算出し、同日の全バーへブロードキャスト
  する。07:00以降のバーが参照する値は必ず自分より過去(同日0-6時台)のバー集約であり、
  00:00〜06:59台のバー自身はtrade_ok=Falseで常にシグナル対象外のため、アジア窓の途中で
  データが打ち切られても(その時点までの)シグナル出力(常に0)には影響しない
  = プレフィックス不変(信号(df) と 信号(df.iloc[:m]) の重複区間は必ず一致する)。
- DailyATRは「UTC日付ごとに完全に確定した日足(groupbyで再構成)のATR20を.shift(1)して
  from broadcast」なので、あるバーiが参照するDailyATRは必ずiの属する日より前の日の
  日足のみに由来し、その前日の日足を構成するM15バーは全てiより時系列で過去にある
  (=プレフィックス不変。df.iloc[:m]へ切り詰めてもm>iである限り前日分のバーは
  削られないため、iで使われるDailyATR値は変化しない)。ER(efficiency_ratio)も
  過去のみを参照するrolling計算(osc_extra.pyの既存実装、min_periodsによる
  ウォームアップNaNのみ)。24hリターンはclose.shift(96)による過去参照のみ。
- 「1日1回」判定はUTC日付ごとのcumsum(同日内で自分以前の累積回数)で実装しており、
  未来のバーの有無によって過去バーの判定結果が変わることはない。
- signal_fn自体はshiftしない生シグナルを返し、翌バー始値執行のshift(1)はengine側が担う
  (ここで追加shiftすると二重shiftになるため行わない)。

近似(明記、続き): SL/TPは既定sl_pips=40/tp_pips=60の固定pipsをパラメータとして持つが、
この戦略の本旨はATR正規化SL/TPである。そのため本テンプレートはシグナルのみを提供し、
sl_pips/tp_pipsをDailyATR×係数から都度上書きする運用は検証スクリプト側の責務とする
(engineは固定pips契約のため、テンプレート単体では構造的SL/TPまでは持たせていない)。
金曜NYクローズ持ち越し・スプレッド上限はEA/検証側、時間切れ決済はengineのmax_hold_bars
パラメータで別途扱う対象とし、いずれも本テンプレートのスコープ外。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.osc_extra import efficiency_ratio
from app.core.strategy_model import Strategy

_ASIAN_HOUR_END = 6   # アジアレンジ形成窓はUTC 00:00〜06:59(hourが0..6のバー)
_TRADE_START_HOUR = 7  # 07:00 UTC以降の確定バーのみトレード判定(アジアレンジ確定後)
_RET24_LOOKBACK = 96   # 24時間 = M15足96本前


def _asian_high_low(
    df: pd.DataFrame, date: pd.Series, asian_mask: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """UTC日付ごとのアジアレンジ(00:00〜06:59)高値・安値を、全バーへブロードキャストして返す。

    アジア窓のバー(asian_mask)だけをUTC日付でgroupbyしてhigh.max/low.minを取り、
    同日の全バー(アジア窓自身も含む)へ map で割り当てる。アジア窓のバーが1本も無い
    日付はNaN(=その日は無取引、呼び出し側の比較はNaN相手には常にFalseとなり安全)。
    """
    asian = df.loc[asian_mask]
    asian_date = date[asian_mask]
    grp_high = asian["high"].groupby(asian_date).max()
    grp_low = asian["low"].groupby(asian_date).min()
    asian_high = date.map(grp_high)
    asian_low = date.map(grp_low)
    return asian_high, asian_low


def _daily_atr_prev_broadcast(df: pd.DataFrame, date: pd.Series, atr_days: int) -> pd.Series:
    """UTC日付でM15足を日足相当に再構成し、そのATR(atr_days)を「前営業日確定値」として
    全バーへブロードキャストして返す(近似1、docstring参照)。

    daily_high/low=当日のM15足のmax/min、daily_close=当日最終M15足の終値、で日足を
    再構成し、atr()で日足ATRを計算したうえで.shift(1)する。groupbyの結果は実データが
    存在する営業日のみを日付昇順で持つため(週末は行自体が存在しない)、shift(1)は
    「暦day-1」ではなく「直前の実在する営業日」を正しく指す。.shift(1)により当日の
    (まだ全バーが確定していない)日足を一切使わず、必ず前営業日までの完全に確定した
    日足のみを使うため、当日中はこの値は更新されない(=前日固定)。
    """
    daily_high = df["high"].groupby(date).max()
    daily_low = df["low"].groupby(date).min()
    daily_close = df["close"].groupby(date).last()
    daily_atr_series = atr(daily_high, daily_low, daily_close, atr_days)
    daily_atr_prev = daily_atr_series.shift(1)
    return date.map(daily_atr_prev)


def _signal_regime_london_break(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    er_period = max(2, int(p["er_period"].value))
    er_trend = float(p["er_trend"].value)
    range_ratio_lo = float(p["range_ratio_lo"].value)
    range_ratio_hi = float(p["range_ratio_hi"].value)
    brk_atr = float(p["brk_atr"].value)
    daily_atr_period = max(2, int(p["daily_atr_period"].value))
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only
    allow_long = dir_mode in (0, 1)
    allow_short = dir_mode in (0, 2)

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    ts = df["timestamp"]
    hour = ts.dt.hour
    date = ts.dt.date
    asian_mask = hour <= _ASIAN_HOUR_END  # 00:00〜06:59 UTC

    asian_high, asian_low = _asian_high_low(df, date, asian_mask)
    asian_range = asian_high - asian_low

    daily_atr_days = max(2, round(daily_atr_period / 96))  # M15本数(既定1920)を日数(既定20)へ換算
    daily_atr = _daily_atr_prev_broadcast(df, date, daily_atr_days)  # 日足ATR20相当・前日確定値(近似1)
    range_ratio = (asian_range / daily_atr).where(daily_atr != 0)
    range_ok = ((range_ratio >= range_ratio_lo) & (range_ratio <= range_ratio_hi)).fillna(False)

    er = efficiency_ratio(df["close"], er_period)  # ER48のM15近似(近似2、period=192)
    trend_ok = (er >= er_trend).fillna(False)

    prev_close_24h = df["close"].shift(_RET24_LOOKBACK)
    ret24 = ((df["close"] / prev_close_24h) - 1.0).where(prev_close_24h != 0)

    trade_ok = hour >= _TRADE_START_HOUR

    brk_long_level = asian_high + brk_atr * daily_atr
    brk_short_level = asian_low - brk_atr * daily_atr

    raw_long = pd.Series(False, index=df.index)
    raw_short = pd.Series(False, index=df.index)
    if allow_long:
        raw_long = (
            trend_ok & range_ok & trade_ok
            & (df["close"] > brk_long_level)
            & (ret24 > 0)
        ).fillna(False)
    if allow_short:
        raw_short = (
            trend_ok & range_ok & trade_ok
            & (df["close"] < brk_short_level)
            & (ret24 < 0)
        ).fillna(False)

    combined = raw_long | raw_short
    # 1通貨1日1回: UTC日付ごとにcombinedの累積回数を取り、その日で初めてTrueになった
    # バーだけを採用する(cumsumは同日内で自分以前のみの合計=過去情報のみで先読みなし)。
    # 日付が変わればカウントは0から再開する=再アーム。
    cum_in_day = combined.astype(int).groupby(date).cumsum()
    first_of_day = combined & (cum_in_day == 1)

    signal[first_of_day & raw_long] = 1
    signal[first_of_day & raw_short] = -1
    return signal


templates.register(
    "regime_london_break",
    defaults={
        "er_period": 192.0,
        "er_trend": 0.25,
        "range_ratio_lo": 0.15,
        "range_ratio_hi": 0.40,
        "brk_atr": 0.05,
        "daily_atr_period": 1920.0,
        "dir_mode": 0.0,
        "sl_pips": 40.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_regime_london_break,
)
