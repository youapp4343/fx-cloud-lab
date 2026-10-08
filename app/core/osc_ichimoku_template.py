"""オシレーター合議 + 一目均衡表雲ゲート 逆張りテンプレート(osc_ichimoku)。

RSI/Stochastic%K/CCI/Williams%R の4オシレーターの「売られすぎ/買われすぎ」判定を
集計し(osc_confluence/osc_confluence_xと同じ流儀)、min_agree個以上一致した状態への
遷移バー(エッジ)で逆張り候補とする。「一目の雲はトレンドフィルタとして性能を上げるか」
の検証用テンプレート: cloud_modeで一目均衡表の雲によるゲートの有無・向きを切り替え、
素の合議逆張り(ゲート無し)と比較できるようにする。

ゾーン(標準値、osc_confluenceと同一の4種を採用): RSI30/70, Stoch20/80, CCI±100,
WPR-80/-20。periodは全オシレーター共通。

一目均衡表はローカル実装(indicators.pyに一目関数がないため cci_ichimoku_template.py
と同一方式で本ファイル内に自前実装、他ファイルへの変更は行わない):
  転換線 = (過去tenkan本の高値+安値)/2、基準線 = (過去kijun本の高値+安値)/2、
  先行スパンA_raw = (転換線+基準線)/2、先行スパンB_raw = (過去senkou_b本の高値+安値)/2。
  ★一目の先行スパンは本来チャート上で未来へcloud_shift(26)本シフトして描画される。
  そのため「現在バーに表示されている雲」はcloud_shift本前に確定した値であり、
  cloud_a_now = span_a_raw.shift(cloud_shift) / cloud_b_now = span_b_raw.shift(cloud_shift)
  という「過去参照のshift」で表現できる(span_a_raw/span_b_raw自体は現在バーまでの高安の
  みから計算する確定値であり、それをcloud_shift本分だけ過去にずらして現在バーへ適用する
  = 未来のデータを現在バーで読むわけではない、先読みなし)。

cloud_mode(トレンドフィルタの検証用スイッチ):
  0 = ゲート無し(素のオシレーター合議逆張りのみ。雲の影響を受けないベースライン)
  1 = 雲方向限定(買いは close>雲上端のみ、売りは close<雲下端のみ。トレンド方向への
      押し目/戻りでの逆張りのみ許可し、逆行トレンド中の逆張りを排除する)
  2 = 雲逆張り限定(買いは close<雲下端のみ=雲の下からの反発狙い、売りは close>雲上端
      のみ=雲の上からの反落狙い。cloud_mode=1とは反対の「雲を背にした逆張り」)
cloud_mode=0/1/2を切り替えて、雲gateが素の逆張りの成績を改善するかどうかを比較する
用途(「一目の雲が性能いい」という主張の検証)。

エントリー: 買い候補(oversold合議のエッジ) & cloud_modeのゲート通過 = LONG。
売り候補(overbought合議のエッジ) & ゲート通過 = SHORT。dir_modeで買い/売り/両方を
選択(0=both,1=long,2=short、cci_ichimokuと同じ規約)。連続発火防止のため合議状態への
遷移バー(raw & ~raw.shift(1))のみでエッジ判定し、ゲートはエッジ判定の後にANDする
(ゲート条件自体は状態量なのでエッジ化しない、オシレーター合議側のみエッジ化する)。

近似(docstring明記): オシレーターはosc_confluence(6種)のうちRSI/Stochastic/CCI/
Williams%Rの4種のみ(タスク仕様が指定した4本、DeMarker/Momentumは含まない)。一目の
ねじれ判定(スパンA/B交差)・遅行スパン(chikou)による追加フィルタは省略し、cci_ichimoku
と同様に雲の上下判定のみを機械化した。

先読み規律: 転換線/基準線/先行スパンA・B_rawは確定バー+自分の過去のみ
(rolling().max()/min()は当バーを含む過去参照で先読みなし)。雲はcloud_shift本の
.shift()で過去に確定した値のみを現在バーに適用(未来シフトではなく過去参照のshift)。
オシレーターは app.core.indicators の rsi/stochastic/cci/williams_r を使用。
再アームの.shift(1)は自分の過去参照のみ。執行のためのシフトはengine.run_backtest側が
行うため、テンプレート内では実行用シフトをしない。
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from app.core import templates
from app.core.indicators import cci, rsi, stochastic, williams_r
from app.core.strategy_model import Strategy


def _ichimoku_cloud_now(
    high: pd.Series,
    low: pd.Series,
    tenkan_period: int,
    kijun_period: int,
    senkou_b_period: int,
    cloud_shift: int,
) -> Tuple[pd.Series, pd.Series]:
    """一目均衡表の先行スパンA/Bをローカル実装で計算し、現在バーに表示される雲の値を返す。

    先行スパンは本来チャート上で未来へcloud_shift本シフトして描画されるため、
    「現在バーの雲」はcloud_shift本前に確定した値(=spanA_raw/spanB_rawをcloud_shift本
    shiftしたもの、過去参照のみで未来データは使わない)。cci_ichimoku_templateと同一方式。
    """
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2.0
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2.0
    cloud_a_now = span_a_raw.shift(cloud_shift)
    cloud_b_now = span_b_raw.shift(cloud_shift)
    return cloud_a_now, cloud_b_now


def _signal_osc_ichimoku(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """オシレーター合議(RSI/Stoch/CCI/WPR) + 一目均衡表の雲ゲートで逆張り。

    モジュールdocstring参照。cloud_modeでゲートの有無・向きを切替、dir_modeで
    買い/売り/両方を切替(0=both,1=long,2=short)。
    """
    p = strategy.params
    period = max(2, int(p["period"].value))
    min_agree = max(1, min(4, int(p["min_agree"].value)))
    cloud_mode = int(p["cloud_mode"].value)
    tenkan_period = max(2, int(p["tenkan"].value))
    kijun_period = max(2, int(p["kijun"].value))
    senkou_b_period = max(2, int(p["senkou_b"].value))
    cloud_shift = max(1, int(p["cloud_shift"].value))
    dir_mode = int(p["dir_mode"].value)

    high, low, close = df["high"], df["low"], df["close"]

    r = rsi(close, period)
    k, _ = stochastic(high, low, close, period, 3, 3)
    cc = cci(high, low, close, period)
    w = williams_r(high, low, close, period)

    oversold = (
        (r <= 30).astype(int) + (k <= 20).astype(int)
        + (cc <= -100).astype(int) + (w <= -80).astype(int)
    )
    overbought = (
        (r >= 70).astype(int) + (k >= 80).astype(int)
        + (cc >= 100).astype(int) + (w >= -20).astype(int)
    )

    buy_state = oversold >= min_agree
    sell_state = overbought >= min_agree
    # 状態への遷移バーのみ発火(滞在中の連続発火を防ぐ。NaN比較はFalseになるため安全)
    buy_edge = buy_state & ~buy_state.shift(1).fillna(False).astype(bool)
    sell_edge = sell_state & ~sell_state.shift(1).fillna(False).astype(bool)

    cloud_a_now, cloud_b_now = _ichimoku_cloud_now(
        high, low, tenkan_period, kijun_period, senkou_b_period, cloud_shift
    )
    # 両スパンが確定して初めて「雲」とみなす(skipna=False、片方だけNaNなら雲もNaN)
    cloud_top = pd.concat([cloud_a_now, cloud_b_now], axis=1).max(axis=1, skipna=False)
    cloud_bottom = pd.concat([cloud_a_now, cloud_b_now], axis=1).min(axis=1, skipna=False)

    if cloud_mode == 0:
        # ゲート無し: 素のオシレーター合議逆張り(雲は計算のみで判定には使わない)
        gate_long = pd.Series(True, index=df.index)
        gate_short = pd.Series(True, index=df.index)
    elif cloud_mode == 1:
        # 雲方向限定: トレンド方向への押し目/戻りでの逆張りのみ許可
        gate_long = close > cloud_top
        gate_short = close < cloud_bottom
    elif cloud_mode == 2:
        # 雲逆張り限定: 雲を背にした逆張り(雲下からの反発/雲上からの反落)
        gate_long = close < cloud_bottom
        gate_short = close > cloud_top
    else:
        raise ValueError(f"unknown cloud_mode: {cloud_mode}")

    gate_long = gate_long.fillna(False)
    gate_short = gate_short.fillna(False)

    long_sig = buy_edge & gate_long
    short_sig = sell_edge & gate_short

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "osc_ichimoku",
    defaults={
        "period": 14.0,
        "min_agree": 2.0,
        "cloud_mode": 1.0,
        "tenkan": 9.0,
        "kijun": 26.0,
        "senkou_b": 52.0,
        "cloud_shift": 26.0,
        "dir_mode": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 45.0,
        "lot": 0.1,
    },
    signal_fn=_signal_osc_ichimoku,
)
