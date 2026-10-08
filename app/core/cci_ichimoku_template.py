"""CCI一目均衡 逆張り テンプレート(note EA記事 Rank5「CCI一目均衡」の機械化)。

出典: note.com/aimjey EAランキング5位。5分足USDJPY、一目均衡表の雲(転換線9/基準線36/
先行スパンB52)の外側でCCI(14)とCCI(12)が両方とも±150を超えた後の戻り(反発)を狙う
逆張り。報告PF 1.69(始値)/1.22(全tick、記事中最高PF主張)。OOS分割なし・上位10選抜=
選択バイアス注意、報告PFの再現性は本テンプレのバックテストで別途検証が必要。

一目均衡表はローカル実装(indicators.pyに一目関数がないため本ファイル内に自前実装):
  転換線 = (過去tenkan本の高値+安値)/2、基準線 = (過去kijun本の高値+安値)/2、
  先行スパンA_raw = (転換線+基準線)/2、先行スパンB_raw = (過去senkou_b本の高値+安値)/2。
  ★一目の先行スパンは本来チャート上で未来へcloud_shift(26)本シフトして描画される。
  そのため「現在バーに表示されている雲」は cloud_shift本前に確定した値であり、
  cloud_a_now = span_a_raw.shift(cloud_shift) / cloud_b_now = span_b_raw.shift(cloud_shift)
  という「過去参照のshift」で表現できる(span_a_raw/span_b_raw自体は現在バーまでの高安の
  みから計算する確定値であり、それをcloud_shift本分だけ過去にずらして現在バーへ適用する
  = 未来のデータを現在バーで読むわけではない、先読みなし)。

エントリー: closeが雲(cloud_a_now, cloud_b_nowの上側)を上抜けている状態(=雲の上)で、
CCI(14)・CCI(12)がともに-150を下回ったら買い(雲の上=中期的には強いが短期的に売られ
すぎからの反発を狙う)。closeが雲の下側にあり、CCI(14)・CCI(12)がともに+150を上回ったら
売り。連続発火を避けるため再アーム方式(raw & ~raw.shift(1))で条件成立の初回バーのみ
シグナルを立てる。決済は engine の sl/tp(記事: SL4.0=400pips/TP5.0=500pipsのワイド
ストップだが、値が極端なため既定値は控えめな40/50pipsに変更、要注記)。

近似(docstring明記): 記事本文にある雲のねじれ(スパンA/B交差)判定・遅行スパン(chikou)に
よる追加フィルタは本テンプレートでは省略し、雲ブレイク+CCIダブル極値のみを機械化した
(記事の主要トリガと判断した部分のみを実装、副次条件は簡略化)。

先読み規律: 転換線/基準線/先行スパンA・B_rawは確定バー+自分の過去のみ(rolling().max()/
min()は当バーを含む過去参照で先読みなし)。雲はcloud_shift本の.shift()で過去に確定した
値のみを現在バーに適用(未来シフトではなく過去参照のshift)。CCIは
app.core.indicators.cci を使用。再アームの.shift(1)は自分の過去参照のみ。執行のための
シフトはengine.run_backtest側が行うため、テンプレート内では実行用シフトをしない。
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from app.core import templates
from app.core.indicators import cci
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
    shiftしたもの、過去参照のみで未来データは使わない)。
    """
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2.0
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2.0
    cloud_a_now = span_a_raw.shift(cloud_shift)
    cloud_b_now = span_b_raw.shift(cloud_shift)
    return cloud_a_now, cloud_b_now


def _signal_cci_ichimoku(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """雲ブレイク方向 + CCIダブル極値からの反発で逆張り(note EA記事Rank5の機械化)。

    モジュールdocstring参照。dir_modeで買い/売り/両方を切替(0=both,1=long,2=short)。
    """
    p = strategy.params
    cci_len1 = max(2, int(p["cci_len1"].value))
    cci_len2 = max(2, int(p["cci_len2"].value))
    tenkan_period = max(2, int(p["tenkan"].value))
    kijun_period = max(2, int(p["kijun"].value))
    senkou_b_period = max(2, int(p["senkou_b"].value))
    cloud_shift = max(1, int(p["cloud_shift"].value))
    cci_th = float(p["cci_th"].value)
    dir_mode = int(p["dir_mode"].value)

    high, low, close = df["high"], df["low"], df["close"]

    cloud_a_now, cloud_b_now = _ichimoku_cloud_now(
        high, low, tenkan_period, kijun_period, senkou_b_period, cloud_shift
    )
    # 両スパンが確定して初めて「雲」とみなす(skipna=False、片方だけNaNなら雲もNaN)
    cloud_top = pd.concat([cloud_a_now, cloud_b_now], axis=1).max(axis=1, skipna=False)
    cloud_bottom = pd.concat([cloud_a_now, cloud_b_now], axis=1).min(axis=1, skipna=False)

    cci1 = cci(high, low, close, cci_len1)
    cci2 = cci(high, low, close, cci_len2)

    long_raw = (close > cloud_top) & (cci1 < -cci_th) & (cci2 < -cci_th)
    short_raw = (close < cloud_bottom) & (cci1 > cci_th) & (cci2 > cci_th)
    long_raw = long_raw.fillna(False)
    short_raw = short_raw.fillna(False)

    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "cci_ichimoku",
    defaults={
        "cci_len1": 14.0,
        "cci_len2": 12.0,
        "tenkan": 9.0,
        "kijun": 36.0,
        "senkou_b": 52.0,
        "cloud_shift": 26.0,
        "cci_th": 150.0,
        "dir_mode": 0.0,
        "sl_pips": 40.0,
        "tp_pips": 50.0,
        "lot": 0.1,
    },
    signal_fn=_signal_cci_ichimoku,
)
