"""SMC/ICT複合手法「ALCHEMIST」継続セットアップ テンプレート。

出典: SMC(Smart Money Concepts)/ICT(Inner Circle Trader)系解説でよく語られる複合
セットアップ「ALCHEMIST」。核となる3要素——BOS(Break of Structure、構造ブレイクに
よるトレンド定義)・プレミアム/ディスカウント(直近ディーリングレンジの50%を基準に
した押し目/戻りゾーン)・IDM(Inducement、エントリー前の一時的な流動性刈り=
ストップ狩り)——のANDを機械化する。OB(オーダーブロック)やFVG(Fair Value Gap)
といった裁量的なゾーンラベリングは機械化不能なため対象外とし、価格アクションのみ
から再現可能な核だけを実装する。

ロジック(バーtの終値時点で確定した情報のみ使用):
1. 構造/トレンド: 直近の確定BOSが上抜け(終値が直近確定スイング高値を上抜け)なら
   trend=+1、下抜け(終値が直近確定スイング安値を下抜け)ならtrend=-1とし、次のBOSが
   起きるまでその状態を保持する。
2. ディーリングレンジ + プレミアム/ディスカウント: レンジ=[直近確定スイング安値,
   直近確定スイング高値]、中点(50%)より下=ディスカウント(買い許可)、上=
   プレミアム(売り許可)。
3. IDM(誘導)スイープ: idm_lookback本以内に「直近安値/高値を一時的に割った/超えた後、
   現在の終値がそのレベルを回復している」ことを要求する(流動性刈り+リクレイムの近似)。
4. トリガー: 上記3条件に加えて、方向と一致する実体のバー(買いは陽線、売りは陰線)。

近似(要求仕様どおり明記):
- スイング確定は center=True の rolling(2*k+1) を用い、「ピボットi=t-kはt=i+k時点で
  確定する」というk本遅れの因果的スキームで判定する(下記ループ参照)。直近1つ(last)
  の確定スイング高安のみを追跡し、1つ前(prev)は本テンプレのBOS/レンジ判定に不要な
  ため追跡しない簡略化とする。同点(プラトー)がある場合は複数バーがピボット判定され
  うるが、値は同じなので実害はない近似として許容する。
- IDMスイープは、裁量的なOB/FVG判定の代わりに「直近安値/高値」をapp.core.indicators.
  donchian_channel(内部でshift(1)済み、当バー除外で因果的)で近似し、「idm_lookback本
  以内にそのレベルを割った/超えたバーがあり、かつ現在の終値がレベルを回復している」
  ことをIDM成立とみなす。厳密なICT用語の「minor high/low」の代理指標であることに注意。
- BOS判定は終値ベースのみ(ヒゲでの一時的な突破は無視する保守的な定義)。

先読み規律: center=True の rolling で得た配列そのものは未来のバーを含むが、ループでは
時刻tにおいて i=t-k のピボット判定のみを参照する。この判定が依存する窓は
[i-k, i+k] = [i-k, t] であり、上限は常にt(現在のバー)以下なので、時刻tの時点で
既に確定済みのデータのみに依存する(先読みは構造的に発生しない)。プレミアム/
ディスカウント・IDMスイープ・トリガー判定も全て自バー+自分の過去のみを参照する。
執行用の.shiftはengine側(run_backtest)が行うため、本テンプレート内には執行shiftを
一切含まない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import donchian_channel
from app.core.strategy_model import Strategy


def _signal_smc_bos(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    k = max(1, int(p["swing_k"].value))                    # スイング確定のラグ本数
    idm_lookback = max(1, int(p["idm_lookback"].value))    # IDMスイープの遡及本数
    dir_mode = int(p["dir_mode"].value)                    # 0=both,1=long,2=short

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    # --- 因果的な確定スイング(指定スニペットのまま) ---
    hh = df["high"].rolling(2 * k + 1, center=True).max().to_numpy()
    ll = df["low"].rolling(2 * k + 1, center=True).min().to_numpy()
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    is_ph = high == hh
    is_pl = low == ll

    # 時刻tではi=t-kのピボットのみ確定する(窓[i-k,i+k]=[i-k,t]がt時点で既知のため
    # 先読みなし)。last_high/last_lowが直近確定スイング高安(価格)。
    trend_arr = np.zeros(n, dtype=int)
    last_high_arr = np.full(n, np.nan)
    last_low_arr = np.full(n, np.nan)

    trend = 0
    last_high = np.nan
    last_low = np.nan

    for t in range(n):
        i = t - k
        if i >= 0:
            if is_ph[i]:
                last_high = high[i]
            if is_pl[i]:
                last_low = low[i]
        if not np.isnan(last_high) and close[t] > last_high:
            trend = 1   # BOS上抜け: 強気構造へ
        if not np.isnan(last_low) and close[t] < last_low:
            trend = -1  # BOS下抜け: 弱気構造へ
        trend_arr[t] = trend
        last_high_arr[t] = last_high
        last_low_arr[t] = last_low

    trend_s = pd.Series(trend_arr, index=df.index)
    last_high_s = pd.Series(last_high_arr, index=df.index)
    last_low_s = pd.Series(last_low_arr, index=df.index)
    has_range = last_high_s.notna() & last_low_s.notna()
    mid = (last_high_s + last_low_s) / 2.0

    close_s = df["close"]
    discount = has_range & (close_s < mid)  # ディスカウント(押し目買いゾーン)
    premium = has_range & (close_s > mid)   # プレミアム(戻り売りゾーン)

    # --- IDM(誘導)スイープ: donchian_channel(shift(1)済み、因果的)で「直近安値/高値」を近似 ---
    minor_high, minor_low = donchian_channel(df["high"], df["low"], idm_lookback)
    swept_low_event = (df["low"] < minor_low).fillna(False)     # 直近安値を一時的に割った
    swept_high_event = (df["high"] > minor_high).fillna(False)  # 直近高値を一時的に超えた
    swept_low_recent = (
        swept_low_event.astype(int).rolling(idm_lookback, min_periods=1).max().astype(bool)
    )
    swept_high_recent = (
        swept_high_event.astype(int).rolling(idm_lookback, min_periods=1).max().astype(bool)
    )
    idm_long = swept_low_recent & (close_s > minor_low)    # 割った後、現在の終値で回復(リクレイム)
    idm_short = swept_high_recent & (close_s < minor_high)  # 超えた後、現在の終値で回復(リクレイム)

    bullish_bar = close_s > df["open"]
    bearish_bar = close_s < df["open"]

    long_raw = ((trend_s == 1) & discount & idm_long & bullish_bar).fillna(False)
    short_raw = ((trend_s == -1) & premium & idm_short & bearish_bar).fillna(False)
    # 再アーム: 条件が連続して成立する間の連続発火を防ぎ、成立直後の1本のみ発火させる
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "smc_bos",
    defaults={
        "swing_k": 5.0,
        "idm_lookback": 10.0,
        "dir_mode": 0.0,
        "sl_pips": 200.0,   # ゴールドスケール寄り(検証スクリプト側で銘柄別に上書きする)
        "tp_pips": 300.0,
        "lot": 0.1,
    },
    signal_fn=_signal_smc_bos,
)
