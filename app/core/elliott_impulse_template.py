"""エリオット波動インパルス(wave1→wave2押し目)の機械化コア テンプレート。

エリオット波動理論はラベリング自体が本来裁量的(同じチャートでもアナリストによってカウントが
割れる)であり、理論全体の厳密な自動判定は原理的に不可能である。本テンプレートは理論全体を
機械化しようとはせず、「機械化できる核」だけを切り出して割り切って実装する:

    核 = zigzag確定スイング + wave1(L0→H1)/wave2(H1→L2)のフィボ押し目判定
         + エリオットの3原則のうち wave1-2 完了時点で判定可能な2つ
           (i)  wave2安値(L2)はwave1起点(L0)を割らない(L2 > L0)
           (ii) wave2の押し幅は wave1値幅の fib_lo〜fib_hi(既定38.2%〜78.6%)に収まる
                = (H1-L2)/(H1-L0) in [fib_lo, fib_hi]

エリオットの残り2原則「wave3はwave1/3/5の中で最短にならない」「wave4はwave1と重複しない」は、
本テンプレートがエントリーする wave1-2 完了時点では wave3 以降がまだ存在せず判定不能なため、
意図的にチェックしない(判定できないものを判定したフリはしない、という誠実性の方針。
fib_harmonic_template.py/quasimodo_template.pyと同じ誠実性スタンス)。

エントリーは「wave3の開始」を厳密同定する代わりに、close > H1(wave2区間中の直近確定スイング
高値=本設計では常にH1そのものと一致する)という単純な構造ブレイクで近似する。ショートは
下降インパルス(wave1=H0→L1, wave2=L1→H2)の鏡像。戻り値は{1,-1,0}のみ(構造的SL/TP列は
出力しない)。SL/TPはstrategy.paramsのsl_pips/tp_pips(固定pips)で近似し、概念上のSL
(L2/L0のやや下、または H2/H0のやや上)は出力に反映しない(quasimodo_template.py/
smc_bos_template.pyと同じ簡略化方針)。

CAUSAL CONFIRMED SWINGS(指定スニペットをそのまま使用、quasimodo_template.py/
smc_bos_template.pyと同型): pivotは中心窓rolling(2*k+1, center=True)の等号一致で検出するが、
時刻tではi=t-kのピボットしか「確定した」とみなさない。centerウィンドウ[i-k,i+k]の上限は
i+k=tなので、tの時点で未来のバーには一切依存しない(centerウィンドウ自体は未来を覗くが、
読み出しをk本遅らせることで先読みを構造的に防いでいる)。

状態機械(up=ロング側、dn=ショート側は鏡像。両方を常に並行計算し、最後にdir_modeで出力を
マスクする):
    0=idle → (Lピボット確定)→ 1=L0確定(H1待ち) → (Hピボット確定)→ 2=L0+H1確定(L2待ち)
    → (Lピボット確定、L2>L0 かつ retrace∈[fib_lo,fib_hi]なら)→ 3=armed(L0+H1+L2確定、
      breakout/invalidation待ち)。条件を満たさなければ 1 へ戻り、そのLをL0として仕切り直す。
dnはH/Lを入れ替えた鏡像(0=idle→1=H0確定→2=H0+L1確定→3=armed)。

近似(Note approximations):
1. 同種ピボットが連続確定した場合は常に最新のもので上書きする(例: H1未確定のうちに複数の
   Lピボットが確定すれば最後のものがL0になる)。wave1の始点/終点は「直近の代表的な極値」に
   自動追随するが、真に厳密な「最初の一手」ではない近似(quasimodo_templateのSH1/SL1
   仕切り直しルールと同じ思想)。
2. armed(state=3)に入った後はピボット確定イベントを一切見ない(L0/H1/L2、またはH0/L1/H2を
   凍結)。以降は当バーの生の高安値/終値のみでbreakout(close>H1、または close<L1)と
   invalidation(low<L0、または high>H0)を判定する(確定を待つとk本遅れるため、armed後は
   即応的な生値判定にする。quasimodo_template.py/smc_bos_template.pyと同じ設計)。同一バーで
   invalidationとbreakoutの両方の条件を満たす場合はinvalidation(無効化)を優先する
   (engine.run_backtestのSL優先ルールと同じ保守的思想)。
3. 1セットアップにつき1回だけ発火し(breakout成立時にstate=0へ戻す)、無効化時も同様に
   state=0へ戻して次のL0/H0候補から仕切り直す(再アーム)。
4. dir_mode=0で同一バーにロング/ショートが同時成立する場合(理論上ほぼ発生しない)、
   シグナル書き込み順によりショートが優先される(quasimodo_templateの重複時の扱いと同じ)。

先読み規律: 使用するのは現在の確定バー+過去のみ。center=Trueのrollingで得た配列そのものは
未来のインデックスを含むが、ループでは時刻tにおいてi=t-kのピボットのみを読み出すため、
tの時点で既に確定済みのデータにしか依存しない(先読みは構造的に発生しない)。armed後の
breakout/invalidation判定も当バー自身の値のみを使う。執行用の.shiftはengine側
(run_backtest内のraw_signal.shift(1))が行うため、本テンプレート内には執行shiftを一切含まない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_elliott_impulse(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    k = max(1, int(p["swing_k"].value))  # スイングピボット確定ラグ(左右幅)
    fib_lo = float(p["fib_lo"].value)  # wave2押しの下限(浅い)
    fib_hi = float(p["fib_hi"].value)  # wave2押しの上限(深い)
    dir_mode = int(p["dir_mode"].value)  # 0=both,1=long,2=short

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)

    # --- CAUSAL CONFIRMED SWINGS(指定スニペットのまま) ---
    hh = df["high"].rolling(2 * k + 1, center=True).max().to_numpy()
    ll = df["low"].rolling(2 * k + 1, center=True).min().to_numpy()
    is_ph = high == hh
    is_pl = low == ll

    up_out = np.zeros(n, dtype=bool)
    dn_out = np.zeros(n, dtype=bool)

    # up(ロング, wave1=L0→H1, wave2=H1→L2): 0=idle,1=L0確定,2=L0+H1確定,3=armed
    up_state = 0
    up_l0 = up_h1 = up_l2 = float("nan")
    # dn(ショート, wave1=H0→L1, wave2=L1→H2): 0=idle,1=H0確定,2=H0+L1確定,3=armed
    dn_state = 0
    dn_h0 = dn_l1 = dn_h2 = float("nan")

    for t in range(n):
        i = t - k
        new_ph = i >= 0 and bool(is_ph[i])
        new_pl = i >= 0 and bool(is_pl[i])

        if new_pl:
            lo = float(low[i])
            # up側: L0(wave1始点)の確定/仕切り直し、またはL2(wave2)の妥当性判定
            if up_state == 0:
                up_l0, up_state = lo, 1
            elif up_state == 1:
                up_l0 = lo  # H1未確定のまま安値更新 → wave1起点を仕切り直し
            elif up_state == 2:
                denom = up_h1 - up_l0
                valid = denom > 0 and lo > up_l0  # 原則(i): wave2安値がwave1起点を割らない
                if valid:
                    retrace = (up_h1 - lo) / denom
                    valid = fib_lo <= retrace <= fib_hi  # 原則(ii): フィボ押しレンジ内
                if valid:
                    up_l2, up_state = lo, 3  # wave2確定 → armed
                else:
                    up_l0, up_state = lo, 1  # 不成立 → このLを新wave1起点として仕切り直し
            # up_state==3(armed)はピボットイベントを凍結(近似2)

            # dn側: L1(wave1終点)の確定/仕切り直し
            if dn_state == 1:
                dn_l1, dn_state = lo, 2
            elif dn_state == 2:
                dn_l1 = lo  # H2未確定のまま安値更新 → wave1終点を仕切り直し

        if new_ph:
            hi = float(high[i])
            # dn側: H0(wave1始点)の確定/仕切り直し、またはH2(wave2)の妥当性判定(up側の鏡像)
            if dn_state == 0:
                dn_h0, dn_state = hi, 1
            elif dn_state == 1:
                dn_h0 = hi
            elif dn_state == 2:
                denom = dn_h0 - dn_l1
                valid = denom > 0 and hi < dn_h0
                if valid:
                    retrace = (hi - dn_l1) / denom
                    valid = fib_lo <= retrace <= fib_hi
                if valid:
                    dn_h2, dn_state = hi, 3
                else:
                    dn_h0, dn_state = hi, 1

            # up側: H1(wave1終点)の確定/仕切り直し
            if up_state == 1:
                up_h1, up_state = hi, 2
            elif up_state == 2:
                up_h1 = hi  # L2未確定のまま高値更新 → wave1終点を仕切り直し

        # --- armed watch: 当バーの生値でbreakout/invalidationを判定(即応、確定待ちしない) ---
        if up_state == 3:
            if low[t] < up_l0:
                up_state = 0  # invalidation: wave2がwave1起点を割った(近似2)
                up_l0 = up_h1 = up_l2 = float("nan")
            elif close[t] > up_h1:
                up_out[t] = True  # wave3開始の近似ブレイク
                up_state = 0  # 1セットアップ1回で idle に戻す(近似3)
                up_l0 = up_h1 = up_l2 = float("nan")

        if dn_state == 3:
            if high[t] > dn_h0:
                dn_state = 0
                dn_h0 = dn_l1 = dn_h2 = float("nan")
            elif close[t] < dn_l1:
                dn_out[t] = True
                dn_state = 0
                dn_h0 = dn_l1 = dn_h2 = float("nan")

    if dir_mode in (0, 1):
        signal.iloc[up_out] = 1
    if dir_mode in (0, 2):
        signal.iloc[dn_out] = -1  # 同一バー衝突時はショートを後書きで優先(近似4)
    return signal


templates.register(
    "elliott_impulse",
    defaults={
        "swing_k": 5.0,
        "fib_lo": 0.382,
        "fib_hi": 0.786,
        "dir_mode": 0.0,
        "sl_pips": 40.0,
        "tp_pips": 80.0,
        "lot": 0.1,
    },
    signal_fn=_signal_elliott_impulse,
)
