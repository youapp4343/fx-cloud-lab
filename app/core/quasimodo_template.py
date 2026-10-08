"""Quasimodo(QM/QML)リバーサル テンプレート(ALCHEMIST/SMC ゴールド手法の機械化)。

パターン定義: 弱気QM(売り)= 確定スイングの中に SH1(左肩高値)→NECK(戻り安値)
→SH2(NECK確定後にSH1を上抜くヘッド高値、SH2>SH1) の並びが成立し、SH2確定後に
価格がNECKを安値で下抜けたら構造崩壊とみなしショートをアーム。QML=SH1(左肩高値)。
その後、当バーhigh >= SH1-band (band = band_atr×ATR) まで戻ってきたら-1を出す。
アーム中に当バーhighがSH2を上抜いたらセットアップを無効化してidleへ戻す。
強気QM(買い)はミラー: SL1(左肩安値)→NECK(戻り高値)→LL2(ヘッド安値、LL2<SL1)、
NECKを高値で上抜けたら構造崩壊としてロングをアーム。QML=SL1。当バーlow<=SL1+band
で+1、アーム中にLL2を下抜いたら無効化。1セットアップにつき1シグナルで、発火/無効化
後は状態をidleに戻し次のSH1/SL1候補から再アームする。

近似(Note approximations):
1. スイングピボットはプロンプト指定どおり中心窓(±k本、rolling center=True)+厳密一致
   方式で検出する。高安がタイ/横ばいの稀なケースでは同種ピボットの重複・欠落があり得る。
   ピボットiはk本後(t=i+k)になって初めて読み出すため、中心窓が未来を覗いていても
   読み出し側は先読みにならない(プロンプト指定の設計をそのまま踏襲)。
2. SH1/SL1(肩)は「まだNECKが無い間」は新規同種ピボットで常に上書きし、さらに
   「NECK確定後にヘッドの条件(SH1超え/SL1割れ)を満たさない同種ピボットが来た場合」も
   その新しいピボットを新SH1/SL1として仕切り直す(=NECKは破棄し次の戻りを待つ)。
   これにより肩の参照が何年も前の値に固定され続けて構造判定が事実上停止する事態を防ぐ
   (直近のスイング構造だけを見る近似)。NECKは「ヘッド確定前に出た直近の戻り側ピボット」
   を都度上書きする(最深値ではなく直近値を採用する近似)。
3. 構造崩壊(NECK割れ/上抜け)の監視は、ヘッドが確定したバー(実ピボットからk本後)
   以降のみ行う。確定待ちのk本の間に既にNECKを割っていた場合はそれを遡って検知しない
   (先読み回避を優先した保守的近似。見逃すことはあっても先読みは絶対にしない)。
4. アーム後の無効化/リテスト判定は当バーの生の高安値(確定ピボットではない)を使うため
   即応的に働く。1本のバーが無効化ライン(ヘッド超え)とリテスト帯を同時に跨いだ場合は
   無効化を優先する(OHLCからは本足内の高安の経路順序が復元できないための近似)。
   dir_mode=0で買い/売りが同一バーで同時成立した場合は売り(-1)の書き込みを優先する
   (実装上の決定、実際にはほぼ発生しない)。
5. SLは概念上ヘッドの外側にあるが、本テンプレートの出力は{-1,1,0}のみでSL価格自体は
   出力しない(engine側のsl_pipsパラメータで近似する)。

先読み規律: シグナルは現在の確定バー+過去の確定ピボットのみを使用する。ATRは
indicators.atr(shift+ewmの因果的計算)を使用。執行用の.shiftはengine側
(raw_signal.shift(1))が行うため、テンプレート内では一切shiftしない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _signal_quasimodo(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    k = max(1, int(p["swing_k"].value))              # スイングピボットの左右幅
    band_mult = float(p["band_atr"].value)            # QML帯の許容幅(×ATR)
    atr_period = max(2, int(p["atr_period"].value))
    dir_mode = int(p["dir_mode"].value)                # 0=both,1=long,2=short

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    band_arr = (band_mult * atr(df["high"], df["low"], df["close"], atr_period)).to_numpy(float)

    # ピボットフラグ(中心窓は未来を覗くが、k本後にしか読み出さないため先読みなし)
    hh = df["high"].rolling(2 * k + 1, center=True).max().to_numpy()
    ll = df["low"].rolling(2 * k + 1, center=True).min().to_numpy()
    is_ph = high == hh
    is_pl = low == ll

    sh_price: list = []; sh_idx: list = []   # 確定済みスイング高値
    sl_price: list = []; sl_idx: list = []   # 確定済みスイング安値

    # 弱気QM状態: 0=idle,1=SH1確定済み,2=NECK確定済み(ヘッド待ち),
    #            3=ヘッド確定済み(NECK割れ待ち),4=アーム済み(リテスト/無効化待ち)
    bear_state = 0
    bear_sh1 = bear_neck = bear_sh2 = float("nan")
    # 強気QM状態: 記号はミラー(SL1/NECK/LL2)
    bull_state = 0
    bull_sl1 = bull_neck = bull_ll2 = float("nan")

    bear_out = np.zeros(n, dtype=bool)
    bull_out = np.zeros(n, dtype=bool)

    for t in range(n):
        i = t - k
        new_ph = i >= 0 and bool(is_ph[i])
        new_pl = i >= 0 and bool(is_pl[i])
        if new_ph:
            sh_price.append(float(high[i])); sh_idx.append(i)
        if new_pl:
            sl_price.append(float(low[i])); sl_idx.append(i)

        # --- 弱気QM: 新規確定ピボットでSH1/NECK/SH2を更新 ---
        if new_ph:
            h = sh_price[-1]
            if bear_state == 0:
                bear_sh1, bear_state = h, 1
            elif bear_state == 1:
                bear_sh1 = h  # NECKなしでの新高値は肩(SH1)候補を更新
            elif bear_state == 2:
                if h > bear_sh1:
                    bear_sh2, bear_state = h, 3  # SH1超え → ヘッド確定
                else:
                    bear_sh1, bear_state = h, 1  # 超えない新高値は仕切り直し(新SH1候補、NECKは破棄)
            elif bear_state == 3:
                if h > bear_sh2:
                    bear_sh2 = h  # NECK割れ前にさらに高値更新 → ヘッドを延長
        if new_pl:
            lo = sl_price[-1]
            if bear_state == 1:
                bear_neck, bear_state = lo, 2
            elif bear_state == 2:
                bear_neck = lo  # ヘッド確定前の戻り安値(NECK候補)を随時更新

        # --- 弱気QM: 構造崩壊(NECK割れ) → アーム → 無効化/リテスト ---
        if bear_state == 3:
            if low[t] < bear_neck:
                bear_state = 4  # NECK割れ確定、ショートをアーム
        elif bear_state == 4:
            band = band_arr[t]
            if not np.isnan(band):
                if high[t] > bear_sh2:
                    bear_state = 0  # ヘッド超え → 無効化
                    bear_sh1 = bear_neck = bear_sh2 = float("nan")
                elif high[t] >= bear_sh1 - band:
                    bear_out[t] = True  # QML帯へのリテスト → 発火
                    bear_state = 0
                    bear_sh1 = bear_neck = bear_sh2 = float("nan")

        # --- 強気QM(ミラー): 新規確定ピボットでSL1/NECK/LL2を更新 ---
        if new_pl:
            lo = sl_price[-1]
            if bull_state == 0:
                bull_sl1, bull_state = lo, 1
            elif bull_state == 1:
                bull_sl1 = lo
            elif bull_state == 2:
                if lo < bull_sl1:
                    bull_ll2, bull_state = lo, 3  # SL1割れ → ヘッド確定
                else:
                    bull_sl1, bull_state = lo, 1  # 割れない新安値は仕切り直し(新SL1候補、NECKは破棄)
            elif bull_state == 3:
                if lo < bull_ll2:
                    bull_ll2 = lo  # NECK上抜け前にさらに安値更新 → ヘッドを延長
        if new_ph:
            h = sh_price[-1]
            if bull_state == 1:
                bull_neck, bull_state = h, 2
            elif bull_state == 2:
                bull_neck = h  # ヘッド確定前の戻り高値(NECK候補)を随時更新

        # --- 強気QM: 構造崩壊(NECK上抜け) → アーム → 無効化/リテスト ---
        if bull_state == 3:
            if high[t] > bull_neck:
                bull_state = 4  # NECK上抜け確定、ロングをアーム
        elif bull_state == 4:
            band = band_arr[t]
            if not np.isnan(band):
                if low[t] < bull_ll2:
                    bull_state = 0  # ヘッド割れ → 無効化
                    bull_sl1 = bull_neck = bull_ll2 = float("nan")
                elif low[t] <= bull_sl1 + band:
                    bull_out[t] = True  # QML帯へのリテスト → 発火
                    bull_state = 0
                    bull_sl1 = bull_neck = bull_ll2 = float("nan")

    if dir_mode in (0, 1):
        signal.iloc[bull_out] = 1
    if dir_mode in (0, 2):
        signal.iloc[bear_out] = -1
    return signal


templates.register(
    "quasimodo",
    defaults={
        "swing_k": 5.0, "band_atr": 0.5, "atr_period": 14.0, "dir_mode": 0.0,
        "sl_pips": 200.0, "tp_pips": 300.0, "lot": 0.1,
    },
    signal_fn=_signal_quasimodo,
)
