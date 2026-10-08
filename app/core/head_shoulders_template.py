"""三尊(ヘッドアンドショルダーズ)/逆三尊 ネックラインブレイク テンプレート。

パターン定義:
弱気 三尊(SHORT) = 確定スイング高値が SH_L(左肩)→NECK1(谷)→HEAD(頭、SH_L超え)
→NECK2(谷)→SH_R(右肩、HEAD未満かつ|SH_L-SH_R|<=shoulder_tol×ATRでSH_Lとほぼ同水準)
の順で並び、右肩確定後に当バーのcloseがネックライン(NECK1とNECK2の低い方)を
下抜けたら構造崩壊とみなし-1(売り)を出す。
強気 逆三尊(LONG)はミラー: SL_L(左肩安値)→NECK1(山)→HEAD(頭、SL_L未満)→NECK2(山)
→SL_R(右肩、HEAD超えかつSL_Lとほぼ同水準)。ネックライン(NECK1とNECK2の高い方)を
当バーcloseが上抜けたら+1(買い)。
1パターンにつき1シグナルで、発火/無効化後は状態をidleに戻し次のSH_L/SL_L候補から
再アームする(dir_modeで方向制御)。

近似(Note approximations):
1. スイングピボットはプロンプト指定どおり中心窓(±k本、rolling center=True)+厳密一致
   方式で検出する。ピボットiはk本後(t=i+k)になって初めて読み出すため、中心窓が未来を
   覗いていても読み出し側は先読みにならない(quasimodo_template.py/smc_bos_template.py
   と同じ設計をそのまま踏襲)。高安がタイ/横ばいの稀なケースでは同種ピボットの重複・
   欠落があり得る。
2. SH_L/SL_L(肩)は「まだNECK1が無い間」は新規同種ピボットで常に直近値へ上書きし、
   さらに「NECK1確定後にHEADの条件(SH_L超え/SL_L未満)を満たさない同種ピボットが来た
   場合」もその新しいピボットを新SH_L/SL_Lとして仕切り直す(NECK1/HEAD/NECK2は破棄)。
   同様に「NECK2確定後に右肩候補が頭以上/以下(頭を超えて三尊の定義を満たさない)、
   または肩の水準がshoulder_tol許容を超えて不一致」の場合も、その候補ピボットを新
   SH_L/SL_Lとして仕切り直す。これにより肩/頭の参照が古い値に固定され続けて構造判定
   が事実上停止する事態を防ぐ(直近のスイング構造だけを見る近似、quasimodo踏襲)。
   NECK1/NECK2は「その脚を確定させるまでに出た直近の戻り側ピボット」を都度上書きする
   (最深値ではなく直近値を採用する近似)。
3. ネックラインは「左肩-頭間の谷と頭-右肩間の谷(山)の低い(高い)方」を採用する簡易化
   (2点を結ぶ傾いた線ではなくフラットな1本の水準として扱う)。平均を取る代替案も
   あり得るが、より保守的な「安値側の壁を完全に割る/高値側の壁を完全に超える」ことを
   ブレイク条件とするため低い方/高い方を採用した。
4. 右肩確定後(パターン完成後)の「頭超え/割れによる無効化」と「ネックラインブレイクに
   よる発火」の判定は、当バーの生値(高安・終値。確定ピボットではない)を使うため即応的
   に働く(quasimodo流)。1本のバーが無効化条件とブレイク条件を同時に満たす場合は
   無効化を優先する(OHLCからは本足内の高安の経路順序が復元できないための近似、
   quasimodo_template.pyの優先順位を踏襲)。dir_mode=0で買い/売りが同一バーで同時成立
   した場合は売り(-1)の書き込みを優先する(実装上の決定、実際にはほぼ発生しない)。
5. shoulder_tol判定に使うATRがウォームアップ期間中でNaNの場合、比較は自動的に不成立
   (False)になり肩候補の仕切り直しが起きる(明示的な分岐を設けない近似、実害は
   系列先頭の僅かな期間に限られる)。
6. SLは概念上HEADの外側(三尊ならHEADの上、逆三尊ならHEADの下)に置くのが定石だが、
   本テンプレートの出力contractは{-1,1,0}のみでSL価格自体は出力しない(engine側の
   sl_pipsパラメータで近似する)。ネックラインの傾き・出来高(ボリューム)確認・
   厳密な"H&S比率"判定は省略する(注記)。

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


def _signal_head_shoulders(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    k = max(1, int(p["swing_k"].value))                 # スイングピボットの左右幅
    shoulder_tol = float(p["shoulder_tol"].value)        # 肩の許容水準差(×ATR)
    atr_period = max(2, int(p["atr_period"].value))
    dir_mode = int(p["dir_mode"].value)                   # 0=both,1=long,2=short

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr_arr = atr(df["high"], df["low"], df["close"], atr_period).to_numpy(float)

    # ピボットフラグ(中心窓は未来を覗くが、k本後にしか読み出さないため先読みなし)
    hh = df["high"].rolling(2 * k + 1, center=True).max().to_numpy()
    ll = df["low"].rolling(2 * k + 1, center=True).min().to_numpy()
    is_ph = high == hh
    is_pl = low == ll

    sh_price: list = []; sh_idx: list = []   # 確定済みスイング高値
    sl_price: list = []; sl_idx: list = []   # 確定済みスイング安値

    # 弱気三尊(bear)状態: 0=idle, 1=SH_L確定済み(NECK1待ち), 2=NECK1確定済み(頭待ち),
    #                     3=頭確定済み(NECK2待ち), 4=NECK2確定済み(右肩待ち),
    #                     5=右肩確定・パターン完成(ネックライン割れ/頭超え待ち)
    bear_state = 0
    bear_shl = bear_head = bear_neck1 = bear_neck2 = bear_neckline = float("nan")
    # 強気逆三尊(bull)状態: 記号はミラー(SL_L/NECK1/頭/NECK2/右肩/ネックライン)
    bull_state = 0
    bull_sll = bull_head = bull_neck1 = bull_neck2 = bull_neckline = float("nan")

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

        # --- 弱気三尊: 新規確定高値ピボットで SH_L/頭/右肩を更新 ---
        if new_ph:
            h = sh_price[-1]
            if bear_state == 0:
                bear_shl, bear_state = h, 1
            elif bear_state == 1:
                bear_shl = h  # NECK1なしでの新高値は左肩候補を更新(直近優先)
            elif bear_state == 2:
                if h > bear_shl:
                    bear_head, bear_state = h, 3  # 左肩超え → 頭確定
                else:
                    bear_shl, bear_state = h, 1  # 超えない新高値は仕切り直し(NECK1破棄)
            elif bear_state == 3:
                if h > bear_head:
                    bear_head = h  # NECK2確定前にさらに高値更新 → 頭を延長
            elif bear_state == 4:
                if h >= bear_head:
                    # 頭以上の新高値 → 「頭>右肩」を満たさないため新SH_Lとして仕切り直し
                    bear_shl, bear_state = h, 1
                    bear_head = bear_neck1 = bear_neck2 = float("nan")
                else:
                    band = shoulder_tol * atr_arr[t]
                    if abs(bear_shl - h) <= band:
                        bear_neckline = min(bear_neck1, bear_neck2)  # ネックライン=2つの谷の低い方(簡易化)
                        bear_state = 5  # 右肩確定・パターン完成 → ブレイク監視へ
                    else:
                        bear_shl, bear_state = h, 1  # 肩の水準が不一致 → 新SH_Lとして仕切り直し
                        bear_head = bear_neck1 = bear_neck2 = float("nan")
        # --- 弱気三尊: 新規確定安値ピボットで NECK1/NECK2 を更新 ---
        if new_pl:
            lo = sl_price[-1]
            if bear_state == 1:
                bear_neck1, bear_state = lo, 2
            elif bear_state == 2:
                bear_neck1 = lo  # 頭確定前の戻り安値(NECK1候補)を随時更新
            elif bear_state == 3:
                bear_neck2, bear_state = lo, 4
            elif bear_state == 4:
                bear_neck2 = lo  # 右肩確定前の戻り安値(NECK2候補)を随時更新

        # --- 弱気三尊: パターン完成後は当バー生値で無効化/ブレイクを判定(quasimodo流) ---
        if bear_state == 5:
            if high[t] > bear_head:
                bear_state = 0  # 頭超え → 無効化(経路順序不明のため無効化を優先)
                bear_shl = bear_head = bear_neck1 = bear_neck2 = bear_neckline = float("nan")
            elif close[t] < bear_neckline:
                bear_out[t] = True  # ネックライン割れ確定 → 売り
                bear_state = 0
                bear_shl = bear_head = bear_neck1 = bear_neck2 = bear_neckline = float("nan")

        # --- 強気逆三尊(ミラー): 新規確定安値ピボットで SL_L/頭/右肩を更新 ---
        if new_pl:
            lo = sl_price[-1]
            if bull_state == 0:
                bull_sll, bull_state = lo, 1
            elif bull_state == 1:
                bull_sll = lo
            elif bull_state == 2:
                if lo < bull_sll:
                    bull_head, bull_state = lo, 3  # 左肩未満 → 頭確定
                else:
                    bull_sll, bull_state = lo, 1
            elif bull_state == 3:
                if lo < bull_head:
                    bull_head = lo  # NECK2確定前にさらに安値更新 → 頭を延長
            elif bull_state == 4:
                if lo <= bull_head:
                    bull_sll, bull_state = lo, 1
                    bull_head = bull_neck1 = bull_neck2 = float("nan")
                else:
                    band = shoulder_tol * atr_arr[t]
                    if abs(bull_sll - lo) <= band:
                        bull_neckline = max(bull_neck1, bull_neck2)  # ネックライン=2つの山の高い方(簡易化)
                        bull_state = 5
                    else:
                        bull_sll, bull_state = lo, 1
                        bull_head = bull_neck1 = bull_neck2 = float("nan")
        # --- 強気逆三尊: 新規確定高値ピボットで NECK1/NECK2 を更新 ---
        if new_ph:
            h = sh_price[-1]
            if bull_state == 1:
                bull_neck1, bull_state = h, 2
            elif bull_state == 2:
                bull_neck1 = h
            elif bull_state == 3:
                bull_neck2, bull_state = h, 4
            elif bull_state == 4:
                bull_neck2 = h

        # --- 強気逆三尊: パターン完成後は当バー生値で無効化/ブレイクを判定 ---
        if bull_state == 5:
            if low[t] < bull_head:
                bull_state = 0  # 頭割れ → 無効化
                bull_sll = bull_head = bull_neck1 = bull_neck2 = bull_neckline = float("nan")
            elif close[t] > bull_neckline:
                bull_out[t] = True  # ネックライン超え確定 → 買い
                bull_state = 0
                bull_sll = bull_head = bull_neck1 = bull_neck2 = bull_neckline = float("nan")

    if dir_mode in (0, 1):
        signal.iloc[bull_out] = 1
    if dir_mode in (0, 2):
        signal.iloc[bear_out] = -1
    return signal


templates.register(
    "head_shoulders",
    defaults={
        "swing_k": 5.0, "shoulder_tol": 1.0, "atr_period": 14.0, "dir_mode": 0.0,
        "sl_pips": 50.0, "tp_pips": 100.0, "lot": 0.1,
    },
    signal_fn=_signal_head_shoulders,
)
