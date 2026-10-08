"""水平線(サポート/レジスタンス)自動検出・反応 テンプレート(「水平線トレード」の機械化)。

確定スイング高安(quasimodo/order_blockと同じ因果的ピボット検出)を価格クラスタとして
併合し、タッチ回数がmin_touches以上に達した価格帯を「有効な水平線(強レベル)」とみなす。
その線に対する当該足の反応を2方式(mode)で機械的にシグナル化する。

アルゴリズム:
1. 確定スイング(causal confirmed swings): quasimodo/order_blockと同一ロジック。中心窓
   (±swing_k本)のローリング最大/最小との厳密一致でピボット候補を検出し、ピボットi(=t-k)は
   swing_k本後のバーtで初めて確定・登録する。中心窓自体は未来を覗くが、読み出しがk本
   遅延するため先読みにはならない(既存2テンプレートと同一の確定ラグ規律)。
2. クラスタリング(=水平線の形成): 確定したスイング価格(高値・安値のどちらも1本の
   「レベル候補」として扱う)を、既存レベル群のうち最も近いものとの距離が
   tol = cluster_atr×ATR(atr_period) 以内なら併合(価格は全タッチの逐次平均へ更新、
   タッチ回数+1)、tol超なら新規レベルを作成する。マージ探索は強レベルに限らず全レベル
   (タッチ1回のみの弱いレベルも含む)を対象にする(弱レベルが2回目のタッチで昇格できる
   ようにするため)。
3. 有効な水平線 = タッチ回数 >= min_touches(既定2)。タッチ回数がmin_touchesへ到達した
   時点でそのレベルを「強レベル」集合へ昇格し、以降の反応判定(mode0/1)の対象にする。
4. 反応判定(mode、パラメータで選択):
   - mode=0 反発(バウンス逆張り): 当該足のlowが強レベルからtouch_atr×ATR以内に接近し
     closeがそのレベルより上で確定 → +1(買い、サポート反発)。highが強レベルに
     touch_atr×ATR以内接近しcloseがレベルより下で確定 → -1(売り、レジスタンス反発)。
   - mode=1 ブレイク: 直前確定終値(close[t-1])から見て直近上側にある強レベルを、当該足の
     終値が上抜け → +1。直近下側の強レベルを終値が下抜け → -1。
5. 再アーム: raw条件(生の反応判定、bool配列)を毎バー独立に計算し、
   raw & ~raw.shift(1) で連続バーの重複発火だけを抑制する(reversal_at_supportと同じ
   簡易デバウンス)。レベルごとの「使用済み」フラグは持たないため、価格が同じ強レベルへ
   離れてから再度接近すれば再発火しうる(仕様上の「再アーム」、1レベル1回=1タッチ
   連続区間につき1回、という近似)。

近似(Note approximations、必読):
1. クラスタリング・接近判定(touch_atr)のATRは「そのイベントを処理している時点(バーt)」
   の値を使う(スイングが実際に形成されたバーiの当時のボラティリティではない)。
2. レベル価格は全タッチの逐次平均(価格=併合平均)を採用する(初出価格に固定する方式では
   ない)。クラスタ中心が新しいタッチのたびに緩やかに動く(concept drift)ことを許容する。
3. レベルは一度作成されると期間中ずっと保持され、期限切れ・自動削除は行わない
   (quasimodo/order_blockのsh_price/sl_priceリストと同じ蓄積方式)。
4. mode=0(反発)の接近判定は対称距離(|low-レベル|<=band、|high-レベル|<=band)で行う。
   レベルがヒゲをわずかに下回って(上回って)いても、closeでの確定方向が条件を満たせば
   反応とみなす(「レベルより真に外側にあるスイングのみ」という厳密な片側判定ではない)。
5. mode=1(ブレイク)は「直前確定終値から見て直近の強レベル」を基準にする。もし
   「終値が強レベルを上回っている」を無条件に毎バー拾う実装にすると、何年も前に価格が
   通過した遠方の水準が恒久的に条件を満たし続け、直近で意味のあるブレイクの検知を汚染する
   (退化ケース)。これを避けるための設計上の近似であり、プロンプト原文の「終値が『上の
   強レベル』を上抜け」を「直前終値から見て最寄りの上側水準」と解釈している。
6. 同一バーで買い・売り条件が同時に成立する稀なケースは買いを優先する(order_block/
   quasimodoと同じ便宜的tie-break)。
7. SLは概念上レベルの外側に置くべきだが、本テンプレートの出力は{-1,1,0}のみで、
   構造的なsl_price/tp_price列は返さない。engine側のsl_pips/tp_pipsによる固定pips近似と
   する(プロンプト指示どおり)。
8. 出来高プロファイル・ラウンドナンバー・上位足の水平線は使わず、当該足のスイング
   クラスタのみを使う。線の水平許容(角度・傾き)は考慮しない(プロンプト指定の近似を
   そのまま踏襲)。
9. ATRウォームアップ中(atr_period分のデータが揃うまでNaN)はクラスタリング・反応判定の
   いずれも行わない(先読み回避のための保守的スキップ、quasimodoのband NaNガードと同じ
   思想)。

先読み規律: シグナルは現在の確定バー+過去の確定スイングのみを使用する。ピボット確定は
swing_k本の遅延を厳守し(中心窓が未来を覗いても読み出しがk本後になるため先読みにならない
ことが構造的に保証される、quasimodo/order_blockと同一ロジック)。ATRはindicators.atr
(shift+ewmの因果的計算)を使用する。mode=1のブレイク判定が参照する「直前確定終値」も
close[t-1]であり未来を見ない。執行用の.shiftはengine側(raw_signal.shift(1))が行うため、
テンプレート内では一切shiftしない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _register_level(
    price: float,
    tol: float,
    lvl_price: list,
    lvl_touch: list,
    strong_idx: list,
    min_touches: int,
) -> None:
    """確定スイング価格priceを既存レベルにtol以内・最近傍で併合(タッチ+1・価格は逐次平均で
    更新)し、なければ新規レベルを作成する。タッチ回数がmin_touchesへ到達したレベルを
    strong_idx(強レベル集合、反応判定で使う)へ昇格させる(マージ探索自体は全レベルが対象、
    弱レベルも次のタッチで昇格できるようにするため除外しない)。"""
    best_j = -1
    best_dist = tol
    for j in range(len(lvl_price)):
        d = abs(lvl_price[j] - price)
        if d <= best_dist:
            best_dist = d
            best_j = j
    if best_j >= 0:
        lvl_touch[best_j] += 1
        lvl_price[best_j] += (price - lvl_price[best_j]) / lvl_touch[best_j]
        if lvl_touch[best_j] >= min_touches and best_j not in strong_idx:
            strong_idx.append(best_j)
    else:
        lvl_price.append(price)
        lvl_touch.append(1)
        if 1 >= min_touches:
            strong_idx.append(len(lvl_price) - 1)


def _bounce_hit(
    strong_idx: list, lvl_price: list, ref: float, band: float, close_t: float, above: bool
) -> bool:
    """強レベルのいずれかがrefからband以内にあり、close_tがそのレベルを挟んで指定方向
    (above=True→レベルより上、False→レベルより下)で確定していればTrue(mode=0反発判定)。"""
    for j in strong_idx:
        lp = lvl_price[j]
        if abs(ref - lp) <= band:
            if (close_t > lp) if above else (close_t < lp):
                return True
    return False


def _nearest_level(strong_idx: list, lvl_price: list, ref: float, above: bool):
    """強レベルのうちref以上(above=True)で最小、またはref以下(above=False)で最大の価格を
    返す(mode=1ブレイク判定で「直前終値から見た最寄りの水準」を求める)。無ければNone。"""
    best = None
    for j in strong_idx:
        lp = lvl_price[j]
        if above:
            if lp >= ref and (best is None or lp < best):
                best = lp
        else:
            if lp <= ref and (best is None or lp > best):
                best = lp
    return best


def _signal_hline_react(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    k = max(1, int(p["swing_k"].value))                 # スイング確定までの遅延本数
    cluster_mult = float(p["cluster_atr"].value)         # クラスタリング許容(×ATR)
    min_touches = max(1, int(p["min_touches"].value))    # 有効な水平線とみなす最小タッチ回数
    touch_mult = float(p["touch_atr"].value)             # 反発(mode0)の接近許容(×ATR)
    atr_period = max(2, int(p["atr_period"].value))
    mode = int(p["mode"].value)                          # 0=反発,1=ブレイク
    dir_mode = int(p["dir_mode"].value)                   # 0=both,1=long,2=short

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr_arr = atr(df["high"], df["low"], df["close"], atr_period).to_numpy(float)

    # --- 確定スイング(quasimodo/order_blockと同一ロジック、指定コードをそのまま使用) ---
    hh = df["high"].rolling(2 * k + 1, center=True).max().to_numpy()
    ll = df["low"].rolling(2 * k + 1, center=True).min().to_numpy()
    is_ph = high == hh
    is_pl = low == ll

    lvl_price: list = []    # レベル価格(逐次平均)
    lvl_touch: list = []    # タッチ回数
    strong_idx: list = []   # touch_count>=min_touchesに達したレベルのインデックス集合

    allow_long = dir_mode in (0, 1)
    allow_short = dir_mode in (0, 2)

    raw_long = np.zeros(n, dtype=bool)
    raw_short = np.zeros(n, dtype=bool)

    for t in range(n):
        atr_t = atr_arr[t]
        i = t - k
        if i >= 0 and not np.isnan(atr_t):
            tol = cluster_mult * atr_t
            if is_ph[i]:
                _register_level(float(high[i]), tol, lvl_price, lvl_touch, strong_idx, min_touches)
            if is_pl[i]:
                _register_level(float(low[i]), tol, lvl_price, lvl_touch, strong_idx, min_touches)

        if strong_idx:
            if mode == 0:
                # --- 反発(バウンス逆張り) ---
                if not np.isnan(atr_t):
                    band = touch_mult * atr_t
                    if allow_long and _bounce_hit(strong_idx, lvl_price, low[t], band, close[t], True):
                        raw_long[t] = True
                    if allow_short and _bounce_hit(strong_idx, lvl_price, high[t], band, close[t], False):
                        raw_short[t] = True
            elif t >= 1:
                # --- ブレイク(直前確定終値から見た最寄りの強レベルを当バー終値が抜けたか) ---
                prev_close = close[t - 1]
                if allow_long:
                    lp = _nearest_level(strong_idx, lvl_price, prev_close, True)
                    if lp is not None and close[t] > lp:
                        raw_long[t] = True
                if allow_short:
                    lp = _nearest_level(strong_idx, lvl_price, prev_close, False)
                    if lp is not None and close[t] < lp:
                        raw_short[t] = True

    raw_long_s = pd.Series(raw_long, index=df.index)
    raw_short_s = pd.Series(raw_short, index=df.index)
    long_sig = raw_long_s & (~raw_long_s.shift(1).fillna(False))
    short_sig = raw_short_s & (~raw_short_s.shift(1).fillna(False))

    if allow_long:
        signal[long_sig] = 1
    if allow_short:
        signal[short_sig & (signal == 0)] = -1  # 同時成立時は買い優先(order_blockと同じtie-break)
    return signal


templates.register(
    "hline_react",
    defaults={
        "swing_k": 5.0,
        "cluster_atr": 0.5,
        "min_touches": 2.0,
        "touch_atr": 0.3,
        "atr_period": 14.0,
        "mode": 0.0,
        "dir_mode": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 45.0,
        "lot": 0.1,
    },
    signal_fn=_signal_hline_react,
)
