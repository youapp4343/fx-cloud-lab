"""水平線ブレイク&リテスト(break_retest)テンプレート。

「日足〜1h足で水平線を引き、ブレイク後に線へ戻ってきたタッチ(役割転換=ロールリバーサル)で
エントリー。RR1:2。利確pips=直近スイング(ジグザグ高安)の7割、損切=その半分」を機械化。

アルゴリズム(先読み無し):
1. 確定スイング高安(causal): 中心窓±swing_k本の最大/最小一致でピボット検出、バーt=i+swing_k
   で初めて確定・登録(読み出しがk本遅延=先読みにならない、quasimodo/hline_reactと同一規律)。
2. 水平線: 確定スイング価格をATR×cluster_atr以内で併合(タッチ+1、価格は逐次平均)。
   タッチ>=min_touchesで「強レベル」昇格。近傍(active_dist_atr×ATR以内)の強レベルのみ対象。
3. ブレイク検知: 終値が強レベルをbreak_atr×ATR超で上抜け→そのレベルを「上ブレイク・リテスト待ち」
   (下抜けは逆)。
4. リテスト・エントリー: 上ブレイク済レベルへ価格が戻り、当該足lowがレベル+retest_atr×ATR以内に
   接近しつつ終値がレベル上で確定→買い(役割転換: 旧レジスタンス→サポート)。下は対称で売り。
   1ブレイクにつき1エントリー(消費)。
5. 動的SL/TP: 直近swing_pivots本の確定ピボットの高安レンジ×tp_frac(0.7)を利確距離、
   その1/rr(=1/2)を損切距離とし、sl_price/tp_price列(絶対価格)で返す。engineがshift(1)。

複数足(D1/H4/H1)の線は将来拡張。本テンプレは渡された単一足のスイング線のみ使用。斜め線も未実装。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _register_level(price, tol, lvl_price, lvl_touch, strong_idx, min_touches):
    best_j, best_dist = -1, tol
    for j in range(len(lvl_price)):
        d = abs(lvl_price[j] - price)
        if d <= best_dist:
            best_dist, best_j = d, j
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


def _signal_break_retest(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    k = max(1, int(p["swing_k"].value))
    cluster_mult = float(p["cluster_atr"].value)
    min_touches = max(1, int(p["min_touches"].value))
    atr_period = max(2, int(p["atr_period"].value))
    break_mult = float(p["break_atr"].value)
    retest_mult = float(p["retest_atr"].value)
    active_dist = float(p["active_dist_atr"].value)
    swing_pivots = max(2, int(p["swing_pivots"].value))
    tp_frac = float(p["tp_frac"].value)
    rr = max(0.5, float(p["rr"].value))

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    a = atr(df["high"], df["low"], df["close"], atr_period).to_numpy(float)
    n = len(df)

    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan)
    tp_price = np.full(n, np.nan)

    lvl_price, lvl_touch, strong_idx = [], [], []
    break_state = {}          # level_j -> +1(上ブレイク待ち)/-1(下ブレイク待ち)
    piv_prices = []           # 確定ピボット価格(高安どちらも, 直近スイングレンジ用)

    for t in range(n):
        at = a[t]
        # ピボット確定: バー i=t-k を窓[i-k, i+k]で判定
        i = t - k
        if i - k >= 0 and not np.isnan(at):
            wl, wr = i - k, i + k + 1
            if high[i] == high[wl:wr].max():
                tol = cluster_mult * at
                _register_level(high[i], tol, lvl_price, lvl_touch, strong_idx, min_touches)
                piv_prices.append(high[i])
            if low[i] == low[wl:wr].min():
                tol = cluster_mult * at
                _register_level(low[i], tol, lvl_price, lvl_touch, strong_idx, min_touches)
                piv_prices.append(low[i])

        if np.isnan(at):
            continue
        ct = close[t]

        # 直近スイングレンジ(動的TP用): 直近swing_pivots本の高安幅
        if len(piv_prices) >= swing_pivots:
            recent = piv_prices[-swing_pivots:]
            swing = max(recent) - min(recent)
        else:
            swing = 10.0 * at  # ピボット不足時のフォールバック
        tp_dist = tp_frac * swing
        if tp_dist <= 0:
            continue
        sl_dist = tp_dist / rr

        fired = False
        for j in strong_idx:
            lp = lvl_price[j]
            if abs(lp - ct) > active_dist * at:   # 遠方レベルは無視(近傍のみ)
                continue
            st = break_state.get(j, 0)
            # ブレイク検知
            if ct > lp + break_mult * at and st <= 0:
                break_state[j] = 1
                continue
            if ct < lp - break_mult * at and st >= 0:
                break_state[j] = -1
                continue
            # リテスト・エントリー
            if st == 1 and low[t] <= lp + retest_mult * at and ct > lp and not fired:
                signal[t] = 1
                sl_price[t] = ct - sl_dist
                tp_price[t] = ct + tp_dist
                break_state[j] = 0
                fired = True
            elif st == -1 and high[t] >= lp - retest_mult * at and ct < lp and not fired:
                signal[t] = -1
                sl_price[t] = ct + sl_dist
                tp_price[t] = ct - tp_dist
                break_state[j] = 0
                fired = True

    return pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price}, index=df.index)


templates.register(
    "break_retest",
    defaults={
        "swing_k": 5.0,
        "cluster_atr": 0.5,
        "min_touches": 2.0,
        "atr_period": 14.0,
        "break_atr": 0.3,
        "retest_atr": 0.5,
        "active_dist_atr": 50.0,
        "swing_pivots": 4.0,
        "tp_frac": 0.7,
        "rr": 2.0,
        "sl_pips": 30.0,   # フォールバック(sl_price無効時)
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_break_retest,
)
