"""2段押し(two_leg_pullback)テンプレート — 事前登録検証(2026-07-24, tick_chart_prereg.md #1+#2統合)。

「トレンド中の押し目が"1段目下落→小反発→2段目下落"の2段構造になった時のみエントリー」を
機械化。単純な1回押し(vp_pullback等の既存null群)と異なり、押し目の"形"(浅い反発+
非破壊的な2段目)を要求する点が新規性。ロング/ショートは対称のミラーとして同一ループで
同時に扱う(side引数は持たない)。

状態機械(バーiは確定情報のみ使用、次バー始値で執行=engine側shift):
- トレンド判定: ロングはEMA21>EMA50かつEMA21が5本前より上昇、ショートはその逆。
- フェーズ PH_LEG1: 直近高値(安値)からの1段目下落(上昇)を検知(隣接バー比較でcausal判定)。
  反転(隣バーより高値側に動く)した時点で安値1(高値1)を確定。安値1がEMA21未達(EMA21より
  上、=浅い押し目)であることを確認してからPH_REBOUNDへ(実測: この条件を素通しすると
  ほぼ全ての反発初手でEMA21を上抜けてしまい発火率がゼロに潰れるため、「浅い戻り」は
  安値1自体がEMA21に届いていないことを指す解釈を採用、事前登録spot-checkで確認済み)。
- フェーズ PH_REBOUND: 反発の高値(安値)を追跡するのみ。再度下落(上昇)が始まったらPH_LEG2へ。
- フェーズ PH_LEG2: 2段目下落(上昇)を追跡。反転時に安値2(高値2)を確定。
  安値2が安値1をdepth_atr×ATR14を超えて下回る(高値2が高値1を超えて上回る)場合は
  構造条件違反として無効化しPH_LEG1へ。条件を満たせばPH_TRIGGERへ(同バーが陽線/陰線
  なら即座にシグナル足として記録)。
- フェーズ PH_TRIGGER: シグナル足(2段目内で最初に現れた陽線/陰線)の高値/安値を記録し、
  以降のバーの終値がそれを上抜け(下抜け)確定した時点でシグナル発火。安値2(高値2)を
  先に割り込んだ場合は構造破壊として無効化しPH_LEG1へ。
- SL = 安値2の下(高値2の上)、微小バッファ sl_buffer_atr×ATR14。
  TP = 動的sl_price/tp_price列(RR=rr倍、engine側でshift(1)適用)。

ロング/ショートの状態機械は符号(sign=+1/-1)でパラメータ化した単一ループ関数を2回
呼び出すことで実装し、コード重複を避けている。5.7M行(USDJPY M1相当)でも1パスの
逐次ループで完走する設計(pure Pythonループだが状態はスカラーのみでO(n))。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema
from app.core.strategy_model import Strategy

_PH_LEG1, _PH_REBOUND, _PH_LEG2, _PH_TRIGGER = 0, 1, 2, 3


def _scan_direction(
    n: int,
    ext: np.ndarray,
    opp: np.ndarray,
    o: np.ndarray,
    c: np.ndarray,
    ema21: np.ndarray,
    trend_ok: np.ndarray,
    atr_arr: np.ndarray,
    depth_atr: float,
    sl_buffer_atr: float,
    rr: float,
    sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ロング(sign=+1)/ショート(sign=-1)共通の2段押し状態機械。

    ext: 追跡対象の極値配列(ロングはlow、ショートはhigh)
    opp: 反対側の極値配列(ロングはhigh、ショートはlow) — 反発追跡とシグナル足の
         ブレイク基準値の両方に使う
    """
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)

    phase = _PH_LEG1
    leg_bars = 0
    run_ext = 0.0
    ext1 = 0.0
    run_opp = 0.0
    ext2 = 0.0
    have_sig = False
    sig_opp = 0.0

    for i in range(1, n):
        if not trend_ok[i] or np.isnan(atr_arr[i]):
            phase = _PH_LEG1
            leg_bars = 0
            continue

        moving_adverse = sign * (ext[i] - ext[i - 1]) <= 0  # 継続下落(ロング)/継続上昇(ショート)

        if phase == _PH_LEG1:
            if moving_adverse:
                run_ext = ext[i] if leg_bars == 0 or sign * (ext[i] - run_ext) <= 0 else run_ext
                leg_bars += 1
            else:
                if leg_bars >= 1:
                    ext1 = run_ext
                    leg_bars = 0
                    if sign * (ema21[i] - ext1) < 0:
                        # 安値1(高値1)がEMA21未達(浅い押し目) → 反発追跡へ
                        run_opp = opp[i]
                        phase = _PH_REBOUND
                    else:
                        phase = _PH_LEG1
                else:
                    leg_bars = 0

        elif phase == _PH_REBOUND:
            run_opp = opp[i] if sign * (opp[i] - run_opp) > 0 else run_opp
            if moving_adverse:
                phase = _PH_LEG2
                run_ext = ext[i]
                leg_bars = 1

        elif phase == _PH_LEG2:
            if moving_adverse:
                run_ext = ext[i] if sign * (ext[i] - run_ext) <= 0 else run_ext
                leg_bars += 1
            else:
                if leg_bars >= 1:
                    ext2 = run_ext
                    if sign * (ext1 - ext2) > depth_atr * atr_arr[i]:
                        # 安値2(高値2)が安値1(高値1)を大きく超えて逸脱 → 構造条件違反
                        phase = _PH_LEG1
                        leg_bars = 0
                    else:
                        phase = _PH_TRIGGER
                        have_sig = False
                        sig_opp = 0.0
                        if sign * (c[i] - o[i]) > 0:
                            have_sig = True
                            sig_opp = opp[i]
                else:
                    phase = _PH_LEG1
                    leg_bars = 0

        elif phase == _PH_TRIGGER:
            if sign * (ext2 - ext[i]) > 0:
                # 安値2(高値2)を割り込み(超え) → 構造破壊で無効化
                phase = _PH_LEG1
                leg_bars = 0
                continue
            if not have_sig:
                if sign * (c[i] - o[i]) > 0:
                    have_sig = True
                    sig_opp = opp[i]
            else:
                if sign * (c[i] - sig_opp) > 0:
                    sig[i] = 1 if sign > 0 else -1
                    sl = ext2 - sign * sl_buffer_atr * atr_arr[i]
                    risk = sign * (c[i] - sl)
                    slp[i] = sl
                    tpp[i] = c[i] + sign * rr * risk
                    phase = _PH_LEG1
                    leg_bars = 0

    return sig, slp, tpp


def _signal_two_leg_pullback(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    ema_fast = max(2, int(p["ema_fast"].value))
    ema_slow = max(2, int(p["ema_slow"].value))
    ema_rise_bars = max(1, int(p["ema_rise_bars"].value))
    atr_period = max(2, int(p["atr_period"].value))
    depth_atr = float(p["depth_atr"].value)
    sl_buffer_atr = float(p["sl_buffer_atr"].value)
    rr = float(p["rr"].value)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    ema_f = ema(df["close"], ema_fast)
    ema_s = ema(df["close"], ema_slow)
    a = atr(df["high"], df["low"], df["close"], atr_period)

    trend_up = (ema_f > ema_s) & (ema_f > ema_f.shift(ema_rise_bars))
    trend_dn = (ema_f < ema_s) & (ema_f < ema_f.shift(ema_rise_bars))
    trend_up = trend_up.fillna(False).to_numpy()
    trend_dn = trend_dn.fillna(False).to_numpy()

    ema_f_arr = ema_f.to_numpy(float)
    atr_arr = a.to_numpy(float)
    n = len(df)

    sig_l, slp_l, tpp_l = _scan_direction(
        n, l, h, o, c, ema_f_arr, trend_up, atr_arr, depth_atr, sl_buffer_atr, rr, sign=1
    )
    sig_s, slp_s, tpp_s = _scan_direction(
        n, h, l, o, c, ema_f_arr, trend_dn, atr_arr, depth_atr, sl_buffer_atr, rr, sign=-1
    )

    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan)
    tp_price = np.full(n, np.nan)
    long_fired = sig_l != 0
    short_fired = (sig_s != 0) & ~long_fired  # 同バー衝突時はロング優先(稀な想定)
    signal[long_fired] = sig_l[long_fired]
    sl_price[long_fired] = slp_l[long_fired]
    tp_price[long_fired] = tpp_l[long_fired]
    signal[short_fired] = sig_s[short_fired]
    sl_price[short_fired] = slp_s[short_fired]
    tp_price[short_fired] = tpp_s[short_fired]

    return pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price}, index=df.index)


templates.register(
    "two_leg_pullback",
    defaults={
        "ema_fast": 21.0,
        "ema_slow": 50.0,
        "ema_rise_bars": 5.0,
        "atr_period": 14.0,
        "depth_atr": 0.5,       # 事前登録で固定(sweepでの探索対象外)
        "sl_buffer_atr": 0.1,
        "rr": 2.0,
        "sl_pips": 50.0,        # フォールバック(sl_price無効時)
        "tp_pips": 100.0,
        "max_hold_bars": 300.0,
        "lot": 0.1,
    },
    signal_fn=_signal_two_leg_pullback,
)
