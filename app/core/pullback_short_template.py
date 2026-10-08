"""上昇トレンド中の短期過熱を売る(pullback_short)。

由来(2026-08-11):
`scripts/trigger_information_scan.py`(ロールオーバー帯 UTC21-23 を除外した版)で
10銘柄 × 26トリガー × 4ホライズン × 3レジーム = 3,100 組合せのグロス方向情報を測り、
Bonferroni相当(t>4.4)・グロス/コスト>2.0・n>=500 を満たしたのは次の2件だけだった。

  USDCHF H=2時間 run3_up_then_dn  / against_trend : n5740 +0.778pips t=4.91 比2.22
  USDCHF H=2時間 rsi7_cross_dn_70 / against_trend : n4508 +0.814pips t=4.64 比2.32

どちらも「上位足が上昇トレンドのUSDCHFで、短期の過熱を売る」という同じ仮説の別表現。
互いに整合しているため、単一セルの引き当てよりは信頼できる。ただし10銘柄中USDCHFのみで
出ている点は弱く、OOS・プラセボ・実ティックを通すまでは仮説にすぎない。

条件(全て確定バーのみ):
  上位足: EMA(fast)>EMA(slow) かつ EMA(slow) が slope_lb 本前より上 = 上昇トレンド
          ★確定した上位足バーのみ参照(その足が閉じた次の実行足から有効)
  引き金: mode=0 → 3本連続陽線の直後の陰線
          mode=1 → RSI(rsi_period) が rsi_level を上から下に抜けた
          mode=2 → 上の2つのどちらか
  方向  : 売り(情報スキャンで効いていたのは売り側のみ)
  除外  : UTC 21-23 のエントリー(dukascopyロールオーバー人工物。engine側のfiltersでも
          二重に落とすが、テンプレート側でも明示しておく)

決済:
  情報は「一定時間後のリターン」に宿っているため、狭いSLは戻る前に刈られる。
  SL = ATR × atr_mult(広め)、TP = SL幅 × rr、加えて max_hold_bars の時間退出。
  RR 1:1.5〜1:2 を使いたい場合は rr で指定するが、この形の情報とは相性が悪いことを
  ext_fade の検証で確認済み(SL=1ATR/RR1:2 で期待値が負に転じる)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, rsi
from app.core.strategy_model import Strategy

BAD_HOURS = {21, 22, 23}


def _signal_pullback_short(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    mult = max(2, int(p["htf_mult"].value))
    ema_fast = max(2, int(p["htf_ema_fast"].value))
    ema_slow = max(3, int(p["htf_ema_slow"].value))
    slope_lb = max(1, int(p["htf_slope_lb"].value))
    mode = int(p["mode"].value)
    rsi_period = max(2, int(p["rsi_period"].value))
    rsi_level = float(p["rsi_level"].value)
    atr_period = max(2, int(p["atr_period"].value))
    atr_mult = abs(float(p["atr_mult"].value))
    rr = abs(float(p["rr"].value))

    o, c = df["open"], df["close"]
    h, l = df["high"], df["low"]
    n = len(df)

    # ---- 上位足トレンド(確定バーのみ) ----
    idx = np.arange(n) // mult
    g_close = df.groupby(idx)["close"].last()
    ef, es = ema(g_close, ema_fast), ema(g_close, ema_slow)
    up = ((ef > es) & (es > es.shift(slope_lb))).fillna(False).to_numpy()
    trend_up = np.zeros(n, dtype=bool)
    for k in range(len(g_close)):
        start = (k + 1) * mult
        if start >= n:
            break
        trend_up[start:min((k + 2) * mult, n)] = up[k]

    # ---- 引き金 ----
    up_bar = (c > o)
    run3 = (up_bar.shift(1) & up_bar.shift(2) & up_bar.shift(3)).fillna(False)
    trig_run = (run3 & (c < o)).to_numpy()

    r = rsi(c, rsi_period)
    trig_rsi = ((r.shift(1) >= rsi_level) & (r < rsi_level)).fillna(False).to_numpy()

    if mode == 0:
        trig = trig_run
    elif mode == 1:
        trig = trig_rsi
    else:
        trig = trig_run | trig_rsi

    # ---- ロールオーバー帯の除外 ----
    if "timestamp" in df.columns:
        hours = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
        trig = trig & ~np.isin(hours, list(BAD_HOURS))

    short_sig = trig & trend_up

    # ---- SL/TP ----
    a = atr(h, l, c, atr_period)
    dist = (atr_mult * a).to_numpy()
    valid = np.isfinite(dist) & (dist > 0)
    short_sig = short_sig & valid

    ca = c.to_numpy()
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    sig[short_sig] = -1
    slp[short_sig] = ca[short_sig] + dist[short_sig]
    tpp[short_sig] = ca[short_sig] - rr * dist[short_sig]
    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


_BASE = {
    "htf_mult": 12.0,        # M5実行 → H1
    "htf_ema_fast": 20.0,
    "htf_ema_slow": 50.0,
    "htf_slope_lb": 3.0,
    "mode": 2.0,             # 0=3連陽線→陰線 / 1=RSI下抜け / 2=どちらか
    "rsi_period": 7.0,
    "rsi_level": 70.0,
    "atr_period": 14.0,
    "atr_mult": 3.0,         # 広いSL(戻りを待つ形)
    "rr": 1.0,
    "sl_pips": 24.0,
    "tp_pips": 24.0,
    "max_hold_bars": 24.0,   # M5で2時間(情報スキャンで効いていたホライズン)
    "lot": 0.1,
}
templates.register("pullback_short", defaults=dict(_BASE), signal_fn=_signal_pullback_short)

_m0 = dict(_BASE); _m0.update({"mode": 0.0})
templates.register("pullback_short_run3", defaults=_m0, signal_fn=_signal_pullback_short)

_m1 = dict(_BASE); _m1.update({"mode": 1.0})
templates.register("pullback_short_rsi", defaults=_m1, signal_fn=_signal_pullback_short)

# 指示どおりのRR 1:1.5〜1:2 も比較用に用意(この情報形状とは相性が悪い想定)
_rr15 = dict(_BASE); _rr15.update({"atr_mult": 1.0, "rr": 1.5, "sl_pips": 8.0, "tp_pips": 12.0})
templates.register("pullback_short_rr15", defaults=_rr15, signal_fn=_signal_pullback_short)
_rr20 = dict(_BASE); _rr20.update({"atr_mult": 1.0, "rr": 2.0, "sl_pips": 8.0, "tp_pips": 16.0})
templates.register("pullback_short_rr20", defaults=_rr20, signal_fn=_signal_pullback_short)


# --- 決済形状を正した版(2026-08-11) ------------------------------------------
# ★重要: 上の _BASE は atr_mult=3.0 だが、M5のATRは2-3pipsなので SL/TP が ±7-9pips
# となり、2時間の保有を待たずに決済される。この戦略の情報は「2時間後のリターン」に
# 宿っているため、その設定では別物になり期待値が消える(実測: 同一シグナルで
# USDCHF OOS -0.069 → 時間退出にすると +1.002)。
# 以後はこちらを使う。SLは実質無効(ATR×30)にして時間退出のみで決済する。
#
# 実測(M5, 実測コスト, UTC21-23除外, IS2023-25 / OOS2026):
#   USDCHF IS PF1.180 +0.695 / OOS PF1.303 +1.002 (boot_p 0.027) 頻度2.63回/日
#   USDCAD IS PF1.117 +0.561 / OOS PF1.113 +0.437
#   AUDUSD IS PF1.057 +0.237 / OOS PF1.108 +0.476
#   EURUSD IS PF1.010 +0.056 / OOS PF1.082 +0.400
#   7銘柄プール: IS +0.295 / OOS +0.283 (t=0.98)
# 弱点: プールでは有意でない。USDCHF単体のp=0.027も7銘柄の最良なので補正すると約0.19。

_TIME_EXIT = dict(_BASE)
_TIME_EXIT.update({
    "mode": 1.0,             # RSI下抜けのみ(run3との合成より素直)
    "atr_mult": 30.0,        # SLを実質無効化
    "rr": 1.0,
    "max_hold_bars": 24.0,   # M5で2時間
})
templates.register("rsi_fade_short", defaults=dict(_TIME_EXIT), signal_fn=_signal_pullback_short)

for _h, _name in ((12, "rsi_fade_short_h1"), (36, "rsi_fade_short_h3")):
    _v = dict(_TIME_EXIT); _v.update({"max_hold_bars": float(_h)})
    templates.register(_name, defaults=_v, signal_fn=_signal_pullback_short)
