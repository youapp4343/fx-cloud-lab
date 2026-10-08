"""行き過ぎフェード(ext_fade): 上位足トレンド方向への過伸張を逆張りする。

由来(2026-08-11):
`scripts/trigger_information_scan.py` で 468 の(トリガー×レジーム×ホライズン)組合せの
**グロス方向情報**を最適化なしで測ったところ、EURUSD M5 で以下が浮上した。

  bigbar_fade_dn / against_trend : 上位足が上昇中の「大陽線」を売る
      H=15分 +0.598pips (t=3.35, n=1962) / 30分 +0.477 / 60分 +0.585
  bigbar_fade_up / against_trend : 上位足が下降中の「大陰線」を買う
      H=30分 +0.377 (t=1.65, n=1983) / 15分 +0.357

両向きとも正、3つのホライズンすべてで正。統一すると
**「上位足トレンドの方向へ行き過ぎたバーをフェードする」** という単一の仮説になる。
EURUSDの実質往復コストは0.22pips(ThreeTrader)なので、グロス0.6に対して余裕がある。

注意: これは468検定の最良付近であり、選択補正すると単独では有意でない(t=3.35 →
Bonferroni換算で p≈0.19)。OOS・プラセボ・実ティックを通すまでは仮説にすぎない。

条件(全て確定バーのみ):
  上位足トレンド: 実行足を htf_mult 本束ねた足で EMA(fast) と EMA(slow) の整列 + 傾き。
                  ★確定した上位足バーのみ参照する(その足が閉じた次の実行足から有効)。
  過伸張バー   : (高値-安値) / ATR >= exp_thr
  方向         : 上昇トレンド中の陽線 → 売り / 下降トレンド中の陰線 → 買い
  決済         : SL = ATR × atr_mult、TP = SL幅 × rr(指示のRR 1:1.5〜1:2)

先読み回避: ATR・EMAは確定バーまでのrolling/ewm。上位足は確定バーのみ。
シグナルのシフトは engine.run_backtest 側が行う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema
from app.core.strategy_model import Strategy


def _signal_ext_fade(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    mult = max(2, int(p["htf_mult"].value))
    ema_fast = max(2, int(p["htf_ema_fast"].value))
    ema_slow = max(3, int(p["htf_ema_slow"].value))
    slope_lb = max(1, int(p["htf_slope_lb"].value))
    exp_thr = abs(float(p["exp_thr"].value))
    atr_period = max(2, int(p["atr_period"].value))
    atr_mult = abs(float(p["atr_mult"].value))
    rr = abs(float(p["rr"].value))
    side_mode = int(p["side_mode"].value)
    # use_trend=0 で上位足トレンド条件を無効化する(情報スキャンでは regime="all" が最強だった)
    use_trend = int(p["use_trend"].value) if "use_trend" in p else 1

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    n = len(df)

    # ---- 上位足トレンド(確定バーのみ) ----
    idx = np.arange(n) // mult
    g_close = df.groupby(idx)["close"].last()
    ef, es = ema(g_close, ema_fast), ema(g_close, ema_slow)
    up = ((ef > es) & (es > es.shift(slope_lb))).fillna(False).to_numpy()
    dn = ((ef < es) & (es < es.shift(slope_lb))).fillna(False).to_numpy()

    trend = np.zeros(n, dtype=int)
    for k in range(len(g_close)):
        start = (k + 1) * mult          # ★確定した次の実行足から有効
        if start >= n:
            break
        stop = min((k + 2) * mult, n)
        trend[start:stop] = 1 if up[k] else (-1 if dn[k] else 0)

    # ---- 過伸張バー ----
    a = atr(h, l, c, atr_period)
    rng_ratio = (h - l) / a.replace(0, np.nan)
    extended = (rng_ratio >= exp_thr).fillna(False).to_numpy()
    bull_bar = (c > o).to_numpy()
    bear_bar = (c < o).to_numpy()

    # 上昇トレンド中の陽線を売る / 下降トレンド中の陰線を買う
    if use_trend == 1:
        short_sig = extended & bull_bar & (trend == 1)
        long_sig = extended & bear_bar & (trend == -1)
    else:
        short_sig = extended & bull_bar
        long_sig = extended & bear_bar
    if side_mode > 0:
        short_sig = np.zeros(n, dtype=bool)
    elif side_mode < 0:
        long_sig = np.zeros(n, dtype=bool)

    # ---- SL/TP(リスクリワード固定) ----
    dist = (atr_mult * a).to_numpy()
    valid = np.isfinite(dist) & (dist > 0)
    long_sig &= valid
    short_sig &= valid

    ca = c.to_numpy()
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    sig[long_sig] = 1
    sig[short_sig] = -1
    slp[long_sig] = ca[long_sig] - dist[long_sig]
    tpp[long_sig] = ca[long_sig] + rr * dist[long_sig]
    slp[short_sig] = ca[short_sig] + dist[short_sig]
    tpp[short_sig] = ca[short_sig] - rr * dist[short_sig]

    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


_BASE = {
    "htf_mult": 12.0,        # M5実行 → H1
    "htf_ema_fast": 20.0,
    "htf_ema_slow": 50.0,
    "htf_slope_lb": 3.0,
    "exp_thr": 2.0,          # レンジ/ATR がこれ以上で「行き過ぎ」
    "atr_period": 14.0,
    "atr_mult": 1.0,
    "rr": 1.75,
    "side_mode": 0.0,
    "use_trend": 1.0,
    "sl_pips": 8.0,          # 構造SL/TPのフォールバック
    "tp_pips": 14.0,
    "max_hold_bars": 12.0,   # M5で60分(情報スキャンで正だった最長ホライズン)
    "lot": 0.1,
}
templates.register("ext_fade", defaults=dict(_BASE), signal_fn=_signal_ext_fade)

_rr15 = dict(_BASE); _rr15.update({"rr": 1.5, "tp_pips": 12.0})
templates.register("ext_fade_rr15", defaults=_rr15, signal_fn=_signal_ext_fade)

_rr20 = dict(_BASE); _rr20.update({"rr": 2.0, "tp_pips": 16.0})
templates.register("ext_fade_rr20", defaults=_rr20, signal_fn=_signal_ext_fade)

_short = dict(_BASE); _short.update({"max_hold_bars": 3.0})   # 15分で時間退出
templates.register("ext_fade_h3", defaults=_short, signal_fn=_signal_ext_fade)


# --- 決済形状の検証(2026-08-11 の情報スキャン拡張を受けて) -------------------
# 全10銘柄×M5/M15の5,389組合せを測ると、bigbar_fade_up(大陰線を買う)だけが
# 突出していた(210組合せ中72が比>2.0、グロス中央値+0.632pips、t=6〜9.6)。
# 一方 ext_fade(RR1:2、SL=ATR×1.0)はOOSでnullだった。
# 理由は決済形状: この情報は「一定時間後のリターン」に宿っており、狭いSLは戻る前に刈られる。
# → 買い専用・広いSL・時間退出 という形で検証する変種を用意する。

_dipbuy = dict(_BASE)
_dipbuy.update({
    "side_mode": 1.0,        # 買いのみ(売り側は情報スキャンで効いていない)
    "atr_mult": 3.0,         # 広いSL(戻りを待つ)
    "rr": 1.0,               # TP = SL幅。実質は時間退出が主
    "max_hold_bars": 6.0,    # M5で30分(t最大のホライズン)
    "sl_pips": 24.0, "tp_pips": 24.0,
})
templates.register("dip_buy_h6", defaults=dict(_dipbuy), signal_fn=_signal_ext_fade)

_d12 = dict(_dipbuy); _d12.update({"max_hold_bars": 12.0})   # 60分
templates.register("dip_buy_h12", defaults=_d12, signal_fn=_signal_ext_fade)

_d24 = dict(_dipbuy); _d24.update({"max_hold_bars": 24.0})   # 2時間
templates.register("dip_buy_h24", defaults=_d24, signal_fn=_signal_ext_fade)

# トレンド条件なし(情報スキャンでは regime="all" が最強: n=4217, t=7.9〜9.6)
for _h in (6, 12, 24):
    _dn = dict(_dipbuy); _dn.update({"use_trend": 0.0, "max_hold_bars": float(_h)})
    templates.register(f"dip_buy_nt{_h}", defaults=_dn, signal_fn=_signal_ext_fade)
