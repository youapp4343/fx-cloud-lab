"""MTFスキャル(mtf_scalp): 上位足で方向、下位足でタイミング。

仕様(2026-08-11 指示):
  - 実行足 M1〜H1(既定は下位足=実行足、上位足はその倍数で内部生成)
  - 上位足で方向性を確認 / 下位足でエントリーポイントを確認
  - リスクリワード 1:1.5〜1:2(SL=ATR×倍率、TP=SL×rr)
  - オシレーターはフィルターにも引き金にも使ってよい
  - 1日3回程度のエントリー頻度を狙う(しきい値で調整)

★上位足の参照は「確定済みバーのみ」
  過去に MTF を ffill で結合して完全な未来参照(t値43-47の偽エッジ)を出している。
  ここでは上位足バーの**終了時刻**をキーにして merge_asof(direction="backward")する。
  つまり時刻 t の判断には、t 以前に**閉じ終わった**上位足バーしか入らない。

方向(上位足):
  ema_fast > ema_slow かつ ema_slow が slope_lb 本前より上 → 上昇。逆は下降。
  さらに上位足RSIが50を上回る/下回るを方向の二重確認に使う(htf_rsi_gate=1で有効)。

タイミング(下位足):
  上昇局面では「押し目からの反転」を待つ:
    RSI(下位足) が rsi_low を下回った後に上抜け、かつ 終値 > 直前終値
  下降局面は鏡像(rsi_high を上抜け後に下抜け)。
  stoch_gate=1 なら ストキャスティクス %K の同方向クロスも要求する(絞り込み)。

決済:
  SL = 終値 -/+ atr_mult × ATR(下位足)、TP = 反対方向に SL幅 × rr。
  sl_price/tp_price を返し、engine が次バー始値で執行する(shiftはengine側)。

先読み回避: 全指標は確定バーまでの rolling/ewm。上位足は上記の通り確定バーのみ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, rsi
from app.core.strategy_model import Strategy

# 実行足 → 上位足の対応(pandasのresample rule)
_TF_RULE = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1h", "H4": "4h", "D1": "1D"}
_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _htf_frame(df: pd.DataFrame, mult: int) -> pd.DataFrame:
    """実行足を mult 倍に束ねた上位足を作る。各行に「そのバーが確定した時刻」を持たせる。"""
    idx = np.arange(len(df)) // mult
    g = df.groupby(idx)
    htf = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        # 確定時刻 = そのグループの最後の実行足バーの位置。これ以降でのみ参照してよい。
        "end_pos": g.apply(lambda x: x.index[-1], include_groups=False)
        if hasattr(g, "apply") else None,
    })
    # end_pos が取れない環境向けのフォールバック(位置ベースで計算)
    if htf["end_pos"].isna().any():
        ends = []
        for k in range(len(htf)):
            ends.append(min((k + 1) * mult - 1, len(df) - 1))
        htf["end_pos"] = ends
    return htf


def _signal_mtf_scalp(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    mult = max(2, int(p["htf_mult"].value))          # 上位足 = 実行足 × mult
    ema_fast = max(2, int(p["htf_ema_fast"].value))
    ema_slow = max(3, int(p["htf_ema_slow"].value))
    slope_lb = max(1, int(p["htf_slope_lb"].value))
    htf_rsi_gate = int(p["htf_rsi_gate"].value)
    htf_rsi_period = max(2, int(p["htf_rsi_period"].value))

    rsi_period = max(2, int(p["rsi_period"].value))
    rsi_low = float(p["rsi_low"].value)
    rsi_high = float(p["rsi_high"].value)
    stoch_gate = int(p["stoch_gate"].value)
    stoch_k = max(2, int(p["stoch_k"].value))

    atr_period = max(2, int(p["atr_period"].value))
    atr_mult = abs(float(p["atr_mult"].value))
    rr = abs(float(p["rr"].value))
    side_mode = int(p["side_mode"].value)             # 0=両方向 1=買いのみ -1=売りのみ

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    n = len(df)

    # ---------------- 上位足(確定バーのみ) ----------------
    htf = _htf_frame(df, mult)
    hf_close = htf["close"]
    hf_ema_f = ema(hf_close, ema_fast)
    hf_ema_s = ema(hf_close, ema_slow)
    hf_rsi = rsi(hf_close, htf_rsi_period)
    hf_up = (hf_ema_f > hf_ema_s) & (hf_ema_s > hf_ema_s.shift(slope_lb))
    hf_dn = (hf_ema_f < hf_ema_s) & (hf_ema_s < hf_ema_s.shift(slope_lb))
    if htf_rsi_gate == 1:
        hf_up = hf_up & (hf_rsi > 50.0)
        hf_dn = hf_dn & (hf_rsi < 50.0)

    # 実行足へ展開: 「そのバーが確定した次の実行足バー」から有効にする
    up_arr = np.zeros(n, dtype=bool)
    dn_arr = np.zeros(n, dtype=bool)
    ends = htf["end_pos"].to_numpy()
    up_vals = hf_up.fillna(False).to_numpy()
    dn_vals = hf_dn.fillna(False).to_numpy()
    for k in range(len(htf)):
        start = int(ends[k]) + 1          # ★確定した次のバーから
        if start >= n:
            break
        stop = int(ends[k + 1]) + 1 if k + 1 < len(htf) else n
        up_arr[start:stop] = up_vals[k]
        dn_arr[start:stop] = dn_vals[k]
    trend_up = pd.Series(up_arr, index=df.index)
    trend_dn = pd.Series(dn_arr, index=df.index)

    # ---------------- 下位足(タイミング) ----------------
    r = rsi(c, rsi_period)
    r_prev = r.shift(1)
    cross_up = (r_prev <= rsi_low) & (r > rsi_low)      # 売られ過ぎからの回復
    cross_dn = (r_prev >= rsi_high) & (r < rsi_high)    # 買われ過ぎからの反落

    if stoch_gate == 1:
        ll = l.rolling(stoch_k).min()
        hh = h.rolling(stoch_k).max()
        k_pct = 100.0 * (c - ll) / (hh - ll).replace(0, np.nan)
        k_prev = k_pct.shift(1)
        cross_up = cross_up & (k_pct > k_prev)
        cross_dn = cross_dn & (k_pct < k_prev)

    long_sig = (trend_up & cross_up).fillna(False)
    short_sig = (trend_dn & cross_dn).fillna(False)
    if side_mode > 0:
        short_sig = short_sig & False
    elif side_mode < 0:
        long_sig = long_sig & False

    # ---------------- SL/TP(リスクリワード固定) ----------------
    a = atr(h, l, c, atr_period)
    dist = (atr_mult * a).to_numpy()
    ca = c.to_numpy()
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    li = long_sig.to_numpy()
    si = short_sig.to_numpy()
    valid = np.isfinite(dist) & (dist > 0)
    li = li & valid
    si = si & valid

    sig[li] = 1
    sig[si] = -1
    slp[li] = ca[li] - dist[li]
    tpp[li] = ca[li] + rr * dist[li]
    slp[si] = ca[si] + dist[si]
    tpp[si] = ca[si] - rr * dist[si]

    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


_DEFAULTS = {
    "htf_mult": 12.0,        # M5実行 → H1(12本)
    "htf_ema_fast": 20.0,
    "htf_ema_slow": 50.0,
    "htf_slope_lb": 3.0,
    "htf_rsi_gate": 1.0,
    "htf_rsi_period": 14.0,
    "rsi_period": 7.0,
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "stoch_gate": 0.0,
    "stoch_k": 14.0,
    "atr_period": 14.0,
    "atr_mult": 1.0,         # SL = ATR × 1.0
    "rr": 1.75,              # TP = SL × 1.75(指定の1:1.5〜1:2の中央)
    "side_mode": 0.0,
    "sl_pips": 10.0,         # 構造SL/TPが無効な時のフォールバック
    "tp_pips": 17.5,
    "max_hold_bars": 36.0,   # M5×36 = 3時間で時間退出
    "lot": 0.1,
}

templates.register("mtf_scalp", defaults=dict(_DEFAULTS), signal_fn=_signal_mtf_scalp)

# --- 変種: 頻度と絞り込みの違いを探索空間に並べる ---------------------------
# グリッドが振るのは sl_pips/tp_pips + キー名順の2つだけなので、
# 「どれくらい絞るか」は別テンプレートとして登録する。

_loose = dict(_DEFAULTS)
_loose.update({"rsi_low": 35.0, "rsi_high": 65.0, "htf_rsi_gate": 0.0})
templates.register("mtf_scalp_loose", defaults=_loose, signal_fn=_signal_mtf_scalp)

_tight = dict(_DEFAULTS)
_tight.update({"rsi_low": 25.0, "rsi_high": 75.0, "stoch_gate": 1.0})
templates.register("mtf_scalp_tight", defaults=_tight, signal_fn=_signal_mtf_scalp)

_rr15 = dict(_DEFAULTS)
_rr15.update({"rr": 1.5, "tp_pips": 15.0})
templates.register("mtf_scalp_rr15", defaults=_rr15, signal_fn=_signal_mtf_scalp)

_rr20 = dict(_DEFAULTS)
_rr20.update({"rr": 2.0, "tp_pips": 20.0})
templates.register("mtf_scalp_rr20", defaults=_rr20, signal_fn=_signal_mtf_scalp)

# M15実行 → H1(4本)。同じロジックを1段上の足で。
_m15 = dict(_DEFAULTS)
_m15.update({"htf_mult": 4.0, "max_hold_bars": 12.0})
templates.register("mtf_scalp_m15", defaults=_m15, signal_fn=_signal_mtf_scalp)
