"""出来高収縮押し目(vp_pullback)テンプレート — Gajjala/SABAI動画手法の機械化核(2026-07-18)。

「上昇インパルス+出来高増 → 低出来高の浅い押しで9EMAタッチ → 押し目高値ブレイクで買い、
SL=前バー安値、R倍数利確」。動画の裁量部分(銘柄選択・地合い)は対象外の
チャートパターン層のみの検証。ミラーショートは対照(β判定用)。

条件(バーi確定情報のみ、事前登録):
- インパルス: (close - close[imp_bars前])/ATR14 >= imp_atr かつ
  出来高rolling(imp_bars)平均 > vol_sma(50)×vol_up_mult
- 直近 recent_bars 以内にインパルス発生
- 押し: 直近 touch_bars 以内に low <= EMA(ema_p) かつ
  押し出来高(3本平均) < インパルス出来高×vol_dn_mult、EMA上向き
- トリガー: close > 前バーhigh
- SL = 前バーlow − 0.1×ATR / TP = エントリー基準 + rr×リスク(動的sl/tp列)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema
from app.core.strategy_model import Strategy


def _signal_vp_pullback(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    ema_p = max(2, int(strategy.params["ema_period"].value))
    imp_bars = max(3, int(strategy.params["imp_bars"].value))
    imp_atr = float(strategy.params["imp_atr"].value)
    vol_up = float(strategy.params["vol_up_mult"].value)
    vol_dn = float(strategy.params["vol_dn_mult"].value)
    recent = max(2, int(strategy.params["recent_bars"].value))
    touch_b = max(1, int(strategy.params["touch_bars"].value))
    rr = float(strategy.params["rr"].value)
    side = int(strategy.params["side"].value)

    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    v = df["volume"].astype(float)
    e9 = ema(c, ema_p)
    a = atr(h, l, c, 14)
    vol50 = v.rolling(50).mean()
    imp_vol = v.rolling(imp_bars).mean()
    pull_vol = v.rolling(3).mean()

    ret_atr = (c - c.shift(imp_bars)) / a
    # --- ロング側 ---
    impulse_up = (ret_atr >= imp_atr) & (imp_vol > vol_up * vol50)
    recent_imp_up = impulse_up.rolling(recent).max().fillna(0) > 0
    touched_up = (l <= e9).rolling(touch_b).max().fillna(0) > 0
    vol_dry = pull_vol < vol_dn * imp_vol.shift(3)
    ema_up = e9 > e9.shift(3)
    long_sig = recent_imp_up & touched_up & vol_dry & ema_up & (c > h.shift(1)) & (c > e9)
    # --- ショート側(ミラー対照) ---
    impulse_dn = (ret_atr <= -imp_atr) & (imp_vol > vol_up * vol50)
    recent_imp_dn = impulse_dn.rolling(recent).max().fillna(0) > 0
    touched_dn = (h >= e9).rolling(touch_b).max().fillna(0) > 0
    ema_dn = e9 < e9.shift(3)
    short_sig = recent_imp_dn & touched_dn & vol_dry & ema_dn & (c < l.shift(1)) & (c < e9)

    if side > 0:
        short_sig &= False
    elif side < 0:
        long_sig &= False

    n = len(df)
    sig = np.zeros(n, dtype=int)
    slp = np.full(n, np.nan)
    tpp = np.full(n, np.nan)
    li = long_sig.fillna(False).to_numpy()
    si = short_sig.fillna(False).to_numpy()
    ca = c.to_numpy(); la = l.to_numpy(); ha = h.to_numpy(); aa = a.to_numpy()
    sig[li] = 1
    sig[si] = -1
    sl_l = la - 0.1 * aa
    risk_l = ca - sl_l
    slp[li] = np.roll(sl_l, 1)[li]
    tpp[li] = ca[li] + rr * (ca[li] - np.roll(sl_l, 1)[li])
    sl_s = ha + 0.1 * aa
    slp[si] = np.roll(sl_s, 1)[si]
    tpp[si] = ca[si] - rr * (np.roll(sl_s, 1)[si] - ca[si])
    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


templates.register(
    "vp_pullback",
    defaults={
        "ema_period": 9.0,
        "imp_bars": 12.0,
        "imp_atr": 2.0,
        "vol_up_mult": 1.3,
        "vol_dn_mult": 0.8,
        "recent_bars": 6.0,
        "touch_bars": 3.0,
        "rr": 2.0,
        "side": 1.0,        # 1=ロングのみ(動画準拠) -1=ミラー
        "sl_pips": 500.0,
        "tp_pips": 500.0,
        "max_hold_bars": 96.0,
        "lot": 0.1,
    },
    signal_fn=_signal_vp_pullback,
)
