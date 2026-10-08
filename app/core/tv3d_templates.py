# -*- coding: utf-8 -*-
"""TradingView catalog 201-300 の第4バッチ。

未読だったB判定26本のうち、条件が完全に確定した2本。
残り24本は「描画オブジェクトの状態機械」または「学習済みフィルタ」を含むため
そのままでは移植できない(理由は検証報告に明記)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr, sma
from app.core.strategy_model import Strategy
from app.core.tv3_templates import _EXIT_DEFAULTS, _atr_exit


# ==========================================================================
# 274 Fair Value Gap Signals [UAlgo]
#     FVG形成(low > high[2] / high < low[2])をATRフィルタで選別し、
#     その後ゾーンに価格が入った(テスト)足で入る。
#     ★ATRフィルタが既定ONで「ギャップ幅がATR未満なら捨てる」。
#       213(Oberlunar版)と違い、ゾーンの鮮度制限でなく大きさで絞る。
# ==========================================================================
def _sig_fvg_ualgo(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    h, l, c = df["high"], df["low"], df["close"]
    a = atr(h, l, c, max(2, int(p["atr_len"].value)))

    bull = l > h.shift(2)
    bear = h < l.shift(2)
    if float(p["use_atr_filter"].value) > 0.5:
        bull &= (l - h.shift(2)) >= a
        bear &= (l.shift(2) - h) >= a

    top_b, bot_b = l.where(bull), h.shift(2).where(bull)
    top_s, bot_s = l.shift(2).where(bear), h.where(bear)
    keep = max(1, int(p["max_age"].value))

    def _test(gap, top, bot, side: int) -> np.ndarray:
        """ゾーン形成後、初めて価格が中に入った足で1回だけ点灯させる。"""
        g = gap.fillna(False).to_numpy()
        tp, bt = top.to_numpy(float), bot.to_numpy(float)
        hh, ll, cc = h.to_numpy(), l.to_numpy(), c.to_numpy()
        out = np.zeros(len(g), dtype=bool)
        born, t, b = -10 ** 9, np.nan, np.nan
        used = True
        for i in range(len(g)):
            if g[i]:
                born, t, b, used = i, tp[i], bt[i], False
                continue
            if used or i - born > keep or not np.isfinite(t):
                continue
            # canTest = bar_index > g.index + 1、isInside = h>=bottom and l<=top
            if i > born + 1 and hh[i] >= b and ll[i] <= t:
                # 反発方向で入る(強気ギャップは買い)
                if (side > 0 and cc[i] > b) or (side < 0 and cc[i] < t):
                    out[i], used = True, True
        return out

    buy = _test(bull, top_b, bot_b, 1)
    sell = _test(bear, top_s, bot_s, -1)
    return _atr_exit(df, p, buy, sell)


templates.register("tv3_274_fvg_ualgo", defaults={
    "atr_len": 14.0, "use_atr_filter": 1.0, "max_age": 30.0, **_EXIT_DEFAULTS},
    signal_fn=_sig_fvg_ualgo)


# ==========================================================================
# 220 Trade Serene - ORB + Price Action Entry & Exit
#     指定セッションのレンジ(ORB)確定後、終値がORB高値を上抜け +
#     強い実体 + 直近構造のブレイク + VWAPの上 + 出来高確認。
#     ★原文のセッションは取引所ローカル。FXはUTC基準なので窓をパラメータに出す。
# ==========================================================================
def _sig_orb_serene(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour.to_numpy()
    day = ts.dt.floor("D").to_numpy()

    s, e = int(p["orb_start_utc"].value), int(p["orb_end_utc"].value)
    in_orb = (hour >= s) & (hour < e)
    d = pd.DataFrame({"day": day, "in": in_orb, "h": h.to_numpy(), "l": l.to_numpy()})
    g = d[d["in"]].groupby("day").agg(oh=("h", "max"), ol=("l", "min"))
    orb_h = pd.Series(day, index=df.index).map(g.oh)
    orb_l = pd.Series(day, index=df.index).map(g.ol)
    complete = pd.Series(hour >= e, index=df.index) & orb_h.notna()

    rng = (h - l).replace(0, np.nan)
    strong_up = (c > o) & (((c - o) / rng) >= float(p["min_body"].value))
    strong_dn = (c < o) & (((o - c) / rng) >= float(p["min_body"].value))
    st = max(2, int(p["struct_len"].value))
    brk_up = c > h.shift(1).rolling(st).max()
    brk_dn = c < l.shift(1).rolling(st).min()

    # VWAP(日次リセット)
    tp = (h + l + c) / 3.0
    dv = pd.DataFrame({"day": day, "pv": (tp * v).to_numpy(), "v": v.to_numpy()})
    vwap = (dv.groupby("day").pv.cumsum() / dv.groupby("day").v.cumsum()).values
    vwap = pd.Series(vwap, index=df.index)
    vol_ok = (v > sma(v, 20) * float(p["vol_mult"].value)
              if float(p["use_volume"].value) > 0.5 else pd.Series(True, index=df.index))
    vw_up = (c > vwap) if float(p["use_vwap"].value) > 0.5 else pd.Series(True, index=df.index)
    vw_dn = (c < vwap) if float(p["use_vwap"].value) > 0.5 else pd.Series(True, index=df.index)

    buy = complete & (c > orb_h) & strong_up & brk_up & vw_up & vol_ok
    sell = complete & (c < orb_l) & strong_dn & brk_dn & vw_dn & vol_ok
    # 1日1回に絞る(原文もORB確定後の初回ブレイクを狙う設計)
    buy = buy & ~buy.groupby(pd.Series(day, index=df.index)).cumsum().shift(1).fillna(0).astype(bool)
    sell = sell & ~sell.groupby(pd.Series(day, index=df.index)).cumsum().shift(1).fillna(0).astype(bool)
    return _atr_exit(df, p, buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy())


templates.register("tv3_220_orb_serene", defaults={
    "orb_start_utc": 7.0, "orb_end_utc": 8.0, "min_body": 0.5, "struct_len": 10.0,
    "use_vwap": 1.0, "use_volume": 1.0, "vol_mult": 1.2, **_EXIT_DEFAULTS},
    signal_fn=_sig_orb_serene)
