"""WEMOF風 BB3σ純度逆張り(wemof)テンプレート(2026-07-17 事前登録)。

わっきゃい氏WEMOF(非公開インジ)の公開情報からの機械化代理:
- USDJPY M5、BB(bb_period=20)の±bb_dev(3)σを終値が超えたら平均回帰方向へ逆張り。
- 純度フィルタ(purity_min>0で有効): 直近 purity_bars(6)本の
  方向支配率 = |Σ(close-open)| / Σ|close-open| が purity_min 以上、かつ支配方向が
  スパイク方向と一致(=「陽線連続・押し目なし・重なり少」の代理)。
- 出口: engine の tp_pips/sl_pips。exit_mid=1 なら TPをBB中央線(動的tp_price)に。
注: 原版の裁量部分(ニュース/指標/ライン/節目の見送り)は再現不能。純度式・BB期間も
非公開のため本検証は「公開骨格の代理モデル」であり原版の有効性を確定しない。
先読み回避: BB・純度は全てバーi確定情報、engineのshift(1)が次バー執行。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_wemof(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    bb_p = max(2, int(strategy.params["bb_period"].value))
    bb_dev = float(strategy.params["bb_dev"].value)
    p_bars = max(2, int(strategy.params["purity_bars"].value))
    p_min = float(strategy.params["purity_min"].value)
    exit_mid = int(strategy.params["exit_mid"].value) if "exit_mid" in strategy.params else 0

    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    cs = df["close"]
    mid = cs.rolling(bb_p).mean()
    sd = cs.rolling(bb_p).std(ddof=0)
    upper = (mid + bb_dev * sd).to_numpy()
    lower = (mid - bb_dev * sd).to_numpy()
    mid_a = mid.to_numpy()

    body = c - o
    s_body = pd.Series(body)
    net = s_body.rolling(p_bars).sum().to_numpy()
    gross = pd.Series(np.abs(body)).rolling(p_bars).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        dominance = np.abs(net) / np.where(gross > 0, gross, np.nan)

    short_sig = c > upper   # 上方スパイク → 売り
    long_sig = c < lower    # 下方スパイク → 買い
    if p_min > 0:
        short_sig &= (dominance >= p_min) & (net > 0)  # 純度高い上昇の末の+3σ
        long_sig &= (dominance >= p_min) & (net < 0)
    valid = np.isfinite(upper) & np.isfinite(dominance if p_min > 0 else upper)
    short_sig &= valid
    long_sig &= valid

    n = len(df)
    signal = np.zeros(n, dtype=int)
    signal[long_sig] = 1
    signal[short_sig] = -1
    out = pd.DataFrame({"signal": signal}, index=df.index)
    if exit_mid:
        tp_price = np.full(n, np.nan)
        tp_price[long_sig] = mid_a[long_sig]
        tp_price[short_sig] = mid_a[short_sig]
        out["tp_price"] = tp_price
    return out


templates.register(
    "wemof",
    defaults={
        "bb_period": 20.0,
        "bb_dev": 3.0,
        "purity_bars": 6.0,
        "purity_min": 0.7,   # 0=フィルタ無効
        "exit_mid": 0.0,     # 1=BB中央線をTPに(動的)
        "tp_pips": 3.0,
        "sl_pips": 10.0,
        "max_hold_bars": 288.0,  # M5×288=24h
        "lot": 0.1,
    },
    signal_fn=_signal_wemof,
)
