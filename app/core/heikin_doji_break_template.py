"""heikin_doji_break(平均足インパルス→ダギ足コンソリ→実勢closeブレイク)テンプレート
(YouTube解析レポート set2 #8由来、初の平均足(Heikin-Ashi)ベーステンプレ)。

出典: scripts/scalp_set2_prereg.md T4。M5+M15(2TF、6ペア×2TF=12検定。平均足の
実体はノイズが乗りやすいM1は対象外)。

**重要な区別(シグナル判定=平均足 / 約定=実勢OHLC)**: 平均足はあくまでシグナル
判定用の変換値であり、実際の価格そのものではない。本テンプレでは
  - インパルス/ダギ足の判定・SL/TPの価格基準  → 平均足(HA_open/close/high/low)
  - ブレイク確定(ダギ足の高安値を上下に抜けたか) → 実勢close(df["close"]そのもの)
の2系統を明確に分けてコードコメントで都度明記する(動画は未言及だが実務上必須)。

平均足変換(事前登録の定義どおり):
  HA_close = (O+H+L+C)/4
  HA_open  = (前HA_open+前HA_close)/2 (初回バーのみ実OHLCで初期化: (O[0]+C[0])/2)
  HA_high  = max(H, HA_open, HA_close)
  HA_low   = min(L, HA_open, HA_close)
HA_openは1階の線形再帰(alpha=0.5の指数平滑と数学的に同一)のため、
「seed[0]=(O[0]+C[0])/2、seed[t]=HA_close[t-1](t>=1)」という系列に対する
ewm(alpha=0.5, adjust=False)で厳密に再現できる(ループ不要、docstring内で検算済み:
y[0]=seed[0]=HA_open[0]、y[t]=0.5*seed[t]+0.5*y[t-1]=0.5*HA_close[t-1]+0.5*HA_open[t-1]
=HA_open[t]の定義式そのもの)。

パターン定義:
- 「綺麗な」平均足: 陽線なら反対側(下)髭=HA_open-HA_low <= 実体の15%、
  陰線なら反対側(上)髭=HA_high-HA_open <= 実体の15%。
- インパルス: 方向一致の「綺麗な」平均足が2本以上連続(直近2本が同方向・綺麗であれば
  それ以前から続いていても自然に満たす)。
- ダギ足: 実体<=レンジの20%、かつ レンジ >= インパルス直近2本のいずれかの実体以上
  (= レンジ >= 両者の実体の小さい方)。
- エントリー: ダギ足確定後、以降のバーで**実勢close**がダギ足のHA高値を上抜け→ロング、
  HA安値を下抜け→ショート(方向はブレイク方向で決まり、元インパルスの向きでは
  ゲートしない、事前登録の記述どおり)。ブレイク確定バー→engineのshift(1)で
  自動的に次バー始値エントリー。
- 1つのダギ足水準につき1エントリー(ブレイクで消費)。ブレイク前に新しいダギ足が
  成立した場合は最新の水準に更新する(break_retest_templateの複数水準管理とは異なり、
  本テンプレは直近1水準のみを追跡する単純化、事前登録に順序の明記なし)。
- SL=ダギ足の反対側極値(ロング=HA安値、ショート=HA高値)、TP=SL幅×1(RR1:1)。
  共にsl_price/tp_price列(動的)で返す。

先読み規律: HA変換はバーi自身とその過去のみで確定(HA_openの再帰も同様)。
ダギ足の水準登録・ブレイク判定は逐次ループでバーi確定時点までの情報のみ使用。
signal列はengine側でshift(1)されるため、実際の使用は信号(ブレイク確定)バーの
次バー(先読みなし)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_heikin_doji_break(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    wick_max_frac = float(p["wick_max_frac"].value)          # 0.15(反対側髭/実体)
    doji_body_max_frac = float(p["doji_body_max_frac"].value)  # 0.20(ダギ実体/レンジ)
    rr = float(p["rr"].value)                                  # 1.0(RR1:1)
    side = int(p["side"].value)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)   # 実勢close(ブレイク判定に使う。HA closeとは別物)
    n = len(df)

    # --- 平均足変換(docstring参照、ewm(alpha=0.5)で再帰を厳密再現) ---
    ha_close = (o + h + l + c) / 4.0
    ha_close_s = pd.Series(ha_close, index=df.index)
    seed = ha_close_s.shift(1)
    if n > 0:
        seed.iloc[0] = (o[0] + c[0]) / 2.0  # 初回バーのみ実OHLCで初期化
    ha_open = seed.ewm(alpha=0.5, adjust=False).mean().to_numpy()
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    body = np.abs(ha_close - ha_open)
    rng = ha_high - ha_low
    bull = ha_close > ha_open
    bear = ha_close < ha_open
    lower_wick = ha_open - ha_low   # 陽線の反対側(下)髭
    upper_wick = ha_high - ha_open  # 陰線の反対側(上)髭

    with np.errstate(invalid="ignore"):
        clean_bull = bull & (body > 0) & (lower_wick <= wick_max_frac * body)
        clean_bear = bear & (body > 0) & (upper_wick <= wick_max_frac * body)

    # インパルス=直近2本が同方向の「綺麗な」平均足(2本以上連続していれば自然に満たす)
    impulse_up_end = clean_bull & np.concatenate(([False], clean_bull[:-1]))
    impulse_dn_end = clean_bear & np.concatenate(([False], clean_bear[:-1]))
    impulse_end = impulse_up_end | impulse_dn_end  # bar i-1 がインパルス最終バー

    doji_ok = (rng > 0) & (body <= doji_body_max_frac * rng)

    n_arr = n
    sig = np.zeros(n_arr, dtype=int)
    slp = np.full(n_arr, np.nan)
    tpp = np.full(n_arr, np.nan)

    pending_high = None  # ダギ足のHA高値(未ブレイク水準)
    pending_low = None   # ダギ足のHA安値

    for i in range(n_arr):
        # --- ブレイク確定判定(実勢closeのみ使用、HA closeは絶対に使わない) ---
        if pending_high is not None:
            if c[i] > pending_high:
                risk = c[i] - pending_low
                sig[i] = 1
                slp[i] = pending_low
                tpp[i] = c[i] + rr * risk
                pending_high = pending_low = None
            elif c[i] < pending_low:
                risk = pending_high - c[i]
                sig[i] = -1
                slp[i] = pending_high
                tpp[i] = c[i] - rr * risk
                pending_high = pending_low = None

        # --- 新規ダギ足登録判定(平均足のみ使用) ---
        if i >= 2 and impulse_end[i - 1] and doji_ok[i]:
            min_impulse_body = min(body[i - 1], body[i - 2])
            if rng[i] >= min_impulse_body:
                pending_high = float(ha_high[i])
                pending_low = float(ha_low[i])

    if side > 0:
        mask = sig < 0
        sig[mask] = 0
        slp[mask] = np.nan
        tpp[mask] = np.nan
    elif side < 0:
        mask = sig > 0
        sig[mask] = 0
        slp[mask] = np.nan
        tpp[mask] = np.nan

    return pd.DataFrame({"signal": sig, "sl_price": slp, "tp_price": tpp}, index=df.index)


templates.register(
    "heikin_doji_break",
    defaults={
        "wick_max_frac": 0.15,
        "doji_body_max_frac": 0.20,
        "rr": 1.0,
        "side": 0.0,
        "sl_pips": 20.0,   # フォールバック(sl_price無効時のみ使用)
        "tp_pips": 20.0,   # フォールバック(tp_price無効時のみ使用)
        "max_hold_bars": 200.0,   # 安全弁(事前登録に明記なし、RR1:1の動的決済が主)
        "lot": 0.1,
    },
    signal_fn=_signal_heikin_doji_break,
)
