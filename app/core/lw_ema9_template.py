"""ラリー・ウィリアムズ 9日EMA戦略 テンプレート(X @mzgaangy 動画の機械化)。

出典: ラリー・ウィリアムズがロビンスカップ(1987, $1万→$110万超)で用いたとされる9EMA日足手法。
インジは9EMAのみ、「9EMAの傾きが下向き(赤)→上向き(緑)に反転したローソク足の高値を、次以降の
足が上抜けたら買い、損切りは反転足の安値の下」。売りは鏡。パッと見でエントリーが分かる簡易手法。

機械化: 9EMAの傾き符号反転(赤→緑 / 緑→赤)を検出し、反転足の高値/安値を「保留ブレイク水準」
として保持。以降の足がその水準を抜けた最初の足でシグナル(engineのshift(1)で翌足始値執行)。
反転足自身では発火しない(トリガ判定を先、保留更新を後にする)。反対方向の反転が来たら保留は失効。

近似(docstringに明記): 原法は反転足高値への逆指値でイントラバー約定するが、engineは翌足始値
成行のため約定は1バー遅く保守側。SL=反転足安値/TP=直近高値の構造的決済はStage1では固定
sl_pips/tp_pips で近似(エントリー優位の判定が目的)。構造的SL/TP(sl_price列)はStage2の忠実版。

先読み規律: ema9[i]・high/low[i] は確定バーまでの情報のみ。トリガも保留水準も過去の反転足に
基づき、未来足を参照しない(1パス逐次走査で自明)。執行shiftはengine側、テンプレ内shiftなし。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import ema
from app.core.strategy_model import Strategy


def _signal_lw_ema9(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ema_period = max(2, int(p["ema_period"].value))
    dir_mode = int(p["dir_mode"].value)  # 0=both,1=long,2=short

    ema9 = ema(df["close"], ema_period).to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    sig = np.zeros(n, dtype=int)
    slope = np.empty(n)
    slope[0] = np.nan
    slope[1:] = ema9[1:] - ema9[:-1]

    pend_long = None   # 直近の「赤→緑」反転足の高値(上抜けで買い)
    pend_short = None  # 直近の「緑→赤」反転足の安値(下抜けで売り)
    for i in range(1, n):
        if np.isnan(slope[i]) or np.isnan(slope[i - 1]):
            continue
        # 1) トリガ判定(過去の反転足に基づく保留水準を、当バーが抜けたか)
        if pend_long is not None and high[i] > pend_long:
            sig[i] = 1
            pend_long = None
        elif pend_short is not None and low[i] < pend_short:
            sig[i] = -1
            pend_short = None
        # 2) 保留更新(反転足自身は上でトリガ済みなので自己発火しない)
        if slope[i] > 0 and slope[i - 1] <= 0:      # 赤→緑(上向き反転)
            pend_long = high[i]
            pend_short = None
        elif slope[i] < 0 and slope[i - 1] >= 0:    # 緑→赤(下向き反転)
            pend_short = low[i]
            pend_long = None

    if dir_mode == 1:
        sig[sig < 0] = 0
    elif dir_mode == 2:
        sig[sig > 0] = 0
    return pd.Series(sig, index=df.index, dtype=int)


templates.register(
    "lw_ema9",
    defaults={
        "ema_period": 9.0, "dir_mode": 0.0,
        "sl_pips": 50.0, "tp_pips": 100.0, "lot": 0.1,
    },
    signal_fn=_signal_lw_ema9,
)
