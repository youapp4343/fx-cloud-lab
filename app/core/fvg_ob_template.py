"""日足Bullish/Bearish FVG + Order Block リテスト テンプレート(手法#12)。

出典: ICT(Inner Circle Trader)/SMCの「Fair Value Gap(FVG、公正価値の欠落=価格の
インバランス)」+「そのFVGを作った起点側の最後の逆色実体ローソク足(Order Block)への
リテストで反転再開する」という組み合わせパターンを機械化する。GBPJPY日足/H4を想定。

定義(本テンプレートでの機械化):
- Bullish FVG: 3本一組の窓 [i-2, i-1, i] で bar[i-2].high と bar[i].low の間に
  ギャップが空く(bar[i].low > bar[i-2].high)。これが「完成」するのはbar iの終値確定時点。
  デマンドOB = bar (i-2) 以前・ob_lookback本以内で見つかる直近の陰線(close<open)実体。
  OBゾーン=[その足のlow, その足のhigh]。ロングをアーム。
- Bearish FVG: bar[i].high < bar[i-2].low のミラー。サプライOB=直近の陽線(close>open)。
  ショートをアーム。
- リテスト: アーム後の「later bar」(次バー以降。当バーでは判定しない)でlow<=OB_high
  (ロング)/high>=OB_low(ショート)に触れたら発火。無効化: close<OB_low(ロング)/
  close>OB_high(ショート)で破棄。同一バーで無効化とリテストが両立する場合は無効化優先
  (ゾーンを終値で丸ごと突き抜けた足は「有効なリテスト」ではなく構造破壊とみなす、
  order_block_template.pyと同じ保守的近似)。

近似(docstring明記、必読):
- OBは「最後の逆色実体ローソク足1本」のみ(複数本にまたがる複合OBは非対応)。
- FVGの有効期限(N本以内に未充填なら失効)・ATRベースの最小ギャップ幅フィルタは未実装
  (どのFVGも失効せず、常に「まだ生きている」候補として扱う簡易版)。
- 1方向につき常に最新1本のOBのみ保持する状態機械。新しいFVGが検出されると、前のOBが
  未使用(リテスト・無効化未達)でも上書きされる(order_block_template.pyと同じ設計)。
- ob_lookback以内に逆色実体が見つからない場合はそのFVGを無視する(OBが定義できないため)。
- 同一バーでロング・ショート両方のリテストが同時発火する稀なケースはロング優先
  (シグナルは1本の値しか持てないための便宜的tie-break、order_block_template.pyと同じ)。
- 1つのOBにつきエントリーは1回のみ(発火後は破棄。無効化された場合もその場で破棄)。

先読み規律: バーtで参照するのは bar[t] と bar[t-2](FVG判定)、および bar[t-2]以前へ
遡及するOB探索と自分の過去状態のみ。リテスト/無効化判定は「前バー以前にアームされた
OBのみ」を対象にしてから当バーの新規FVGを検出する順序のループにしており
(order_block_template.pyと同型)、当バーで新規アームされたOBは次バー以降にしか
判定されない。執行shiftはengine側(raw_signal.shift(1))が行うため、テンプレート内での
シフトは一切行わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_fvg_ob(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ob_lookback = max(1, int(p["ob_lookback"].value))  # OB探索の遡及本数
    dir_mode = int(p["dir_mode"].value)                 # 0=both,1=long,2=short

    n = len(df)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    bearish = close < open_  # 陰線(実体ベース、ヒゲは無視する近似)
    bullish = close > open_  # 陽線

    allow_long = dir_mode in (0, 1)
    allow_short = dir_mode in (0, 2)

    sig = np.zeros(n, dtype=int)

    long_ob: dict | None = None   # アーム中のデマンドOB {"low":.., "high":..}
    short_ob: dict | None = None  # アーム中のサプライOB

    for t in range(n):
        # --- アーム済みOBのリテスト/無効化判定(このバーで新規アームされるOBは対象外、
        #     次バー以降に判定=「later bar」を素直に反映) ---
        if long_ob is not None:
            if close[t] < long_ob["low"]:
                long_ob = None  # リテスト前に終値でOB_low割れ -> 無効化(同時ヒット時優先)
            elif low[t] <= long_ob["high"]:
                sig[t] = 1
                long_ob = None  # 1OBにつき1エントリーのみ

        if short_ob is not None:
            if close[t] > short_ob["high"]:
                short_ob = None
            elif high[t] >= short_ob["low"]:
                if sig[t] == 0:  # ロング・ショート同時発火時はロング優先(稀なtie-break)
                    sig[t] = -1
                short_ob = None

        # --- FVG検出(3本窓 [t-2, t-1, t] が当バーtで完成)とOBの新規アーム/再アーム ---
        if t >= 2:
            left = t - 2  # ギャップ起点側の足(OB探索の基準はここ以前)
            j_start = max(0, left - ob_lookback)

            if allow_long and low[t] > high[left]:  # Bullish FVG完成
                ob_j = None
                for j in range(left, j_start - 1, -1):
                    if bearish[j]:
                        ob_j = j
                        break
                if ob_j is not None:
                    long_ob = {"low": float(low[ob_j]), "high": float(high[ob_j])}

            if allow_short and high[t] < low[left]:  # Bearish FVG完成
                ob_j = None
                for j in range(left, j_start - 1, -1):
                    if bullish[j]:
                        ob_j = j
                        break
                if ob_j is not None:
                    short_ob = {"low": float(low[ob_j]), "high": float(high[ob_j])}

    return pd.Series(sig, index=df.index, dtype=int)


templates.register(
    "fvg_ob",
    defaults={
        "ob_lookback": 5.0,
        "dir_mode": 0.0,
        "sl_pips": 50.0,
        "tp_pips": 100.0,
        "lot": 0.1,
    },
    signal_fn=_signal_fvg_ob,
)
