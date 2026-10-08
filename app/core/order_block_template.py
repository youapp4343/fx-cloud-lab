"""オーダーブロック(OB)・リテスト テンプレート(SMC「BOS後のOBリテスト」の機械化)。

出典: Smart Money Concepts (SMC) / ICTの中核パターンの一つ。「構造の変化(Break of
Structure = BOS)が起きた直後、そのブレイクを作った直前の"最後の逆行足"(Order Block)
まで価格が一度戻ってから反転再開する」という経験則を、観測可能なOHLCのみから機械化する。

近似(docstring明記、必読):
- OBの本来の定義(機関投資家の指値注文が集積した価格帯)は非公開情報であり直接観測不可能。
  本テンプレートでは実務で広く使われる簡易定義「BOS直前の最後の逆色実体ローソク足」
  (ロング側=直近安値更新方向を破る前の最後の陰線、ショート側=最後の陽線)で代理する。
- スイング高安の確定はフラクタル判定(前後swing_k本を含む2*swing_k+1本ウィンドウの
  最大/最小)をcenter=Trueローリングで検出する。あるバーがスイングだったと判明するのは
  swing_k本後(=確定バーの2*swing_k+1本ウィンドウが埋まった時点)であり、先読み不可能。
- BOSは終値ベース(ヒゲではなく確定終値が直近確定スイング高安を上/下抜け)で判定する
  保守的な定義。
- BOSはエッジトリガ(その基準水準を初めて上/下抜けたバーのみ)で発火し、そのたびにOBを
  新規アーム(=再アーム)する。前のOBが未使用(リテスト・無効化未達)でも新しいBOSが
  来れば上書きされる(1方向につき常に最新1本のOBのみを保持する簡易状態機械)。
- リテストは「安値(ロング)/高値(ショート)がOBゾーンへ接触」で判定する簡易近似
  (ゾーン内でのクローズや出来高確認などの追加条件は課さない、浅いヒゲタッチも拾う)。
- 無効化条件(終値がOB_lowを割る/OB_highを超える)とリテスト条件が同一バーで両立する
  場合は無効化を優先する(ゾーンを丸ごと終値で突き抜けた足は「有効なリテスト」ではなく
  「構造破壊」とみなす近似。engine.pyの「SL/TP同時ヒット時はSL優先」と同じ保守的思想)。
- 同一バーでロング・ショート両方のリテストが同時発火する稀なケース(広いレンジ足が
  両ゾーンを同時に通過)はロング優先で採用する(シグナルは1本の値しか持てないための
  便宜的tie-break)。
- 1つのOBにつきエントリーは1回のみ(発火後は破棄。無効化された場合もその場で破棄)。

先読み規律: スイング確定はswing_k本遅延の因果パターンで、当バー時点で未来データを参照
しないことは構造的に保証される(確定インデックス i=t-swing_k のローリング窓は
[i-swing_k, i+swing_k] = [t-2*swing_k, t] であり、上限が常にt以下に収まるため)。
BOS/OB探索/リテスト/無効化はいずれも当バーの終値・高値・安値+自分の過去状態のみで
判定するステートフルループであり、未来のバーを一切参照しない。執行shiftはengine側
(raw_signal.shift(1))が行うため、テンプレート内でのシフトは一切行わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy


def _signal_order_block(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    k = max(1, int(p["swing_k"].value))                 # スイング確定までの遅延本数
    ob_lookback = max(1, int(p["ob_lookback"].value))    # OB探索の遡及本数
    dir_mode = int(p["dir_mode"].value)                  # 0=both,1=long,2=short

    n = len(df)

    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    # --- 因果的な確定スイング高安(指定パターンをそのまま使用) ---
    # pivot i=t-k は「前後k本(計2k+1本)ウィンドウの最大/最小」として定義され、
    # そのウィンドウ[i-k, i+k]の上限がi+k=tでtを超えないため、バーtの時点で
    # is_ph[i]/is_pl[i]を読んでも先読みは構造的に発生しない(確定はi+k=t本後)。
    hh = df["high"].rolling(2 * k + 1, center=True).max().to_numpy()
    ll = df["low"].rolling(2 * k + 1, center=True).min().to_numpy()
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    is_ph = high == hh
    is_pl = low == ll

    bearish = close < open_  # 陰線(実体ベース、ヒゲは無視する近似)
    bullish = close > open_  # 陽線

    allow_long = dir_mode in (0, 1)
    allow_short = dir_mode in (0, 2)

    sig = np.zeros(n, dtype=int)

    conf_sh_price: float | None = None  # 直近の確定済みスイング高値(BOS判定の基準)
    conf_sl_price: float | None = None  # 直近の確定済みスイング安値
    was_above = False   # 前バー時点でclose>conf_sh_priceだったか(BOSエッジ検知用)
    was_below = False

    long_ob: dict | None = None   # アーム中のデマンドOB {"low":.., "high":..}
    short_ob: dict | None = None  # アーム中のサプライOB

    for t in range(n):
        # --- スイング確定の反映: pivot i=t-k がこのバーで初めて確定する ---
        i = t - k
        if i >= 0:
            if is_ph[i]:
                conf_sh_price = float(high[i])
            if is_pl[i]:
                conf_sl_price = float(low[i])

        # --- アーム済みOBのリテスト/無効化判定(前バー以前にアームされたOBのみ対象、
        #     このバーで新規アームされるOBは次バー以降に判定=「Later」を素直に反映) ---
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

        # --- BOS検知(エッジトリガ)とOBの新規アーム/再アーム ---
        now_above = allow_long and conf_sh_price is not None and close[t] > conf_sh_price
        now_below = allow_short and conf_sl_price is not None and close[t] < conf_sl_price

        if now_above and not was_above:
            # ブレイク直前の最後の陰線 = デマンドOB候補(at or before t、ob_lookback本まで遡及)
            j_start = max(0, t - ob_lookback)
            ob_j = None
            for j in range(t, j_start - 1, -1):
                if bearish[j]:
                    ob_j = j
                    break
            if ob_j is not None:
                long_ob = {"low": float(low[ob_j]), "high": float(high[ob_j])}

        if now_below and not was_below:
            # ブレイク直前の最後の陽線 = サプライOB候補
            j_start = max(0, t - ob_lookback)
            ob_j = None
            for j in range(t, j_start - 1, -1):
                if bullish[j]:
                    ob_j = j
                    break
            if ob_j is not None:
                short_ob = {"low": float(low[ob_j]), "high": float(high[ob_j])}

        was_above = now_above
        was_below = now_below

    return pd.Series(sig, index=df.index, dtype=int)


templates.register(
    "order_block",
    defaults={
        "swing_k": 5.0,
        "ob_lookback": 10.0,
        "dir_mode": 0.0,
        "sl_pips": 200.0,
        "tp_pips": 300.0,
        "lot": 0.1,
    },
    signal_fn=_signal_order_block,
)
