"""フィボナッチ押し目(fib_pullback)とハーモニクス(harmonic)の戦略テンプレート。

両テンプレートとも app/core/swings.py の detect_swings が付与する confirmed_at_idx を厳格に守り、
「バーiの時点では confirmed_at_idx <= i のスイングのみ参照してよい」という先読み回避の規律を
そのまま踏襲する(swings.py自体のdocstring・app/core/engine.pyのシグナルshift(1)と同じ思想)。

誠実性についての設計方針(両テンプレート共通):
本モジュールは「フィボナッチ比率やハーモニクスXABCD比率に予測力がある」と主張するものではない。
どちらも「価格の反応しやすそうな水準」という仮説をパラメータ(fib_low/fib_high、pattern_index/tolerance等)
として明示的に区分けしただけであり、優位性の有無はテンプレート自体ではなく既存の発掘(discovery)・
グリッドサーチ・バックテスト基盤が経験的に判定する。各signal_fnのdocstringにも同旨を明記する。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy
from app.core.swings import detect_swings, fib_extension_levels, fib_retracement_levels

# ---------------------------------------------------------------------------
# fib_pullback
# ---------------------------------------------------------------------------


def _latest_confirmed_extremes(n: int, swings: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """各バーiに対し、その時点(confirmed_at_idx<=i)で参照可能な直近の確定済みスイングハイ/ローを割り当てる。

    merge_asof(direction="backward")で「bar_idx(=i)以下の最大confirmed_at_idx」を持つ行を選ぶことで、
    先読み回避の規律(confirmed_at_idx<=iのみ参照可)をバーごとのループなしにベクトル化して実現する
    (swings全体を毎バー再スキャンするより効率的)。

    戻り値: (h_bar_idx, h_price, l_bar_idx, l_price)。いずれも長さnのfloat配列で、
    そのバー時点でまだH/Lどちらかが一度も確定していない場合はNaN。
    """
    bars = pd.DataFrame({"bar_idx": np.arange(n, dtype=np.int64)})

    highs = (
        swings.loc[swings["kind"] == "H", ["confirmed_at_idx", "bar_idx", "price"]]
        .sort_values("confirmed_at_idx", kind="stable")
        .rename(columns={"bar_idx": "h_bar_idx", "price": "h_price"})
    )
    lows = (
        swings.loc[swings["kind"] == "L", ["confirmed_at_idx", "bar_idx", "price"]]
        .sort_values("confirmed_at_idx", kind="stable")
        .rename(columns={"bar_idx": "l_bar_idx", "price": "l_price"})
    )

    h_merged = pd.merge_asof(bars, highs, left_on="bar_idx", right_on="confirmed_at_idx", direction="backward")
    l_merged = pd.merge_asof(bars, lows, left_on="bar_idx", right_on="confirmed_at_idx", direction="backward")

    return (
        h_merged["h_bar_idx"].to_numpy(dtype=float),
        h_merged["h_price"].to_numpy(dtype=float),
        l_merged["l_bar_idx"].to_numpy(dtype=float),
        l_merged["l_price"].to_numpy(dtype=float),
    )


def _retracement_price_at_ratio(swing_start_price: float, swing_end_price: float, ratio: float) -> float:
    """fib_retracement_levelsと同一の式(level = end - (end-start)*ratio)をratioについて直接評価する。

    fib_low/fib_highはグリッドサーチで既定の5水準("0.236"等)以外の任意の値になり得るため、
    まずlevels辞書を文字列キーで引き(既定値0.382/0.618は辞書キーと完全一致する)、
    一致しない場合だけこの式にフォールバックする。辞書アクセスのみだとKeyErrorで
    最適化パイプライン全体を止めかねないための安全策。
    """
    return swing_end_price - (swing_end_price - swing_start_price) * ratio


def _signal_fib_pullback(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    """フィボナッチ押し目(リトレースメント)テンプレート。

    誠実性の注記: これは「フィボナッチ比率に予測上の魔力がある」という主張ではない。
    「押し目の深さ」を fib_low〜fib_high という2つのパラメータで区間化して表現しただけであり、
    swing_order/fib_low/fib_high は sl_pips 等と全く同じ通常の StrategyParam なので、
    既存のグリッドサーチ/ランダムサーチにそのままかけて優位性の有無を検証できる設計にしてある
    (優位性はこの関数ではなくバックテスト結果が判断する)。

    ロジック(バーiの終値時点で確定した情報のみ使用):
    1. 直近の確定済みスイングハイ/ロー(confirmed_at_idx<=i)を求める。
    2. 直近の確定極値がH(h_bar_idx>l_bar_idx、つまり直近レグがL->H上昇)なら押し目買い候補。
       安値がフィボゾーンまで到達し、終値がゾーンを割らずに陽線で引ければエントリー。
    3. 直近の確定極値がL(l_bar_idx>h_bar_idx、直近レグがH->L下降)なら戻り売り候補(対称ロジック)。
    """
    swing_order = max(1, int(strategy.params["swing_order"].value))  # argrelextremaはorder>=1必須
    fib_low = float(strategy.params["fib_low"].value)
    fib_high = float(strategy.params["fib_high"].value)

    n = len(df)
    result_index = df.index
    if n == 0:
        empty = pd.Series(dtype=float)
        return pd.DataFrame({"signal": empty.astype(int), "sl_price": empty, "tp_price": empty})

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    swings = detect_swings(df, order=swing_order)
    h_bar_idx, h_price, l_bar_idx, l_price = _latest_confirmed_extremes(n, swings)

    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan, dtype=float)
    tp_price = np.full(n, np.nan, dtype=float)

    has_both = ~np.isnan(h_bar_idx) & ~np.isnan(l_bar_idx)
    up_candidates = np.nonzero(has_both & (h_bar_idx > l_bar_idx))[0]  # 直近レグ L->H(上昇) -> 押し目買い候補
    down_candidates = np.nonzero(has_both & (l_bar_idx > h_bar_idx))[0]  # 直近レグ H->L(下降) -> 戻り売り候補

    for i in up_candidates:
        l_p, h_p = float(l_price[i]), float(h_price[i])
        levels = fib_retracement_levels(l_p, h_p)
        zone_upper = levels.get(str(fib_low), _retracement_price_at_ratio(l_p, h_p, fib_low))
        zone_lower = levels.get(str(fib_high), _retracement_price_at_ratio(l_p, h_p, fib_high))
        touched_zone = low[i] <= zone_upper
        held_zone = close[i] > zone_lower
        bullish = close[i] > open_[i]
        if touched_zone and held_zone and bullish:
            signal[i] = 1
            sl_price[i] = l_p  # スイング安値を下抜けたらセットアップ無効、という保守的なSL
            tp_price[i] = fib_extension_levels(l_p, h_p, h_p)["1.272"]

    for i in down_candidates:
        h_p, l_p = float(h_price[i]), float(l_price[i])
        levels = fib_retracement_levels(h_p, l_p)
        zone_upper = levels.get(str(fib_high), _retracement_price_at_ratio(h_p, l_p, fib_high))
        zone_lower = levels.get(str(fib_low), _retracement_price_at_ratio(h_p, l_p, fib_low))
        touched_zone = high[i] >= zone_lower
        held_zone = close[i] < zone_upper
        bearish = close[i] < open_[i]
        if touched_zone and held_zone and bearish:
            signal[i] = -1
            sl_price[i] = h_p  # スイング高値を上抜けたらセットアップ無効
            tp_price[i] = fib_extension_levels(h_p, l_p, l_p)["1.272"]

    return pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price}, index=result_index)


templates.register(
    "fib_pullback",
    defaults={
        "swing_order": 5.0,
        "fib_low": 0.382,
        "fib_high": 0.618,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_fib_pullback,
)


# ---------------------------------------------------------------------------
# harmonic
# ---------------------------------------------------------------------------

# b_min/b_max: B点のXA比率レンジ。d_ratio: D点のXA比率。d_is_extensionは参考情報として保持
# (d_price = A - xa*d_ratio という1本の式がd_ratio<1でretracement/>1でextensionを自動的に表現するため
# 実際の分岐には使わないが、パターンの意味を読み手が確認できるようテーブルに残す)。
_HARMONIC_RATIOS: dict[str, dict[str, float | bool]] = {
    "bat": {"b_min": 0.382, "b_max": 0.50, "d_ratio": 0.886, "d_is_extension": False},
    "butterfly": {"b_min": 0.786, "b_max": 0.786, "d_ratio": 1.27, "d_is_extension": True},
    "crab": {"b_min": 0.382, "b_max": 0.618, "d_ratio": 1.618, "d_is_extension": True},
    "gartley": {"b_min": 0.618, "b_max": 0.618, "d_ratio": 0.786, "d_is_extension": False},
}
_HARMONIC_PATTERN_NAMES: list[str] = sorted(_HARMONIC_RATIOS.keys())  # ["bat","butterfly","crab","gartley"]

_SwingPoint = Tuple[str, float, int]  # (kind, price, bar_idx)


def _signal_harmonic(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    """ハーモニクスパターン(XABCD比率)テンプレート。

    誠実性の注記: ハーモニクスパターン(Gartley/Bat/Butterfly/Crab等)の優位性を裏付ける
    頑健な学術的エビデンスは乏しい。本テンプレートは「これらの比率に優位性がある」と
    主張するものではなく、XABCD比率という仮説を他のテンプレートと全く同じ土俵
    (発掘・グリッドサーチ・バックテスト)に乗せて公平に検証できるようにすることが目的。

    アルゴリズム:
    1. 確定済みスイング(confirmed_at_idx<=i)をbar_idx昇順(=confirmed_at_idx昇順と同義。
       orderは呼び出し内で一定なのでbar_idx+orderは単調変換であり順序は保たれる)に読み、
       「厳密にH/Lが交互になる直近4点」をdeque(maxlen=4)で維持する。
       次に確定したスイングのkindが直前に追加したものと異なれば追加(4点を超えたら最古を自動排出)、
       同じなら直前を最新のもので置き換える(規約: 同種の極値が連続確定した場合、
       押し目/戻りの完了を待たずに古い極値をXABCDの頂点として使い続けるのは実勢に合わないため、
       常に最新の同種極値で上書きする)。
    2. バーiの時点でX,A,B,C(古い順)が揃っていれば、選択パターンのb_min~b_max(±tolerance)に
       ab_ratio=|B-A|/|A-X|が収まるか判定し、収まればD点の理論価格を1本の式で算出する
       (d_ratio<1ならXA区間内へのretracement、>1ならAを超えるextensionを自動的に表現する)。
    3. バーiの価格がDのPRZ(Potential Reversal Zone)に到達し、かつ反発方向に引ければシグナル。
    """
    pattern_idx = int(strategy.params["pattern_index"].value)
    pattern_idx = max(0, min(pattern_idx, len(_HARMONIC_PATTERN_NAMES) - 1))  # range探索でのout-of-rangeを端にクリップ
    pattern_name = _HARMONIC_PATTERN_NAMES[pattern_idx]
    ratios = _HARMONIC_RATIOS[pattern_name]

    tolerance = float(strategy.params["tolerance"].value)
    swing_order = max(1, int(strategy.params["swing_order"].value))  # argrelextremaはorder>=1必須

    b_min = float(ratios["b_min"])
    b_max = float(ratios["b_max"])
    ab_lower = b_min - tolerance * b_min
    ab_upper = b_max + tolerance * b_max
    d_ratio = float(ratios["d_ratio"])

    n = len(df)
    result_index = df.index
    if n == 0:
        empty = pd.Series(dtype=float)
        return pd.DataFrame({"signal": empty.astype(int), "sl_price": empty, "tp_price": empty})

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    swings = detect_swings(df, order=swing_order)
    swing_bar_idx = swings["bar_idx"].to_numpy()
    swing_price = swings["price"].to_numpy(dtype=float)
    swing_kind = swings["kind"].to_numpy()
    swing_confirmed = swings["confirmed_at_idx"].to_numpy()
    num_swings = len(swings)

    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan, dtype=float)
    tp_price = np.full(n, np.nan, dtype=float)

    points: Deque[_SwingPoint] = deque(maxlen=4)
    pointer = 0  # confirmed_at_idx昇順ソート済みswingsに対して1方向にのみ進むポインタ(再スキャンしない)

    for i in range(n):
        while pointer < num_swings and swing_confirmed[pointer] <= i:
            kind = str(swing_kind[pointer])
            point: _SwingPoint = (kind, float(swing_price[pointer]), int(swing_bar_idx[pointer]))
            if not points or points[-1][0] != kind:
                points.append(point)
            else:
                points[-1] = point  # 同kind連続時は直前を最新の同種極値で上書き(上記docstring参照)
            pointer += 1

        if len(points) < 4:
            continue

        x_price = points[0][1]
        a_price = points[1][1]
        b_price = points[2][1]
        c_price = points[3][1]

        xa = a_price - x_price
        if xa == 0:
            continue
        ab_ratio = abs(b_price - a_price) / abs(xa)
        if not (ab_lower <= ab_ratio <= ab_upper):
            continue

        d_price = a_price - xa * d_ratio

        if xa > 0:  # X->A上昇 -> Dは下方のPRZ -> 買い候補
            if low[i] <= d_price and close[i] > d_price:
                signal[i] = 1
                sl_price[i] = d_price - abs(xa) * 0.1
                tp_price[i] = c_price
        else:  # X->A下降 -> Dは上方のPRZ -> 売り候補
            if high[i] >= d_price and close[i] < d_price:
                signal[i] = -1
                sl_price[i] = d_price + abs(xa) * 0.1
                tp_price[i] = c_price

    return pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price}, index=result_index)


templates.register(
    "harmonic",
    defaults={
        "pattern_index": 0.0,
        "tolerance": 0.08,
        "swing_order": 5.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_harmonic,
)
