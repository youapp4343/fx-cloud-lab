"""VGRSI(Visibility Graph RSI)テンプレート(@iwachan_trader 紹介の学術指標の機械化)。

出典: arXiv "Visibility graphs can make money in financial markets"(2605.01300)。価格系列を
自然可視性グラフ(natural visibility graph)に変換し、各バーから「後方に見通せる(visible)」
過去バー群の値動きを、RSI流に0-100へ正規化した新規オシレーター。RSIが直前差分のみを見るのに
対し、可視性で選ばれた幾何学的に重要な過去バーの上げ下げを集計する点が違い。

アルゴリズム(論文の定義に忠実):
- 各バー j から後方windowだけ遡り、自然可視性で visible な過去 index i を抽出
  (i を j-1 から減らしつつ slope=(p[i]-p[j])/(i-j) が走査中の最大を更新した時のみ visible。
   これは「i,j 間の全 k が結線より下」= 標準の natural visibility と等価で O(window))。
- visible な i の価格変化 delta[i]=p[i]-p[i-1] を上げ/下げに分類し、
  S+=Σ上げ幅, S-=Σ下げ幅, N+=上げ本数, N-=下げ本数。
- r_S=S+/S-, r_N=N+/N-, r=½(r_S+r_N)  (論文の A0=平均バリアント)
- VGRSI = 100 - 100/(1+r)   (RSIと同じ0-100スケール)
売買ルール(論文): VGRSIが売られすぎ域を下抜けたら買い / 買われすぎ域を上抜けたら売り
(=平均回帰オシレーター)。論文は M1/M5/M30 の3時間足合議 + SL/TP=直近ローソク高さ中央値×Z
だが、本テンプレは単一足で os/ob クロスのみを検証(多時間足合議・構造SL/TPはStage2、と明記)。

先読み規律: VGRSI[j] はバー j までの確定 close のみで計算(後方可視性=未来参照なし)。
クロス検出の .shift(1) は自分の過去参照のみ。執行shiftはengine側、テンプレ内shiftなし。
計算量 O(n×window) のため、スイープでは H1/H4/D1 に限定(M1/M5は重い)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.strategy_model import Strategy

_EPS = 1e-9


def vgrsi(close: pd.Series, window: int = 30) -> pd.Series:
    """後方自然可視性グラフに基づく VGRSI(A0バリアント、0-100)。"""
    p = close.to_numpy(dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=close.index)
    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = p[1:] - p[:-1]
    for j in range(2, n):
        lo = max(1, j - window)  # i>=1 (delta[0]は未定義のため除外)
        running_min = np.inf
        Sp = Sm = 0.0
        Np = Nm = 0
        for i in range(j - 1, lo - 1, -1):
            s = (p[i] - p[j]) / (i - j)   # i<j → 分母<0(=jから見た後方傾き)
            # 自然可視性: i,j間の全kが結線より下 ⇔ slope(i) が走査済み(=jに近い)kの最小slopeを下回る
            if s < running_min:           # visible
                running_min = s
                d = delta[i]
                if d > 0:
                    Sp += d
                    Np += 1
                elif d < 0:
                    Sm += -d
                    Nm += 1
        if Sp == 0.0 and Sm == 0.0:
            continue  # 可視域が完全フラット→中立未定義、無シグナル
        r_S = Sp / (Sm + _EPS)
        r_N = Np / (Nm + _EPS)
        r = 0.5 * (r_S + r_N)
        out[j] = 100.0 - 100.0 / (1.0 + r)
    return pd.Series(out, index=close.index)


def _signal_vgrsi(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    window = max(5, int(p["window"].value))
    os_level = float(p["os_level"].value)   # 売られすぎ域(下抜けで買い)
    ob_level = float(p["ob_level"].value)   # 買われすぎ域(上抜けで売り)
    dir_mode = int(p["dir_mode"].value)     # 0=both,1=long,2=short

    vg = vgrsi(df["close"], window)
    prev = vg.shift(1)
    cross_dn_os = (vg < os_level) & (prev >= os_level)   # 売られすぎへ突入 → 買い(平均回帰)
    cross_up_ob = (vg > ob_level) & (prev <= ob_level)   # 買われすぎへ突入 → 売り

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[cross_dn_os.fillna(False)] = 1
    if dir_mode in (0, 2):
        signal[cross_up_ob.fillna(False)] = -1
    return signal


templates.register(
    "vgrsi",
    defaults={
        "window": 30.0, "os_level": 30.0, "ob_level": 70.0, "dir_mode": 0.0,
        "sl_pips": 30.0, "tp_pips": 45.0, "lot": 0.1,
    },
    signal_fn=_signal_vgrsi,
)
