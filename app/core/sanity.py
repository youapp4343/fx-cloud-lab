"""検査ライブラリ — バックテストの「沈黙するもっともらしい誤り」を機械的に防ぐ。

設計: docs/plan_sanity_safeguards.md。本セッションで実際に起きたバグ(偽0相関・
金コスト10倍・楽観コスト)を二度と埋もれさせないための再利用関数群。
思想は「良すぎる/綺麗すぎる結果を自動で疑う」。tests/test_sanity.py が実バグを回帰固定する。

主要関数:
- check_spread(symbol, pips): 銘柄別妥当レンジ外を例外に(単位取り違え検出)。
- safe_corr(series_dict, min_overlap): 共通観測が少なすぎる相関を例外に(偽の独立=決済時刻index+fillna0を殺す)。
- combine_returns(series_dict, weights): ポートフォリオ合成の唯一の正しい入口(トレード日整列)。
- sanity_flags(returns, ...): Sharpe/PF/DD/年集中/IS-OOS符号反転などの赤旗リスト。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 片道スプレッドの妥当レンジ(pips, リテール実勢の常識範囲)。範囲外は単位取り違えを疑う。
# JPYクロスはpip=0.01, XAUUSDはpip=0.1(=$0.1)基準の"pips"。
SPREAD_RANGES: Dict[str, tuple] = {
    "EURUSD": (0.1, 3.0), "GBPUSD": (0.2, 4.0), "AUDUSD": (0.2, 4.0),
    "USDJPY": (0.2, 3.0), "USDCHF": (0.2, 4.0), "USDCAD": (0.3, 4.0),
    "NZDUSD": (0.3, 5.0), "EURJPY": (0.3, 5.0), "GBPJPY": (0.4, 6.0),
    "AUDJPY": (0.3, 5.0), "EURGBP": (0.3, 5.0), "GBPAUD": (0.5, 8.0),
    "XAUUSD": (1.0, 20.0), "XAGUSD": (1.0, 30.0),  # 金通常2-5pips($0.2-0.5), 35pips($3.5)=10倍ミスは弾く
}


def check_spread(symbol: str, spread_pips: float) -> float:
    """スプレッド(pips)が銘柄の妥当レンジ内か検査。範囲外はValueError(金35 vs 3.5型の単位ミス検出)。"""
    lo, hi = SPREAD_RANGES.get(symbol, (0.0, 100.0))
    if not (lo <= spread_pips <= hi):
        raise ValueError(
            f"{symbol} のスプレッド {spread_pips}pips は妥当レンジ[{lo},{hi}]外。"
            f"単位取り違え(pips vs 価格, 10倍誤り)を疑ってください。"
        )
    return spread_pips


def _to_date_index(s: pd.Series) -> pd.Series:
    """indexをトレード日(normalize)に落とす。同日複数は合算。決済時刻index起因の偽独立を潰す核。"""
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series indexはDatetimeIndexである必要があります(トレード日整列のため)")
    g = s.groupby(s.index.normalize()).sum()
    return g.sort_index()


def safe_corr(series_dict: Dict[str, pd.Series], min_overlap: int = 30) -> pd.DataFrame:
    """トレード日で整列した相関行列。共通観測が min_overlap 未満のペアは例外。

    決済時刻でindexしてfillna(0)→共通行ゼロ→偽の相関0、を構造的に防ぐ。
    相関計算では絶対に fillna(0) しない(欠測日を0とみなすと相関が希釈され偽独立になる)。
    """
    dated = {k: _to_date_index(v) for k, v in series_dict.items()}
    keys = list(dated)
    corr = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = dated[keys[i]], dated[keys[j]]
            common = a.index.intersection(b.index)
            if len(common) < min_overlap:
                raise ValueError(
                    f"相関計算不可: {keys[i]} と {keys[j]} の共通取引日が {len(common)}(< {min_overlap})。"
                    f"決済時刻index等で整列がずれている可能性(偽の独立)。異なる曜日戦略なら共通日ゼロが正常"
                    f"=その2つは相関未定義(独立扱い可)だが、fillna(0)で0相関と誤認しないこと。"
                )
            c = np.corrcoef(a.loc[common].to_numpy(), b.loc[common].to_numpy())[0, 1]
            corr.iloc[i, j] = corr.iloc[j, i] = c
    return corr


def combine_returns(
    series_dict: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """ポートフォリオ合成の唯一の正しい入口。トレード日で整列し重み付き合算。

    各戦略をトレード日(normalize)に落として全取引日のunionで整列、欠測日は0(その日その戦略は
    無取引=寄与0)。**合成P&Lのfillna(0)はここだけ許可**(相関計算では禁止=safe_corr)。
    weights未指定は逆ボラ等リスク。
    """
    dated = {k: _to_date_index(v) for k, v in series_dict.items()}
    df = pd.DataFrame(dated)
    if weights is None:
        vol = df.std().replace(0, np.nan)
        w = (1.0 / vol)
        w = (w / w.sum()).to_dict()
    else:
        w = weights
    port = df.fillna(0.0).mul(pd.Series(w)).sum(axis=1)
    return port.sort_index()


def _pf(r: pd.Series) -> float:
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float("inf")


def _sharpe(r: pd.Series, ann: int = 252) -> float:
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0


def _max_dd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def sanity_flags(
    returns: pd.Series,
    *,
    name: str = "",
    is_returns: Optional[pd.Series] = None,
    oos_returns: Optional[pd.Series] = None,
) -> List[str]:
    """「良すぎる/怪しい」結果に赤旗を立てる。赤旗=即不採用でなく精査トリガ。

    トリガ: Sharpe>2.5, PF>3.0, |maxDD|<1%, n<100, 単一年に利益>60%集中, IS/OOS符号反転。
    報告前にこれを通し、赤旗があれば本文に明記して精査すること(報告前ゲート)。
    """
    flags: List[str] = []
    r = returns.dropna()
    if len(r) < 100:
        flags.append(f"標本小 n={len(r)}(<100): 統計的主張を過大にしない")
    sh = _sharpe(r)
    if sh > 2.5:
        flags.append(f"Sharpe {sh:.2f} が高すぎ(>2.5): バグ/先読み/コスト過小を疑え")
    pf = _pf(r)
    if pf > 3.0:
        flags.append(f"PF {pf:.2f} が高すぎ(>3.0): 同上")
    dd = _max_dd(r)
    if dd > -0.01:
        flags.append(f"最大DD {dd*100:.2f}% が小さすぎ(>-1%): リスク過小評価/整列バグを疑え")
    if isinstance(r.index, pd.DatetimeIndex) and len(r):
        yr = r.groupby(r.index.year).sum()
        tot = yr.sum()
        if tot > 0 and (yr.max() / tot) > 0.6:
            flags.append(f"利益が単一年に集中(最大年={yr.idxmax()} が総益の{yr.max()/tot*100:.0f}%): レジーム依存")
    if is_returns is not None and oos_returns is not None:
        si, so = np.sign(is_returns.mean()), np.sign(oos_returns.mean())
        if si != 0 and so != 0 and si != so:
            flags.append("IS/OOSで平均リターンの符号反転: 安定エッジでない(レジーム不安定)")
    return [f"[{name}] {f}" if name else f for f in flags]


def assert_monotonic_index(s: pd.Series) -> None:
    """indexが昇順・重複なしを保証(時系列処理の前提、先読み/重複集計事故を防ぐ)。"""
    if not s.index.is_monotonic_increasing:
        raise ValueError("indexが昇順でありません(時系列前提が崩れています)")
    if not s.index.is_unique:
        raise ValueError("indexに重複があります(同一キー多重計上の恐れ)")
