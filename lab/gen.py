"""ルール自動生成器: 特徴量条件のAND結合から売買ルールを乱択で合成する。

既存テンプレートに無い組み合わせを作るための経路。
- 特徴量はすべて当該バー終値までの情報のみ(先読みなし)。執行は engine 側で翌バーにシフトされる
- しきい値は train 区間の分位点で固定する(confirm/holdout の分布を見ない)
- 他銘柄の値動き(ref_*)も特徴量に含める(リードラグ探索)
- 生成したシグナルは lab_replay テンプレート経由で既存エンジンに流す
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core import datafeed, templates
from app.core.indicators import atr, rsi, sma
from app.core.strategy_model import Strategy

BARS_PER_DAY = {"M15": 96, "M30": 48, "H1": 24, "H4": 6}

LOW_Q = [0.05, 0.1, 0.2, 0.3]
HIGH_Q = [0.7, 0.8, 0.9, 0.95]
SL_ATR = [1.0, 1.5, 2.0, 3.0]
TP_ATR = [1.0, 1.5, 2.0, 3.0, 5.0]
HOLD_BARS = [4, 8, 16, 32, 64]
N_COND_WEIGHTS = {1: 0.3, 2: 0.5, 3: 0.2}
N_REFS = 2
MIN_COVERAGE = 0.85

# lab_replay が返すシグナル(直前にセットしたものをそのまま返す)
_REPLAY: Dict[str, np.ndarray] = {}


def _signal_replay(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    sig = _REPLAY["sig"]
    if len(sig) != len(df):
        raise ValueError(f"replay長不一致: sig={len(sig)} df={len(df)}")
    return pd.Series(sig, index=df.index, dtype=int)


templates.register(
    "lab_replay",
    defaults={"sl_pips": 30.0, "tp_pips": 30.0, "lot": 0.1},
    signal_fn=_signal_replay,
    exportable=False,
)


def set_replay(sig: np.ndarray) -> None:
    _REPLAY["sig"] = sig


def _streak(close: pd.Series) -> pd.Series:
    """連続陽線(+)/連続陰線(-)の本数。"""
    d = np.sign(close.diff().fillna(0.0))
    grp = (d != d.shift()).cumsum()
    return d * (d.groupby(grp).cumcount() + 1)


def build_features(df: pd.DataFrame, refs: Dict[str, pd.DataFrame]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """特徴量行列(float32)と ATR14(価格単位)を返す。"""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = atr(h, l, c, 14).replace(0.0, np.nan)
    f: Dict[str, pd.Series] = {}

    for k in (1, 3, 6, 12, 24, 48):
        f[f"ret_{k}"] = (c - c.shift(k)) / a
    for n in (7, 14, 28):
        f[f"rsi_{n}"] = rsi(c, n)
    for n in (20, 50, 100, 200):
        f[f"sma_dev_{n}"] = (c - sma(c, n)) / a
    for n in (20, 55, 120):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"don_pos_{n}"] = (c - lo) / (hi - lo).replace(0.0, np.nan)
    f["vol_ratio"] = a / atr(h, l, c, 100).replace(0.0, np.nan)
    rng = (h - l).replace(0.0, np.nan)
    f["range_ratio"] = (h - l) / a
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["streak"] = _streak(c)

    day = df["timestamp"].dt.floor("D")
    day_open = o.groupby(day).transform("first")
    f["day_pos"] = (c - day_open) / a
    daily = df.groupby(day).agg(dh=("high", "max"), dl=("low", "min")).shift(1)
    pdh, pdl = day.map(daily["dh"]), day.map(daily["dl"])
    f["prev_day_pos"] = (c - pdl) / (pdh - pdl).replace(0.0, np.nan)

    # 他銘柄の同時刻までの値動き(自身のボラで正規化)。欠損バーは直前値で埋めない(NaN=条件不成立)
    for name, ref in refs.items():
        rc = df["timestamp"].map(ref.set_index("timestamp")["close"])
        r1 = rc.pct_change(fill_method=None)
        # 相手銘柄の休場バー(NaN)が窓に混ざっても推定できるよう min_periods を緩める
        vol = r1.rolling(100, min_periods=50).std().replace(0.0, np.nan)
        for k in (1, 4, 12):
            f[f"ref_{name}_ret_{k}"] = (rc / rc.shift(k) - 1.0) / (vol * np.sqrt(k))

    names = list(f)
    mat = np.column_stack([f[n].to_numpy(dtype=np.float32) for n in names])
    return names, mat, a.to_numpy(dtype=np.float64)


def load_refs(pair: str, tf: str, pairs: List[str], rng: random.Random) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in rng.sample([p for p in pairs if p != pair], k=len(pairs) - 1):
        path = datafeed.OHLC_DIR / f"{name}_{tf}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path, columns=["timestamp", "close"])
        if len(out) >= N_REFS:
            break
    return out


class Block:
    """1銘柄×1時間足ぶんの特徴量と train 分位点。"""

    def __init__(self, df: pd.DataFrame, bounds: Tuple[int, int], pip: float,
                 refs: Dict[str, pd.DataFrame]) -> None:
        names, mat, atr_arr = build_features(df, refs)
        a, b = bounds
        self.slices = [slice(0, a), slice(a, b), slice(b, len(df))]
        # coverageゲート: どの区間でも欠損が多い特徴量は使わない(欠損の有無が代理変数化するため)
        keep = [j for j in range(len(names))
                if all(np.isfinite(mat[sl, j]).mean() >= MIN_COVERAGE for sl in self.slices)]
        self.names, self.mat = [names[j] for j in keep], mat[:, keep]
        self.qs = sorted(LOW_Q + HIGH_Q)
        self.qtab = np.nanquantile(self.mat[:a].astype(np.float64), self.qs, axis=0)
        self.atr_pips = float(np.nanmedian(atr_arr[:a]) / pip)

    def sample_rule(self, rng: random.Random) -> Dict[str, Any]:
        n = rng.choices(list(N_COND_WEIGHTS), list(N_COND_WEIGHTS.values()))[0]
        conds = []
        for j in rng.sample(range(len(self.names)), k=n):
            op = rng.choice("<>")
            q = rng.choice(LOW_Q if op == "<" else HIGH_Q)
            thr = float(self.qtab[self.qs.index(q), j])
            conds.append({"f": self.names[j], "op": op, "q": q, "thr": round(thr, 5)})
        return {
            "side": rng.choice([1, -1]),
            "conds": conds,
            "sl_pips": round(max(1.0, rng.choice(SL_ATR) * self.atr_pips), 1),
            "tp_pips": round(max(1.0, rng.choice(TP_ATR) * self.atr_pips), 1),
            "hold": rng.choice(HOLD_BARS),
        }

    def signal(self, rule: Dict[str, Any], part: int) -> Optional[np.ndarray]:
        """条件成立の立ち上がりバーだけに side を立てる。しきい値がNaNならNone。"""
        m = self.mat[self.slices[part]]
        mask = np.ones(len(m), dtype=bool)
        for cnd in rule["conds"]:
            if not np.isfinite(cnd["thr"]):
                return None
            col = m[:, self.names.index(cnd["f"])]
            with np.errstate(invalid="ignore"):
                mask &= (col < cnd["thr"]) if cnd["op"] == "<" else (col > cnd["thr"])
        edge = mask & ~np.concatenate([[False], mask[:-1]])
        return edge.astype(np.int64) * rule["side"]


def describe(rule: Dict[str, Any]) -> str:
    conds = " & ".join(f"{c['f']}{c['op']}{c['thr']:g}(q{int(c['q'] * 100)})" for c in rule["conds"])
    side = "LONG" if rule["side"] > 0 else "SHORT"
    return f"{side} if {conds}; SL {rule['sl_pips']:g} TP {rule['tp_pips']:g} hold {rule['hold']}"


def rule_params(rule: Dict[str, Any]) -> Dict[str, float]:
    return {"sl_pips": rule["sl_pips"], "tp_pips": rule["tp_pips"], "lot": 0.1,
            "max_hold_bars": float(rule["hold"])}
