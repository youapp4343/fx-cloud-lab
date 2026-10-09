"""ルール自動生成器: 特徴量条件のAND結合から売買ルールを乱択で合成する。

既存テンプレートに無い組み合わせを作るための経路。
- 特徴量はすべて当該バー終値までの情報のみ(先読みなし)。執行は engine 側で翌バーにシフトされる
- しきい値は train 区間の分位点で固定する(confirm/holdout の分布を見ない)
- 他銘柄の値動き(ref_*)も特徴量に含める(リードラグ探索)
- COT建玉・金利(cot_* / us2y_* / rate_diff*)は公表時刻(available_at)以降のバーにだけ結合する
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


def _ns(s: pd.Series) -> pd.Series:
    """時刻の単位をnsに揃える。parquetを書いたpandasの版でms/us/nsが混ざり、merge_asofが拒否するため。"""
    return pd.to_datetime(s).astype("datetime64[ns]")


def _streak(close: pd.Series) -> pd.Series:
    """連続陽線(+)/連続陰線(-)の本数。"""
    d = np.sign(close.diff().fillna(0.0))
    grp = (d != d.shift()).cumsum()
    return d * (d.groupby(grp).cumcount() + 1)


def build_features(df: pd.DataFrame, refs: Dict[str, pd.DataFrame], pair: str = "",
                   macro: Optional[Dict[str, pd.DataFrame]] = None,
                   tf: str = "H1") -> Tuple[List[str], np.ndarray, np.ndarray]:
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

    f.update(horizon_features(df, a, BARS_PER_DAY.get(tf, 24)))
    f.update(calendar_features(df))
    if macro and pair:
        f.update(macro_features(df, pair, macro))
        if "strength" in macro:
            f.update(strength_features(df, pair, macro["strength"]))

    names = list(f)
    mat = np.column_stack([f[n].to_numpy(dtype=np.float32) for n in names])
    return names, mat, a.to_numpy(dtype=np.float64)


def horizon_features(df: pd.DataFrame, a: pd.Series, bpd: int) -> Dict[str, pd.Series]:
    """日〜月単位の特徴量。GitHubスカウト(scout/REPORT.md)で検証の質が高い群に多かった
    時系列モメンタム・ボラレジーム・セッションレンジを、既存の短期特徴量に足す。"""
    f: Dict[str, pd.Series] = {}
    c, h, l = df["close"], df["high"], df["low"]
    for days in (5, 20, 60):
        k = days * bpd
        f[f"ret_{days}d"] = (c - c.shift(k)) / (a * np.sqrt(k))
    f["vol_ratio_20d"] = a / a.rolling(20 * bpd, min_periods=10 * bpd).mean().replace(0.0, np.nan)

    # 直近に完了したアジア時間(UTC 0-8時)のレンジに対する現在値の位置。
    # 8時以降は当日分、8時より前は前日分を使う(形成途中のレンジは使わない)
    ts = df["timestamp"]
    day = ts.dt.floor("D")
    asia = df[ts.dt.hour < 8].groupby(day[ts.dt.hour < 8]).agg(ah=("high", "max"), al=("low", "min"))
    asia["available_at"] = asia.index + pd.Timedelta(hours=8)
    asia = asia.reset_index(drop=True).sort_values("available_at")
    asia["available_at"] = _ns(asia["available_at"])
    m = pd.merge_asof(pd.DataFrame({"timestamp": _ns(ts).to_numpy()}), asia, left_on="timestamp",
                      right_on="available_at", direction="backward", tolerance=pd.Timedelta(days=4))
    rng = (m["ah"] - m["al"]).replace(0.0, np.nan)
    f["asia_pos"] = pd.Series(((c.to_numpy() - m["al"]) / rng).to_numpy(), index=df.index)
    return f


def calendar_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """暦の0/1特徴量。名前が is_ で始まるものは分位点ではなく「成立しているか」で条件化する。"""
    ts = df["timestamp"]
    f: Dict[str, pd.Series] = {}
    for i, name in enumerate(("mon", "tue", "wed", "thu", "fri")):
        f[f"is_{name}"] = (ts.dt.dayofweek == i).astype(float)
    # 月末・月初は暦日で定義(営業日数は将来の休場を知らないと数えられないため使わない)
    f["is_month_end"] = (ts.dt.days_in_month - ts.dt.day < 3).astype(float)
    f["is_month_start"] = (ts.dt.day <= 3).astype(float)
    return f


STRENGTH_PAIRS = {  # 通貨 -> (対USDペア, USDが基軸なら-1)
    "EUR": ("EURUSD", 1), "GBP": ("GBPUSD", 1), "AUD": ("AUDUSD", 1), "NZD": ("NZDUSD", 1),
    "JPY": ("USDJPY", -1), "CHF": ("USDCHF", -1), "CAD": ("USDCAD", -1),
}


def load_strength() -> Optional[pd.DataFrame]:
    """通貨強弱: 各通貨の対USD日次リターンから、8通貨の平均に対する相対リターンを作る。

    日足の終値(UTC 0時区切り)で計算し、翌日0時以降のバーにだけ結合する(available_at = 日付+1日)。
    """
    rets = {}
    for ccy, (pair, sign) in STRENGTH_PAIRS.items():
        path = datafeed.OHLC_DIR / f"{pair}_H1.parquet"
        if not path.exists():
            return None
        d = pd.read_parquet(path, columns=["timestamp", "close"]).set_index("timestamp")["close"]
        rets[ccy] = np.log(d.resample("1D").last().dropna()).diff() * sign
    r = pd.DataFrame(rets).dropna(how="all")
    r["USD"] = 0.0
    out = {"available_at": r.index + pd.Timedelta(days=1)}
    for days in (5, 20):
        cum = r.rolling(days, min_periods=days).sum()
        rel = cum.sub(cum.mean(axis=1), axis=0)
        for ccy in rel.columns:
            out[f"{ccy}_{days}"] = rel[ccy].to_numpy()
    return pd.DataFrame(out).sort_values("available_at").reset_index(drop=True)


def strength_features(df: pd.DataFrame, pair: str, strength: pd.DataFrame) -> Dict[str, pd.Series]:
    base, quote = pair[:3], pair[3:]
    f: Dict[str, pd.Series] = {}
    for days in (5, 20):
        kb, kq = f"{base}_{days}", f"{quote}_{days}"
        if kb in strength.columns and kq in strength.columns:
            s = strength[["available_at"]].copy()
            s["v"] = strength[kb] - strength[kq]
            f[f"ccy_strength_{days}d"] = _asof(df["timestamp"], s, "v", pd.Timedelta(days=5))
    return f


def load_refs(pair: str, tf: str, pairs: List[str], rng: random.Random) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in rng.sample([p for p in pairs if p != pair], k=len(pairs) - 1):
        path = datafeed.OHLC_DIR / f"{name}_{tf}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path, columns=["timestamp", "close"])
        if len(out) >= N_REFS:
            break
    return out


# 公表データの鮮度上限。これより古い値しか無いバーは欠損扱い(止まった系列を引き延ばさない)
COT_MAX_AGE = pd.Timedelta(days=21)
DAILY_RATE_MAX_AGE = pd.Timedelta(days=10)
MONTHLY_RATE_MAX_AGE = pd.Timedelta(days=150)


def load_macro() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in ("cot", "rates"):
        path = datafeed.BASE_DIR / "data" / "macro" / f"{name}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path)
    strength = load_strength()
    if strength is not None:
        out["strength"] = strength
    return out


def _asof(ts: pd.Series, src: pd.DataFrame, col: str, max_age: pd.Timedelta) -> pd.Series:
    """各バー時刻について available_at <= 時刻 の最新値を引く(公表前の値は使わない)。"""
    src = src.dropna(subset=[col]).sort_values("available_at")[["available_at", col]].copy()
    src["available_at"] = _ns(src["available_at"])
    left = pd.DataFrame({"timestamp": _ns(ts).to_numpy()})
    m = pd.merge_asof(left, src, left_on="timestamp", right_on="available_at",
                      direction="backward", tolerance=max_age)
    return pd.Series(m[col].to_numpy(), index=ts.index)


def macro_features(df: pd.DataFrame, pair: str, macro: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """COT建玉と金利の特徴量。変換(zスコア・差分)は公表系列の上で行い、その後にバーへ結合する。"""
    f: Dict[str, pd.Series] = {}
    ts = df["timestamp"]
    base, quote = pair[:3], pair[3:]

    cot = macro.get("cot")
    if cot is not None:
        legs: Dict[str, Dict[str, pd.Series]] = {}
        for ccy in (base, quote):
            g = cot[cot["ccy"] == ccy].sort_values("as_of").copy()
            if g.empty:
                continue
            for src in ("spec_net", "comm_net"):
                roll = g[src].rolling(52, min_periods=40)
                g[f"{src}_z"] = (g[src] - roll.mean()) / roll.std().replace(0.0, np.nan)
            chg = g["spec_net"].diff(4)
            g["spec_chg"] = chg / g["spec_net"].diff().rolling(52, min_periods=40).std().replace(0.0, np.nan)
            legs[ccy] = {c: _asof(ts, g, c, COT_MAX_AGE) for c in ("spec_net_z", "comm_net_z", "spec_chg")}
        # USDは先物が無いので0扱い(相手通貨の建玉だけで決まる)。XAUUSDはXAU側のみ
        if legs and (base in legs or base == "USD") and (quote in legs or quote == "USD"):
            zero = pd.Series(0.0, index=ts.index)
            for c, name in (("spec_net_z", "cot_spec_z"), ("comm_net_z", "cot_comm_z"), ("spec_chg", "cot_spec_chg4")):
                f[name] = legs.get(base, {}).get(c, zero) - legs.get(quote, {}).get(c, zero)

    rates = macro.get("rates")
    if rates is not None:
        wide = {k: g.sort_values("as_of").rename(columns={"value": k}) for k, g in rates.groupby("series")}
        if "US2Y" in wide:
            g = wide["US2Y"].copy()
            g["chg5"], g["chg20"] = g["US2Y"].diff(5), g["US2Y"].diff(20)
            f["us2y_chg5"] = _asof(ts, g, "chg5", DAILY_RATE_MAX_AGE)
            f["us2y_chg20"] = _asof(ts, g, "chg20", DAILY_RATE_MAX_AGE)
            if "US10Y" in wide:
                c = g.merge(wide["US10Y"][["as_of", "US10Y"]], on="as_of")
                c["curve"] = c["US10Y"] - c["US2Y"]
                f["us_curve"] = _asof(ts, c, "curve", DAILY_RATE_MAX_AGE)
        kb, kq = f"R3M_{base}", f"R3M_{quote}"
        if kb in wide and kq in wide:
            d = wide[kb][["as_of", "available_at", kb]].merge(wide[kq][["as_of", kq]], on="as_of")
            d["diff"] = d[kb] - d[kq]
            d["diff_chg3"] = d["diff"].diff(3)
            f["rate_diff"] = _asof(ts, d, "diff", MONTHLY_RATE_MAX_AGE)
            f["rate_diff_chg3"] = _asof(ts, d, "diff_chg3", MONTHLY_RATE_MAX_AGE)
    return f


class Block:
    """1銘柄×1時間足ぶんの特徴量と train 分位点。"""

    def __init__(self, df: pd.DataFrame, bounds: Tuple[int, int], pip: float,
                 refs: Dict[str, pd.DataFrame], pair: str = "",
                 macro: Optional[Dict[str, pd.DataFrame]] = None, tf: str = "H1") -> None:
        names, mat, atr_arr = build_features(df, refs, pair, macro, tf)
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
            if self.names[j].startswith("is_"):
                conds.append({"f": self.names[j], "op": ">", "q": 1.0, "thr": 0.5})
                continue
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
    conds = " & ".join(c["f"] if c["f"].startswith("is_")
                       else f"{c['f']}{c['op']}{c['thr']:g}(q{int(c['q'] * 100)})" for c in rule["conds"])
    side = "LONG" if rule["side"] > 0 else "SHORT"
    return f"{side} if {conds}; SL {rule['sl_pips']:g} TP {rule['tp_pips']:g} hold {rule['hold']}"


def rule_params(rule: Dict[str, Any]) -> Dict[str, float]:
    return {"sl_pips": rule["sl_pips"], "tp_pips": rule["tp_pips"], "lot": 0.1,
            "max_hold_bars": float(rule["hold"])}
