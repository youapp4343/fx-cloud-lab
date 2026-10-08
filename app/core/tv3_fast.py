# -*- coding: utf-8 -*-
"""tv3 テンプレのループ部分を numba でJIT化する。

Supertrend系は band のクランプが逐次依存なのでベクトル化できない。
純Pythonループだと M5(35万本)× 8倍率 × 3区間(full/is/oos)× 84セルで
1テンプレ2,500秒に達した(tv3_226 実測)。

★数値が変わってはいけないので、純Python版と同一結果であることを
   `verify_same()` で確認してから使う。一致しなければ純Python版に落とす。
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # numba が無い環境では純Pythonにフォールバック
    _HAS_NUMBA = False

    def njit(*a, **k):  # noqa: D103
        def deco(f):
            return f
        return deco if not a else a[0]


@njit(cache=True)
def _st_dir_core(mid, cl, a, factor):
    """Supertrend の方向(-1=強気, 1=弱気)。Pine の ta.supertrend と同じ符号。"""
    n = len(cl)
    d = np.ones(n)
    fu = np.nan
    fl = np.nan
    for i in range(1, n):
        if not np.isfinite(a[i]):
            d[i] = d[i - 1]
            continue
        up = mid[i] + factor * a[i]
        lo = mid[i] - factor * a[i]
        pu = fu if np.isfinite(fu) else up
        pl = fl if np.isfinite(fl) else lo
        fu = up if (up < pu or cl[i - 1] > pu) else pu
        fl = lo if (lo > pl or cl[i - 1] < pl) else pl
        prev = d[i - 1]
        if prev == 1 and cl[i] > fu:
            d[i] = -1.0
        elif prev == -1 and cl[i] < fl:
            d[i] = 1.0
        else:
            d[i] = prev
    return d


@njit(cache=True)
def _st_dirs_core(mid, cl, a, factors):
    """複数倍率を一度に。(n, len(factors)) を返す。1=強気, -1=弱気(232の符号)。"""
    n = len(cl)
    m = len(factors)
    out = np.zeros((n, m))
    for j in range(m):
        k = factors[j]
        s = np.nan
        d = 1.0
        for i in range(n):
            if not np.isfinite(a[i]):
                out[i, j] = d
                continue
            off = k * a[i]
            if not np.isfinite(s):
                s = mid[i] - off
                d = 1.0
            elif d == 1.0:
                if mid[i] - off > s:
                    s = mid[i] - off
                if cl[i] < s:
                    d = -1.0
                    s = mid[i] + off
            else:
                if mid[i] + off < s:
                    s = mid[i] + off
                if cl[i] > s:
                    d = 1.0
                    s = mid[i] - off
            out[i, j] = d
    return out


@njit(cache=True)
def _entropy_core(r, n, nbin):
    """対数リターンの移動シャノンエントロピー(0-1正規化)。226用。"""
    N = len(r)
    ent = np.full(N, 0.5)
    max_ent = np.log(float(nbin)) / np.log(2.0)
    counts = np.zeros(nbin, dtype=np.int64)
    for i in range(n, N):
        lo = 1e18
        hi = -1e18
        cnt = 0
        for k in range(i - n + 1, i + 1):
            v = r[k]
            if np.isfinite(v):
                cnt += 1
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
        if cnt < 3 or hi == lo:
            continue
        for b in range(nbin):
            counts[b] = 0
        w = (hi - lo) / nbin
        for k in range(i - n + 1, i + 1):
            v = r[k]
            if not np.isfinite(v):
                continue
            b = int((v - lo) / w)
            if b >= nbin:
                b = nbin - 1
            if b < 0:
                b = 0
            counts[b] += 1
        tot = 0
        for b in range(nbin):
            tot += counts[b]
        e = 0.0
        for b in range(nbin):
            if counts[b] > 0:
                p = counts[b] / tot
                e -= p * np.log(p) / np.log(2.0)
        ent[i] = e / max_ent
    return ent


def verify_same(fn_fast, fn_slow, *args, tol: float = 0.0) -> bool:
    """JIT版と純Python版が一致するか。差があれば使わない。"""
    a = np.asarray(fn_fast(*args), dtype=float)
    b = np.asarray(fn_slow(*args), dtype=float)
    if a.shape != b.shape:
        return False
    d = np.abs(a - b)
    d = d[np.isfinite(d)]
    return bool(len(d) == 0 or d.max() <= tol)
