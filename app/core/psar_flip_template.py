"""パラボリックSARドテン順張り(psar_flip)テンプレート。

SARが価格の反対側にフリップした瞬間に順張りエントリーするドテン型
(GitHubコーパスのParabolicSAR型EAの典型構成)。

先読み回避: parabolic_sarの各バー値は前バーまでの情報で確定している(indicators.py参照)。
フリップ検知はengine._signal_ma_crossと同じくshift(1)した生値同士の比較で行い、
シグナル自体はシフトせず、engine.run_backtestのshift(1)が次バー始値執行を保証する。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import parabolic_sar
from app.core.strategy_model import Strategy

_DEFAULT_STEP = 0.02
_DEFAULT_MAX_STEP = 0.2


def _signal_psar_flip(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """SARが価格の上→下にフリップしたバーで買い、下→上にフリップしたバーで売り。"""
    step = float(strategy.params["step"].value)
    max_step = float(strategy.params["max_step"].value)
    if step <= 0:
        step = _DEFAULT_STEP
    if max_step <= 0:
        max_step = _DEFAULT_MAX_STEP

    sar = parabolic_sar(df["high"], df["low"], step, max_step)
    close = df["close"]
    prev_sar = sar.shift(1)
    prev_close = close.shift(1)
    flip_up = (prev_sar >= prev_close) & (sar < close)
    flip_down = (prev_sar <= prev_close) & (sar > close)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[flip_up.fillna(False)] = 1
    signal[flip_down.fillna(False)] = -1
    return signal


templates.register(
    "psar_flip",
    defaults={
        "step": 0.02,
        "max_step": 0.2,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_psar_flip,
)
