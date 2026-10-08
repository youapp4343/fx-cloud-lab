"""Bagovino テンプレート — EMA5×EMA12 クロス + RSI21 フィルタ。

出典: Forex Factory 等で流通する "Bagovino" スキャル/デイ手法。速いEMA(5)が遅いEMA(12)を
クロスした瞬間に、RSI21 が中立50の順張り側にあることを確認して仕掛ける単純なトレンド
フォロー。任意でCCIのゼロライン方向を追加フィルタにできる。

- LONG: EMA5 が EMA12 を上抜け、かつ RSI21 > 50。
- SHORT: EMA5 が EMA12 を下抜け、かつ RSI21 < 50。
- use_cci=1 のとき: 上記に加え CCI > 0(買い)/ CCI < 0(売り)を要求する。

注記: CCI の期間は indicators.cci の既定値14を用いる(専用パラメータは設けない)。原法の
裁量的な時間帯選好・手動利確は捨て、決済はengineの sl/tp に委ねる。

先読み規律: signal_fn はshiftしない生シグナル(1/-1/0)を返す。翌バー始値執行のshiftは
engine.run_backtest 側が担う。EMAクロス検出の `.shift(1)` は自分の過去バーのみを参照する
ため先読みではない(macd_cross / tdi_cross と同じ流儀)。クロスは本質的に単発イベントの
ため追加の再アームは不要。ウォームアップのNaNは比較でFalseに落ち無シグナルとなる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import cci, ema, rsi
from app.core.strategy_model import Strategy


def _signal_bagovino(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    fast = max(1, int(p["fast"].value))
    slow = max(2, int(p["slow"].value))
    rsi_period = max(2, int(p["rsi_period"].value))
    use_cci = int(p["use_cci"].value)  # 0=使わない, 1=CCIゼロライン方向も要求
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    ema_fast = ema(df["close"], fast)
    ema_slow = ema(df["close"], slow)
    r = rsi(df["close"], rsi_period)

    # EMAクロス検出(自分の過去バー参照)。
    pf, ps = ema_fast.shift(1), ema_slow.shift(1)
    cross_up = (ema_fast > ema_slow) & (pf <= ps)
    cross_dn = (ema_fast < ema_slow) & (pf >= ps)

    long_ok = cross_up & (r > 50.0)
    short_ok = cross_dn & (r < 50.0)

    if use_cci == 1:
        c = cci(df["high"], df["low"], df["close"])  # 期間は既定14
        long_ok = long_ok & (c > 0.0)
        short_ok = short_ok & (c < 0.0)

    long_sig = long_ok.fillna(False)
    short_sig = short_ok.fillna(False)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "bagovino",
    defaults={
        "fast": 5.0, "slow": 12.0, "rsi_period": 21.0,
        "use_cci": 0.0, "dir_mode": 0.0,
        "sl_pips": 40.0, "tp_pips": 60.0, "lot": 0.1,
    },
    signal_fn=_signal_bagovino,
)
