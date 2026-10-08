"""ベガストンネル(Vegas Tunnel)ブレイクアウト テンプレート。

出典: 為替コミュニティ発祥の "Vegas Tunnel" 手法。EMA144 と EMA169 の2本で作る帯
(=トンネル)を価格ブレイクの基準にする順張り手法。終値がトンネル上端(2本のEMAの
うち高い方)を上抜けたら買い、トンネル下端(低い方)を下抜けたら売り、というのが核。
元手法では利確を 55/89/144pips のフィボ級数で分割(スケールアウト)し、損切りは
「反対側のトンネル端」に置く、さらに上位足の長期EMA(610等)でトレンドフィルタを
掛ける、といった運用が乗る。

本テンプレートは機械検証できる核だけを実装する:
  - トンネル = (max(EMA_a, EMA_b), min(EMA_a, EMA_b))。
  - LONG = 終値がトンネル上端を上抜けた最初のバー(再アームで初回ブレイクのみ発火)。
  - SHORT = 終値がトンネル下端を下抜けた最初のバー(同上)。
  - 任意のトレンドガード no_counter: 長期EMA(ema_slow=610)が下降中なら買いを、
    上昇中なら売りを抑止(強い逆張りブレイクを避ける)。既定OFF。

近似・簡略化(重要):
  - 「損切り=反対側のトンネル端(可変距離)」と「利確 55/89/144 の分割決済」は、
    engine 側の固定 sl_pips / tp_pips 一本に単純化している。したがって建玉サイズ可変・
    トレーリング・部分利確は再現していない(決済は engine の sl/tp/max_hold に委譲)。
  - EMA周期をATRで動的に切り替える版(ATR Vegas)は別物として本テンプレートでは扱わない。

先読み規律: signal_fn はシフトしない生シグナル(1/-1/0 の pd.Series)を返すのみ。翌バー
始値執行の shift は engine.run_backtest 側(raw_signal.shift(1))が担う。各バーは自バーの
確定OHLC + 自分の過去(shift(1)で得る前バー値)のみを参照し、未来バーは一切見ない。
トンネル端は EMA_a と EMA_b の両方が確定するまで NaN(np.maximum/np.minimum が NaN を
伝播)となり、その間はブレイク判定が False に落ちてシグナル無しとなる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import ema
from app.core.strategy_model import Strategy


def _signal_vegas_tunnel(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    period_a = max(2, int(p["ema_a"].value))
    period_b = max(2, int(p["ema_b"].value))
    slow_period = max(2, int(p["ema_slow"].value))
    no_counter = int(p["no_counter"].value)  # 0=ガード無し, 1=長期EMA逆行時は抑止
    dir_mode = int(p["dir_mode"].value)      # 0=both, 1=long_only, 2=short_only

    ema_a = ema(df["close"], period_a)
    ema_b = ema(df["close"], period_b)
    # トンネル上端/下端。両EMAが確定するまでNaN(np.maximum/minimumがNaNを伝播)。
    tunnel_top = pd.Series(np.maximum(ema_a.to_numpy(), ema_b.to_numpy()), index=df.index)
    tunnel_bot = pd.Series(np.minimum(ema_a.to_numpy(), ema_b.to_numpy()), index=df.index)

    close = df["close"]
    above = close > tunnel_top  # トンネル上端より上(上抜け状態)
    below = close < tunnel_bot  # トンネル下端より下(下抜け状態)
    # 再アーム: トンネルを抜けた最初のバーのみ発火。帯の外に張り付く間の連続発火を防ぐ。
    long_sig = above & (~above.shift(1).fillna(False))
    short_sig = below & (~below.shift(1).fillna(False))

    if no_counter == 1:
        slow = ema(close, slow_period)
        slow_falling = (slow < slow.shift(1)).fillna(False)  # 長期EMAが下降中
        slow_rising = (slow > slow.shift(1)).fillna(False)   # 長期EMAが上昇中
        long_sig = long_sig & (~slow_falling)  # 下降トレンド下での買いブレイクを抑止
        short_sig = short_sig & (~slow_rising)  # 上昇トレンド下での売りブレイクを抑止

    long_sig = long_sig.fillna(False)
    short_sig = short_sig.fillna(False)

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "vegas_tunnel",
    defaults={
        "ema_a": 144.0, "ema_b": 169.0, "ema_slow": 610.0,
        "no_counter": 0.0, "dir_mode": 0.0,
        "sl_pips": 30.0, "tp_pips": 89.0, "lot": 0.1,
    },
    signal_fn=_signal_vegas_tunnel,
)
