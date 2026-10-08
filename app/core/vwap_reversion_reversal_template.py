"""VWAP乖離反転(vwap_reversion_reversal)テンプレート — 事前登録検証(2026-07-24,
tick_chart_prereg.md #10簡略版)。

「セッションVWAPから大きく乖離後、構造的な反転確認(安値切り上げ/高値切り下げ+実体方向)で
エントリー」を機械化。出来高デルタ確認の代わりに価格構造確認(切り上げ/切り下げ)で代替する
簡略版(FXデータに真の売買方向別出来高が無いため)。

- VWAP: 日次リセットのセッションVWAP。価格=典型値(H+L+C)/3、出来高=ティックボリューム。
  日付(タイムスタンプの暦日)ごとにgroupby.cumsumで累積し直すため日をまたいで蓄積しない。
- ロング: close <= VWAP - dev_atr×ATR14 に到達(当日のこれまでの累積のみ参照、先読み無し)。
  到達した「次のバー以降」armedフラグが立ち、直近安値 > 1本前の安値(切り上げ)かつ
  陽線確定(close>open)の最初のバーでエントリー。armedはエントリー発火時に消費(reset)
  される(1回の乖離イベント=最大1トレード)。日をまたいだ場合や乖離が解消されないまま
  何度も閾値へ触れても、同一armedエピソード中は複数発火しない設計。
  (実装当初はarmedを当日内cummaxで持続させたが、これだと1銘柄1日あたり数十回の
  ドテン連打が発生し明らかに過剰発火だったため、イベント消費型に修正した。)
- ショートはミラー(高値切り下げ+陰線)。
- SL = 反転確認バーの安値(高値)の下(上)、微小バッファ sl_buffer_atr×ATR14。
  TP = 動的sl_price/tp_price列(RR=rr倍、engine側でshift(1)適用)。

armed/consumeは経路依存(前バーの状態を引き継ぐ)ため単純ループで実装。VWAP/ATR自体は
pandasのgroupby+ewmでベクトル化計算する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import atr
from app.core.strategy_model import Strategy


def _scan_armed_trigger(
    n: int,
    reached: np.ndarray,
    rev_ext: np.ndarray,   # ロング=low、ショート=high(反転確認の切り上げ/切り下げ判定用)
    o: np.ndarray,
    c: np.ndarray,
    day_id: np.ndarray,
    sign: int,
) -> np.ndarray:
    """到達イベントで armed=True、日替わりまたは発火でリセット。
    戻り値: 発火バーのインデックスに True を立てたbool配列。
    """
    fired = np.zeros(n, dtype=bool)
    armed = False
    cur_day = day_id[0] if n > 0 else -1
    for i in range(1, n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            armed = False
        if armed:
            rev_ok = sign * (rev_ext[i] - rev_ext[i - 1]) > 0
            candle_ok = sign * (c[i] - o[i]) > 0
            if rev_ok and candle_ok:
                fired[i] = True
                armed = False
        if reached[i]:
            armed = True
    return fired


def _signal_vwap_reversion_reversal(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    p = strategy.params
    dev_atr = float(p["dev_atr"].value)
    atr_period = max(2, int(p["atr_period"].value))
    sl_buffer_atr = float(p["sl_buffer_atr"].value)
    rr = float(p["rr"].value)

    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    vol = df["volume"].astype(float)
    date = pd.to_datetime(df["timestamp"]).dt.normalize()
    day_id = date.factorize()[0]

    tp_price_typical = (h + l + c) / 3.0
    cum_pv = (tp_price_typical * vol).groupby(date, sort=False).cumsum()
    cum_v = vol.groupby(date, sort=False).cumsum()
    vwap = (cum_pv / cum_v).where(cum_v > 0)

    a = atr(h, l, c, atr_period)

    reached_long = (c <= (vwap - dev_atr * a)).fillna(False).to_numpy()
    reached_short = (c >= (vwap + dev_atr * a)).fillna(False).to_numpy()

    n = len(df)
    oa = o.to_numpy(float); ca = c.to_numpy(float)
    la = l.to_numpy(float); ha = h.to_numpy(float)
    aa = a.to_numpy(float)

    fired_long = _scan_armed_trigger(n, reached_long, la, oa, ca, day_id, sign=1)
    fired_short = _scan_armed_trigger(n, reached_short, ha, oa, ca, day_id, sign=-1)
    fired_short &= ~fired_long  # 同バー衝突時はロング優先(稀な想定)

    signal = np.zeros(n, dtype=int)
    sl_price = np.full(n, np.nan)
    tp_price = np.full(n, np.nan)

    signal[fired_long] = 1
    sl_l = la[fired_long] - sl_buffer_atr * aa[fired_long]
    sl_price[fired_long] = sl_l
    tp_price[fired_long] = ca[fired_long] + rr * (ca[fired_long] - sl_l)

    signal[fired_short] = -1
    sl_s = ha[fired_short] + sl_buffer_atr * aa[fired_short]
    sl_price[fired_short] = sl_s
    tp_price[fired_short] = ca[fired_short] - rr * (sl_s - ca[fired_short])

    return pd.DataFrame({"signal": signal, "sl_price": sl_price, "tp_price": tp_price}, index=df.index)


templates.register(
    "vwap_reversion_reversal",
    defaults={
        "dev_atr": 2.0,          # 事前登録で固定(sweepでの探索対象外)
        "atr_period": 14.0,
        "sl_buffer_atr": 0.1,
        "rr": 2.0,
        "sl_pips": 50.0,         # フォールバック(sl_price無効時)
        "tp_pips": 100.0,
        "max_hold_bars": 300.0,
        "lot": 0.1,
    },
    signal_fn=_signal_vwap_reversion_reversal,
)
