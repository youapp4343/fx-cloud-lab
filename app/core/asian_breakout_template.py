"""アジアレンジ・ブレイクアウト + EMA55方向フィルタ テンプレート。

東京(アジア)時間に形成される値幅(アジアレンジ)を、ロンドン勢が参入する早朝以降に
ブレイクした方向へ順張りする定番手法。だましを減らすため、中期トレンド方向を EMA(55) の
向きで確認し、上抜けは EMA55 上昇中のみロング、下抜けは EMA55 下降中のみショートに限定する。

時間帯の定義(UTC、tz-naive がプロジェクト規約):
- アジアレンジ形成窓 = 前日21:00 〜 当日06:00 UTC(=UTC時のhourが [21,22,23]∪[0,1,2,3,4,5])。
  06:00時点でその日のアジア高値・安値を確定(ロック)する。
- トレードセッション = 06:00〜20:00 UTC(hourが [6..20])。この間に初めて高値がロックした
  アジア高値を上抜け、かつ EMA55 が上昇(ema55 > ema55.shift(1))していれば +1(初回のみ)。
  初めて安値がアジア安値を下抜け、かつ EMA55 が下降していれば -1(初回のみ)。
- ガードは1セッションにつきロング1回・ショート1回まで(month_end_fix と同じ「重複防止」思想)。

session_date の割り当て(アジア窓の日付跨ぎ処理): アジア窓の 21:00〜23:00 のバーは翌カレンダー日
の東京セッションに属する。そこで全バーに session_date = (timestamp + 3時間).date() を与える。
+3時間により 21:00 UTC → 00:00(翌日)、20:59 UTC → 23:59(同日)へ写像されるため、各 session_date は
「前日21:00〜当日20:59 UTC」の連続1ブロックに一致する。この単一キーで groupby すれば、アジア窓
(21:00〜05:00)と同じ日のトレードセッション(06:00〜20:00)が1つの session_date にまとまる
(=その日06:00にロックされるレンジと、それを使って建てる当日のトレードが同じキーで揃う)。

先読み回避: アジア高値/安値は「アジア時間帯のバーのみ」から groupby で算出し、同 session_date の
全バーへブロードキャストする。トレードバー(hour 6..20)は必ず全アジアバー(最遅で hour 5 台)より
後に来るため、トレードバーで参照するアジアレンジは 06:00 までに確定済み=過去情報のみ(リーク無し)。
EMA55 の向きも ema55 と ema55.shift(1)(自分の前バー値)の比較で、確定済みバーのみを使う。
signal_fn は shift しない生シグナルを返し、翌バー始値執行の shift(1) は engine 側が担う。

近似: 原法は SL30/TP15 の半利確 + 建値移動(ブレイクイーブン)といった段階決済だが、本テンプレート
では engine の単一 sl_pips/tp_pips へ単純化している(既定 SL30/TP30)。SL/TP・ブレイク幅は M15/H1 の
バー粒度を想定して調整するのが前提(H4以上ではアジア窓・トレード窓のバー数が少なく機能しにくい)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.indicators import ema
from app.core.strategy_model import Strategy

_ASIAN_HOURS = frozenset({21, 22, 23, 0, 1, 2, 3, 4, 5})  # アジアレンジ形成窓(UTC hour)
_TRADE_START_HOUR = 6   # トレードセッション開始(この時刻以降でアジアレンジをロック)
_TRADE_END_HOUR = 20    # トレードセッション終了(含む)


def _asian_range_by_session(
    df: pd.DataFrame, session_date: pd.Series, asian_mask: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """session_date ごとのアジア高値・安値を、全バーへブロードキャストした2本のSeriesで返す。

    アジア時間帯のバー(asian_mask)だけを session_date で groupby して high.max / low.min を取り、
    それを各バーの session_date へ map で割り当てる。アジアバーが1本も無い session_date は NaN。
    """
    asian = df.loc[asian_mask]
    asian_sd = session_date[asian_mask]
    grp_high = asian["high"].groupby(asian_sd).max()
    grp_low = asian["low"].groupby(asian_sd).min()
    asian_high = session_date.map(grp_high)
    asian_low = session_date.map(grp_low)
    return asian_high, asian_low


def _signal_asian_breakout(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ema_period = max(2, int(p["ema_period"].value))
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only
    allow_long = dir_mode in (0, 1)
    allow_short = dir_mode in (0, 2)

    n = len(df)
    signal = pd.Series(0, index=df.index, dtype=int)
    if n == 0:
        return signal

    ts = df["timestamp"]
    hour = ts.dt.hour
    # session_date = (timestamp + 3時間) の日付。21:00 UTC を翌セッション日へ写像する(docstring参照)。
    session_date = (ts + pd.Timedelta(hours=3)).dt.date
    asian_mask = hour.isin(_ASIAN_HOURS)

    asian_high, asian_low = _asian_range_by_session(df, session_date, asian_mask)

    line = ema(df["close"], ema_period)
    rising = (line > line.shift(1)).fillna(False)   # EMA55 上昇中(自バー確定、前バー比)
    falling = (line < line.shift(1)).fillna(False)  # EMA55 下降中

    sd_arr = session_date.to_numpy()
    hour_arr = hour.to_numpy()
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ah_arr = asian_high.to_numpy(dtype=float)
    al_arr = asian_low.to_numpy(dtype=float)
    rising_arr = rising.to_numpy()
    falling_arr = falling.to_numpy()
    out = np.zeros(n, dtype=int)

    cur_session = None
    done_long = False   # 当セッションで既にロング発火したか(1セッション1回まで)
    done_short = False
    for i in range(n):
        s = sd_arr[i]
        if s != cur_session:  # セッション切替でガードをリセット
            cur_session = s
            done_long = False
            done_short = False
        h = hour_arr[i]
        if h < _TRADE_START_HOUR or h > _TRADE_END_HOUR:
            continue  # アジア窓・深夜はレンジ形成の監視のみ。トレードは06:00〜20:00 UTCに限る
        ah = ah_arr[i]
        al = al_arr[i]
        # ロング: 当日アジア高値を高値が初めて上抜け、かつ EMA55 上昇中
        if allow_long and not done_long and not np.isnan(ah) and high[i] > ah and rising_arr[i]:
            out[i] = 1
            done_long = True
        # ショート: 当日アジア安値を安値が初めて下抜け、かつ EMA55 下降中
        # (rising と falling は排他なので同一バーで両建てにはならない)
        if allow_short and not done_short and not np.isnan(al) and low[i] < al and falling_arr[i]:
            out[i] = -1
            done_short = True

    return pd.Series(out, index=df.index, dtype=int)


templates.register(
    "asian_breakout",
    defaults={
        "ema_period": 55.0,
        "dir_mode": 0.0,
        "sl_pips": 30.0,
        "tp_pips": 30.0,
        "lot": 0.1,
    },
    signal_fn=_signal_asian_breakout,
)
