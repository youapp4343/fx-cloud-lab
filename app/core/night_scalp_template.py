"""ナイトスキャルピングテンプレート(市販EA再現シリーズ)。

GogoJungle/MQL5マーケットで長年人気の「ナイトスキャルパー」型EA(Night Hunter Pro、
Evening Scalper Pro、Triple Scalper等)の公開されているロジック概要を、本ツールの
テンプレート契約で再現したもの: NY市場引け後〜アジア時間序盤の低ボラティリティ帯に
限定して、ボリンジャーバンドタッチからの平均回帰(逆張り)を小さめのTPで狙う。

「時間帯限定」はseasonalテンプレート、「BB逆張り」はbb_reversionテンプレートと
それぞれ同じ部品だが、市販ナイトスキャルEAの本質は両者の組合せ(静かな時間帯だけ
逆張りする)にあるため、独立したテンプレートとして提供する。

時間はデータのタイムスタンプ基準(本ツールのdukascopyデータはUTC)。市販EAが謳う
「NYクローズ後」はおおむねUTC 21〜24時、東京仲値前後まで含めるならUTC 0〜1時。
デフォルトは21〜1時(日またぎ)とする。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import bollinger_bands
from app.core.strategy_model import Strategy


def _signal_night_scalp(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    hour_start = int(strategy.params["hour_start"].value) % 24
    # hour_end=24は「その日の終端まで」(seasonalテンプレートと同じ扱い、%24でラップしない)
    hour_end = max(0, min(24, int(strategy.params["hour_end"].value)))
    period = max(2, int(strategy.params["bb_period"].value))
    sigma = float(strategy.params["bb_sigma"].value)

    ts = df["timestamp"]
    if hour_start <= hour_end:
        in_window = (ts.dt.hour >= hour_start) & (ts.dt.hour < hour_end)
    else:
        # 日またぎ(例: 21→1時)。夜間窓は曜日をまたいでも連続した1つの窓として扱う
        # (seasonalと違い曜日条件が無いため、単純なOR条件で正しい)。
        in_window = (ts.dt.hour >= hour_start) | (ts.dt.hour < hour_end)

    _, upper, lower = bollinger_bands(df["close"], period, sigma)
    touch_upper = df["close"] >= upper
    touch_lower = df["close"] <= lower

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(in_window & touch_upper).fillna(False)] = -1
    signal[(in_window & touch_lower).fillna(False)] = 1
    return signal


templates.register(
    "night_scalp",
    defaults={
        "hour_start": 21.0,
        "hour_end": 1.0,
        "bb_period": 20.0,
        "bb_sigma": 2.0,
        "sl_pips": 15.0,
        "tp_pips": 8.0,
        "lot": 0.1,
    },
    signal_fn=_signal_night_scalp,
)
