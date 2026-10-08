"""レジーム適応(regime_adaptive)テンプレート。

app/core/novel_indicators.pyのvariance_ratio(分散比)でバーごとに相場のレジーム
(トレンド優位/平均回帰優位/不明)を判定し、レジームごとに異なる既存ロジック相当の
手法へ自動的に切り替える。既存テンプレート(ma_cross/rsi_reversal/breakout/bb_reversion等)が
単一ロジックを常時適用するのに対し、本テンプレートは「どの手法を使うべきか」自体を
市場状態(VR)から検出する点が質的に異なる。

- トレンドレジーム(VR > vr_trend_threshold): 価格がSMA(mom_period)を上抜け/下抜けした
  瞬間にモメンタムへ乗る。engine._signal_ma_crossのgolden/dead cross判定
  (シフト済みの生値同士を比較してクロスを検知する)と同じ流儀を、fast/slow SMAの代わりに
  close/SMAに適用したもの。
- 平均回帰レジーム(VR < vr_revert_threshold): RSIが極値から戻ってきた瞬間に逆張り
  (engine._signal_rsi_reversalと同じ判定)。
- vr_trend_thresholdとvr_revert_thresholdの間(どちらのレジームでもない)は様子見でシグナルを
  出さない。閾値が正しい大小関係(vr_trend_threshold > vr_revert_threshold)であれば
  trending/revertingが同時にTrueになることはない。閾値を逆転させた場合はこの限りではないが、
  その場合も単に両方のロジックが同時評価されるだけでクラッシュはしない
  (同一バーで両方のシグナルが立つ場合は後段のsell代入がbuy代入を上書きする)。

先読み回避: variance_ratio/sma/rsiはいずれもバーiまでの過去windowのみを使うためシフト不要。
シグナル自体はシフトせず、engine.run_backtestのshift(1)が次バー始値執行を保証する
(既存の_signal_ma_cross等と同じ規律)。

NaN処理: 比較演算(`>` `<` `<=` `>=`)はNaNオペランドに対して常にFalseを返すため、
vr/sma/rsiのウォームアップ期間は自動的に「シグナルなし」として扱われる。ただしbool Seriesを
そのままshift()すると欠損表現のためobject dtypeへ暗黙アップキャストされ、その後のfillna(False)が
FutureWarning(暗黙ダウンキャスト警告)を出す。これを避けるため_signal_ma_crossに倣い、
shift()するのは常に生の数値(close/sma_mom)のみとし、bool化(比較)はshift後に行う。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import rsi, sma
from app.core.novel_indicators import variance_ratio
from app.core.strategy_model import Strategy


def _signal_regime_adaptive(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    # vr_period<=0/rsi_period<=0はvariance_ratio/rsi内部のewm(alpha=1/period)等でZeroDivisionError・
    # ValueErrorになる(FABLE監査で発見、optimizerのrange探索で0/負値グリッドを踏むとクラッシュしうる)。
    # vr_kは既にmax(1,...)でガード済みだったため、他パラメータも同様にクリップする。
    vr_period = max(1, int(strategy.params["vr_period"].value))
    vr_k = max(1, int(strategy.params["vr_k"].value))
    vr_trend_threshold = float(strategy.params["vr_trend_threshold"].value)
    vr_revert_threshold = float(strategy.params["vr_revert_threshold"].value)
    mom_period = max(1, int(strategy.params["mom_period"].value))
    rsi_period = max(1, int(strategy.params["rsi_period"].value))
    rsi_lower = float(strategy.params["rsi_lower"].value)
    rsi_upper = float(strategy.params["rsi_upper"].value)

    vr = variance_ratio(df["close"], vr_period, vr_k)
    trending = vr > vr_trend_threshold
    reverting = vr < vr_revert_threshold

    # トレンドレジーム: 価格がSMA(mom_period)を上抜け/下抜けした瞬間。
    # _signal_ma_crossのgolden/dead cross判定と同じく、シフト後の生値同士を比較してクロスを
    # 検知する(bool Seriesを直接shiftしない)。
    sma_mom = sma(df["close"], mom_period)
    prev_close = df["close"].shift(1)
    prev_sma_mom = sma_mom.shift(1)
    cross_up = (prev_close <= prev_sma_mom) & (df["close"] > sma_mom)
    cross_down = (prev_close >= prev_sma_mom) & (df["close"] < sma_mom)
    momentum_buy = trending & cross_up
    momentum_sell = trending & cross_down

    # 平均回帰レジーム: RSIが極値から戻ってきた瞬間に逆張り(_signal_rsi_reversalと同じ判定)。
    r = rsi(df["close"], rsi_period)
    prev_r = r.shift(1)
    revert_buy = reverting & (prev_r <= rsi_lower) & (r > rsi_lower)
    revert_sell = reverting & (prev_r >= rsi_upper) & (r < rsi_upper)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(momentum_buy | revert_buy).fillna(False)] = 1
    signal[(momentum_sell | revert_sell).fillna(False)] = -1
    return signal


templates.register(
    "regime_adaptive",
    defaults={
        "vr_period": 60.0,
        "vr_k": 4.0,
        "vr_trend_threshold": 1.2,
        "vr_revert_threshold": 0.8,
        "mom_period": 20.0,
        "rsi_period": 14.0,
        "rsi_lower": 30.0,
        "rsi_upper": 70.0,
        "sl_pips": 30.0,
        "tp_pips": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_regime_adaptive,
)
