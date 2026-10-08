"""ORZ手法 H1トレンドフォロー押し目/戻り目 テンプレート(ORZ氏ブログの機械化)。

出典: ORZ手法(トレーダーORZ氏がブログで解説したH1順張り押し目/戻り目手法)を復元・機械化。
本質は「3本のSMA(20/50/100)で確立したトレンドの中で、価格がSMA20/50帯まで押し戻った
ところを、下位足の反転パターン確定を待って順張りで仕掛ける」もの。ターゲットは60-80pips、
損切りは押し目/戻り目を作った直近の抵抗/支持の外側に置く、という設計。

核となる判定:
  1. トレンド確立(パーフェクトオーダー): 上昇=sma20>sma50>sma100 かつ close>sma100
     (逆順が下降)。3本のSMAが同順に整列した強いトレンドのみ対象とする。
  2. 押し目/戻り目: トレンド方向に対し価格が中期(sma50)帯まで押し戻ったこと。
  3. エントリートリガ: 押し目形成後、トレンド方向へダウ理論の継続(直近スイング高安の更新)
     が確定した最初のバー。

近似(docstring明記):
  - 【一目雲を省略】原手法はSMA3本に加え一目均衡表の雲(先行スパンA/B)をトレンド/支持抵抗の
    フィルタに使うが、本テンプレでは雲を省略した(近似)。雲は「価格>雲」を上昇フィルタに
    足す形で将来拡張しうるが、先行スパンは26本先行=描画上のシフトのみで値自体は確定バーから
    計算されるため先読みではない。現状はSMA100(close>sma100)を大局フィルタで代理する。
  - 【押し目形成を近似】「価格がSMA20/50帯へ押し戻った」を、直近pull_lb本の安値最小が
    (sma50 + band_atr×ATR)以下(上昇)/直近pull_lb本の高値最大が(sma50 - band_atr×ATR)
    以上(下降)で代理する。厳密なSMA20とSMA50の間への接触判定ではなくATRバンドで緩めた近似。
  - 【下位足の反転パターン確定を近似】原手法は執行足(下位足)でのプライスアクション反転
    パターン(ピンバー/包み足等)を待つが、本テンプレは当TF上での「ダウ継続ブレイク」=直近
    dow_lb本のスイング高安の更新(donchian, 内部shift(1)=当バー除外で因果)で代理する。
  - 【決済を固定pips化】原手法のTP=60-80pips・SL=抵抗/支持の外側という構造的決済は、Stage1
    では固定pips(sl_pips/tp_pips)で近似する。engine側がSL/TPを適用する。

先読み規律: SMA/ATR/donchianは確定バー+自分の過去のみを参照。donchianは当バー除外済みで
先読みなし。トリガの.shift(1)は自分の過去バー参照のみ(再アーム用)。翌バー始値執行のshiftは
engine.run_backtest側(raw_signal.shift(1))が担い、テンプレ内では執行用shiftをしない。
ウォームアップNaNは比較でFalseに落ち(最終マスクは.fillna(False)済み)、シグナル無しとなる。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, donchian_channel, sma
from app.core.strategy_model import Strategy


def _signal_orz_pullback(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ma_fast = max(2, int(p["ma_fast"].value))       # 短期SMA(既定20)
    ma_mid = max(2, int(p["ma_mid"].value))         # 中期SMA(既定50)=押し目帯の基準
    ma_slow = max(2, int(p["ma_slow"].value))       # 長期SMA(既定100)=大局フィルタ
    pull_lb = max(1, int(p["pull_lb"].value))       # 押し目形成(sma50帯接触)の遡及本数
    band_atr = float(p["band_atr"].value)           # 押し目帯の許容幅(×ATR)
    dow_lb = max(2, int(p["dow_lb"].value))         # ダウ継続ブレイクのスイング参照本数
    atr_period = max(2, int(p["atr_period"].value))
    dir_mode = int(p["dir_mode"].value)             # 0=both,1=long,2=short

    high, low, close = df["high"], df["low"], df["close"]
    fast = sma(close, ma_fast)
    mid = sma(close, ma_mid)
    slow = sma(close, ma_slow)
    band = band_atr * atr(high, low, close, atr_period)
    prior_high, prior_low = donchian_channel(high, low, dow_lb)  # 当バー除外の直近スイング高安

    # トレンド確立(パーフェクトオーダー): 3本SMAが同順に整列 + 大局フィルタ
    trend_up = (fast > mid) & (mid > slow) & (close > slow)
    trend_dn = (fast < mid) & (mid < slow) & (close < slow)
    # 押し目/戻り目: 直近pull_lb本で価格が中期(sma50)帯へ押し戻した
    pulled_up = low.rolling(pull_lb).min() <= (mid + band)
    pulled_dn = high.rolling(pull_lb).max() >= (mid - band)
    # エントリートリガ: ダウ継続(直近スイング高安の更新)
    break_up = high > prior_high
    break_dn = low < prior_low

    long_raw = (trend_up & pulled_up & break_up).fillna(False)
    short_raw = (trend_dn & pulled_dn & break_dn).fillna(False)
    # 再アーム: 連続ブレイク発火を1回に絞る(離れて再度発火した最初のバーのみ)
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "orz_pullback",
    defaults={
        "ma_fast": 20.0, "ma_mid": 50.0, "ma_slow": 100.0,
        "pull_lb": 20.0, "band_atr": 0.5, "dow_lb": 10.0, "atr_period": 14.0,
        "dir_mode": 0.0, "sl_pips": 40.0, "tp_pips": 70.0, "lot": 0.1,
    },
    signal_fn=_signal_orz_pullback,
)
