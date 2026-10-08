"""200MA押し目/戻りテンプレート(裁量トレーダー記事の機械化シリーズ)。

出典: ザイFX ramenKing記事(https://zai.diamond.jp/articles/-/433417)。200SMA/200EMAで
トレンド方向を判定し、価格がそのMAへ押し戻ったところで順張りエントリーする裁量手法
(日足で方向判断→H4/H1でエントリー、2-3回分割建て)。累計実績は謳われているが
バックテスト数値は非公開で、過去11年間通期でプラスの裏付けはなく、直近の一方向
トレンド(2022-23年USD/JPY)に乗った逸話の域を出ない。

本テンプレートはこの記事から「トレンド方向に、価格が200MAへ押し戻った場面で順張り
エントリーする」という検証可能な核だけを機械化したもの。待ち時間のスキャルピング・
建玉サイズの裁量・時間足切替の裁量・分割建て(ナンピン化)など再現不可能な部分は
捨てている。押し目買いはブレイク追いより約定価格が有利という構造上、生存戦略
プロファイル(長保有・遅いMA・トレンドフォロー・買いバイアス)に合致する狙いがある。

先読み規律: signal_fnはshiftしない生シグナル(1/-1/0のpd.Series)を返すのみ。翌バー
始値執行のshiftはengine.run_backtest側(raw_signal.shift(1)、engine.py)が担うため、
テンプレート側ではshiftしない。各バーは自バーのOHLC + 自分の過去(shift(1)で得られる
値)のみを参照し、未来のバーは一切参照しない。MA/ATRウォームアップのNaNは比較で
Falseに落ち、シグナル無しとなる(最終マスクは.fillna(False)してから代入)。
"""

from __future__ import annotations

import pandas as pd

from app.core import templates
from app.core.indicators import atr, ema, sma
from app.core.strategy_model import Strategy


def _signal_ma_pullback(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    p = strategy.params
    ma_period = max(2, int(p["ma_period"].value))
    use_ema = int(p["use_ema"].value)  # 0=SMA, 1=EMA
    atr_period = max(2, int(p["atr_period"].value))
    touch_atr_mult = float(p["touch_atr_mult"].value)
    dir_mode = int(p["dir_mode"].value)  # 0=both, 1=long_only, 2=short_only

    line = ema(df["close"], ma_period) if use_ema == 1 else sma(df["close"], ma_period)
    band = touch_atr_mult * atr(df["high"], df["low"], df["close"], atr_period)

    up = df["close"] > line  # 上昇レジーム(自バー確定)
    dn = df["close"] < line  # 下降レジーム
    low_touch = df["low"] <= (line + band)  # 上昇中に価格が下からMA帯へ押し戻した
    high_touch = df["high"] >= (line - band)  # 下降中に価格が上からMA帯へ戻した

    long_raw = (up & low_touch).fillna(False)
    short_raw = (dn & high_touch).fillna(False)
    # 再アーム: MA帯に張り付く間の連続発火を防ぎ、離れて戻った最初のバーのみ発火
    long_sig = long_raw & (~long_raw.shift(1).fillna(False))
    short_sig = short_raw & (~short_raw.shift(1).fillna(False))

    signal = pd.Series(0, index=df.index, dtype=int)
    if dir_mode in (0, 1):
        signal[long_sig] = 1
    if dir_mode in (0, 2):
        signal[short_sig] = -1
    return signal


templates.register(
    "ma_pullback",
    defaults={
        "ma_period": 200.0, "use_ema": 0.0, "atr_period": 14.0,
        "touch_atr_mult": 0.5, "dir_mode": 0.0,
        "sl_pips": 100.0, "tp_pips": 300.0, "lot": 0.1,
    },
    signal_fn=_signal_ma_pullback,
)
