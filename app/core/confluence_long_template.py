"""多要素合議ロング・スキャル(confluence_long)テンプレート — 事前登録検証専用。

背景(重要): 本テンプレは「トレンド判定+抵抗帯判定+一目均衡表の雲+MTF判定を1通貨に複数
ロジックで合議(AND/多数決)すればエッジが製造されるか」という仮説の最終確認として実装する。
構成要素は個別に全てnull実証済み: 一目均衡表 → cci_ichimoku_template null、押し目/トレンド
順張り → 8クラス全滅(project_ml_factor_round1等)、抵抗帯ブレイク → break_retest_template
0/216生存、MTF合議 → three_ducks_template null。事前予測は強いnull(合議はトレード数を
減らすだけで期待値を作らない)だが、scripts/sweep_confluence_long.py で誠実に測定する。

シグナル構成(全て確定バー情報のみ、engine.run_backtestがshift(1)で翌バー始値執行する。
本テンプレート内では実行用シフトをしない):

- F1 日足トレンド: 同一銘柄のD1 parquet(data/ohlc/{symbol}_D1.parquet)を読み、
  close > SMA(sma_d1=200) かつ SMA上向き(SMA[t] > SMA[t-5])をD1バー自身の確定値で計算した後、
  1本shiftして「そのD1バーが開始した時点までに確定済みだった直近D1バーの値」に置き換えてから
  merge_asof(direction="backward")でM5の各バーへ割当てる(engine.py の mtf_confirm フィルタと
  同一パターン)。D1バーhの値はhバー自身の終値で計算するため、そのバーの期間が終わるまでは
  確定しない。shift(1)によって「1本前のD1バー(=現在のD1バー開始時刻までに終値確定済み)」の
  値のみを使うことで、当日の未確定な日足を先読みしない。
- F2 一目の雲: 当該TF(M5)自身で標準パラメータ(転換9/基準26/先行スパンB52、雲は26先行
  シフト)のローカル一目実装(indicators.pyに一目関数がないため本ファイル内に実装、
  cci_ichimoku_templateと同方式)。現在バーに表示される雲は cloud_shift 本前に確定した
  先行スパンA/B(spanA_raw/spanB_rawをcloud_shift本shiftした過去参照値)。close > 雲上端 で
  「価格が雲の上」と判定する。
- F3 抵抗帯ブレイク: 直近 res_lookback=100 本の高値(rolling max)を1本shiftして当バーを
  除外し、現在のcloseがそれを上抜けたら発火。
- F4 MTF一致: 同一銘柄のH1 parquetを読み、close > EMA(mtf_ema=50) をH1バー自身の確定値で
  計算し、F1と同じ shift(1) + merge_asof(backward) パターンでM5へ割当てる。
- n_required: F1〜F4のうち何個が真かの合議数で発火(4=全AND、3=3個以上の多数決)。

出口はengineのsl_pips/tp_pips(RR1:1、sl_pips==tp_pips)・max_hold_barsに委譲する。
ロングオンリー(signal=1のみ、ショートなし)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from app.core import templates
from app.core.indicators import ema, sma
from app.core.strategy_model import Strategy

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OHLC_DIR = BASE_DIR / "data" / "ohlc"  # sweep実行中は複数銘柄×複数run読むためキャッシュする

_HTF_CACHE: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}


def _load_htf(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """{symbol}_{timeframe}.parquet(data/ohlc/、読み取り専用)をキャッシュ付きで読む。

    未取得の場合はNoneを返す(呼び出し側でF1/F4を全Falseにフォールバックし、
    バックテスト自体は失敗させない。engine.py の mtf_confirm フィルタと同じ方針)。
    """
    key = (symbol, timeframe)
    if key not in _HTF_CACHE:
        path = OHLC_DIR / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            _HTF_CACHE[key] = None
        else:
            htf = pd.read_parquet(path)
            htf["timestamp"] = pd.to_datetime(htf["timestamp"])
            _HTF_CACHE[key] = htf.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return _HTF_CACHE[key]


def _confirmed_htf_bool(base_ts: pd.Series, htf_df: pd.DataFrame, raw_bool: pd.Series) -> pd.Series:
    """上位足の真偽値系列(raw_bool、そのバー自身の終値で計算した確定値)を、下位足の各バーへ
    「その時点で確定済みだった直近上位足バーの値」として割当てる(先読み回避)。

    手順: raw_bool.shift(1)で「1本前の上位足バー(=現在の上位足バーが開始した時刻までに
    終値が確定済み)」の値に置き換えてから、merge_asof(direction="backward")で下位足の各
    バーへ「その下位足バー時刻以下で最も新しい上位足バー時刻」の値を割当てる。

    先読みチェック: 割当てに使われた上位足バーの生タイムスタンプ(htf_ts、shift前の元の
    バー時刻ラベル)は、必ずbase_tsの対応する下位足バー時刻以下でなければならない
    (assertで検証、違反時は例外)。htf_ts == base_ts のケースは、shift(1)により
    そのhtf_ts時点で"1本前のバーの確定済み終値"のみを使っているため安全(そのバーが
    開始した瞬間に既に確定している情報のみを参照)。
    """
    confirmed = raw_bool.astype(float).shift(1)
    lower_key = pd.DataFrame({"ts": base_ts.to_numpy(), "_pos": range(len(base_ts))})
    lower_key = lower_key.sort_values("ts", kind="stable")
    htf_key = pd.DataFrame({"htf_ts": htf_df["timestamp"].to_numpy(), "value": confirmed.to_numpy()})
    htf_key = htf_key.sort_values("htf_ts", kind="stable")
    merged = pd.merge_asof(lower_key, htf_key, left_on="ts", right_on="htf_ts", direction="backward")

    matched = merged["htf_ts"].notna()
    if matched.any():
        assert bool((merged.loc[matched, "htf_ts"] <= merged.loc[matched, "ts"]).all()), (
            "confluence_long: 上位足merge_asofが現バーtimestamp超過を参照しています(先読みバイアス)"
        )

    out = merged.sort_values("_pos", kind="stable")["value"].reset_index(drop=True)
    out.index = base_ts.index
    return out.fillna(0.0) > 0.5


def _ichimoku_cloud_now(
    high: pd.Series,
    low: pd.Series,
    tenkan_period: int,
    kijun_period: int,
    senkou_b_period: int,
    cloud_shift: int,
) -> Tuple[pd.Series, pd.Series]:
    """一目均衡表の先行スパンA/Bをローカル実装で計算し、現在バーに表示される雲の値を返す
    (cci_ichimoku_templateと同方式)。先行スパンは本来チャート上でcloud_shift本未来へ
    シフトして描画されるため、「現在バーの雲」はcloud_shift本前に確定したspanA_raw/
    spanB_raw(自分の過去のみを参照する.rolling()から算出、先読みなし)をcloud_shift本
    shiftした過去参照値。
    """
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2.0
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2.0
    cloud_a_now = span_a_raw.shift(cloud_shift)
    cloud_b_now = span_b_raw.shift(cloud_shift)
    return cloud_a_now, cloud_b_now


def _signal_confluence_long(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    """F1(日足トレンド)/F2(一目雲上抜け)/F3(抵抗帯ブレイク)/F4(H1 MTF一致)の合議。

    n_required個以上が真ならロングシグナル(1)、それ以外は0。ショートなし。
    """
    p = strategy.params
    sma_d1_period = max(2, int(p["sma_d1"].value))
    res_lookback = max(2, int(p["res_lookback"].value))
    mtf_ema_period = max(2, int(p["mtf_ema"].value))
    tenkan_period = max(2, int(p["tenkan"].value))
    kijun_period = max(2, int(p["kijun"].value))
    senkou_b_period = max(2, int(p["senkou_b"].value))
    cloud_shift = max(1, int(p["cloud_shift"].value))
    n_required = min(4, max(1, int(round(p["n_required"].value))))

    high, low, close, ts = df["high"], df["low"], df["close"], df["timestamp"]

    # --- F1: 日足トレンド(D1 SMA200 上向き) ---
    d1_df = _load_htf(strategy.symbol, "D1")
    if d1_df is not None and len(d1_df) > 0:
        d1_close = d1_df["close"]
        d1_sma = sma(d1_close, sma_d1_period)
        d1_up_raw = ((d1_close > d1_sma) & (d1_sma > d1_sma.shift(5))).fillna(False)
        f1 = _confirmed_htf_bool(ts, d1_df, d1_up_raw)
    else:
        f1 = pd.Series(False, index=df.index)

    # --- F2: 一目均衡表の雲上抜け(M5自TF) ---
    cloud_a, cloud_b = _ichimoku_cloud_now(high, low, tenkan_period, kijun_period, senkou_b_period, cloud_shift)
    cloud_top = pd.concat([cloud_a, cloud_b], axis=1).max(axis=1, skipna=False)
    f2 = (close > cloud_top).fillna(False)

    # --- F3: 抵抗帯ブレイク(直近res_lookback本高値、当バー除く) ---
    res_level = high.rolling(res_lookback, min_periods=res_lookback).max().shift(1)
    f3 = (close > res_level).fillna(False)

    # --- F4: MTF一致(H1 EMA50上) ---
    h1_df = _load_htf(strategy.symbol, "H1")
    if h1_df is not None and len(h1_df) > 0:
        h1_close = h1_df["close"]
        h1_ema = ema(h1_close, mtf_ema_period)
        h1_up_raw = (h1_close > h1_ema).fillna(False)
        f4 = _confirmed_htf_bool(ts, h1_df, h1_up_raw)
    else:
        f4 = pd.Series(False, index=df.index)

    votes = f1.astype(int) + f2.astype(int) + f3.astype(int) + f4.astype(int)
    long_raw = votes >= n_required

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[long_raw] = 1
    return signal


templates.register(
    "confluence_long",
    defaults={
        "sma_d1": 200.0,
        "res_lookback": 100.0,
        "mtf_ema": 50.0,
        "tenkan": 9.0,
        "kijun": 26.0,
        "senkou_b": 52.0,
        "cloud_shift": 26.0,
        "n_required": 4.0,
        "sl_pips": 10.0,
        "tp_pips": 10.0,
        "max_hold_bars": 60.0,
        "lot": 0.1,
    },
    signal_fn=_signal_confluence_long,
)
