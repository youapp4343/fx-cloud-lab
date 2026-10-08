"""月次クロスセクショナル/タイムシリーズ通貨ポートフォリオ バックテストエンジン。

仕様: docs/plan_xs_portfolio.md §2-4。既存の単一銘柄バーレベルエンジン
(app/core/engine.py)とは独立したモジュール(月次リバランス、複数通貨同時保有)。
このモジュールからengine.py側への依存・逆依存は無い。

先読み厳禁の中核規約(run_portfolio、仕様§3):
    月tに形成する目標ウェイト w_t = signal_fn(V, R, t) は、V/R の index<=t の
    行だけを参照してよい。R[t+1]・V[t+1]以降は絶対に参照しない。実現は
    R[t+1] で行う(w_tはt+1月保有分のウェイトとして使う)。run_portfolioは
    V/Rをスライスせずそのままsignal_fnに渡すため、この規約を守る責任は
    signal_fn自身にある(§4の3戦略はいずれも `_formation_return` 経由で
    V.iloc[t]/V.iloc[t-f]-1 のみを参照し、Rやt+1以降のVには触れない)。

戦略固有パラメータ(formation/k等)はfunctools.partialやlambdaで事前バインド
してからrun_portfolioのsignal_fn引数に渡すこと(run_portfolio自体は
signal_fn(V, R, t) の3引数のみで呼び出す、仕様§3のシグネチャ通り)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# 通貨コード -> (対USDペア名, 逆数フラグ)。逆数=USDXXXペアでforeign価値=1/close。
# 仕様§2の値をそのまま使用。
USD_PAIRS: dict[str, tuple[str, bool]] = {
    "EUR": ("EURUSD", False),
    "GBP": ("GBPUSD", False),
    "AUD": ("AUDUSD", False),
    "JPY": ("USDJPY", True),
    "CAD": ("USDCAD", True),
    "CHF": ("USDCHF", True),
}

# 片道スプレッド(pips、プロジェクト実勢)。仕様§2の値をそのまま使用。
SPREAD_PIPS: dict[str, float] = {
    "EUR": 0.6,
    "GBP": 0.9,
    "AUD": 0.9,
    "JPY": 0.7,
    "CAD": 1.0,
    "CHF": 0.8,
}


def _pip_size(pair: str) -> float:
    """pipサイズ: JPYクロス(USDJPY)は0.01、他は0.0001(仕様§2)。"""
    return 0.01 if pair == "USDJPY" else 0.0001


def load_monthly(ohlc_dir: str = "data/ohlc") -> tuple[pd.DataFrame, pd.DataFrame]:
    """USD_PAIRSの6ペアのD1 parquetから月次のV(通貨価値)・price_pair(生ペア価格)を作る。

    各ペアの{pair}_D1.parquetを読み、timestamp索引でcloseを月末リサンプル
    (resample("ME").last())。foreign価値 v_c = 1/close if invert else close。
    price_pair_c は逆数化しない生ペア価格(コスト計算用)。
    全通貨を共通の月末索引で内部結合(inner)し、いずれかがNaNの月は落とす。

    Returns:
        V: index=月末、columns=6通貨コード(USD_PAIRS順)、値=1単位外貨のUSD価値。
        price_pair: 同index・同columns、値=生ペア価格。
    """
    base_dir = Path(ohlc_dir)
    codes = list(USD_PAIRS.keys())

    v_series: dict[str, pd.Series] = {}
    price_series: dict[str, pd.Series] = {}
    for code in codes:
        pair, invert = USD_PAIRS[code]
        path = base_dir / f"{pair}_D1.parquet"
        raw = pd.read_parquet(path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw = raw.set_index("timestamp").sort_index()
        monthly_close = raw["close"].resample("ME").last()
        price_series[code] = monthly_close
        v_series[code] = (1.0 / monthly_close) if invert else monthly_close

    # 共通月末索引で内部結合(inner)。resample自体が生む欠測(その月にデータ皆無)
    # に備え、結合後さらにdropnaで「いずれかがNaNの月」を落とす。
    V = pd.concat(v_series, axis=1, join="inner")[codes].dropna(how="any")
    price_pair = pd.concat(price_series, axis=1, join="inner")[codes]
    price_pair = price_pair.loc[V.index]

    return V, price_pair


def monthly_returns(V: pd.DataFrame) -> pd.DataFrame:
    """月次保有リターン。R.loc[m,c] = 月mの通貨c保有リターン(m-1末→m末)。先頭行NaN。"""
    return V.pct_change()


def cost_oneway(price_pair: pd.DataFrame) -> pd.DataFrame:
    """片道コスト率(月次リターン尺度、通貨別・月別)。仕様§2。

    cost_oneway[c,m] = SPREAD_PIPS[c] * pip_size(pair_c) / price_pair[c,m]
    (名目に対するスプレッド割合。逆数化しても一次で不変)。
    """
    codes = list(price_pair.columns)
    out = {}
    for code in codes:
        pair, _invert = USD_PAIRS[code]
        out[code] = SPREAD_PIPS[code] * _pip_size(pair) / price_pair[code]
    return pd.DataFrame(out, index=price_pair.index)[codes]


# ---------------------------------------------------------------------------
# §4 戦略(signal_fn)。各関数は V.iloc[t] 以下の行のみを参照し(t末情報)、
# 長さ6の符号付きndarrayを返す。formation/k等の戦略固有パラメータはpartial等で
# 事前バインドしてrun_portfolioに渡す。
# ---------------------------------------------------------------------------


def _formation_return(V: pd.DataFrame, t: int, formation: int) -> pd.Series:
    """形成fヶ月リターン: V.iloc[t]/V.iloc[t-formation]-1(水準比、複利バグ回避)。

    t-formation<0はデータが無い(pandasの負インデックスは末尾へラップアラウンド
    =未来行を暗黙参照してしまう先読みバグの温床)ため、ValueErrorで明示的に拒否する。
    """
    if t - formation < 0:
        raise ValueError(
            f"formation={formation}に対してt={t}が小さすぎます(t-formation<0、"
            "warmupがformation/lookbackより小さい可能性)"
        )
    return V.iloc[t] / V.iloc[t - formation] - 1.0


def xs_momentum(V: pd.DataFrame, R: pd.DataFrame, t: int, formation: int = 1, k: int = 2) -> np.ndarray:
    """クロスセクショナル・モメンタム(仕様§4-1)。t末情報のみ使用。

    各通貨のformationヶ月形成リターンで順位付け。上位k=+1/(2k)、下位k=-1/(2k)、
    中間0。(6通貨ではk=1が1/6分位、k=2はtercile相当)
    """
    n_assets = V.shape[1]
    if not (1 <= k <= n_assets // 2):
        raise ValueError(f"kは1〜{n_assets // 2}の範囲で指定してください: k={k}")

    form_ret = _formation_return(V, t, formation)
    rank_desc = form_ret.rank(method="first", ascending=False)  # 1=最高リターン

    w = pd.Series(0.0, index=V.columns)
    w[rank_desc <= k] = 1.0 / (2 * k)
    w[rank_desc > n_assets - k] = -1.0 / (2 * k)
    return w.to_numpy()


def ts_momentum(V: pd.DataFrame, R: pd.DataFrame, t: int, formation: int = 1) -> np.ndarray:
    """タイムシリーズ・モメンタム(仕様§4-2)。t末情報のみ使用。

    各通貨sign(formationヶ月形成リターン)。ウェイト=sign/6(常に全6通貨を
    ロング/ショート、gross=Σ|w|=1)。
    """
    form_ret = _formation_return(V, t, formation)
    n_assets = V.shape[1]
    sign = np.sign(form_ret.to_numpy())
    return sign / n_assets


def composite_trend(V: pd.DataFrame, R: pd.DataFrame, t: int) -> np.ndarray:
    """複合トレンド(仕様§4-3)。t末情報のみ使用。

    各通貨 s_c = sign(1moret)+sign(3moret)+sign(12moret) ∈{-3..3}。
    ウェイト∝s_c、Σ|w|=1へ正規化(全0なら全0)。
    """
    r1 = _formation_return(V, t, 1)
    r3 = _formation_return(V, t, 3)
    r12 = _formation_return(V, t, 12)
    s = np.sign(r1.to_numpy()) + np.sign(r3.to_numpy()) + np.sign(r12.to_numpy())
    total = float(np.abs(s).sum())
    if total == 0:
        return np.zeros_like(s, dtype=float)
    return s / total


def _carry_diff(rates: pd.DataFrame, columns: pd.Index, t: int) -> pd.Series:
    """t末に既知の対USD金利差 rate_c - rate_USD(cはV.columnsの6通貨)。

    rates は run_portfolio に渡す V と同一index(月末)へ整列済みのDataFrameで、
    columns に6通貨 + "USD" を含むこと。rates.iloc[t] は月tの金利観測=月t末に
    既知なので、これで月t+1のポジションを組むのは先読みにならない
    (形成リターンが V.iloc[t] までを使うのと同じ規律)。
    """
    row = rates.iloc[t]
    return row[columns] - row["USD"]


def carry_rank(V: pd.DataFrame, R: pd.DataFrame, t: int, *, rates: pd.DataFrame, k: int = 2) -> np.ndarray:
    """キャリー(実装方法B: 金利差ランキング、仕様§4-4)。t末情報のみ使用。

    対USD金利差 rate_c - rate_USD で6通貨を順位付け。上位k=+1/(2k)、下位k=-1/(2k)。
    rates は V と同一indexへ整列済み・"USD"列を含むこと(実金利注入前提、手打ち近似禁止)。
    """
    n_assets = V.shape[1]
    if not (1 <= k <= n_assets // 2):
        raise ValueError(f"kは1〜{n_assets // 2}の範囲で指定してください: k={k}")
    diff = _carry_diff(rates, V.columns, t)
    rank_desc = diff.rank(method="first", ascending=False)  # 1=最高金利差
    w = pd.Series(0.0, index=V.columns)
    w[rank_desc <= k] = 1.0 / (2 * k)
    w[rank_desc > n_assets - k] = -1.0 / (2 * k)
    return w.to_numpy()


def carry_sign(V: pd.DataFrame, R: pd.DataFrame, t: int, *, rates: pd.DataFrame) -> np.ndarray:
    """キャリー(実装方法A: 金利差の符号、仕様§4-4)。t末情報のみ使用。

    各通貨 sign(rate_c - rate_USD)/6。金利>USDならロング、<USDならショート、全6通貨保有。
    """
    diff = _carry_diff(rates, V.columns, t).to_numpy()
    return np.sign(diff) / V.shape[1]


def combo_carry_mom(V: pd.DataFrame, R: pd.DataFrame, t: int, *, rates: pd.DataFrame, formation: int = 1) -> np.ndarray:
    """キャリー50% + TSモメンタム50%(仕様§5の推奨構成)。t末情報のみ使用。

    w = 0.5*carry_sign + 0.5*ts_momentum。両者とも符号/6でΣ|w|≈1のため合成もその規模。
    相関の低い2シグナルの合成でvol低下・Sharpe改善を狙う設計。
    """
    return 0.5 * carry_sign(V, R, t, rates=rates) + 0.5 * ts_momentum(V, R, t, formation=formation)


def load_daily_returns(ohlc_dir: str = "data/ohlc") -> pd.DataFrame:
    """6通貨の日次リターン(対USD、V変換=逆数化込み)。research/vol_spillover/PREREG.md §シグナル計算1。

    load_monthly()と同じUSD_PAIRS変換(invertならUSDXXXの1/close)を日次closeへ適用し、
    pct_change()で日次リターン系列を作る。伝播シグナルのVAR/FEVD推定はこの日次系列の
    ローリング窓のみを参照する(月次Vとは別ロード、月末までの確定日次バーのみ使うのは
    呼び出し側=vol_spilloverの責務)。
    """
    base_dir = Path(ohlc_dir)
    codes = list(USD_PAIRS.keys())
    v_daily: dict[str, pd.Series] = {}
    for code in codes:
        pair, invert = USD_PAIRS[code]
        raw = pd.read_parquet(base_dir / f"{pair}_D1.parquet")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw = raw.set_index("timestamp").sort_index()
        v_daily[code] = (1.0 / raw["close"]) if invert else raw["close"]
    V_daily = pd.concat(v_daily, axis=1, join="inner")[codes].dropna(how="any")
    return V_daily.pct_change().dropna(how="any")


def _fevd_generalized(daily_window: pd.DataFrame, p: int, H: int) -> np.ndarray:
    """Pesaran-Shin一般化FEVD(H期先)。research/vol_spillover/PREREG.md記載の式をそのまま実装。

    daily_window: 直近window日分の確定日次リターン(先読み禁止は呼び出し側の責務)。
    VAR(p)をstatsmodelsで推定し、係数行列(MA表現 A_h, h=0..H-1)と残差共分散Σを取得。
        θ_ij(H) = (σ_jj^-1 * Σ_h (e_i' A_h Σ e_j)^2) / (Σ_h e_i' A_h Σ A_h' e_i)
        θ~_ij = θ_ij / Σ_j θ_ij   (行方向に正規化)
    Returns: θ~ (n x n ndarray, 行=i変数の予測誤差分散、列=j変数由来の寄与割合)。
    健全性: 独立な系列ならθ~≈diag(1)(スクリプト側test_fevd.pyで確認済み)。
    """
    n = daily_window.shape[1]
    res = VAR(daily_window.to_numpy()).fit(p, trend="c")
    Sigma = res.sigma_u  # (n,n) 残差共分散
    ma = res.ma_rep(maxn=H - 1)  # (H,n,n)、ma[0]=単位行列
    sigma_diag = np.diag(Sigma)

    theta = np.zeros((n, n))
    denom = np.zeros(n)
    for i in range(n):
        d = 0.0
        for h in range(H):
            Ah = ma[h]
            v = Ah @ Sigma @ Ah.T
            d += v[i, i]
        denom[i] = d
    for i in range(n):
        for j in range(n):
            num = 0.0
            for h in range(H):
                Ah = ma[h]
                num += (Ah @ Sigma)[i, j] ** 2
            theta[i, j] = (num / sigma_diag[j]) / denom[i] if denom[i] > 0 else 0.0

    row_sums = theta.sum(axis=1, keepdims=True)
    return np.divide(theta, row_sums, out=np.zeros_like(theta), where=row_sums > 0)


def _spillover_to_others(theta_norm: np.ndarray) -> np.ndarray:
    """通貨jの「他者への伝播」= Σ_{i≠j} θ~_ij(列jを行方向に合計、対角除く)。PREREG §シグナル計算4。"""
    return theta_norm.sum(axis=0) - np.diag(theta_norm)


def vol_spillover(
    V: pd.DataFrame,
    R: pd.DataFrame,
    t: int,
    *,
    V_daily: pd.DataFrame,
    window: int,
    p: int,
    H: int,
    k: int,
    cache: dict | None = None,
) -> np.ndarray:
    """ボラティリティ・ショック伝播ランキング(research/vol_spillover/PREREG.md)。t末情報のみ使用。

    V.index[t](月末)までの確定日次リターン(V_daily)の直近window営業日でVAR(p)を推定し、
    H期先Pesaran-Shin一般化FEVDで各通貨の「他者への伝播」を計算。伝播が最も弱いk通貨を
    ロング(+1/2k)・最も強いk通貨をショート(-1/2k)、残りはフラット。

    cache: {(month_end, window, p, H): spillover ndarray} の辞書を渡すと、同一(window,p,H,月)の
    VAR/FEVD計算を使い回す(グリッド探索でk違いの3通りを毎回再計算しない高速化。呼び出し側の
    責務で、Noneなら常に再計算=正しさに影響しない)。
    """
    n_assets = V.shape[1]
    if not (1 <= k <= n_assets // 2):
        raise ValueError(f"kは1〜{n_assets // 2}の範囲で指定してください: k={k}")

    month_end = V.index[t]
    cache_key = (month_end, window, p, H)
    if cache is not None and cache_key in cache:
        spill = cache[cache_key]
    else:
        daily_upto = V_daily.loc[:month_end]
        if len(daily_upto) < window:
            raise ValueError(
                f"window={window}に対し日次データ不足: t={t}({month_end.date()}), "
                f"利用可能{len(daily_upto)}日"
            )
        win = daily_upto.iloc[-window:][list(V.columns)]
        theta_norm = _fevd_generalized(win, p=p, H=H)
        spill = _spillover_to_others(theta_norm)
        if cache is not None:
            cache[cache_key] = spill

    spill_s = pd.Series(spill, index=V.columns)
    rank_asc = spill_s.rank(method="first", ascending=True)  # 1=最弱伝播者
    w = pd.Series(0.0, index=V.columns)
    w[rank_asc <= k] = 1.0 / (2 * k)  # 弱い伝播者=ロング
    w[rank_asc > n_assets - k] = -1.0 / (2 * k)  # 強い伝播者=ショート
    return w.to_numpy()


def value_signal(cpi_df: pd.DataFrame, V: pd.DataFrame, t: int) -> np.ndarray:
    """Phase2用: バリュー(実質為替)シグナル(仕様§4-5、未実装スタブ)。

    実質為替(名目×相対CPI)の5年変化で割安通貨を買う設計。
    実CPIデータの注入が前提。Phase1では未使用。
    """
    raise NotImplementedError(
        "value_signalはPhase2(実CPIデータ注入後)まで未実装です。手打ち近似での代用は禁止。"
    )


# ---------------------------------------------------------------------------
# §3 バックテストコア
# ---------------------------------------------------------------------------


def _max_drawdown(monthly: pd.Series) -> float:
    """累積(複利)ネットリターンからの最大ドローダウン(負の値、例: -0.23=-23%)。

    仕様§3はmax_drawdownを必須statsキーとして列挙するのみで厳密な定義式は
    明記していないため、標準的な「累積ウェルスのピーク比」定義を採用する。
    """
    if len(monthly) == 0:
        return 0.0
    wealth = (1.0 + monthly).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def _compute_stats(monthly: pd.Series, turnover: pd.Series) -> dict[str, float]:
    ann_return = float(monthly.mean() * 12)
    ann_vol = float(monthly.std(ddof=1) * np.sqrt(12))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(monthly),
        "skew": float(monthly.skew()),
        "avg_turnover": float(turnover.mean()),
        "hit_rate": float((monthly > 0).mean()),
        "n_months": int(len(monthly)),
    }


def run_portfolio(
    signal_fn: Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray],
    V: pd.DataFrame,
    R: pd.DataFrame,
    price_pair: pd.DataFrame,
    *,
    weighting: str = "equal",
    target_vol: float | None = None,
    apply_cost: bool = True,
    warmup: int = 13,
) -> dict[str, Any]:
    """月次クロスセクショナル/TSポートフォリオのバックテストコア(仕様§3)。

    月tをwarmupからT-1(T=最終月index=len(V)-1)まで進める。各月t:
      形成: w_t = signal_fn(V, R, t) (index<=tの行だけ参照。R[t+1]は絶対に使わない)
      実現: gross[t+1] = Σ_c w_t[c]*R[t+1,c]
      コスト: turnover[t+1]=Σ_c|w_t[c]-w_prev[c]|、
              cost[t+1]=Σ_c|w_t[c]-w_prev[c]|*cost_oneway[c,t+1](apply_cost時)。初回w_prev=0。
      net[t+1] = gross[t+1] - cost[t+1]

    weighting="equal"はsignal_fnの生ウェイトをそのまま使う。"invvol"は生ウェイトに
    1/σ_c(直近12ヶ月Rの標準偏差、t末まで)を掛けてΣ|w|を元の合計に再正規化する。

    target_vol指定時は各月のポジションを
    min(1, target_vol/(√12・σ_strat_trailing))で縮小する。σ_strat_trailingは
    このrun_portfolio自身が生成したnetリターンの直近12ヶ月標準偏差をt末まで
    (ラグ)で計算する(先読みなし)。手前12ヶ月未満は縮小1.0。

    Returns:
        {"monthly": pd.Series(net, index=月t+1), "stats": {...}, "gross_monthly": ...}
    """
    if weighting not in ("equal", "invvol"):
        raise ValueError(f"未対応のweightingです: {weighting}")

    n_assets = V.shape[1]
    n = len(V)
    T = n - 1  # 最終月index

    cost_rate = cost_oneway(price_pair)

    w_prev = np.zeros(n_assets)
    net_history: list[float] = []  # target_vol縮小用。自身のnetリターンをラグ参照する

    idx: list[Any] = []
    net_list: list[float] = []
    gross_list: list[float] = []
    turnover_list: list[float] = []

    for t in range(warmup, T):
        raw_w = np.asarray(signal_fn(V, R, t), dtype=float)

        if weighting == "invvol":
            window = R.iloc[max(0, t - 11) : t + 1]
            sigma = window.std(ddof=1).to_numpy()
            inv = np.divide(1.0, sigma, out=np.zeros_like(sigma, dtype=float), where=sigma > 0)
            adj = raw_w * inv
            raw_abs_sum = np.abs(raw_w).sum()
            adj_abs_sum = np.abs(adj).sum()
            if raw_abs_sum > 0 and adj_abs_sum > 0:
                w_t = adj * (raw_abs_sum / adj_abs_sum)
            else:
                w_t = np.zeros(n_assets)
        else:  # "equal"
            w_t = raw_w

        if target_vol is not None:
            if len(net_history) >= 12:
                sigma_strat = float(np.std(net_history[-12:], ddof=1))
                scale = min(1.0, target_vol / (np.sqrt(12) * sigma_strat)) if sigma_strat > 0 else 1.0
            else:
                scale = 1.0
            w_t = w_t * scale

        diff = w_t - w_prev
        turnover = float(np.abs(diff).sum())
        gross = float(np.dot(w_t, R.iloc[t + 1].to_numpy()))
        cost = float(np.dot(np.abs(diff), cost_rate.iloc[t + 1].to_numpy())) if apply_cost else 0.0
        net = gross - cost

        idx.append(V.index[t + 1])
        net_list.append(net)
        gross_list.append(gross)
        turnover_list.append(turnover)
        net_history.append(net)
        w_prev = w_t

    result_index = pd.Index(idx)
    monthly = pd.Series(net_list, index=result_index)
    gross_monthly = pd.Series(gross_list, index=result_index)
    turnover_s = pd.Series(turnover_list, index=result_index)

    return {
        "monthly": monthly,
        "stats": _compute_stats(monthly, turnover_s),
        "gross_monthly": gross_monthly,
    }
