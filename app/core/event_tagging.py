"""手動イベントタグ(Phase9: 突発ニュース事後統計分析)。

ユーザー本来の要望である「SNS(X/Twitter等)発の突発ニュースに反応する売買」は、
リアルタイムで信頼できる無料のSNS API・十分な速度のインフラが存在しないため実現不可能
(本プロジェクトが一貫して採用している誠実な設計判断)。本モジュールはその代替として、
ユーザーが記憶・記録している過去の突発ニュース発生時刻を手動でCSVタグ付けし、
その前後の値動きを事後統計分析する機能を提供する。リアルタイムでSNSを監視して
自動タグ付けする機能ではない(data/events/README.md 参照)。

統計的な枠組みは app/core/discovery.py のローソク足パターン発掘(パターン出現後の
forward returnをt検定で検証)と同一であり、「パターン出現」の代わりに「ユーザー指定の
イベント時刻」を仮説の起点として使う。forward_bars分の未来を参照するのはdiscovery.py
同様、意図的な事後参照でありシミュレーション上の先読みバイアスとは別物。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVENTS_DIR = BASE_DIR / "data" / "events"

EVENT_COLUMNS = ["datetime_utc", "label"]

# discovery.py _DEFAULT_PIP_SIZEと同じ規約: analyze_event_reactionはsymbolを受け取らないため
# 非JPYペア想定のこの固定値でpipsを換算する。p値は正のスケーリングに不変なため、
# 統計的な有意性判定(p_value)には影響せず、mean_pips/std_pipsの表示スケールにのみ影響する。
_DEFAULT_PIP_SIZE = 0.0001

# ラベル別集計を表示する最低件数(要件で例示された「5件未満は省略」を採用)。
MIN_LABEL_SAMPLE = 5


def _events_path() -> Path:
    # EVENTS_DIRはテスト時にmonkeypatchされうるため、モジュールロード時に固定せず都度参照する。
    return EVENTS_DIR / "manual_events.parquet"


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """app/core/newsfeed.py _to_naive_utcと同じ規約(UTC基準のtz-naive)に揃える。"""
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def import_events(csv_path: str) -> pd.DataFrame:
    """CSV(列: datetime_utc, label)を読み込み、EVENTS_DIR/manual_events.parquetに保存する。

    列が不足していればValueErrorを送出する。datetime_utcはUTC基準tz-naiveのpandas datetime
    に変換し、昇順ソートしたうえで保存し、DataFrameを返す。labelは任意の文字列
    (例: "Trump tweet on tariffs")で、ラベル毎の分析比較(analyze_event_reactionのby_label)に使う。
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSVに必須列が不足しています: {missing}"
            f"(必須列: {EVENT_COLUMNS}, 実際の列: {list(df.columns)})"
        )

    df = df[EVENT_COLUMNS].copy()
    df["datetime_utc"] = _to_naive_utc(df["datetime_utc"])
    df["label"] = df["label"].astype(str)
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_events_path(), index=False)
    return df


def load_events() -> pd.DataFrame:
    """EVENTS_DIR/manual_events.parquetを読み込んで返す。未取得ならFileNotFoundErrorを送出する。"""
    path = _events_path()
    if not path.exists():
        raise FileNotFoundError(
            f"手動イベントタグが見つかりません: {path} 。"
            "先に event_tagging.import_events(csv_path) でCSVを取り込んでください"
            "(data/events/README.md 参照)。"
        )
    return pd.read_parquet(path)


def _one_sided_p_value(pips: np.ndarray) -> float:
    """平均pips>0を対立仮説とする片側t検定のp値(discovery.py _one_sided_p_valueと同じ考え方)。

    n<2または分散0の退化ケースはscipyのinf/nanを避け、有意性なしとしてp_value=1.0を返す。
    """
    if len(pips) < 2:
        return 1.0
    std = float(np.std(pips, ddof=1))
    if std == 0.0:
        return 1.0
    result = stats.ttest_1samp(pips, popmean=0.0, alternative="greater")
    return float(result.pvalue)


def _horizon_stats(pips: np.ndarray, forward_bars: int, note: str | None = None) -> dict:
    n = int(pips.size)
    result = {
        "forward_bars": int(forward_bars),
        "n": n,
        "mean_pips": float(np.mean(pips)) if n > 0 else 0.0,
        "p_value": _one_sided_p_value(pips),
        "std_pips": float(np.std(pips, ddof=1)) if n > 1 else 0.0,
    }
    if note is not None:
        result["note"] = note
    return result


def _pips_for_horizon(
    entry_idx: np.ndarray,
    direction_confirm_idx: np.ndarray,
    direction_sign: np.ndarray,
    close_arr: np.ndarray,
    n_bars: int,
    forward_bars: int,
) -> np.ndarray:
    """entry_idx(次バー始値)からforward_bars本後の終値までの、direction_sign方向のpips配列を返す。

    重要(FABLE監査で発見・修正した系統的バイアス対策): 損益の起点は「エントリー時点
    (open[entry_idx])」ではなく「方向確定時点(close[direction_confirm_idx])」にする。
    direction_signはentry_idx→direction_confirm_idxの値動きの符号で決まるため、損益も
    同じentry_idxを起点に測ると、「方向を決めた初動そのものの絶対値」が全ホライズンに
    定数バイアスとして加算され続け、純粋ランダムウォークのデータでも全ホライズンで
    人為的に有意な結果(p<<0.01)が出てしまうバグがあった(実測で確認・修正)。
    起点をdirection_confirm_idxにずらし「方向確定に使った区間」と「損益を測る区間」を
    独立にすることでこの水増しを防ぐ。呼び出し側でforward_bars<=direction_barsの
    ホライズンは意味を持たない(measurement区間が空になる)ため事前に除外すること。

    exit_idx = entry_idx + forward_bars が n_bars を超えるものは計算不能なため除外する
    (discovery.py _forward_returns_pipsと同じ「将来バーが存在しないだけ」の理由による除外)。
    """
    exit_idx = entry_idx + forward_bars
    valid = exit_idx < n_bars
    base_price = close_arr[direction_confirm_idx[valid]]
    exit_price = close_arr[exit_idx[valid]]
    return (exit_price - base_price) * direction_sign[valid] / _DEFAULT_PIP_SIZE


def analyze_event_reaction(
    df: pd.DataFrame,
    events: pd.DataFrame,
    forward_bars_list: list[int] = [1, 3, 5, 10, 20],
    direction_bars: int = 1,
) -> dict:
    """イベントタグ時刻の直後(次バー始値)からのforward returnを事後統計分析する。

    各イベント時刻について、その時刻以前の最後のバー(直前バー)の次のバーの始値を
    エントリー価格とする(discovery.py _forward_returns_pipsと同じ「次バー始値」規約)。
    方向(買い/売り)はエントリーからdirection_bars本後の終値までの値動きの符号で決める
    (「イベント後最初の1本の初動方向に乗った場合、その後forward_bars_list各ホライズンで
    どれだけ伸びるか/反転するか」を見る設計)。方向を後から(イベント後の値動きから)決めるため
    厳密には売買シグナルではなく事後統計だが、discovery.pyのforward return分析
    (パターン出現後の将来バーを参照する)と同じ「意図的な事後参照であり、シミュレーション上の
    リアルタイム売買判断に用いる先読みバイアスとは別物」という位置づけである。

    各ホライズン(forward_bars_listの各値)について、方向に沿ったpipsの平均が0より大きい
    ことを対立仮説とする片側t検定(scipy.stats.ttest_1sampベース、_one_sided_p_valueは
    discovery.py _one_sided_p_valueと同じ考え方)のp値を付与する。

    イベントのdatetime_utcがdfの範囲外(直前バーが存在しない、または直前バーの次のバーが
    存在しない)場合はそのイベントをスキップし、除外はn_events_skippedに計上する
    (結果から静かに消すのではなく、件数として必ず可視化する)。

    symbolを受け取らないため、pipsはdiscovery.py _DEFAULT_PIP_SIZEと同じ0.0001固定
    (非JPYペア想定)で換算する。p値は正のスケーリングに不変なため、この選択は
    統計的な有意性判定には影響せず、mean_pips/std_pipsの表示スケールにのみ影響する。

    誠実性のnote: n_events_total件という少数サンプルでの統計は非常に不安定であり、
    discovery.pyのような多重検定補正の対象にできるほどの仮説数も無い(仮説数は
    forward_bars_list×labelの組合せ程度しかなく、パターン発掘の数百仮説スケールとは
    全く異なる)。よってここで返すp値はあくまで参考値であり、この結果をもって
    「これで戦略の優位性が証明された」と主張しないこと。

    戻り値: {"n_events_total": int, "n_events_matched": int, "n_events_skipped": int,
             "by_horizon": [{"forward_bars", "n", "mean_pips", "p_value", "std_pips"}, ...],
             "by_label": {ラベル文字列: {"n_events_matched": int, "by_horizon": [...]}}
             (件数がMIN_LABEL_SAMPLE未満のラベルは省略)}
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = _to_naive_utc(df["timestamp"]).to_numpy()
    open_arr = df["open"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    n_bars = len(df)

    n_events_total = len(events)
    if n_events_total == 0:
        empty_horizon = [_horizon_stats(np.empty(0, dtype=float), fb) for fb in forward_bars_list]
        return {
            "n_events_total": 0,
            "n_events_matched": 0,
            "n_events_skipped": 0,
            "by_horizon": empty_horizon,
            "by_label": {},
        }

    event_times = _to_naive_utc(events["datetime_utc"]).to_numpy()
    labels = events["label"].astype(str).to_numpy()

    # anchor_idx: イベント時刻以前の最後のバー(直前バー)のindex。見つからなければ-1。
    anchor_idx = np.searchsorted(timestamps, event_times, side="right") - 1
    entry_idx = anchor_idx + 1
    direction_exit_idx = entry_idx + direction_bars

    # 「dfの範囲外」(直前バーが無い/次バーが無い)と「方向判定用バーが無い」を併せてマッチ条件とする。
    matched_mask = (anchor_idx >= 0) & (entry_idx < n_bars) & (direction_exit_idx < n_bars)
    n_events_matched = int(matched_mask.sum())
    n_events_skipped = n_events_total - n_events_matched

    m_entry_idx = entry_idx[matched_mask]
    m_direction_exit_idx = direction_exit_idx[matched_mask]
    m_labels = labels[matched_mask]

    direction_change = close_arr[m_direction_exit_idx] - open_arr[m_entry_idx]
    direction_sign = np.where(direction_change >= 0, 1.0, -1.0)

    def _horizon_entry(entry_idx_arr, confirm_idx_arr, sign_arr, fb):
        # fb<=direction_barsは「方向を決めた区間そのもの」を測ることになり無意味
        # (常に非負のトートロジー、FABLE監査で発見)なので検定対象外として明示する。
        if fb <= direction_bars:
            return _horizon_stats(
                np.empty(0, dtype=float), fb, note="初動(direction_bars以下)のため検定対象外"
            )
        pips = _pips_for_horizon(entry_idx_arr, confirm_idx_arr, sign_arr, close_arr, n_bars, fb)
        return _horizon_stats(pips, fb)

    by_horizon = [
        _horizon_entry(m_entry_idx, m_direction_exit_idx, direction_sign, fb) for fb in forward_bars_list
    ]

    by_label: dict[str, dict] = {}
    label_counts = pd.Series(m_labels).value_counts()
    for label in sorted(str(lbl) for lbl in label_counts.index):
        label_mask = m_labels == label
        count = int(label_mask.sum())
        if count < MIN_LABEL_SAMPLE:
            continue
        label_entry_idx = m_entry_idx[label_mask]
        label_confirm_idx = m_direction_exit_idx[label_mask]
        label_direction_sign = direction_sign[label_mask]
        by_label[label] = {
            "n_events_matched": count,
            "by_horizon": [
                _horizon_entry(label_entry_idx, label_confirm_idx, label_direction_sign, fb)
                for fb in forward_bars_list
            ],
        }

    return {
        "n_events_total": n_events_total,
        "n_events_matched": n_events_matched,
        "n_events_skipped": n_events_skipped,
        "by_horizon": by_horizon,
        "by_label": by_label,
    }
