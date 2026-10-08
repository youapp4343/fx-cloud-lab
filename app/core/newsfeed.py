"""ニュースカレンダー(手動整備CSV)の読み込みと、バーがイベント近傍かどうかの判定モジュール。

信頼できる無料のリアルタイム経済ニュースAPIは実在しないため、本モジュールは
ユーザーが手動整備するCSVカレンダー(data/news/README.md 参照)を正式な入力とする設計。
自動スクレイピング等はこのツールでは実装しない。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NEWS_DIR = BASE_DIR / "data" / "news"

CALENDAR_COLUMNS = ["datetime_utc", "currency", "impact", "event_name"]
_IMPACT_RANK = {"high": 3, "medium": 2, "low": 1}


def _calendar_path() -> Path:
    # NEWS_DIRはテスト時にmonkeypatchされうるため、モジュールロード時に固定せず都度参照する。
    return NEWS_DIR / "calendar.parquet"


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """app/core/datafeed.py download()と同じ規約(UTC基準のtz-naive)に揃える。"""
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def import_calendar(csv_path: str) -> pd.DataFrame:
    """CSVカレンダー(列: datetime_utc,currency,impact,event_name)を読み込み正規化してparquetに保存する。

    列が不足していればValueErrorを送出する。datetime_utcはUTC基準tz-naiveのpandas datetime
    に変換し、昇順ソートしたうえで NEWS_DIR/calendar.parquet に保存し、DataFrameを返す。
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in CALENDAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSVに必須列が不足しています: {missing}"
            f"(必須列: {CALENDAR_COLUMNS}, 実際の列: {list(df.columns)})"
        )

    df = df[CALENDAR_COLUMNS].copy()
    df["datetime_utc"] = _to_naive_utc(df["datetime_utc"])
    df["currency"] = df["currency"].astype(str).str.upper()
    df["impact"] = df["impact"].astype(str).str.lower()
    df["event_name"] = df["event_name"].astype(str)
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    calendar_path = _calendar_path()
    df.to_parquet(calendar_path, index=False)
    return df


def load_calendar() -> pd.DataFrame:
    """NEWS_DIR/calendar.parquetを読み込んで返す。未取得ならFileNotFoundErrorを送出する。"""
    calendar_path = _calendar_path()
    if not calendar_path.exists():
        raise FileNotFoundError(
            f"ニュースカレンダーが見つかりません: {calendar_path} 。"
            "先に newsfeed.import_calendar(csv_path) でCSVカレンダーを取り込んでください"
            "(data/news/README.md 参照)。"
        )
    return pd.read_parquet(calendar_path)


def is_near_news(
    timestamps: pd.Series,
    calendar: pd.DataFrame,
    currencies: Optional[List[str]],
    min_impact: str,
    before_min: int,
    after_min: int,
) -> pd.Series:
    """timestamps(tz-naive UTC)の各要素がcalendar該当イベントの[-before_min,+after_min]分以内かを判定する。

    calendarが空、min_impact/currencies条件で該当イベントが0件の場合は全てFalseを返す
    (エラーにしない)。impactの強弱は high > medium > low として扱う
    (min_impact="high"ならhighのみ、"medium"ならhigh+medium、"low"なら全て)。
    """
    result = pd.Series(False, index=timestamps.index)
    if calendar is None or len(calendar) == 0:
        return result

    min_rank = _IMPACT_RANK.get(str(min_impact).lower(), _IMPACT_RANK["high"])
    impact_rank = calendar["impact"].astype(str).str.lower().map(_IMPACT_RANK).fillna(0)
    events = calendar[impact_rank >= min_rank]

    if currencies:
        wanted = {c.upper() for c in currencies}
        events = events[events["currency"].astype(str).str.upper().isin(wanted)]

    if len(events) == 0:
        return result

    ts = _to_naive_utc(timestamps)
    event_times = _to_naive_utc(events["datetime_utc"])
    before = pd.Timedelta(minutes=before_min)
    after = pd.Timedelta(minutes=after_min)

    mask = pd.Series(False, index=timestamps.index)
    for event_time in event_times:
        mask |= (ts >= event_time - before) & (ts <= event_time + after)

    return mask
