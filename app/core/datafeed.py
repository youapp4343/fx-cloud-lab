"""FXヒストリカルデータ取得・変換モジュール。

要: pip install dukascopy-python 後に実データで動作確認すること。
dukascopy_python.fetch(instrument, interval, offer_side, start, end) 相当の
公開API(PyPI: https://pypi.org/project/dukascopy-python/)を前提に実装している。
instrument定数名・interval定数名は同パッケージの実バージョンで変更される可能性があるため、
実行環境で `import dukascopy_python.instruments` して属性名を確認のうえ調整すること。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OHLC_DIR = BASE_DIR / "data" / "ohlc"

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# resample()向け: 正規化した足種別 -> pandas resampleルール
_RESAMPLE_RULES: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

# download()向け: 正規化した足種別 -> dukascopy_python.INTERVAL_* 属性名
_INTERVAL_ATTR: dict[str, str] = {
    "M1": "INTERVAL_MIN_1",
    "M5": "INTERVAL_MIN_5",
    "M15": "INTERVAL_MIN_15",
    "M30": "INTERVAL_MIN_30",
    "H1": "INTERVAL_HOUR_1",
    "H4": "INTERVAL_HOUR_4",
    "D1": "INTERVAL_DAY_1",
}


def _resolve_instrument(dukascopy_python: Any, symbol: str) -> Any:
    """symbol文字列(例: 'EURUSD')からdukascopy_python.instrumentsの定数を探す。"""
    instruments_mod = dukascopy_python.instruments
    normalized = symbol.upper().replace("/", "").replace("_", "")

    candidates: list[str] = []
    if len(normalized) == 6:
        pair = f"{normalized[:3]}_{normalized[3:]}"
        candidates += [
            f"INSTRUMENT_FX_MAJORS_{pair}",
            f"INSTRUMENT_FX_MINORS_{pair}",
            f"INSTRUMENT_FX_CROSSES_{pair}",
            f"INSTRUMENT_FX_EXOTICS_{pair}",
        ]

    for name in candidates:
        if hasattr(instruments_mod, name):
            return getattr(instruments_mod, name)

    # フォールバック: 属性名にsymbolが含まれるものを総当たりで探す
    for name in dir(instruments_mod):
        if name.startswith("INSTRUMENT_") and normalized in name.replace("_", ""):
            return getattr(instruments_mod, name)

    raise ValueError(f"dukascopyのinstrument定数が見つかりません: symbol={symbol}")


def download(symbol: str, start: date, end: date, timeframe: str = "m1") -> dict:
    """dukascopyから指定通貨ペア・期間のヒストリカルデータを取得しparquet保存する。

    要: pip install dukascopy-python 後に実データで動作確認すること。
    ネットワークエラー・インストール未了・instrument未解決などの失敗時は
    分かりやすいメッセージ付きの例外をraiseし直す。
    """
    try:
        import dukascopy_python
        import dukascopy_python.instruments  # noqa: F401 - サブモジュールを親パッケージ属性として使うため明示import必須
    except ImportError as exc:
        raise RuntimeError(
            "dukascopy-python がインストールされていません。"
            "`pip install dukascopy-python` を実行してください。"
        ) from exc

    tf_key = timeframe.upper()
    interval_attr = _INTERVAL_ATTR.get(tf_key)
    if interval_attr is None:
        raise ValueError(f"未対応のtimeframeです: {timeframe}")

    try:
        instrument = _resolve_instrument(dukascopy_python, symbol)
        interval = getattr(dukascopy_python, interval_attr)
        offer_side = dukascopy_python.OFFER_SIDE_BID

        start_dt = datetime(start.year, start.month, start.day)
        end_dt = datetime(end.year, end.month, end.day)

        df = dukascopy_python.fetch(
            instrument=instrument,
            interval=interval,
            offer_side=offer_side,
            start=start_dt,
            end=end_dt,
        )
    except Exception as exc:  # noqa: BLE001 - ネットワーク/API例外を包んで再送出
        raise RuntimeError(
            f"dukascopyからのデータ取得に失敗しました: symbol={symbol}, "
            f"start={start}, end={end}, timeframe={timeframe}: {exc}"
        ) from exc

    try:
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "timestamp" not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "timestamp"})
        if "volume" not in df.columns:
            df["volume"] = 0.0

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            # dukascopyはtz-aware(UTC)で返す。CSVインポート等のtz-naiveデータと
            # resample()で結合できるよう、UTC基準のtz-naiveに統一する。
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        df = df[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    except KeyError as exc:
        raise RuntimeError(
            f"dukascopyの返却データの列が想定と異なります(実際の列: {list(df.columns)}): {exc}"
        ) from exc

    out_dir = RAW_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{start}_{end}.parquet"
    df.to_parquet(out_path, index=False)

    return {
        "symbol": symbol,
        "start": str(start),
        "end": str(end),
        "timeframe": timeframe,
        "rows": len(df),
        "path": str(out_path),
    }


def resample(symbol: str, timeframe: str) -> pd.DataFrame:
    """data/raw/{symbol}/配下の1分足データを結合し指定timeframeにリサンプルする。"""
    tf_key = timeframe.upper()
    rule = _RESAMPLE_RULES.get(tf_key)
    if rule is None:
        raise ValueError(f"未対応のtimeframeです: {timeframe}")

    raw_dir = RAW_DIR / symbol
    files = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []
    if not files:
        raise FileNotFoundError(f"生データが見つかりません: {raw_dir}")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")

    resampled = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()

    OHLC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OHLC_DIR / f"{symbol}_{tf_key}.parquet"
    resampled.to_parquet(out_path, index=False)

    return resampled


def import_csv(
    file_path: str,
    symbol: str,
    column_mapping: dict,
    timeframe: str = "m1",
) -> pd.DataFrame:
    """手動CSVインポート。column_mappingで任意のCSV列名をOHLCV標準名に正規化する。

    column_mapping例: {"timestamp":"Date","open":"Open","high":"High",
                        "low":"Low","close":"Close","volume":"Volume"}
    """
    df = pd.read_csv(file_path)

    missing = [csv_col for csv_col in column_mapping.values() if csv_col not in df.columns]
    if missing:
        raise ValueError(f"CSVに指定された列が見つかりません: {missing}")

    inverse_mapping = {csv_col: std_col for std_col, csv_col in column_mapping.items()}
    df = df.rename(columns=inverse_mapping)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_std = [c for c in required if c not in df.columns]
    if missing_std:
        raise ValueError(f"column_mappingに必須列が不足しています: {missing_std}")

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    tf_key = timeframe.upper()
    if tf_key == "M1":
        out_dir = RAW_DIR / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        start = df["timestamp"].min().date()
        end = df["timestamp"].max().date()
        out_path = out_dir / f"{symbol}_{start}_{end}_import.parquet"
    else:
        OHLC_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OHLC_DIR / f"{symbol}_{tf_key}.parquet"

    df.to_parquet(out_path, index=False)
    return df


def list_available() -> dict:
    """data/raw/とdata/ohlc/をスキャンして利用可能なデータ一覧を返す。"""
    raw_list: list[dict] = []
    if RAW_DIR.exists():
        for symbol_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
            files = sorted(f.name for f in symbol_dir.glob("*.parquet"))
            if files:
                raw_list.append({"symbol": symbol_dir.name, "files": files})

    ohlc_list: list[dict] = []
    if OHLC_DIR.exists():
        for f in sorted(OHLC_DIR.glob("*.parquet")):
            stem = f.stem
            if "_" in stem:
                symbol, timeframe = stem.rsplit("_", 1)
            else:
                symbol, timeframe = stem, ""
            ohlc_list.append({"symbol": symbol, "timeframe": timeframe})

    return {"raw": raw_list, "ohlc": ohlc_list}
