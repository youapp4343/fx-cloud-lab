"""時間帯別の実測コストを損益に反映する。

エンジンは銘柄ごとの固定スプレッド(config.SPREAD_PIPS)を建値に1回だけ課している。
ここでは各トレードについて「実測の時間帯別コスト − 固定スプレッド」を追加で差し引き、
固定値では見えないロールオーバー帯の拡大を損益に入れる。

- コスト = その時間の平均スプレッド(lab/spread_profile.json、時間加重) + 手数料
- 買いは建てた時刻、売りは決済した時刻のスプレッドを払う(バーはbid。askで約定する側が払う)
- 表の時間軸は「米国夏時間のUTC時」。ロールオーバー(NY17時)は夏はUTC21時、冬はUTC22時なので、
  米国標準時の期間は1時間ずらして引く(us_summer_hour)
- 未反映: 売りポジションのSLがスプレッド拡大だけで刈られる効果(askの高値は持っていない)。
  そのぶん、ロールオーバーをまたぐ売りは実際より良く見える
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from lab import config

MODEL = config.COST_MODEL
COMMISSION_PIPS = config.COMMISSION_PIPS
PROFILE_PATH = Path(__file__).resolve().parent / "spread_profile.json"

_profile: Dict[str, Any] = json.loads(PROFILE_PATH.read_text(encoding="utf-8")) if PROFILE_PATH.exists() else {}
_cache: Dict[str, np.ndarray] = {}
_dst_bounds: Dict[int, tuple] = {}


def _nth_sunday(year: int, month: int, nth: int) -> datetime:
    first = datetime(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (nth - 1))


def is_us_dst(ts: datetime) -> bool:
    """米国夏時間か(UTC基準)。3月第2日曜 07:00 UTC 〜 11月第1日曜 06:00 UTC。"""
    if ts.year not in _dst_bounds:
        _dst_bounds[ts.year] = (_nth_sunday(ts.year, 3, 2) + timedelta(hours=7),
                                _nth_sunday(ts.year, 11, 1) + timedelta(hours=6))
    start, end = _dst_bounds[ts.year]
    return start <= ts < end


def us_summer_hour(ts: datetime) -> int:
    """UTC時刻を、表の時間軸(米国夏時間のUTC時)に直す。冬は1時間戻す。"""
    return ts.hour if is_us_dst(ts) else (ts.hour - 1) % 24


def hourly_cost(pair: str) -> np.ndarray:
    """時間帯(米国夏時間のUTC時 0-23)ごとの往復コスト(pips)。表に無い銘柄は固定スプレッドのまま。"""
    if pair not in _cache:
        if pair in _profile:
            _cache[pair] = np.array(_profile[pair]["mean"], dtype=np.float64) + COMMISSION_PIPS
        else:
            _cache[pair] = np.full(24, config.SPREAD_PIPS[pair], dtype=np.float64)
    return _cache[pair]


def adjust(trades: List[Dict[str, Any]], pair: str) -> np.ndarray:
    """エンジンのトレード一覧から、時間帯別コスト反映後のpips配列を返す。"""
    if not trades:
        return np.array([], dtype=np.float64)
    cost, base = hourly_cost(pair), config.SPREAD_PIPS[pair]
    pips = np.array([float(t["pips"]) for t in trades], dtype=np.float64)
    hours = np.array([us_summer_hour(t["entry_time"] if t["side"] == "long" else t["exit_time"]) for t in trades])
    return pips - (cost[hours] - base)
