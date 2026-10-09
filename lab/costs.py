"""時間帯別の実測コストを損益に反映する。

エンジンは銘柄ごとの固定スプレッド(config.SPREAD_PIPS)を建値に1回だけ課している。
ここでは各トレードについて「実測の時間帯別コスト − 固定スプレッド」を追加で差し引き、
固定値では見えないロールオーバー帯(UTC21時前後)の拡大を損益に入れる。

- コスト = その時間の平均スプレッド(lab/spread_profile.json) + 手数料
- 買いは建てた時刻、売りは決済した時刻のスプレッドを払う(バーはbid。askで約定する側が払う)
- 未反映: 売りポジションのSLがスプレッド拡大だけで刈られる効果(askの高値は持っていない)。
  そのぶん、ロールオーバーをまたぐ売りは実際より良く見える
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from lab import config

MODEL = config.COST_MODEL
COMMISSION_PIPS = config.COMMISSION_PIPS
PROFILE_PATH = Path(__file__).resolve().parent / "spread_profile.json"

_profile: Dict[str, Any] = json.loads(PROFILE_PATH.read_text(encoding="utf-8")) if PROFILE_PATH.exists() else {}
_cache: Dict[str, np.ndarray] = {}


def hourly_cost(pair: str) -> np.ndarray:
    """UTC時(0-23)ごとの往復コスト(pips)。実測が無い銘柄は固定スプレッドのまま。"""
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
    hours = np.array([(t["entry_time"] if t["side"] == "long" else t["exit_time"]).hour for t in trades])
    return pips - (cost[hours] - base)
