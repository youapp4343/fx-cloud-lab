"""探索ユニバースと判定ゲートの設定(事前固定。結果を見てから緩めない)。"""

from __future__ import annotations

PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "XAUUSD",
]

# 取得する足と取得開始日
FETCH_TFS = ["H1", "M15"]
FETCH_START = "2016-01-01"

# 取得せずリサンプルで作る足: 生成先 -> (元の足, pandasのルール)
DERIVED_TFS = {"H4": ("H1", "4h"), "M30": ("M15", "30min")}

# 探索時の時間足と抽選の重み(M15は1本あたり重いので低め)
TF_WEIGHTS = {"H4": 0.25, "H1": 0.4, "M30": 0.2, "M15": 0.15}

# 取引時間帯フィルタ(UTC)
WINDOWS = {
    "all": None,
    "asia0_8": [0, 8],
    "eu8_16": [8, 16],
    "ny13_21": [13, 21],
    "night20_0": [20, 0],
}

# 想定コスト(pips)。ブローカー実測ではなく保守的な仮定値。XAUUSDは1pip=0.1ドル。
SPREAD_PIPS = {
    "EURUSD": 0.6, "USDJPY": 0.7, "GBPUSD": 1.0, "AUDUSD": 0.8, "USDCHF": 1.0,
    "USDCAD": 1.0, "NZDUSD": 1.2, "EURJPY": 1.2, "GBPJPY": 1.8, "AUDJPY": 1.3,
    "EURGBP": 1.0, "XAUUSD": 2.5,
}
SLIPPAGE_PIPS = 0.2

# コストモデルの識別子。lab/costs.py が SPREAD_PIPS の上に時間帯別の実測コストを重ねる。
# 変えたら過去の結果とは比較できないので、report は数え直し、recheck で再判定する
COST_MODEL = "tt_hourly_v1"
COMMISSION_PIPS = 0.4   # ThreeTrader Raw: 往復 $4/lot 相当 ≒ 0.4pips(XAUUSDは1pip=0.1ドルで同じ0.4)

# index/曜日指定など個別入力が要る特殊テンプレートは対象外(grand_sweepと同じ)
EXCLUDE_TEMPLATES = {"feature_rule", "candle_pattern", "seasonal", "lab_replay"}

# 試行のうちルール自動生成(lab.gen)に回す割合。残りは既存テンプレートの摂動
GEN_SHARE = 0.6

# 時系列分割(古い順): train 60% / confirm 20% / holdout 20%
SPLIT = (0.6, 0.8)

# 段階ゲート。holdoutはtrain・confirmを両方通過した試行だけが見る。
TRAIN_MIN_N, TRAIN_MIN_PF = 100, 1.10
CONFIRM_MIN_N, CONFIRM_MIN_PF = 30, 1.10

# 生存判定(holdout)
HOLDOUT_MIN_N = 30
FDR_Q = 0.10
BOOTSTRAP_N = 2000
# プラセボ(シグナルの日単位巡回シフト)の本数と許容p値。p最小値は 1/(N+1)
PLACEBO_N = 20
PLACEBO_MAX_P = 0.10

# パラメータ摂動の倍率候補(1.0を厚めに)
PARAM_FACTORS = [0.5, 0.67, 0.8, 1.0, 1.0, 1.25, 1.5, 2.0]
