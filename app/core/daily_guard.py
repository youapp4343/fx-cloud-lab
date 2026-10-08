"""日次ガード資金管理(デイリーロスリミット+日次利益退避)。

コンセプト: 1日の負け額上限を設定し、日次でプラスになった資金は運用口座から退避(sweep)
しながら運用する(プロップファーム式)。爆益系(高ロット)戦略の破綻リスクを日単位で限定する。

実装はエンジン非改変の「トレード列の事後シミュレーション」:
既存エンジン(engine.py / nanpin_engine.py)のresult["trades"]を入力に、
- exit_time(UTC)の日付でトレードを日次グルーピング
- 日内累積損益(決済ベース)が -daily_loss_limit 到達で、その時刻以降に「新規エントリー
  するはずだった」取引のみ不実行として除外(エントリー済みポジションの決済損益は消せない)
- 日末に運用残高がベース超過なら超過分をsweptへ退避し運用残高をベースへ戻す
  (負けた日はそのまま=翌日は目減りした残高で再開。勝ちはまずベース復帰に充当され、
  ベースを超えた分だけが退避される)
- 運用残高<=0で全停止(口座破綻)

誠実性の注記(docstring契約):
- 「日次停止」はトレード単位の近似である。実運用では保有中ポジションの途中強制決済で
  上限ちょうどに損失を止められるが、本シミュレーションはトレードを丸ごと実行/不実行の
  二択で扱うため、上限を最後の1トレード分だけ超過してから停止する(悲観側の近似)。
- sweepは出金コスト・税・最低出金額を考慮しない。
- 元エンジンのequity_curve(バー単位)とは独立した日次粒度の再構成であり、バー内の
  含み損は考慮しない(ナンピン系はmax_floating_lossを別途必ず確認すること)。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

DAILY_GUARD_DISCLAIMER = (
    "日次ガードはトレード単位の近似シミュレーションです。日次損失上限は最後のトレード分だけ"
    "超過してから停止します(悲観側)。バー内の含み損・出金コストは考慮しません。"
    "元戦略に期待値が無い場合、資金管理では期待値を作れません(破綻を遅らせるだけです)。"
)


def apply_daily_guard(
    trades: List[Dict[str, Any]],
    initial_balance: float = 10000.0,
    daily_loss_limit: float = 300.0,
    sweep: bool = True,
    daily_profit_target_pips: float | None = None,
) -> Dict[str, Any]:
    """トレード列に日次ロスリミット+日次利益退避を適用した資金推移を再構成する。

    trades: engine.run_backtest / nanpin_engine.run_nanpin_backtest のresult["trades"]。
            exit_time昇順である前提(既存エンジンは時系列順に追記するため通常満たされる)。
    daily_loss_limit: 1日の許容損失額(口座通貨、正の値)。
    sweep: Trueなら日末にベース超過分を退避(運用残高をinitial_balanceへリセット)。

    戻り値: {"executed_trades": 実行された取引数, "skipped_trades": 停止で弾かれた取引数,
             "stop_days": 日次停止が発動した日数, "trading_days": 取引があった日数,
             "swept_total": 退避累計, "final_operating": 最終運用残高,
             "final_total": swept+運用残高, "ruined": bool(運用残高<=0で停止したか),
             "daily_curve": [{"date", "pnl", "stopped", "swept", "operating"}...],
             "disclaimer": str}
    """
    if daily_loss_limit <= 0:
        raise ValueError(f"daily_loss_limit は正の値である必要があります: {daily_loss_limit!r}")

    operating = initial_balance
    swept_total = 0.0
    executed = 0
    skipped = 0
    stop_days = 0
    ruined = False
    daily_curve: List[Dict[str, Any]] = []

    # 停止の意味論(重要): 日次上限到達で止められるのは「その後に新規エントリーするはず
    # だった取引」だけである。既にエントリー済みのポジションの損失は消せない(決済日基準で
    # 損失取引を丸ごと不実行にすると、数週間前に建てたナンピンバスケットのstopout損失を
    # 「その日は取引しなかったから無かった」ことにでき、負け戦略が偽の爆益に化ける。
    # 初版実装で実際に発生した欠陥への対策)。よって:
    # - 日次PnLは exit_time の日付で集計し、上限到達時刻を stop_map[日付] に記録する
    # - 取引のスキップ判定は entry_time 基準: エントリー日の停止時刻以降にエントリーする
    #   はずだった取引のみ不実行(それ以外は既存ポジションとして決済され損益が計上される)
    if trades:
        df = pd.DataFrame(trades)
        df["xts"] = pd.to_datetime(df["exit_time"])
        df["ets"] = pd.to_datetime(df["entry_time"])
        # exit_ts昇順に処理する(exit日は単調非減少になるため、日境界でsweepを確定できる)
        df = df.sort_values("xts", kind="stable").reset_index(drop=True)

        stop_map: Dict[Any, Any] = {}  # {エントリー不可となった日付: 停止時刻}
        ruin_ts = None  # 破綻時刻(以降のエントリーを全て不実行)
        cur_date = None
        cur_pnl = 0.0
        cur_pips = 0.0
        cur_stopped = False
        cur_hit_target = False

        def _close_day() -> None:
            nonlocal operating, swept_total, stop_days, cur_pnl, cur_pips, cur_stopped, cur_hit_target
            if cur_date is None:
                return
            day_swept = 0.0
            if cur_stopped:
                stop_days += 1
            if sweep and not ruined and operating > initial_balance:
                day_swept = operating - initial_balance
                swept_total += day_swept
                operating = initial_balance
            daily_curve.append(
                {"date": str(cur_date), "pnl": cur_pnl, "pips": cur_pips, "stopped": cur_stopped,
                 "hit_target": cur_hit_target, "swept": day_swept, "operating": operating}
            )

        for row in df.itertuples(index=False):
            e_ts, x_ts, profit = row.ets, row.xts, float(row.profit)
            e_date, x_date = e_ts.date(), x_ts.date()
            if x_date != cur_date:
                _close_day()  # 前日を確定(sweepは日末に実施=破綻判定にも退避が正しく反映される)
                cur_date, cur_pnl, cur_stopped = x_date, 0.0, False
                cur_pips, cur_hit_target = 0.0, False
            # エントリー時点で既に停止/破綻していた取引は「建てられなかった」ので不実行
            if (ruin_ts is not None and e_ts >= ruin_ts) or (
                e_date in stop_map and e_ts >= stop_map[e_date]
            ):
                skipped += 1
                continue
            cur_pnl += profit
            cur_pips += float(getattr(row, 'pips', 0.0) or 0.0)
            operating += profit
            executed += 1
            if (
                daily_profit_target_pips is not None
                and cur_pips >= daily_profit_target_pips
                and x_date not in stop_map
            ):
                stop_map[x_date] = x_ts  # 日次利益目標到達、当日新規エントリー停止
                cur_hit_target = True
            if cur_pnl <= -daily_loss_limit and x_date not in stop_map:
                stop_map[x_date] = x_ts  # この時刻以降の当日新規エントリーを禁止
                cur_stopped = True
            if operating <= 0 and ruin_ts is None:
                ruin_ts = x_ts
                ruined = True
        _close_day()

    target_days = sum(1 for d in daily_curve if d.get("hit_target"))
    return {
        "executed_trades": executed,
        "target_days": target_days,
        "skipped_trades": skipped,
        "stop_days": stop_days,
        "trading_days": len(daily_curve),
        "swept_total": swept_total,
        "final_operating": operating,
        "final_total": swept_total + operating,
        "ruined": ruined,
        "daily_curve": daily_curve,
        "disclaimer": DAILY_GUARD_DISCLAIMER,
    }
