"""週次OCOストラドル近似テンプレート(Wave5 案A)。

月曜の週初レンジ(range_bars本)を基準に、レンジ形成後active_bars本の間に終値が
レンジ上限+buffer_pipsを超えたら買い(+1)、下限-buffer_pipsを割ったら売り(-1)に追随する。
「週初レンジの上下に逆指値(stop)を同時に置き、片方約定で他方を取り消す」週次OCO
ストラドルのテンプレート近似。brokenフラグで週1回しかシグナルを出さないことが
「片方約定で他方キャンセル」の近似にあたる。決済はsl_pips/tp_pipsに加え、defaultsの
max_hold_bars(engineが自動認識する任意パラメータ)による時間切れ決済で
「金曜(週内)クローズ」を近似する。

エンジンへのstop_entry(逆指値エントリー)注文タイプ追加を却下し、テンプレート近似を
採用した理由(オーケストレーター設計判断、正典=docs/plan_wave5.md §5):
- engine.pyへの新注文タイプ追加は既存全戦略・全テンプレートに波及する回帰リスクが最大級
  (バー内判定順序はdocs/exit_rules_spec.mdの正規仕様であり、注文タイプ追加はその改訂を伴う)。
- ブレイク瞬間のstop約定は、実際にはスプレッド拡大・スリッページが最悪化するタイミングで
  あり、固定スプレッド前提のバックテストでは約定価格が系統的に楽観側へ壊れる型。

誠実性(この近似が本物のOCOストラドル運用と異なる点):
- 終値確認エントリーはstop注文より約定が遅く不利=保守側の近似だが、厳密には「別物の戦略」
  であり、本テンプレートの成績はstop注文運用の成績を保証しない。
- 金曜クローズはmax_hold_barsによる近似であり、週末持ち越しが稀に発生しうる
  (祝日等でバーが欠けた週は時間切れが翌週にずれ込む)。
- ブレイク系テンプレートはグランドスイープで総じて弱かった、という前提を引き継ぐ
  (本テンプレートへの期待値も控えめに置く)。

defaultsはM15基準。時間足別の推奨値(検証スクリプトvalidate_weekly_oco.pyもこの表に従う):

    時間足   range_bars            active_bars      max_hold_bars
    M15     16 (月曜冒頭4時間)      120 (約30時間)    384 (約4営業日)
    H1      4                     30               96
    H4      1                     8                24

先読み回避についての設計メモ: 週の状態(アンカー位置・レンジ高安・ブレイク済みか)は
週境界をまたいで前のバーの結果に依存する経路依存(path-dependent)ロジックのため、
session_range_template.pyと同じ「時系列順の単純な1パス走査」を採用する。1パスの逐次走査
であれば「バーiの時点でi以前(i自身を含む、確定済みの高値・安値・終値)の情報しか参照して
いない」ことがループを読むだけで自明であり、監査が容易になる。なおsession_range_templateで
必要だった週末ギャップのblocked_date対策は本テンプレートでは不要: アンカー条件が
「新しいISO週の最初のバーかつ月曜(dayofweek==0)」という構造条件であり、日曜のブローカー
再開バー(21-23時台)はISOカレンダー上は前週の最終日(day=7)に分類されるため、
そもそもアンカー候補になり得ない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import templates
from app.core.engine import _pip_size
from app.core.strategy_model import Strategy


def _signal_weekly_oco_straddle(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    range_bars = max(1, int(strategy.params["range_bars"].value))
    active_bars = max(1, int(strategy.params["active_bars"].value))
    buffer_pips = float(strategy.params["buffer_pips"].value)
    # buffer_pipsは価格距離に変換して使う。pip定義はengine._pip_size(シンボル別pip定義の
    # 単一の真実。案E導入後はsymbols.pyへの委譲に自動追随する)。
    buffer_price = buffer_pips * _pip_size(strategy.symbol)

    iso = df["timestamp"].dt.isocalendar()
    # ISO週キー: isocalendarのyear(暦年ではなくISO年)を使うこと。暦年を使うと
    # 12/29-1/3周辺の週(例: 2024-12-30月曜はISO 2025-W01)が1月1日で偽の週境界を作り、
    # 月曜以外の曜日で「新週の最初のバー」と誤判定して週全体をブロックしてしまう。
    week_key = (iso["year"].astype(np.int64) * 100 + iso["week"].astype(np.int64)).to_numpy()
    dow = df["timestamp"].dt.dayofweek.to_numpy()  # 月曜=0 ... 日曜=6
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    signal = np.zeros(n, dtype=int)

    cur_week = None  # 現在処理中のISO週キー(iso_year*100+iso_week)
    anchor_i = -1  # 今週のアンカーバー(週初の月曜最初のバー)のインデックス。-1=今週は無効
    range_high = -np.inf
    range_low = np.inf
    broken = False  # 今週すでにレンジを抜けたか(週1回のみシグナル=OCOの片方約定近似)

    for i in range(n):
        wk = week_key[i]
        if wk != cur_week:
            # 新しいISO週の最初のバー。ISO週は月曜0:00に切り替わるため、このバーの曜日は
            # 月曜(通常)か、月曜が祝日等で欠損していれば火曜以降になる。
            cur_week = wk
            if dow[i] == 0:
                # 週初アンカー確定。アンカーバー自身の高安がレンジの初期値になる
                # (以降range_bars-1本で拡張し、計range_bars本の週初レンジを形成)。
                anchor_i = i
                range_high = high[i]
                range_low = low[i]
                broken = False
            else:
                # 月曜欠損週はアンカーせず週全体をスキップする(火曜アンカー禁止)。
                # 火曜以降を起点にすると「週初レンジ」の意味が変わり、月曜に既に動いた
                # 相場の途中をレンジと誤認してブレイク判定が別物になるため。
                # データ先頭が週の途中から始まる場合も同じ扱いで安全側に倒れる。
                anchor_i = -1
            continue
        if anchor_i < 0:
            continue  # 今週は無効(月曜欠損 or データ先頭が週途中)
        bars_since_anchor = i - anchor_i
        if bars_since_anchor < range_bars:
            # 週初レンジ形成中: 高安を拡張するだけでシグナルは出さない
            range_high = max(range_high, high[i])
            range_low = min(range_low, low[i])
            continue
        if bars_since_anchor >= range_bars + active_bars:
            continue  # アクティブ期間終了。今週はもうシグナルを出さない
        if not broken:
            if close[i] > range_high + buffer_price:
                signal[i] = 1
                broken = True
            elif close[i] < range_low - buffer_price:
                signal[i] = -1
                broken = True

    return pd.Series(signal, index=df.index, dtype=int)


templates.register(
    "weekly_oco_straddle",
    defaults={
        "range_bars": 16.0,
        "active_bars": 120.0,
        "buffer_pips": 5.0,
        "sl_pips": 40.0,
        "tp_pips": 80.0,
        "max_hold_bars": 384.0,  # engineが自動認識(週内クローズ近似、M15で約4営業日)
        "lot": 0.1,
    },
    signal_fn=_signal_weekly_oco_straddle,
)
