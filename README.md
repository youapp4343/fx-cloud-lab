# FX Cloud Lab

FX戦略のランダム探索を GitHub Actions 上で回し、結果を GitHub Pages に出す。手元PCは使わない。

## 使い方(スマホ)

1. GitHub アプリ or ブラウザ → このリポジトリ → **Actions** → **search** → **Run workflow**
   - `minutes`: 1シャードあたりの探索時間(8シャード並列。20なら合計約160分ぶん)
2. 完了後、Pages のレポートを開く: `https://<user>.github.io/fx-cloud-lab/`

毎日 03:17 JST に自動実行もされる。結果は `gh-pages` ブランチに累積する。

## 構成

- `app/core/` — バックテストエンジンと戦略テンプレート(FXプロジェクトのコピー)
- `lab/config.py` — ユニバース・コスト・判定ゲート(事前固定)
- `lab/fetch.py` — Dukascopy から H1/M15 を差分取得、H4・M30 はリサンプルで生成
- `lab/fetch_macro.py` — CFTC COT建玉・FRED金利を取得。公表時刻(available_at)つきで保存
- `lab/gen.py` — ルール自動生成。特徴量は値動き・他銘柄・日〜月単位のモメンタム・暦・通貨強弱・COT・金利
- `lab/gh_scout.py` — GitHubのFX戦略リポジトリを「検証の質」で採点(結果は `scout/REPORT.md`)
- `lab/costs.py` — 時間帯別の実測コストをトレード損益に反映
- `lab/recheck.py` — 過去の holdout 到達分を現行コストで再判定
- `lab/search.py` — 時間予算つきランダム探索(train → confirm → holdout の段階ゲート)
- `lab/report.py` — 累積統合、BH-FDR、HTML生成

## 判定

train 60% / confirm 20% / holdout 20% の時系列分割。holdout を見るのは train・confirm 通過試行のみ。
生存 = holdout で n≥30、FDR q≤0.10、bootstrap CI95 下限>0、上位3トレード除外後も合計pips>0。

コストは ThreeTrader の実測スプレッド(UTC時間帯別、`lab/spread_profile.json`)+手数料。表は `python -m lab.build_spread_profile <収集CSVのフォルダ>` でローカルから作り直す。コストモデルを変えたら Run workflow の `recheck=true` で過去の holdout 到達分を再判定する。

「生存」は実運用可ではなく、ブローカー実ティック再検証とデモ運用の候補という意味。

## app/core の更新

FXプロジェクト側でテンプレートを足したら `app/core` をコピーし直して push する。
