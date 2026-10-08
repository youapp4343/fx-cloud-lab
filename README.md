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
- `lab/fetch.py` — Dukascopy から H1/M15 を差分取得、H4 は H1 から生成
- `lab/search.py` — 時間予算つきランダム探索(train → confirm → holdout の段階ゲート)
- `lab/report.py` — 累積統合、BH-FDR、HTML生成

## 判定

train 60% / confirm 20% / holdout 20% の時系列分割。holdout を見るのは train・confirm 通過試行のみ。
生存 = holdout で n≥30、FDR q≤0.10、bootstrap CI95 下限>0、上位3トレード除外後も合計pips>0。

コストは仮定値。「生存」は実運用可ではなく、ブローカー実ティック再検証とデモ運用の候補という意味。

## app/core の更新

FXプロジェクト側でテンプレートを足したら `app/core` をコピーし直して push する。
