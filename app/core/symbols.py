"""シンボル仕様(pip定義・契約サイズ・クオート通貨)の単一の真実(Wave5 案E)。

これまで3エンジン(app/core/engine.py / nanpin_engine.py / arb_engine.py)に同一ロジックとして
再掲されていた _pip_size / _pip_value_per_lot の定義を本モジュールへ集約する。各エンジンの
同名関数は関数名・シグネチャ・呼び出し箇所を一切変えず、中身だけを本モジュールへ委譲する
(discovery.py / seasonal_scan.py は engine._pip_size を import しているため自動追随する)。

pip定義と契約サイズ:
- XAUUSD: pip=0.1ドル / 1lot=100oz  → pip価値 $10/lot(USD建て、価格に依存しない)
- XAGUSD: pip=0.01ドル / 1lot=5,000oz → pip価値 $50/lot(同上)
- 6文字FXペアでクオート通貨がJPY: pip=0.01 / 1lot=100,000通貨(現行挙動維持)
- その他の6文字FXペア: pip=0.0001 / 1lot=100,000通貨(現行挙動維持)
- 未知シンボル(金銀以外かつ6文字以外): pip=0.0001へフォールバック(従来の
  engine._pip_size と同じ安全側デフォルト)

誠実性の注記:
- XAU pip=0.1 は業界多数派の慣例を採用したもの。0.01(=1セント刻み)を「pip」と呼ぶ
  ブローカーも実在するため、sl_pips/tp_pips/スプレッドpips指定の意味が2倍〜10倍ずれうる。
  実運用前に必ず使用ブローカーのpip/point定義と照合すること。
- XAUEUR等のUSD建て以外の金銀クロスは未対応: 6文字FXペア規則に落ちて誤った仕様
  (pip=0.0001/契約100,000)になるため本ツールでは使用しないこと。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str  # 正規化済み(大文字)シンボル
    pip_size: float  # 1pipの価格単位
    contract_size: float  # 1lotあたりの取引数量(FX=通貨単位、金銀=トロイオンス)
    quote_ccy: str  # クオート通貨。pip価値のUSD換算分岐(JPYのみprice割り)に使う


_FX_CONTRACT_SIZE = 100_000.0
_FALLBACK_PIP_SIZE = 0.0001  # 従来の engine._pip_size と同じフォールバック値

# 金銀はUSD建て(quote_ccy="USD")のため、JPY換算分岐には決して流れない。
_METAL_SPECS: Dict[str, SymbolSpec] = {
    "XAUUSD": SymbolSpec(symbol="XAUUSD", pip_size=0.1, contract_size=100.0, quote_ccy="USD"),
    "XAGUSD": SymbolSpec(symbol="XAGUSD", pip_size=0.01, contract_size=5_000.0, quote_ccy="USD"),
}


def get_spec(symbol: str) -> SymbolSpec:
    """シンボル文字列を SymbolSpec に解決する。

    優先順位: 金銀の完全一致 → 6文字FXペア規則(末尾3文字=クオート通貨) →
    フォールバック(pip=0.0001/契約100,000/USD建て扱い)。未知シンボルで例外は
    投げない(旧 _pip_size がフォールバックで動いていた挙動を維持するため)。
    """
    s = symbol.upper()
    metal = _METAL_SPECS.get(s)
    if metal is not None:
        return metal
    if len(s) == 6:
        quote = s[3:]
        pip = 0.01 if quote == "JPY" else _FALLBACK_PIP_SIZE
        return SymbolSpec(symbol=s, pip_size=pip, contract_size=_FX_CONTRACT_SIZE, quote_ccy=quote)
    # 6文字以外の未知シンボル。旧実装は endswith("JPY") のみで判定していたが、6文字でない
    # 時点でFXペア表記として不正なため0.0001に倒す(実データは全て6文字FXペア+XAU/XAG)。
    return SymbolSpec(
        symbol=s, pip_size=_FALLBACK_PIP_SIZE, contract_size=_FX_CONTRACT_SIZE, quote_ccy="USD"
    )


def pip_size(symbol: str) -> float:
    """3エンジンの _pip_size の委譲先(単一の真実)。"""
    return get_spec(symbol).pip_size


def pip_value_per_lot(
    symbol: str, price: float, pip_size_override: Optional[float] = None
) -> float:
    """1lotあたりのpip価値をUSD建てで概算する(3エンジンの _pip_value_per_lot の委譲先)。

    - USD建て(XAUUSD/XAGUSD/EURUSD等): pip_size × contract_size がそのままUSD/pip。
      XAUUSD=0.1×100oz=$10/lot、XAGUSD=0.01×5,000oz=$50/lot(価格に依存しない)。
    - JPYクオート(USDJPY/EURJPY等): JPY建てpip価値をpriceで割ってUSDへ換算する従来近似を
      維持する(対USD建てペアで正確、対円クロスは厳密な換算ではない)。
    - JPY/USD以外のクオート(EURGBP等): 従来同様「クオート通貨建て値≈USD」の近似で
      換算せずそのまま返す(現行挙動維持)。
    - pip_size_override: エンジン側の既存シグネチャ _pip_value_per_lot(symbol, pip_size, price)
      互換のための引数。Noneなら本モジュールのspec値を使う(3エンジンは常に
      _pip_size(symbol) と同値を渡してくるため結果は同一になる)。
    """
    spec = get_spec(symbol)
    pip = spec.pip_size if pip_size_override is None else pip_size_override
    raw = pip * spec.contract_size
    if spec.quote_ccy == "JPY":
        return raw / price
    return raw
