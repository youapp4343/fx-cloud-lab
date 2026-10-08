from app.core import engine  # noqa: F401 - import時に組み込みテンプレートを登録する
from app.core import candle_template  # noqa: F401 - candle_patternテンプレートを登録する
from app.core import vol_breakout_template  # noqa: F401 - vol_breakoutテンプレートを登録する
from app.core import feature_rule_template  # noqa: F401 - feature_ruleテンプレートを登録する
from app.core import fib_harmonic_template  # noqa: F401 - fib_pullback/harmonicテンプレートを登録する
from app.core import bb_reversion_template  # noqa: F401 - bb_reversionテンプレートを登録する
from app.core import volume_spike_template  # noqa: F401 - volume_spikeテンプレートを登録する
from app.core import seasonal_template  # noqa: F401 - seasonalテンプレートを登録する
from app.core import regime_adaptive_template  # noqa: F401 - regime_adaptiveテンプレートを登録する
from app.core import vol_squeeze_template  # noqa: F401 - vol_squeeze_breakoutテンプレートを登録する
from app.core import session_range_template  # noqa: F401 - session_range_breakoutテンプレートを登録する
from app.core import orderflow_divergence_template  # noqa: F401 - orderflow_divergenceテンプレートを登録する
from app.core import night_scalp_template  # noqa: F401 - night_scalpテンプレートを登録する
from app.core import macd_cross_template  # noqa: F401 - macd_crossテンプレートを登録する
from app.core import stoch_reversal_template  # noqa: F401 - stoch_reversalテンプレートを登録する
from app.core import envelope_reversion_template  # noqa: F401 - envelope_reversionテンプレートを登録する
from app.core import psar_flip_template  # noqa: F401 - psar_flipテンプレートを登録する
from app.core import osc_reversal_template  # noqa: F401 - osc_reversal(CCI/WPR/DeMarker/Momentum統合)テンプレートを登録する
from app.core import osc_confluence_template  # noqa: F401 - osc_confluence(オシレーター合議)テンプレートを登録する
from app.core import osc_confluence_x_template  # noqa: F401 - osc_confluence_x(14種拡張合議)テンプレートを登録する
from app.core import swing_trend_template  # noqa: F401 - swing_trend(方向制限スイング)テンプレートを登録する
from app.core import spike_fade_template  # noqa: F401 - spike_fade(急変逆張り)テンプレートを登録する
from app.core import spike_fade_delayed_template  # noqa: F401 - spike_fade_delayed(遅延急変逆張り)テンプレートを登録する
from app.core import month_end_fix_template  # noqa: F401
from app.core import weekly_oco_template  # noqa: F401
from app.core import ma_pullback_template  # noqa: F401 - ma_pullbackテンプレートを登録する
from app.core import tdi_cross_template  # noqa: F401 - tdi_cross(TDIクロス 手法#7)を登録する
from app.core import choros_touch_template  # noqa: F401 - choros_touch(MA初回タッチ 手法#8)を登録する
from app.core import gotobi_nakane_template  # noqa: F401 - gotobi_nakane(ゴトー日仲値 手法#1)を登録する
from app.core import gotobi_postfix_template  # noqa: F401 - 仲値後ショート(実稼働中)
# 追加9手法(batch2)
from app.core import momo_template  # noqa: F401 - momo(5分Momo)
from app.core import bagovino_template  # noqa: F401 - bagovino(EMA5/12+RSI21)
from app.core import three_ducks_template  # noqa: F401 - three_ducks(MTF SMA60)
from app.core import vegas_tunnel_template  # noqa: F401 - vegas_tunnel(EMA144/169)
from app.core import double_bb_template  # noqa: F401 - double_bb(2σ/1σ帯)
from app.core import turtle_soup_template  # noqa: F401 - turtle_soup(ブレイク失敗逆張り)
from app.core import asian_breakout_template  # noqa: F401 - asian_breakout(東京レンジ+EMA55)
from app.core import pivot_rsi_div_template  # noqa: F401 - pivot_rsi_div(ピボット+RSIダイバー)
from app.core import ema10_easy_template  # noqa: F401 - ema10_easy(EMA10押し目)
from app.core import reversal_at_support_template  # noqa: F401 - reversal_at_support(反発足+サポート文脈)
from app.core import lw_ema9_template  # noqa: F401 - lw_ema9(ラリー・ウィリアムズ9EMA)
from app.core import tetsuban_pullback_template  # noqa: F401 - tetsuban_pullback(鉄板①MTF20SMA押し目)
from app.core import vgrsi_template  # noqa: F401 - vgrsi(Visibility Graph RSI)
from app.core import orz_pullback_template  # noqa: F401 - orz_pullback(ORZ手法 MTF押し目)
from app.core import kou_slope_template  # noqa: F401 - kou_slope(MA傾き順張り押し目)
from app.core import liquidity_sweep_template  # noqa: F401 - liquidity_sweep(SMC流動性狩り逆張り)
from app.core import quasimodo_template  # noqa: F401 - quasimodo(QML反転)
from app.core import order_block_template  # noqa: F401 - order_block(OBリテスト)
from app.core import smc_bos_template  # noqa: F401 - smc_bos(構造+premium/discount+IDM)
from app.core import cci_reversal_template  # noqa: F401 - cci_reversal(CCI±200折返し 手法#9)
from app.core import roundnum_fade_template  # noqa: F401 - roundnum_fade(キリ番逆張り 手法#15/16)
from app.core import fvg_ob_template  # noqa: F401 - fvg_ob(FVG+OBリテスト 手法#12)
from app.core import prevday_sweep_template  # noqa: F401 - prevday_sweep(前日安値スイープ 手法#13)
from app.core import gold_vote_template  # noqa: F401 - gold_vote(ゴールド買い限定投票 手法#14)
from app.core import gmma_engulf_template  # noqa: F401 - gmma_engulf(GMMA押し目+包み足)
from app.core import elliott_impulse_template  # noqa: F401 - elliott_impulse(エリオットwave2押し)
from app.core import head_shoulders_template  # noqa: F401 - head_shoulders(三尊/逆三尊ネックラインブレイク)
from app.core import inside_range_break_template  # noqa: F401 - inside_range_break(レンジ収束ブレイク)
from app.core import gmma_stack_template  # noqa: F401 - gmma_stack(GMMA完全整列 note EA#1)
from app.core import supertrend_template  # noqa: F401 - supertrend(ATR Supertrend note EA#2)
from app.core import cci_ichimoku_template  # noqa: F401 - cci_ichimoku(note EA#5)
from app.core import osc_ichimoku_template  # noqa: F401
from app.core import hline_react_template  # noqa: F401
from app.core import regime_london_break_template  # noqa: F401 - regime_london_break(戦略A アジアレンジ収縮→ロンドンブレイク)
from app.core import range_scalp_template  # noqa: F401 - range_scalp(コペルニクス風 USDJPY M5低ボラ平均回帰)
from app.core import break_retest_template  # noqa: F401 - break_retest(水平線ブレイク&リテスト RR1:2 動的ジグザグTP)
from app.core import consec_break_template  # noqa: F401 - consec_break(連続陰線→陽線ブレイク gold H1)
from app.core import gold_break_fusion_template  # noqa: F401 - gold_break_fusion(gold 3モジュールブレイク Rolling/AsiaLondon/Compression)
from app.core import afs_experts  # noqa: F401 - Adaptive Flow Scalper 3Expert(afs_momentum/afs_reversion/afs_session, M1版)
from app.core import envelope_template  # noqa: F401 - envelope_fade(ぶせな1分足エンベロープ逆張り 5ゾーン+EMAリセット)
from app.core import morning_orb_template  # noqa: F401 - morning_orb(EURUSD朝レンジブレイク OCO/レンジ幅SLTP)
from app.core import ema_dev_candle_template  # noqa: F401 - ema_dev_candle(200EMA大乖離×反転足 X投稿検証)
from app.core import wemof_template  # noqa: F401 - wemof(BB3σ純度逆張り わっきゃいWEMOF代理)
from app.core import vp_pullback_template  # noqa: F401 - vp_pullback(出来高収縮押し目 Gajjala/SABAI検証)
from app.core import vol_osc_template  # noqa: F401 - vol_osc(MFI/OBVダイバー/Force/CMF 出来高オシレーター)
from app.core import confluence_long_template  # noqa: F401 - confluence_long(トレンド+一目雲+抵抗帯+MTF 多要素合議ロングスキャル 事前登録検証)
from app.core import sma120_wpr_template  # noqa: F401 - sma120_wpr(120SMA+WilliamsR YouTube#8 事前登録検証)
from app.core import ema3_stack_template  # noqa: F401 - ema3_stack(EMA50/100/150押し目 YouTube#2 事前登録検証)
from app.core import ema3_fractal_template  # noqa: F401 - ema3_fractal(EMA20/50/100+Fractal YouTube#7 事前登録検証)
from app.core import sma1030_rsi_template  # noqa: F401 - sma1030_rsi(SMA10/30+RSI8 YouTube#9 事前登録検証)
from app.core import rollover_marubozu_template  # noqa: F401 - rollover_marubozu(時間帯クラスタリング YouTube set2#10 事前登録検証)
from app.core import atr_dynamic_stoch_template  # noqa: F401 - atr_dynamic_stoch(EMA3本+StochRSIクロス+ATR動的SLTP YouTube set2#5 事前登録検証)
from app.core import fractal_rsi50_fixed10_template  # noqa: F401 - fractal_rsi50_fixed10(EMA21/50/200+Fractal+RSI50 YouTube set2#4 事前登録検証)
from app.core import heikin_doji_break_template  # noqa: F401 - heikin_doji_break(平均足インパルス+ダギ足ブレイク YouTube set2#8 事前登録検証)
from app.core import two_leg_pullback_template  # noqa: F401 - two_leg_pullback(2段押し tick_chart_prereg #1+#2 事前登録検証)
from app.core import vwap_reversion_reversal_template  # noqa: F401 - vwap_reversion_reversal(VWAP乖離反転 tick_chart_prereg #10簡略版 事前登録検証)
from app.core import flow_scalp_template  # noqa: F401 - flow_scalp(符号付き出来高フロー×レンジ拡大 スキャル土俵検証)
from app.core import mtf_scalp_template  # noqa: F401 - mtf_scalp(上位足で方向・下位足でタイミング、RR固定のMTFスキャル)
from app.core import ext_fade_template  # noqa: F401 - ext_fade(上位足トレンドへの過伸張フェード、情報スキャン由来)
from app.core import pullback_short_template  # noqa: F401 - pullback_short(上昇トレンド中の短期過熱売り、クリーン情報スキャン由来)
from app.core import catalog_templates  # noqa: F401 - cat_*(検証候補100件のnote手法を機械化)
from app.core import catalog2_templates  # noqa: F401 - cat2_*(検証候補101-200のA判定手法)
from app.core import tv3_templates  # noqa: F401 - tv3_*(TradingView 201-300のA判定手法)
from app.core import tv3b_templates  # noqa: F401 - tv3_*第2弾(B判定から回収したもの)
from app.core import tv3s_templates  # noqa: F401 - tv3s_*(TradingView 201-300のstrategy 5本)
from app.core import tv3c_templates  # noqa: F401 - tv3_*第3弾(B判定の再確認で回収)
from app.core import tv3d_templates  # noqa: F401 - tv3_*第4弾(未読B判定から回収)
from app.core import tv4_templates  # noqa: F401 - tv4_*(TradingView 301-400 FX向け)
from app.core import catalog2b_templates  # noqa: F401 - cat2_*(101-200のB判定手法)
