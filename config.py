"""Central configuration for paths, model hyperparameters, and feature settings."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def env_bool(key: str, default: str = "false") -> bool:
    """Parse ENABLE_PROPS-style flags; tolerates quotes and whitespace."""
    raw = os.getenv(key)
    if raw is None:
        raw = default
    return str(raw).strip().strip('"').strip("'").lower() in ("1", "true", "yes", "on")


# Standalone layout: .env at project root or ufc_betting_bot/.env
# Frozen EXE: defer to project_paths.bootstrap() — config.py may live in _MEIPASS.
ROOT_DIR = Path(__file__).resolve().parent
if not getattr(__import__("sys"), "frozen", False):
    load_dotenv(ROOT_DIR / ".env", override=False)
    load_dotenv(ROOT_DIR / "ufc_betting_bot" / ".env", override=False)

# --- Paths ---
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT_DIR / "models"

RAW_FIGHTS_CSV = RAW_DIR / "fights.csv"
PROCESSED_FEATURES_CSV = PROCESSED_DIR / "fight_features.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "ensemble_winner.joblib"
LEGACY_MODEL_PATH = MODELS_DIR / "lgbm_winner.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
BACKTEST_DIR = MODELS_DIR / "backtest"
BACKTEST_SUMMARY_CSV = BACKTEST_DIR / "backtest_summary.csv"
BACKTEST_PREDICTIONS_CSV = BACKTEST_DIR / "walk_forward_predictions.csv"
BACKTEST_THRESHOLD_CSV = BACKTEST_DIR / "threshold_roi.csv"
BACKTEST_IMPORTANCE_CSV = BACKTEST_DIR / "importance_timeline.csv"
BACKTEST_METRICS_BY_YEAR_CSV = BACKTEST_DIR / "metrics_by_year.csv"
BACKTEST_CALIBRATION_PNG = BACKTEST_DIR / "calibration_plot.png"
BACKTEST_ROI_PNG = BACKTEST_DIR / "roi_threshold_plot.png"
PLOTS_DIR = DATA_DIR / "plots"
BACKTEST_2025_CSV = DATA_DIR / "backtest_2025_results.csv"
BACKTEST_2025_YEAR = int(os.getenv("BACKTEST_2025_YEAR", "2025"))
GYMS_CSV = DATA_DIR / "gyms.csv"

# --- Data ---
UFC_STATS_BASE_URL = os.getenv(
    "UFC_STATS_BASE_URL", "http://ufcstats.com/statistics/events/completed?page=all"
)
UFC_STATS_UPCOMING_URL = os.getenv(
    "UFC_STATS_UPCOMING_URL", "http://ufcstats.com/statistics/events/upcoming"
)
UFC_EVENTS_URL = os.getenv("UFC_EVENTS_URL", "https://www.ufc.com/events")
ESPN_UFC_SCOREBOARD_URL = os.getenv(
    "ESPN_UFC_SCOREBOARD_URL",
    "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard",
)
HISTORICAL_DATA_URL = os.getenv(
    "HISTORICAL_DATA_URL",
    "https://raw.githubusercontent.com/jansen88/ufc-data/main/data/complete_ufc_data.csv",
)
HF_UFC_DATASET = os.getenv("HF_UFC_DATASET", "JesterLabs/UFC_FIGHT_DATA")
HF_UFC_SPLIT = os.getenv("HF_UFC_SPLIT", "train")
HF_UFC_PAGE_SIZE = int(os.getenv("HF_UFC_PAGE_SIZE", "100"))

REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
REQUEST_DELAY_SEC = float(os.getenv("REQUEST_DELAY_SEC", "1.0"))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

# Canonical fights.csv columns (user-facing)
FIGHTS_COLUMNS = [
    "fight_id",
    "event",
    "date",
    "location",
    "fighter1",
    "fighter2",
    "winner",
    "weight_class",
    "method",
    "round",
    "time",
    "is_title_fight",
    "is_main_event",
    "sig_strikes_landed_f1",
    "sig_strikes_attempted_f1",
    "sig_strikes_landed_f2",
    "sig_strikes_attempted_f2",
    "takedowns_landed_f1",
    "takedowns_attempted_f1",
    "takedowns_landed_f2",
    "takedowns_attempted_f2",
    "finish",
    "f1_odds",
    "f2_odds",
    "source",
]

HISTORICAL_META_PATH = CACHE_DIR / "historical_meta.json"
UPCOMING_CARD_CACHE = CACHE_DIR / "upcoming_card.csv"
LOCAL_KAGGLE_CANDIDATES = [
    RAW_DIR / "ufc-master.csv",
    RAW_DIR / "complete_ufc_data.csv",
    RAW_DIR / "raw_total_fight_data.csv",
    RAW_DIR / "data.csv",
]
KAGGLE_UFC_BETTING_ODDS_SLUG = os.getenv(
    "KAGGLE_UFC_BETTING_ODDS_SLUG",
    "jerzyszocik/ufc-betting-odds-daily-dataset",
)
KAGGLE_ODDS_DIR = RAW_DIR / "kaggle" / "ufc-betting-odds-daily-dataset"
LOCAL_ODDS_CANDIDATES = [
    KAGGLE_ODDS_DIR / "ufc-master.csv",
    KAGGLE_ODDS_DIR / "data.csv",
    KAGGLE_ODDS_DIR / "ufc_betting_odds.csv",
    RAW_DIR / "ufc_betting_odds_daily.csv",
    RAW_DIR / "ufc-master.csv",
    RAW_DIR / "cleaned_odds.csv",
    RAW_DIR / "complete_ufc_data.csv",
    RAW_DIR / "ufc_odds.csv",
]
ULTIMATE_UFC_DATASET_URL = os.getenv(
    "ULTIMATE_UFC_DATASET_URL",
    "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/main/ufc-master.csv",
)
JANSEN_CLEANED_ODDS_URL = os.getenv(
    "JANSEN_CLEANED_ODDS_URL",
    "https://raw.githubusercontent.com/jansen88/ufc-data/master/data/cleaned_odds.csv",
)
JANSEN_COMPLETE_URL = os.getenv(
    "JANSEN_COMPLETE_URL",
    "https://raw.githubusercontent.com/jansen88/ufc-data/main/data/complete_ufc_data.csv",
)
HISTORICAL_ODDS_CACHE = CACHE_DIR / "historical_odds_unified.csv"
ODDS_API_CACHE_PATH = CACHE_DIR / "ufc_odds_api.csv"

# --- UFCstats / Greco1899 enrichment (fighter profiles + career stats) ---
GRECO_UFCSTATS_BASE_URL = os.getenv(
    "GRECO_UFCSTATS_BASE_URL",
    "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/",
)
UFCSTATS_GRECO_CACHE_DIR = CACHE_DIR / "ufcstats_greco"
UFCSTATS_ENRICH_META_PATH = CACHE_DIR / "ufcstats_enrich_meta.json"
UFCSTATS_ENRICH_TTL_HOURS = int(os.getenv("UFCSTATS_ENRICH_TTL_HOURS", "12"))

FIGHTS_ENRICHMENT_COLUMNS = [
    "fighter1_height",
    "fighter2_height",
    "fighter1_reach",
    "fighter2_reach",
    "fighter1_dob",
    "fighter2_dob",
    "fighter1_stance",
    "fighter2_stance",
    "fighter1_sig_strikes_landed_pm",
    "fighter2_sig_strikes_landed_pm",
    "fighter1_sig_strikes_accuracy",
    "fighter2_sig_strikes_accuracy",
    "fighter1_takedown_accuracy",
    "fighter2_takedown_accuracy",
    "fighter1_takedown_defence",
    "fighter2_takedown_defence",
    "fighter1_submission_avg_attempted_per15m",
    "fighter2_submission_avg_attempted_per15m",
]
FIGHTS_SAVE_COLUMNS = list(
    dict.fromkeys(FIGHTS_COLUMNS + FIGHTS_ENRICHMENT_COLUMNS)
)

# --- Features ---
ROLLING_FIGHTS = int(os.getenv("ROLLING_FIGHTS", "5"))
# Strength-of-schedule window (prior N opponents for avg opp win rate / Elo).
SOS_WINDOW = int(os.getenv("SOS_WINDOW", "5"))
MIN_FIGHTS_PER_FIGHTER = int(os.getenv("MIN_FIGHTS_PER_FIGHTER", "3"))
# Bump when FEATURE_COLUMNS / feature defs change so models auto-retrain.
# v3: Sherdog career + CompuBox-style striking (KD / target / range mix).
# v4: Prior-sport base tiers (wrestling/BJJ/boxing/MT/…) + matchup level advantage.
FEATURE_SCHEMA_VERSION = int(os.getenv("FEATURE_SCHEMA_VERSION", "5"))

# Optional high-value feature block (Phase 1). Always computed in FE; gated into the model list.
ENABLE_HIGH_VALUE_FEATURES = env_bool("ENABLE_HIGH_VALUE_FEATURES", "true")
HIGH_VALUE_FEATURE_COLUMNS = [
    "hv_short_notice_flag_diff",
    "hv_long_layoff_flag_diff",
    "first_fight_new_wc_flag_diff",
    "finish_rate_l5_diff",
    "division_age_adj_diff",
    "hv_td_pressure_diff",
    "hv_control_clash",
    "wins_vs_better_record_l5_diff",
    "ko_losses_career_flag_diff",
]

# UFC-only pathway + market blocks (A/B research). Default OFF — do not affect production
# until pathway_market_ab_2025 keep rule passes. Names are UFC-scoped (not shared with trading bot).
ENABLE_PATHWAY_FEATURES = env_bool("ENABLE_PATHWAY_FEATURES", "false")
ENABLE_MARKET_FEATURES = env_bool("ENABLE_MARKET_FEATURES", "false")
# Research-only: post-hoc shrink of wide-CI probs toward market (not a train feature).
ENABLE_PATHWAY_MARKET_CAL = env_bool("ENABLE_PATHWAY_MARKET_CAL", "false")
PATHWAY_MARKET_CAL_WIDTH = float(os.getenv("PATHWAY_MARKET_CAL_WIDTH", "0.40"))
PATHWAY_MARKET_CAL_SHRINK = float(os.getenv("PATHWAY_MARKET_CAL_SHRINK", "0.35"))

PATHWAY_FEATURE_COLUMNS = [
    "ko_win_rate_l5_diff",
    "ko_win_rate_career_diff",
    "sub_win_rate_l5_diff",
    "sub_win_rate_career_diff",
    "dec_win_rate_l5_diff",
    "dec_win_rate_career_diff",
    "ko_loss_rate_l5_diff",
    "ko_loss_rate_career_diff",
    "sub_loss_rate_l5_diff",
    "sub_loss_rate_career_diff",
    "dec_loss_rate_l5_diff",
    "dec_loss_rate_career_diff",
    "r1_finish_rate_l5_diff",
    "r1_finish_rate_career_diff",
    "late_finish_rate_l5_diff",
    "late_finish_rate_career_diff",
    "distance_rate_l5_diff",
    "distance_rate_career_diff",
    "cardio_decay_proxy_diff",
    "finish_timing_skew_diff",
    "last_loss_opp_elo_diff",
    "path_opp_ko_x_own_ko_loss",
    "path_opp_td_att_x_own_td_def",
    "path_opp_sub_x_own_sub_loss",
    "path_pace_product_diff",
    "path_stance_mismatch",
    "is_five_round",
]
MARKET_FEATURE_COLUMNS = [
    "mkt_implied_prob",
    "line_move",
]

FEATURE_COLUMNS = [
    # Differential (primary signals — fighter1 minus fighter2)
    "elo_diff",
    "win_rate_diff",
    "last5_winrate_diff",
    "momentum_diff",
    "striking_acc_diff",
    "takedown_acc_diff",
    "sub_avg_diff",
    "ko_rate_diff",
    "sig_strikes_per_min_diff",
    "td_defense_diff",
    "control_time_diff",
    "wc_age_advantage_diff",
    "similar_opp_win_rate_diff",
    "sos_opp_win_rate_diff",
    "avg_opp_elo_diff",
    "short_notice_perf_diff",
    "long_layoff_perf_diff",
    "short_notice_flag_diff",
    "long_layoff_flag_diff",
    "height_diff",
    "reach_diff",
    "stance_matchup",
    "southpaw_advantage",
    "striker_score_diff",
    "grappler_score_diff",
    "striker_vs_grappler",
    "style_clash",
    "days_since_last_fight_diff",
    "experience_diff",
    # CompuBox-style striking (Greco detail; real CompuBox when cached)
    "kd_rate_diff",
    "head_strike_pct_diff",
    "body_strike_pct_diff",
    "leg_strike_pct_diff",
    "distance_strike_pct_diff",
    "clinch_strike_pct_diff",
    "ground_strike_pct_diff",
    "power_proxy_diff",
    # Sherdog career (leakage-safe as-of record when available)
    "sherdog_win_rate_diff",
    "sherdog_experience_diff",
    "sherdog_finish_rate_diff",
    # Prior-sport background tiers / matchup
    "base_level_diff",
    "same_primary_base",
    "base_family_clash",
    "multi_base_flag_diff",
    # Optional news sentiment (0 when API disabled)
    "sentiment_diff",
    # Context
    "is_title_fight",
    "is_main_event",
    "scheduled_rounds",
]
# Append HV block when enabled (A/B via ENABLE_HIGH_VALUE_FEATURES=0)
if ENABLE_HIGH_VALUE_FEATURES:
    FEATURE_COLUMNS = list(FEATURE_COLUMNS) + list(HIGH_VALUE_FEATURE_COLUMNS)
if ENABLE_PATHWAY_FEATURES:
    FEATURE_COLUMNS = list(FEATURE_COLUMNS) + list(PATHWAY_FEATURE_COLUMNS)
if ENABLE_MARKET_FEATURES:
    FEATURE_COLUMNS = list(FEATURE_COLUMNS) + list(MARKET_FEATURE_COLUMNS)

# --- Interaction feature discovery (candidates generated in feature_engineering) ---
INTERACTION_DISCOVERY_ENABLED = os.getenv(
    "INTERACTION_DISCOVERY_ENABLED", "true"
).lower() in ("1", "true", "yes")
INTERACTION_MIN_FEATURES = int(os.getenv("INTERACTION_MIN_FEATURES", "8"))
INTERACTION_MAX_FEATURES = int(os.getenv("INTERACTION_MAX_FEATURES", "12"))
DISCOVERED_INTERACTIONS_PATH = MODELS_DIR / "discovered_interactions.json"

TARGET_COLUMN = "f1_win"
DATE_COLUMN = "event_date"
FIGHT_ID_COLUMN = "fight_id"
# Expected mean(TARGET) after canonical fighter slots (alphabetical f1/f2)
TARGET_MEAN_MIN = float(os.getenv("TARGET_MEAN_MIN", "0.48"))
TARGET_MEAN_MAX = float(os.getenv("TARGET_MEAN_MAX", "0.62"))

# --- Model (ensemble: LightGBM + XGBoost) ---
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CALIBRATION_SIZE = float(os.getenv("CALIBRATION_SIZE", "0.15"))
CALIBRATION_METHOD = os.getenv("CALIBRATION_METHOD", "isotonic")  # isotonic | sigmoid
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "true").lower() in ("1", "true", "yes")
TUNE_ON_TRAIN = os.getenv("TUNE_ON_TRAIN", "false").lower() in ("1", "true", "yes")
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))
CONFORMAL_ALPHA = float(os.getenv("CONFORMAL_ALPHA", "0.10"))
UNCERTAINTY_HIGH_WIDTH = float(os.getenv("UNCERTAINTY_HIGH_WIDTH", "0.22"))

# --- Uncertainty gates (ensemble disagreement + conformal interval width) ---
# Fail-closed: missing metrics → treat as high uncertainty (skip).
UNCERTAINTY_GATES_ENABLED = env_bool("UNCERTAINTY_GATES_ENABLED", "true")

# Paper: selective research velocity (few high-conviction bets)
# Prior generation (2026-07 relaxed): disagree 0.14/0.08; interval 0.58/0.42
PAPER_DISAGREEMENT_SKIP = float(os.getenv("PAPER_DISAGREEMENT_SKIP", "0.11"))
PAPER_DISAGREEMENT_TIGHTEN = float(os.getenv("PAPER_DISAGREEMENT_TIGHTEN", "0.06"))
PAPER_INTERVAL_WIDTH_SKIP = float(os.getenv("PAPER_INTERVAL_WIDTH_SKIP", "0.52"))
PAPER_INTERVAL_WIDTH_TIGHTEN = float(os.getenv("PAPER_INTERVAL_WIDTH_TIGHTEN", "0.36"))
PAPER_UNCERTAINTY_EDGE_BUMP = float(os.getenv("PAPER_UNCERTAINTY_EDGE_BUMP", "0.025"))
PAPER_UNCERTAINTY_KELLY_MULT = float(os.getenv("PAPER_UNCERTAINTY_KELLY_MULT", "0.55"))

# Paper-only: allow tiny tickets on strong +EV rows that fail solely on wide_interval.
# Live stays fail-closed (no override). Does not loosen disagreement / missing gates.
PAPER_WIDE_OVERRIDE_ENABLED = env_bool("PAPER_WIDE_OVERRIDE_ENABLED", "true")
PAPER_WIDE_OVERRIDE_MIN_EDGE = float(os.getenv("PAPER_WIDE_OVERRIDE_MIN_EDGE", "0.08"))
PAPER_WIDE_OVERRIDE_MIN_PROB = float(os.getenv("PAPER_WIDE_OVERRIDE_MIN_PROB", "0.70"))
PAPER_WIDE_OVERRIDE_KELLY_MULT = float(os.getenv("PAPER_WIDE_OVERRIDE_KELLY_MULT", "0.20"))
PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC = float(
    os.getenv("PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC", "0.01")
)  # 1% of bankroll
PAPER_WIDE_OVERRIDE_MAX_PER_CARD = int(os.getenv("PAPER_WIDE_OVERRIDE_MAX_PER_CARD", "2"))

# Live: stricter
LIVE_DISAGREEMENT_SKIP = float(os.getenv("LIVE_DISAGREEMENT_SKIP", "0.08"))
LIVE_DISAGREEMENT_TIGHTEN = float(os.getenv("LIVE_DISAGREEMENT_TIGHTEN", "0.04"))
LIVE_INTERVAL_WIDTH_SKIP = float(os.getenv("LIVE_INTERVAL_WIDTH_SKIP", str(UNCERTAINTY_HIGH_WIDTH)))
LIVE_INTERVAL_WIDTH_TIGHTEN = float(os.getenv("LIVE_INTERVAL_WIDTH_TIGHTEN", "0.14"))
LIVE_UNCERTAINTY_EDGE_BUMP = float(os.getenv("LIVE_UNCERTAINTY_EDGE_BUMP", "0.03"))
LIVE_UNCERTAINTY_KELLY_MULT = float(os.getenv("LIVE_UNCERTAINTY_KELLY_MULT", "0.45"))

# Prior Paper defaults (for comparison logs)
_PAPER_UNCERTAINTY_THRESHOLDS_PREV = {
    "disagreement_skip": 0.14,
    "disagreement_tighten": 0.08,
    "interval_width_skip": 0.58,
    "interval_width_tighten": 0.42,
}

WF_MIN_TRAIN_RATIO = float(os.getenv("WF_MIN_TRAIN_RATIO", "0.60"))
WF_IMPORTANCE_INTERVAL = int(os.getenv("WF_IMPORTANCE_INTERVAL", "400"))
STYLE_BONUS_MAX = float(os.getenv("STYLE_BONUS_MAX", "0.05"))
EDGE_RANK_MIN = float(os.getenv("EDGE_RANK_MIN", "0.0"))
RUN_BACKTEST_ON_TRAIN = os.getenv("RUN_BACKTEST_ON_TRAIN", "true").lower() in (
    "1",
    "true",
    "yes",
)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": RANDOM_STATE,
    "n_estimators": int(os.getenv("LGBM_N_ESTIMATORS", "300")),
}
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "300")),
    "verbosity": 0,
}
DEFAULT_ENSEMBLE_WEIGHTS = [
    float(x)
    for x in os.getenv("ENSEMBLE_WEIGHTS", "0.55,0.45").split(",")
    if x.strip()
]

# --- Sentiment / news (optional) ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
SENTIMENT_CACHE_TTL_HOURS = int(os.getenv("SENTIMENT_CACHE_TTL_HOURS", "12"))
ATTACH_SENTIMENT_ON_INFERENCE = os.getenv(
    "ATTACH_SENTIMENT_ON_INFERENCE", "true"
).lower() in ("1", "true", "yes")

# --- Grok / Ollama narrative analysis (optional, non-blocking, model-first sizing) ---
GROK_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY", "")
GROK_API_KEY_SOURCE = ""
# xAI Grok is disabled by default — local Ollama is the analysis engine.
GROK_ENABLED = env_bool("GROK_ENABLED", "false")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")
GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")
GROK_MAX_FIGHTS = int(os.getenv("GROK_MAX_FIGHTS", "6"))
GROK_MAX_PROPS = int(os.getenv("GROK_MAX_PROPS", "6"))
GROK_TIMEOUT_SEC = int(os.getenv("GROK_TIMEOUT_SEC", "600"))
# Paper default ±10% Kelly tilt; Live uses tighter LIVE_NARRATIVE_* bounds.
GROK_KELLY_ADJ_MIN = float(os.getenv("GROK_KELLY_ADJ_MIN", "0.90"))
GROK_KELLY_ADJ_MAX = float(os.getenv("GROK_KELLY_ADJ_MAX", "1.10"))
GROK_CACHE_TTL_HOURS = int(os.getenv("GROK_CACHE_TTL_HOURS", "12"))
NARRATIVE_TILT_ENABLED = env_bool("NARRATIVE_TILT_ENABLED", "true")
LIVE_NARRATIVE_KELLY_MIN = float(os.getenv("LIVE_NARRATIVE_KELLY_MIN", "0.95"))
LIVE_NARRATIVE_KELLY_MAX = float(os.getenv("LIVE_NARRATIVE_KELLY_MAX", "1.05"))
NARRATIVE_LOW_CONVICTION_FORCE_ONE = env_bool("NARRATIVE_LOW_CONVICTION_FORCE_ONE", "true")
# Path set after LOG_DIR is defined (see below).
NARRATIVE_TILT_LOG = DATA_DIR / "logs" / "narrative_tilts.jsonl"

# Local Ollama analysis (default engine for dashboard "Run Ollama Analysis")
# Prefer a mid-size coder model; 14b often times out on CPU for card narrate.
OLLAMA_ENABLED = env_bool("OLLAMA_ENABLED", "true")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_FALLBACK_MODELS = os.getenv(
    "OLLAMA_FALLBACK_MODELS",
    "qwen2.5-coder:7b,llama3.2:3b,qwen2.5:7b",
)
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "600"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_USE_CHAT_API = env_bool("OLLAMA_USE_CHAT_API", "true")
OLLAMA_JSON_FORMAT = env_bool("OLLAMA_JSON_FORMAT", "true")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
# Local vision model for fighter-photo desk (empty = auto-detect llava / qwen2.5vl / …)
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "")
OLLAMA_VISION_TIMEOUT_SEC = int(os.getenv("OLLAMA_VISION_TIMEOUT_SEC", "45"))
PHOTO_ANALYSIS_ENABLED = env_bool("PHOTO_ANALYSIS_ENABLED", "true")

# --- Backtest / bankroll ---
INITIAL_BANKROLL = float(os.getenv("INITIAL_BANKROLL", "75"))
CARD_BUDGET = float(os.getenv("CARD_BUDGET", "12"))
FLAT_STAKE = float(os.getenv("FLAT_STAKE", "10"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.03"))  # model prob minus implied prob
EDGE_THRESHOLDS = [
    float(x)
    for x in os.getenv("EDGE_THRESHOLDS", "0,0.02,0.03,0.05,0.08,0.10").split(",")
    if x.strip()
]

# --- Monte Carlo risk analysis ---
MC_SIMULATIONS = int(os.getenv("MC_SIMULATIONS", "10000"))
MC_CARD_SIMULATIONS = int(os.getenv("MC_CARD_SIMULATIONS", "5000"))
MC_CONFIDENCE_LEVEL = float(os.getenv("MC_CONFIDENCE_LEVEL", "0.95"))
MC_RUIN_THRESHOLD_FRACTION = float(os.getenv("MC_RUIN_THRESHOLD_FRACTION", "0.5"))
MC_MAX_CARD_RISK_FRACTION = float(os.getenv("MC_MAX_CARD_RISK_FRACTION", "0.08"))
MC_MIN_CARD_RISK_FRACTION = float(os.getenv("MC_MIN_CARD_RISK_FRACTION", "0.02"))
MC_MAX_BET_FRACTION = float(os.getenv("MC_MAX_BET_FRACTION", "0.02"))
MC_MIN_BET_FRACTION = float(os.getenv("MC_MIN_BET_FRACTION", "0.005"))
MC_HIGH_DRAWDOWN_WARN_PCT = float(os.getenv("MC_HIGH_DRAWDOWN_WARN_PCT", "25"))
MC_HIGH_RUIN_WARN_PROB = float(os.getenv("MC_HIGH_RUIN_WARN_PROB", "0.05"))
MC_ROLLING_CARD_WINDOW = int(os.getenv("MC_ROLLING_CARD_WINDOW", "3"))

# --- Inference ---
CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "0.65"))
CONFIDENCE_MEDIUM = float(os.getenv("CONFIDENCE_MEDIUM", "0.58"))

# --- The Odds API (https://the-odds-api.com) — free tier primary ---
# Set THE_ODDS_API_KEY or ODDS_API_KEY in .env — never commit the real key.
# Free tier is quota-limited; cache aggressively and refresh sparingly.
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY", "")
ODDS_API_KEY_SOURCE = ""
ODDS_API_BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
ODDS_API_SPORT = os.getenv("ODDS_API_SPORT", "mma_mixed_martial_arts")
ODDS_API_REGIONS = os.getenv("ODDS_API_REGIONS", "us,eu,uk")
ODDS_API_MARKETS = os.getenv("ODDS_API_MARKETS", "h2h")
ODDS_API_PROP_MARKETS = os.getenv("ODDS_API_PROP_MARKETS", "totals")
ODDS_API_ODDS_FORMAT = os.getenv("ODDS_API_ODDS_FORMAT", "decimal")  # decimal | american
# Cap per-event prop API calls (each event burns free-tier credits).
ODDS_API_PROP_MAX_EVENTS = int(os.getenv("ODDS_API_PROP_MAX_EVENTS", "6"))
ODDS_CACHE_PATH = CACHE_DIR / "ufc_odds_api.csv"
# Prefer minutes (15–30) so free-tier quota lasts. Default 20 minutes.
# Legacy ODDS_CACHE_TTL_HOURS still honored when ODDS_CACHE_TTL_MINUTES is unset.
_ODDS_TTL_MIN_ENV = os.getenv("ODDS_CACHE_TTL_MINUTES")
_ODDS_TTL_H_ENV = os.getenv("ODDS_CACHE_TTL_HOURS")
if _ODDS_TTL_MIN_ENV is not None:
    ODDS_CACHE_TTL_MINUTES = max(1, int(_ODDS_TTL_MIN_ENV or "20"))
elif _ODDS_TTL_H_ENV is not None:
    ODDS_CACHE_TTL_MINUTES = max(1, int(round(float(_ODDS_TTL_H_ENV or "0.33") * 60)))
else:
    ODDS_CACHE_TTL_MINUTES = 20
ODDS_CACHE_TTL_HOURS = ODDS_CACHE_TTL_MINUTES / 60.0
# When true (default): reuse any non-empty odds cache forever until deleted —
# Soft Update / Refresh / Quick Odds will not burn another Odds API credit.
ODDS_FETCH_ONCE = env_bool("ODDS_FETCH_ONCE", "true")

# Action Network UFC scoreboard (free backup scrape — fail-soft)
ACTION_NETWORK_ENABLED = env_bool("ACTION_NETWORK_ENABLED", "true")
ACTION_NETWORK_UFC_URL = os.getenv(
    "ACTION_NETWORK_UFC_URL",
    "https://api.actionnetwork.com/web/v1/scoreboard/ufc",
)

# Optional book scrapers / DK API pass — OFF by default (free sources only)
BETNOW_ENABLED = env_bool("BETNOW_ENABLED", "false")
DRAFTKINGS_ENABLED = env_bool("DRAFTKINGS_ENABLED", "false")
BETNOW_PROPS_URL = os.getenv(
    "BETNOW_PROPS_URL",
    "https://www.betnow.eu/sportsbook-info/fighting/ufc/",
)
BETNOW_COOKIE = os.getenv("BETNOW_COOKIE", "")
# URL session token for BetNow (BETNOW_SESSION / BETNOW_SESSION_TOKEN / SESSION_TOKEN)
BETNOW_SESSION_TOKEN = (
    os.getenv("BETNOW_SESSION")
    or os.getenv("BETNOW_SESSION_TOKEN")
    or os.getenv("SESSION_TOKEN")
    or ""
).strip()
MYBOOKIE_ENABLED = env_bool("MYBOOKIE_ENABLED", "false")
MYBOOKIE_UFC_URL = os.getenv("MYBOOKIE_UFC_URL", "https://www.mybookie.ag/sportsbook/ufc/")
MYBOOKIE_PROPS_URL = os.getenv("MYBOOKIE_PROPS_URL", "https://www.mybookie.ag/sportsbook/ufc/props/")
MYBOOKIE_COOKIE = os.getenv("MYBOOKIE_COOKIE", "")

# --- Live alerts (Discord / Telegram) ---
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ALERT_MIN_EDGE = float(os.getenv("ALERT_MIN_EDGE", "0.04"))
ALERT_MIN_PARLAY_EV = float(os.getenv("ALERT_MIN_PARLAY_EV", "0.08"))
ALERT_PARLAY_MIN_EDGE = float(os.getenv("ALERT_PARLAY_MIN_EDGE", "0.03"))
ALERT_PARLAY_MIN_COMBINED_PROB = float(os.getenv("ALERT_PARLAY_MIN_COMBINED_PROB", "0.25"))
ALERT_PARLAY_MAX_LEGS = int(os.getenv("ALERT_PARLAY_MAX_LEGS", "3"))
ALERT_MAX_PARLAYS = int(os.getenv("ALERT_MAX_PARLAYS", "5"))
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "60"))
ALERT_POLL_MINUTES = int(os.getenv("ALERT_POLL_MINUTES", "45"))
ALERT_DRY_RUN = os.getenv("ALERT_DRY_RUN", "false").lower() in ("1", "true", "yes")
ALERT_BOT_NAME = os.getenv("ALERT_BOT_NAME", "UFC Predictor")
ALERT_REQUEST_TIMEOUT_SEC = int(os.getenv("ALERT_REQUEST_TIMEOUT_SEC", "15"))
ALERT_STATE_PATH = CACHE_DIR / "alert_state.json"

# --- Ops / production (ported from trading bot) ---
LOG_DIR = DATA_DIR / "logs"
BET_JOURNAL_CSV = DATA_DIR / "bet_journal.csv"
PREDICTION_BANK_CSV = DATA_DIR / "prediction_bank.csv"
PREDICTION_LESSONS_JSON = DATA_DIR / "prediction_lessons.json"
PREDICTION_BANK_THINK_MODEL = os.getenv("PREDICTION_BANK_THINK_MODEL", "deepseek-r1:8b")
PREDICTION_BANK_THINK_TIMEOUT_SEC = int(os.getenv("PREDICTION_BANK_THINK_TIMEOUT_SEC", "600"))
PREDICTION_BANK_AUTO_LOG = env_bool("PREDICTION_BANK_AUTO_LOG", "true")
NARRATIVE_TILT_LOG = LOG_DIR / "narrative_tilts.jsonl"
HEARTBEAT_PATH = CACHE_DIR / "heartbeat.json"
CIRCUIT_BREAKER_STATE_PATH = CACHE_DIR / "circuit_breaker_state.json"
DRAWDOWN_STATE_PATH = CACHE_DIR / "drawdown_state.json"
RISK_EVENTS_LOG = LOG_DIR / "risk_events.log"

# --- Strategy rating feedback (trading-bot style segment Kelly clamps) ---
# Paper on by default; Live requires explicit STRATEGY_RATING_LIVE_ENABLED=true.
STRATEGY_RATING_ENABLED = env_bool("STRATEGY_RATING_ENABLED", "true")
STRATEGY_RATING_LIVE_ENABLED = env_bool("STRATEGY_RATING_LIVE_ENABLED", "false")
STRATEGY_RATING_MIN_TRADES = int(os.getenv("STRATEGY_RATING_MIN_TRADES", "8"))
STRATEGY_RATING_LOOKBACK_DAYS = int(os.getenv("STRATEGY_RATING_LOOKBACK_DAYS", "365"))
STRATEGY_RATING_MULT_MIN = float(os.getenv("STRATEGY_RATING_MULT_MIN", "0.80"))
STRATEGY_RATING_MULT_MAX = float(os.getenv("STRATEGY_RATING_MULT_MAX", "1.20"))
STRATEGY_RATING_CACHE_SEC = float(os.getenv("STRATEGY_RATING_CACHE_SEC", "3600"))
STRATEGY_RATING_STACK_BLEND = float(os.getenv("STRATEGY_RATING_STACK_BLEND", "0.5"))
STRATEGY_METRICS_DB = DATA_DIR / "strategy_metrics.db"
STRATEGY_PERFORMANCE_JSON = DATA_DIR / "strategy_performance.json"

# --- Profile: paper (simulation) vs live (real money) ---

def normalize_profile(name: str | None) -> str:
    """Map legacy 'research' → 'paper'; default paper."""
    n = (name or "paper").strip().lower()
    if n in ("research", "paper", "sim", "simulation"):
        return "paper"
    if n == "live":
        return "live"
    return "paper"


UFC_PROFILE = normalize_profile(os.getenv("UFC_PROFILE", "paper"))

_PROFILE_PAPER = {
    "max_card_risk_fraction": float(os.getenv("PAPER_MAX_CARD_RISK", "0.55")),
    "max_bet_fraction": float(os.getenv("PAPER_MAX_BET_FRACTION", "0.10")),
    "max_card_stake_usd": float(os.getenv("PAPER_MAX_CARD_STAKE_USD", "0")) or None,
    "daily_loss_limit_fraction": float(os.getenv("PAPER_DAILY_LOSS_LIMIT", "0.08")),
    "max_drawdown_fraction": float(os.getenv("PAPER_MAX_DRAWDOWN", "0.22")),
    "resume_drawdown_fraction": float(os.getenv("PAPER_RESUME_DRAWDOWN", "0.16")),
    # High-accuracy / low-volume (floors enforced again in high_accuracy_strategy)
    "alert_min_edge": float(os.getenv("PAPER_ALERT_MIN_EDGE", "0.06")),
    "singles_min_model_prob": float(os.getenv("PAPER_SINGLES_MIN_MODEL_PROB", "0.70")),
    "singles_min_confidence": os.getenv("PAPER_SINGLES_MIN_CONFIDENCE", "medium").strip().lower(),
    "max_bets_per_card": int(os.getenv("PAPER_MAX_BETS_PER_CARD", "3")),
    "parlay_min_edge": float(os.getenv("PAPER_PARLAY_MIN_EDGE", "0.055")),
    "parlay_min_combined_prob": float(os.getenv("PAPER_PARLAY_MIN_COMBINED_PROB", "0.40")),
    "parlay_min_ev": float(os.getenv("PAPER_PARLAY_MIN_EV", "0.10")),
    "kelly_fraction": float(os.getenv("PAPER_KELLY_FRACTION", "0.30")),
    "prop_min_model_prob": float(os.getenv("PAPER_PROP_MIN_MODEL_PROB", "0.78")),
    "prop_min_edge": float(os.getenv("PAPER_PROP_MIN_EDGE", "0.05")),
    "prop_max_results": int(os.getenv("PAPER_PROP_MAX_RESULTS", "12")),
    "max_singles_show": int(os.getenv("PAPER_MAX_SINGLES_SHOW", "3")),
    "max_parlays_show": int(os.getenv("PAPER_MAX_PARLAYS_SHOW", "2")),
    "alert_max_parlays": int(os.getenv("PAPER_ALERT_MAX_PARLAYS", "2")),
    "parlay_max_legs": int(os.getenv("PAPER_PARLAY_MAX_LEGS", "2")),
}

# Legacy alias (old env vars / cached manifests)
_PROFILE_RESEARCH = _PROFILE_PAPER

_PROFILE_LIVE = {
    "max_card_risk_fraction": float(os.getenv("LIVE_MAX_CARD_RISK", "0.18")),
    "max_bet_fraction": float(os.getenv("LIVE_MAX_BET_FRACTION", "0.05")),
    "max_card_stake_usd": float(os.getenv("LIVE_MAX_CARD_STAKE_USD", "12")),
    "daily_loss_limit_fraction": float(os.getenv("LIVE_DAILY_LOSS_LIMIT", "0.012")),
    "max_drawdown_fraction": float(os.getenv("LIVE_MAX_DRAWDOWN", "0.06")),
    "resume_drawdown_fraction": float(os.getenv("LIVE_RESUME_DRAWDOWN", "0.05")),
    "alert_min_edge": float(os.getenv("LIVE_ALERT_MIN_EDGE", "0.09")),
    "singles_min_model_prob": float(os.getenv("LIVE_SINGLES_MIN_MODEL_PROB", "0.72")),
    "singles_min_confidence": os.getenv("LIVE_SINGLES_MIN_CONFIDENCE", "high").strip().lower(),
    "max_bets_per_card": int(os.getenv("LIVE_MAX_BETS_PER_CARD", "2")),
    "parlay_min_edge": float(os.getenv("LIVE_PARLAY_MIN_EDGE", "0.08")),
    "parlay_min_combined_prob": float(os.getenv("LIVE_PARLAY_MIN_COMBINED_PROB", "0.45")),
    "parlay_min_ev": float(os.getenv("LIVE_PARLAY_MIN_EV", "0.18")),
    "kelly_fraction": float(os.getenv("LIVE_KELLY_FRACTION", "0.12")),
    "prop_min_model_prob": float(os.getenv("LIVE_PROP_MIN_MODEL_PROB", "0.80")),
    "prop_min_edge": float(os.getenv("LIVE_PROP_MIN_EDGE", "0.06")),
    "prop_max_results": int(os.getenv("LIVE_PROP_MAX_RESULTS", "6")),
    "max_singles_show": int(os.getenv("LIVE_MAX_SINGLES_SHOW", "2")),
    "max_parlays_show": int(os.getenv("LIVE_MAX_PARLAYS_SHOW", "1")),
    "alert_max_parlays": int(os.getenv("LIVE_ALERT_MAX_PARLAYS", "1")),
    "parlay_max_legs": int(os.getenv("LIVE_PARLAY_MAX_LEGS", "2")),
}

CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() in ("1", "true", "yes")
DRAWDOWN_HALT_ENABLED = os.getenv("DRAWDOWN_HALT_ENABLED", "true").lower() in ("1", "true", "yes")
DYNAMIC_THRESHOLDS_ENABLED = os.getenv("UFC_DYNAMIC_THRESHOLDS", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Segment health lookback for threshold feedback (Paper shorter / Live longer)
PAPER_HEALTH_LOOKBACK_DAYS = int(os.getenv("PAPER_HEALTH_LOOKBACK_DAYS", "90"))
LIVE_HEALTH_LOOKBACK_DAYS = int(os.getenv("LIVE_HEALTH_LOOKBACK_DAYS", "180"))
HEALTH_MIN_SETTLED_BETS = int(os.getenv("HEALTH_MIN_SETTLED_BETS", "8"))
# Fail-closed: missing segment health never loosens thresholds
HEALTH_FEEDBACK_ENABLED = env_bool("HEALTH_FEEDBACK_ENABLED", "true")

# Skip-reason scorecard (journal + JSONL + weekly rollup)
SKIP_SCORECARD_JSONL = LOG_DIR / "skip_scorecard.jsonl"
SKIP_SCORECARD_JSON = DATA_DIR / "skip_scorecard.json"
SKIP_SCORECARD_LOOKBACK_DAYS = int(os.getenv("SKIP_SCORECARD_LOOKBACK_DAYS", "7"))

# --- Dashboard / watch intervals (minutes) ---
WATCH_CARD_CHECK_MINUTES = int(os.getenv("UFC_WATCH_CARD_CHECK_MINUTES", "45"))
WATCH_AUTO_ODDS_MINUTES = int(os.getenv("UFC_WATCH_AUTO_ODDS_MINUTES", "12"))
DASHBOARD_AUTO_ODDS_MINUTES = int(os.getenv("UFC_DASHBOARD_AUTO_ODDS_MINUTES", "12"))
DASHBOARD_CARD_CHECK_MINUTES = int(os.getenv("UFC_DASHBOARD_CARD_CHECK_MINUTES", "45"))

# --- Prop betting (method, rounds, decision) ---
ENABLE_PROPS = env_bool("ENABLE_PROPS", "false")
PROP_MIN_EDGE = float(os.getenv("PROP_MIN_EDGE", "0.05"))
PROP_MIN_MODEL_PROB = float(os.getenv("PROP_MIN_MODEL_PROB", "0.78"))
PROP_SHOW_ALL_MIN_PROB = float(os.getenv("PROP_SHOW_ALL_MIN_PROB", "0.12"))
PAPER_PROPS_SHOW_ALL_DEFAULT = env_bool("PAPER_PROPS_SHOW_ALL_DEFAULT", "false")
ARB_NEAR_MARGIN_PCT = float(os.getenv("ARB_NEAR_MARGIN_PCT", "3.0"))
ARB_STAKE_TOTAL = float(os.getenv("ARB_STAKE_TOTAL", "100"))
ARB_ALERT_THRESHOLD_PCT = float(os.getenv("ARB_ALERT_THRESHOLD_PCT", "2.5"))
ARB_ALERT_POLL_SEC = int(os.getenv("ARB_ALERT_POLL_SEC", "45"))
ARB_ALERT_SOUND = env_bool("ARB_ALERT_SOUND", "true")
PROP_MAX_RESULTS = int(os.getenv("PROP_MAX_RESULTS", "12"))
PROP_SYNTHETIC_VIG = float(os.getenv("PROP_SYNTHETIC_VIG", "0.08"))
PROP_PARLAY_MIN_EV = float(os.getenv("PROP_PARLAY_MIN_EV", "0.08"))
PROP_PARLAY_MIN_COMBINED_PROB = float(os.getenv("PROP_PARLAY_MIN_COMBINED_PROB", "0.20"))
PROP_PARLAY_MAX_LEGS_DK = int(os.getenv("PROP_PARLAY_MAX_LEGS_DK", "2"))
PROP_CORRELATION_DISCOUNT = float(os.getenv("PROP_CORRELATION_DISCOUNT", "0.12"))
# High-accuracy: Over 1.5 Rounds is the only bettable prop (reliability study).
_DEFAULT_PROP_MARKETS = "over_1_5_rounds"
PROP_MARKETS = [
    x.strip()
    for x in os.getenv("PROP_MARKETS", _DEFAULT_PROP_MARKETS).split(",")
    if x.strip()
]
ROUND_ROBINS_ENABLED = False
BOOK_PROP_RULES: dict[str, dict[str, Any]] = {
    # Book policy: BetNow + Odds API = prop singles only.
    # DraftKings + MyBookie allow prop/mixed parlays (research / book rules).
    # Live HA still gates ticket parlays via PROP_PARLAYS_ENABLED=False.
    "Odds API": {
        "allow_prop_parlays": False,
        "allow_mixed_parlays": False,
        "max_prop_parlay_legs": 1,
    },
    "BetNow.eu": {
        "allow_prop_parlays": False,
        "allow_mixed_parlays": False,
        "max_prop_parlay_legs": 1,
    },
    "DraftKings": {
        "allow_prop_parlays": True,
        "allow_mixed_parlays": True,
        "max_prop_parlay_legs": int(os.getenv("PROP_PARLAY_MAX_LEGS_DK", "2")),
    },
    "MyBookie": {
        "allow_prop_parlays": True,
        "allow_mixed_parlays": True,
        "max_prop_parlay_legs": 3,
    },
}


def refresh_runtime_env() -> None:
    """Re-read env-backed flags after bootstrap load_dotenv (required for frozen EXE)."""
    global ENABLE_PROPS, MYBOOKIE_ENABLED, ODDS_API_KEY, BETNOW_COOKIE, MYBOOKIE_COOKIE
    global BETNOW_SESSION_TOKEN, BETNOW_ENABLED, DRAFTKINGS_ENABLED
    global ACTION_NETWORK_ENABLED, ACTION_NETWORK_UFC_URL
    global ODDS_CACHE_TTL_MINUTES, ODDS_CACHE_TTL_HOURS, ODDS_FETCH_ONCE
    global ODDS_API_KEY_SOURCE, ODDS_API_BASE_URL, ODDS_API_SPORT
    global ODDS_API_REGIONS, ODDS_API_MARKETS, ODDS_API_PROP_MARKETS, ODDS_API_ODDS_FORMAT
    global INITIAL_BANKROLL, CARD_BUDGET, DEFAULT_TOTAL_BANKROLL, DEFAULT_CARD_BUDGET
    global PROP_MIN_EDGE, PROP_MIN_MODEL_PROB, PROP_MAX_RESULTS, UFC_PROFILE
    global PROP_MARKETS
    global GROK_ENABLED, GROK_API_KEY, GROK_API_KEY_SOURCE, GROK_MODEL, GROK_API_BASE
    global GROK_MAX_FIGHTS, GROK_MAX_PROPS, GROK_TIMEOUT_SEC
    global GROK_KELLY_ADJ_MIN, GROK_KELLY_ADJ_MAX, GROK_CACHE_TTL_HOURS
    global NARRATIVE_TILT_ENABLED, LIVE_NARRATIVE_KELLY_MIN, LIVE_NARRATIVE_KELLY_MAX
    global NARRATIVE_LOW_CONVICTION_FORCE_ONE
    global OLLAMA_ENABLED, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_FALLBACK_MODELS
    global OLLAMA_TIMEOUT_SEC, OLLAMA_NUM_PREDICT, OLLAMA_USE_CHAT_API, OLLAMA_JSON_FORMAT, OLLAMA_NUM_CTX
    global OLLAMA_VISION_MODEL, OLLAMA_VISION_TIMEOUT_SEC, PHOTO_ANALYSIS_ENABLED
    global NEWS_API_KEY, ARB_ALERT_THRESHOLD_PCT, ARB_ALERT_POLL_SEC, ARB_ALERT_SOUND
    global STRATEGY_RATING_ENABLED, STRATEGY_RATING_LIVE_ENABLED
    global STRATEGY_RATING_MIN_TRADES, STRATEGY_RATING_LOOKBACK_DAYS
    global STRATEGY_RATING_MULT_MIN, STRATEGY_RATING_MULT_MAX
    global STRATEGY_RATING_CACHE_SEC, STRATEGY_RATING_STACK_BLEND
    global UNCERTAINTY_GATES_ENABLED, UNCERTAINTY_HIGH_WIDTH
    global PAPER_DISAGREEMENT_SKIP, PAPER_DISAGREEMENT_TIGHTEN
    global PAPER_INTERVAL_WIDTH_SKIP, PAPER_INTERVAL_WIDTH_TIGHTEN
    global PAPER_UNCERTAINTY_EDGE_BUMP, PAPER_UNCERTAINTY_KELLY_MULT
    global PAPER_WIDE_OVERRIDE_ENABLED, PAPER_WIDE_OVERRIDE_MIN_EDGE
    global PAPER_WIDE_OVERRIDE_MIN_PROB, PAPER_WIDE_OVERRIDE_KELLY_MULT
    global PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC, PAPER_WIDE_OVERRIDE_MAX_PER_CARD
    global LIVE_DISAGREEMENT_SKIP, LIVE_DISAGREEMENT_TIGHTEN
    global LIVE_INTERVAL_WIDTH_SKIP, LIVE_INTERVAL_WIDTH_TIGHTEN
    global LIVE_UNCERTAINTY_EDGE_BUMP, LIVE_UNCERTAINTY_KELLY_MULT
    global ENABLE_HIGH_VALUE_FEATURES, FEATURE_COLUMNS, HIGH_VALUE_FEATURE_COLUMNS
    global ENABLE_PATHWAY_FEATURES, ENABLE_MARKET_FEATURES, ENABLE_PATHWAY_MARKET_CAL
    global PATHWAY_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
    global PATHWAY_MARKET_CAL_WIDTH, PATHWAY_MARKET_CAL_SHRINK
    global INTERACTION_DISCOVERY_ENABLED

    ENABLE_PROPS = env_bool("ENABLE_PROPS", "false")
    ENABLE_HIGH_VALUE_FEATURES = env_bool("ENABLE_HIGH_VALUE_FEATURES", "true")
    ENABLE_PATHWAY_FEATURES = env_bool("ENABLE_PATHWAY_FEATURES", "false")
    ENABLE_MARKET_FEATURES = env_bool("ENABLE_MARKET_FEATURES", "false")
    ENABLE_PATHWAY_MARKET_CAL = env_bool("ENABLE_PATHWAY_MARKET_CAL", "false")
    PATHWAY_MARKET_CAL_WIDTH = float(os.getenv("PATHWAY_MARKET_CAL_WIDTH", "0.40"))
    PATHWAY_MARKET_CAL_SHRINK = float(os.getenv("PATHWAY_MARKET_CAL_SHRINK", "0.35"))
    INTERACTION_DISCOVERY_ENABLED = env_bool("INTERACTION_DISCOVERY_ENABLED", "true")
    # Rebuild model feature list so A/B scripts can flip flags at runtime.
    _extra = set(HIGH_VALUE_FEATURE_COLUMNS) | set(PATHWAY_FEATURE_COLUMNS) | set(
        MARKET_FEATURE_COLUMNS
    )
    _base_cols = [c for c in FEATURE_COLUMNS if c not in _extra]
    FEATURE_COLUMNS = list(_base_cols)
    if ENABLE_HIGH_VALUE_FEATURES:
        FEATURE_COLUMNS = list(FEATURE_COLUMNS) + list(HIGH_VALUE_FEATURE_COLUMNS)
    if ENABLE_PATHWAY_FEATURES:
        FEATURE_COLUMNS = list(FEATURE_COLUMNS) + list(PATHWAY_FEATURE_COLUMNS)
    if ENABLE_MARKET_FEATURES:
        FEATURE_COLUMNS = list(FEATURE_COLUMNS) + list(MARKET_FEATURE_COLUMNS)
    MYBOOKIE_ENABLED = env_bool("MYBOOKIE_ENABLED", "false")
    BETNOW_ENABLED = env_bool("BETNOW_ENABLED", "false")
    DRAFTKINGS_ENABLED = env_bool("DRAFTKINGS_ENABLED", "false")
    ACTION_NETWORK_ENABLED = env_bool("ACTION_NETWORK_ENABLED", "true")
    ACTION_NETWORK_UFC_URL = os.getenv(
        "ACTION_NETWORK_UFC_URL",
        "https://api.actionnetwork.com/web/v1/scoreboard/ufc",
    )
    _ttl_min = os.getenv("ODDS_CACHE_TTL_MINUTES")
    _ttl_h = os.getenv("ODDS_CACHE_TTL_HOURS")
    if _ttl_min is not None:
        ODDS_CACHE_TTL_MINUTES = max(1, int(_ttl_min or "20"))
    elif _ttl_h is not None:
        ODDS_CACHE_TTL_MINUTES = max(1, int(round(float(_ttl_h or "0.33") * 60)))
    else:
        ODDS_CACHE_TTL_MINUTES = 20
    ODDS_CACHE_TTL_HOURS = ODDS_CACHE_TTL_MINUTES / 60.0
    ODDS_FETCH_ONCE = env_bool("ODDS_FETCH_ONCE", "true")
    # Prefer shared Odds API reload (source path + forced sport + session clear)
    try:
        from src.odds_providers.odds_api_client import refresh_odds_api_runtime

        refresh_odds_api_runtime()
    except Exception:
        ODDS_API_KEY = (
            (os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY", "")).strip().strip('"').strip("'")
        )
        ODDS_API_KEY_SOURCE = "process environment" if ODDS_API_KEY else ""
        ODDS_API_BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
        ODDS_API_SPORT = "mma_mixed_martial_arts"
        ODDS_API_REGIONS = os.getenv("ODDS_API_REGIONS", "us,eu,uk")
        ODDS_API_MARKETS = os.getenv("ODDS_API_MARKETS", "h2h")
        ODDS_API_PROP_MARKETS = os.getenv("ODDS_API_PROP_MARKETS", "totals")
        ODDS_API_ODDS_FORMAT = os.getenv("ODDS_API_ODDS_FORMAT", "decimal")
    INITIAL_BANKROLL = float(os.getenv("INITIAL_BANKROLL", str(INITIAL_BANKROLL)))
    CARD_BUDGET = float(os.getenv("CARD_BUDGET", str(CARD_BUDGET)))
    DEFAULT_TOTAL_BANKROLL = INITIAL_BANKROLL
    DEFAULT_CARD_BUDGET = CARD_BUDGET
    BETNOW_COOKIE = os.getenv("BETNOW_COOKIE", "")
    BETNOW_SESSION_TOKEN = (
        os.getenv("BETNOW_SESSION")
        or os.getenv("BETNOW_SESSION_TOKEN")
        or os.getenv("SESSION_TOKEN")
        or ""
    ).strip()
    MYBOOKIE_COOKIE = os.getenv("MYBOOKIE_COOKIE", "")
    PROP_MIN_EDGE = float(os.getenv("PROP_MIN_EDGE", "0.04"))
    PROP_MIN_MODEL_PROB = float(os.getenv("PROP_MIN_MODEL_PROB", "0.21"))
    PROP_MAX_RESULTS = int(os.getenv("PROP_MAX_RESULTS", "24"))
    PROP_MARKETS = [
        x.strip()
        for x in os.getenv("PROP_MARKETS", _DEFAULT_PROP_MARKETS).split(",")
        if x.strip()
    ]
    UFC_PROFILE = normalize_profile(os.getenv("UFC_PROFILE", "paper"))
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    GROK_API_KEY = (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()
    GROK_ENABLED = env_bool("GROK_ENABLED", "false")
    if not GROK_API_KEY_SOURCE and GROK_API_KEY:
        GROK_API_KEY_SOURCE = "environment"
    GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")
    GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")
    GROK_MAX_FIGHTS = int(os.getenv("GROK_MAX_FIGHTS", "6"))
    GROK_MAX_PROPS = int(os.getenv("GROK_MAX_PROPS", "6"))
    GROK_TIMEOUT_SEC = int(os.getenv("GROK_TIMEOUT_SEC", "600"))
    GROK_KELLY_ADJ_MIN = float(os.getenv("GROK_KELLY_ADJ_MIN", "0.90"))
    GROK_KELLY_ADJ_MAX = float(os.getenv("GROK_KELLY_ADJ_MAX", "1.10"))
    GROK_CACHE_TTL_HOURS = int(os.getenv("GROK_CACHE_TTL_HOURS", "12"))
    NARRATIVE_TILT_ENABLED = env_bool("NARRATIVE_TILT_ENABLED", "true")
    LIVE_NARRATIVE_KELLY_MIN = float(os.getenv("LIVE_NARRATIVE_KELLY_MIN", "0.95"))
    LIVE_NARRATIVE_KELLY_MAX = float(os.getenv("LIVE_NARRATIVE_KELLY_MAX", "1.05"))
    NARRATIVE_LOW_CONVICTION_FORCE_ONE = env_bool("NARRATIVE_LOW_CONVICTION_FORCE_ONE", "true")
    OLLAMA_ENABLED = env_bool("OLLAMA_ENABLED", "true")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    OLLAMA_FALLBACK_MODELS = os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "qwen2.5-coder:7b,llama3.2:3b,qwen2.5:7b",
    )
    OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "600"))
    OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
    OLLAMA_USE_CHAT_API = env_bool("OLLAMA_USE_CHAT_API", "true")
    OLLAMA_JSON_FORMAT = env_bool("OLLAMA_JSON_FORMAT", "true")
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "")
    OLLAMA_VISION_TIMEOUT_SEC = int(os.getenv("OLLAMA_VISION_TIMEOUT_SEC", "45"))
    PHOTO_ANALYSIS_ENABLED = env_bool("PHOTO_ANALYSIS_ENABLED", "true")
    ARB_ALERT_THRESHOLD_PCT = float(os.getenv("ARB_ALERT_THRESHOLD_PCT", "2.5"))
    ARB_ALERT_POLL_SEC = int(os.getenv("ARB_ALERT_POLL_SEC", "45"))
    ARB_ALERT_SOUND = env_bool("ARB_ALERT_SOUND", "true")
    STRATEGY_RATING_ENABLED = env_bool("STRATEGY_RATING_ENABLED", "true")
    STRATEGY_RATING_LIVE_ENABLED = env_bool("STRATEGY_RATING_LIVE_ENABLED", "false")
    STRATEGY_RATING_MIN_TRADES = int(os.getenv("STRATEGY_RATING_MIN_TRADES", "8"))
    STRATEGY_RATING_LOOKBACK_DAYS = int(os.getenv("STRATEGY_RATING_LOOKBACK_DAYS", "365"))
    STRATEGY_RATING_MULT_MIN = float(os.getenv("STRATEGY_RATING_MULT_MIN", "0.80"))
    STRATEGY_RATING_MULT_MAX = float(os.getenv("STRATEGY_RATING_MULT_MAX", "1.20"))
    STRATEGY_RATING_CACHE_SEC = float(os.getenv("STRATEGY_RATING_CACHE_SEC", "3600"))
    STRATEGY_RATING_STACK_BLEND = float(os.getenv("STRATEGY_RATING_STACK_BLEND", "0.5"))
    UNCERTAINTY_GATES_ENABLED = env_bool("UNCERTAINTY_GATES_ENABLED", "true")
    UNCERTAINTY_HIGH_WIDTH = float(os.getenv("UNCERTAINTY_HIGH_WIDTH", "0.22"))
    PAPER_DISAGREEMENT_SKIP = float(os.getenv("PAPER_DISAGREEMENT_SKIP", "0.11"))
    PAPER_DISAGREEMENT_TIGHTEN = float(os.getenv("PAPER_DISAGREEMENT_TIGHTEN", "0.06"))
    PAPER_INTERVAL_WIDTH_SKIP = float(os.getenv("PAPER_INTERVAL_WIDTH_SKIP", "0.52"))
    PAPER_INTERVAL_WIDTH_TIGHTEN = float(os.getenv("PAPER_INTERVAL_WIDTH_TIGHTEN", "0.36"))
    PAPER_UNCERTAINTY_EDGE_BUMP = float(os.getenv("PAPER_UNCERTAINTY_EDGE_BUMP", "0.025"))
    PAPER_UNCERTAINTY_KELLY_MULT = float(os.getenv("PAPER_UNCERTAINTY_KELLY_MULT", "0.55"))
    global PAPER_WIDE_OVERRIDE_ENABLED, PAPER_WIDE_OVERRIDE_MIN_EDGE
    global PAPER_WIDE_OVERRIDE_MIN_PROB, PAPER_WIDE_OVERRIDE_KELLY_MULT
    global PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC, PAPER_WIDE_OVERRIDE_MAX_PER_CARD
    PAPER_WIDE_OVERRIDE_ENABLED = env_bool("PAPER_WIDE_OVERRIDE_ENABLED", "true")
    PAPER_WIDE_OVERRIDE_MIN_EDGE = float(os.getenv("PAPER_WIDE_OVERRIDE_MIN_EDGE", "0.08"))
    PAPER_WIDE_OVERRIDE_MIN_PROB = float(os.getenv("PAPER_WIDE_OVERRIDE_MIN_PROB", "0.70"))
    PAPER_WIDE_OVERRIDE_KELLY_MULT = float(os.getenv("PAPER_WIDE_OVERRIDE_KELLY_MULT", "0.20"))
    PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC = float(
        os.getenv("PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC", "0.01")
    )
    PAPER_WIDE_OVERRIDE_MAX_PER_CARD = int(os.getenv("PAPER_WIDE_OVERRIDE_MAX_PER_CARD", "2"))
    LIVE_DISAGREEMENT_SKIP = float(os.getenv("LIVE_DISAGREEMENT_SKIP", "0.08"))
    LIVE_DISAGREEMENT_TIGHTEN = float(os.getenv("LIVE_DISAGREEMENT_TIGHTEN", "0.04"))
    LIVE_INTERVAL_WIDTH_SKIP = float(os.getenv("LIVE_INTERVAL_WIDTH_SKIP", str(UNCERTAINTY_HIGH_WIDTH)))
    LIVE_INTERVAL_WIDTH_TIGHTEN = float(os.getenv("LIVE_INTERVAL_WIDTH_TIGHTEN", "0.14"))
    LIVE_UNCERTAINTY_EDGE_BUMP = float(os.getenv("LIVE_UNCERTAINTY_EDGE_BUMP", "0.03"))
    LIVE_UNCERTAINTY_KELLY_MULT = float(os.getenv("LIVE_UNCERTAINTY_KELLY_MULT", "0.45"))
    try:
        log_paper_uncertainty_threshold_delta()
    except Exception:
        pass
    try:
        log_decision_layer_thresholds()
    except Exception:
        pass


def is_live_profile() -> bool:
    return UFC_PROFILE == "live"


def is_paper_profile() -> bool:
    return not is_live_profile()


def profile_label() -> str:
    return "LIVE" if is_live_profile() else "PAPER"


def effective_strategy_rating_enabled() -> bool:
    """Paper uses STRATEGY_RATING_ENABLED; Live also requires LIVE flag."""
    if not STRATEGY_RATING_ENABLED:
        return False
    if is_live_profile():
        return bool(STRATEGY_RATING_LIVE_ENABLED)
    return True


def uncertainty_gate_settings() -> dict[str, float | bool]:
    """Paper vs Live disagreement / interval thresholds for betting gates."""
    if is_live_profile():
        return {
            "enabled": bool(UNCERTAINTY_GATES_ENABLED),
            "disagreement_skip": float(LIVE_DISAGREEMENT_SKIP),
            "disagreement_tighten": float(LIVE_DISAGREEMENT_TIGHTEN),
            "interval_width_skip": float(LIVE_INTERVAL_WIDTH_SKIP),
            "interval_width_tighten": float(LIVE_INTERVAL_WIDTH_TIGHTEN),
            "edge_bump": float(LIVE_UNCERTAINTY_EDGE_BUMP),
            "kelly_mult": float(LIVE_UNCERTAINTY_KELLY_MULT),
        }
    return {
        "enabled": bool(UNCERTAINTY_GATES_ENABLED),
        "disagreement_skip": float(PAPER_DISAGREEMENT_SKIP),
        "disagreement_tighten": float(PAPER_DISAGREEMENT_TIGHTEN),
        "interval_width_skip": float(PAPER_INTERVAL_WIDTH_SKIP),
        "interval_width_tighten": float(PAPER_INTERVAL_WIDTH_TIGHTEN),
        "edge_bump": float(PAPER_UNCERTAINTY_EDGE_BUMP),
        "kelly_mult": float(PAPER_UNCERTAINTY_KELLY_MULT),
    }


def decision_layer_settings() -> dict[str, Any]:
    """
    Active profile decision-layer thresholds (singles selectivity + uncertainty).

    Used by alerts, strategy, dashboard summary, and startup logs.
    Includes hard-coded high-accuracy strategy rules.
    """
    ps = profile_settings()
    unc = uncertainty_gate_settings()
    ha: dict[str, Any] = {}
    try:
        from src.high_accuracy_strategy import strategy_rules_summary, format_strategy_rules_line

        ha = strategy_rules_summary()
        ha_line = format_strategy_rules_line()
    except Exception:
        ha_line = ""
    return {
        "profile": UFC_PROFILE,
        "min_edge": float(ps.get("alert_min_edge") or ALERT_MIN_EDGE),
        "singles_min_model_prob": float(ps.get("singles_min_model_prob") or 0.0),
        "singles_min_confidence": str(ps.get("singles_min_confidence") or "low"),
        "max_bets_per_card": int(ps.get("max_bets_per_card") or 3),
        "max_tickets_per_card": int(ps.get("max_bets_per_card") or 3),
        "max_singles_show": int(ps.get("max_singles_show") or 3),
        "parlay_max_legs": int(ps.get("parlay_max_legs") or 2),
        "prop_markets": list(PROP_MARKETS),
        "prop_min_model_prob": float(ps.get("prop_min_model_prob") or PROP_MIN_MODEL_PROB),
        "prop_min_edge": float(ps.get("prop_min_edge") or PROP_MIN_EDGE),
        "round_robins_enabled": bool(ROUND_ROBINS_ENABLED),
        "high_accuracy": ha,
        "strategy_line": ha_line,
        "uncertainty": {
            "enabled": bool(unc.get("enabled", True)),
            "disagreement_skip": float(unc["disagreement_skip"]),
            "disagreement_tighten": float(unc["disagreement_tighten"]),
            "interval_width_skip": float(unc["interval_width_skip"]),
            "interval_width_tighten": float(unc["interval_width_tighten"]),
            "edge_bump": float(unc.get("edge_bump") or 0.0),
            "kelly_mult": float(unc.get("kelly_mult") or 1.0),
        },
    }


def log_decision_layer_thresholds(*, force: bool = False) -> None:
    """Log active decision-layer thresholds once per process."""
    import logging

    log = logging.getLogger("config")
    if not force and getattr(log_decision_layer_thresholds, "_done", False):
        return
    log_decision_layer_thresholds._done = True  # type: ignore[attr-defined]
    d = decision_layer_settings()
    u = d["uncertainty"]
    log.info(
        "Decision layer [%s]: min_edge=%.1f%% min_model_prob=%.0f%% "
        "min_confidence=%s max_tickets/card=%s parlay_legs=%s props=%s RR=%s | uncertainty "
        "disagree skip=%.2f/tighten=%.2f interval skip=%.2f/tighten=%.2f "
        "edge_bump=%.3f kelly_mult=%.2f",
        d["profile"],
        100.0 * float(d["min_edge"]),
        100.0 * float(d["singles_min_model_prob"]),
        d["singles_min_confidence"],
        d["max_bets_per_card"],
        d.get("parlay_max_legs"),
        ",".join(d.get("prop_markets") or []) or "none",
        d.get("round_robins_enabled"),
        u["disagreement_skip"],
        u["disagreement_tighten"],
        u["interval_width_skip"],
        u["interval_width_tighten"],
        u["edge_bump"],
        u["kelly_mult"],
    )
    if d.get("strategy_line"):
        log.info("%s", d["strategy_line"])


def log_paper_uncertainty_threshold_delta(*, force: bool = False) -> None:
    """Log Paper uncertainty old vs new thresholds once per process (comparison aid)."""
    import logging

    log = logging.getLogger("config")
    if not force and getattr(log_paper_uncertainty_threshold_delta, "_done", False):
        return
    log_paper_uncertainty_threshold_delta._done = True  # type: ignore[attr-defined]
    prev = _PAPER_UNCERTAINTY_THRESHOLDS_PREV
    cur = {
        "disagreement_skip": float(PAPER_DISAGREEMENT_SKIP),
        "disagreement_tighten": float(PAPER_DISAGREEMENT_TIGHTEN),
        "interval_width_skip": float(PAPER_INTERVAL_WIDTH_SKIP),
        "interval_width_tighten": float(PAPER_INTERVAL_WIDTH_TIGHTEN),
    }
    live = {
        "disagreement_skip": float(LIVE_DISAGREEMENT_SKIP),
        "disagreement_tighten": float(LIVE_DISAGREEMENT_TIGHTEN),
        "interval_width_skip": float(LIVE_INTERVAL_WIDTH_SKIP),
        "interval_width_tighten": float(LIVE_INTERVAL_WIDTH_TIGHTEN),
    }
    parts = []
    for key in cur:
        o, n = prev[key], cur[key]
        mark = " (changed)" if abs(o - n) > 1e-12 else ""
        parts.append(f"{key}: {o:.2f}->{n:.2f}{mark}")
    log.info(
        "Paper uncertainty gates (old -> new): %s | Live (unchanged): "
        "disagree skip=%.2f/tighten=%.2f interval skip=%.2f/tighten=%.2f | "
        "fail-closed if metrics missing",
        "; ".join(parts),
        live["disagreement_skip"],
        live["disagreement_tighten"],
        live["interval_width_skip"],
        live["interval_width_tighten"],
    )


# Emit comparison once when config is imported (logger may be root/basic).
try:
    log_paper_uncertainty_threshold_delta()
except Exception:
    pass
try:
    log_decision_layer_thresholds()
except Exception:
    pass


def narrative_kelly_bounds() -> tuple[float, float]:
    """Profile-aware Kelly tilt clamp. Paper ±10%; Live tighter by default."""
    if is_live_profile():
        lo = float(LIVE_NARRATIVE_KELLY_MIN)
        hi = float(LIVE_NARRATIVE_KELLY_MAX)
    else:
        lo = float(GROK_KELLY_ADJ_MIN)
        hi = float(GROK_KELLY_ADJ_MAX)
    if lo > hi:
        lo, hi = hi, lo
    # Hard safety: never wider than ±15% even if misconfigured
    lo = max(0.85, min(1.0, lo))
    hi = min(1.15, max(1.0, hi))
    return lo, hi


def profile_settings() -> dict[str, Any]:
    raw = dict(_PROFILE_LIVE if is_live_profile() else _PROFILE_PAPER)
    try:
        from src.high_accuracy_strategy import apply_hardcoded_profile_defaults

        return apply_hardcoded_profile_defaults(raw, live=is_live_profile())
    except Exception:
        return raw


def profile_value(key: str) -> Any:
    return profile_settings()[key]


def max_card_stake_cap(bankroll: float | None = None) -> float:
    """Max dollars to risk on one card (fraction cap + live USD hard cap)."""
    br = max(float(bankroll if bankroll is not None else INITIAL_BANKROLL), 1.0)
    ps = profile_settings()
    cap = br * float(ps["max_card_risk_fraction"])
    usd_cap = ps.get("max_card_stake_usd")
    if is_live_profile() and usd_cap:
        cap = min(cap, float(usd_cap))
    return cap


def default_card_budget_usd(
    bankroll: float | None = None,
    *,
    profile: str | None = None,
) -> float:
    """
    Default card budget = bankroll × profile card-risk % (Paper/Live).

    Live also respects the hard USD card stake cap.
    """
    br = max(float(bankroll if bankroll is not None else INITIAL_BANKROLL), 0.0)
    live = is_live_profile() if profile is None else normalize_profile(profile) == "live"
    raw = dict(_PROFILE_LIVE if live else _PROFILE_PAPER)
    frac = float(raw.get("max_card_risk_fraction") or (0.18 if live else 0.55))
    cap = br * frac
    if live:
        usd_cap = float(raw.get("max_card_stake_usd") or LIVE_MAX_CARD_BUDGET_USD)
        cap = min(cap, usd_cap)
    return round(max(cap, 0.0), 2)


def profile_card_risk_fraction(*, profile: str | None = None) -> float:
    """Paper/Live max_card_risk_fraction used for default card budget."""
    live = is_live_profile() if profile is None else normalize_profile(profile) == "live"
    raw = dict(_PROFILE_LIVE if live else _PROFILE_PAPER)
    return float(raw.get("max_card_risk_fraction") or (0.18 if live else 0.55))


def effective_max_card_risk_fraction(bankroll: float | None = None) -> float:
    br = max(float(bankroll if bankroll is not None else INITIAL_BANKROLL), 1.0)
    return max_card_stake_cap(br) / br


def profile_int(key: str) -> int:
    return int(profile_settings()[key])


def estimated_card_stake_usd(alerts: dict[str, Any]) -> float:
    """Sum suggested stakes from alert singles + parlays."""
    total = 0.0
    for s in alerts.get("singles") or []:
        total += float(s.get("suggested_stake") or 0)
    for p in alerts.get("parlays") or []:
        total += float(p.get("suggested_stake") or 0)
    return total


def live_card_risk_warning(alerts: dict[str, Any], bankroll: float | None = None) -> str | None:
    """Live-only warning when suggested card exposure exceeds the profile cap."""
    if not is_live_profile():
        return None
    br = float(bankroll if bankroll is not None else INITIAL_BANKROLL)
    cap = max_card_stake_cap(br)
    stake = estimated_card_stake_usd(alerts)
    if stake <= cap:
        return None
    return (
        f"HIGH CARD RISK: suggested stakes ${stake:,.2f} exceed live cap "
        f"${cap:,.2f} (${br:,.0f} bankroll). Reduce exposure before betting."
    )


def apply_profile_overrides() -> None:
    """Apply profile caps to module-level defaults (call at startup)."""
    global ALERT_MIN_EDGE, MC_MAX_CARD_RISK_FRACTION, MC_MAX_BET_FRACTION
    global ALERT_MIN_PARLAY_EV, ALERT_PARLAY_MIN_EDGE, ALERT_PARLAY_MIN_COMBINED_PROB
    global PROP_MIN_EDGE, PROP_MIN_MODEL_PROB, PROP_MAX_RESULTS, PROP_MARKETS
    global ALERT_MAX_PARLAYS, ALERT_PARLAY_MAX_LEGS
    ps = profile_settings()
    ALERT_MIN_EDGE = ps["alert_min_edge"]
    ALERT_PARLAY_MIN_EDGE = ps["parlay_min_edge"]
    ALERT_PARLAY_MIN_COMBINED_PROB = ps["parlay_min_combined_prob"]
    ALERT_MIN_PARLAY_EV = ps["parlay_min_ev"]
    MC_MAX_CARD_RISK_FRACTION = effective_max_card_risk_fraction(INITIAL_BANKROLL)
    MC_MAX_BET_FRACTION = ps["max_bet_fraction"]
    PROP_MIN_EDGE = ps["prop_min_edge"]
    PROP_MIN_MODEL_PROB = ps["prop_min_model_prob"]
    PROP_MAX_RESULTS = int(ps["prop_max_results"])
    ALERT_MAX_PARLAYS = int(ps["alert_max_parlays"])
    # Hard-code: 2-leg parlays only
    ALERT_PARLAY_MAX_LEGS = 2
    try:
        from src.high_accuracy_strategy import ALLOWED_PROP_KEYS, PARLAY_MAX_LEGS

        ALERT_PARLAY_MAX_LEGS = int(PARLAY_MAX_LEGS)
        # Keep only allowed prop markets (Over 1.5)
        PROP_MARKETS = [m for m in PROP_MARKETS if m in ALLOWED_PROP_KEYS] or list(ALLOWED_PROP_KEYS)
    except Exception:
        PROP_MARKETS = ["over_1_5_rounds"]
        ALERT_PARLAY_MAX_LEGS = 2


# --- Budget manager (dashboard) ---
BUDGET_JSON_PATH = DATA_DIR / "budget.json"
DEFAULT_TOTAL_BANKROLL = INITIAL_BANKROLL
DEFAULT_CARD_BUDGET = CARD_BUDGET
LIVE_MAX_CARD_BUDGET_USD = float(os.getenv("LIVE_MAX_CARD_BUDGET_USD", "12"))
LIVE_SMALL_BANKROLL_USD = 100.0

BUDGET_BOOKS: tuple[str, ...] = ("BetNow.eu", "DraftKings", "MyBookie")
BUDGET_BALANCE_KEYS: dict[str, str] = {
    "BetNow.eu": "betnow_balance",
    "DraftKings": "draftkings_balance",
    "MyBookie": "mybookie_balance",
}
BUDGET_USE_KEYS: dict[str, str] = {
    "BetNow.eu": "use_betnow",
    "DraftKings": "use_draftkings",
    "MyBookie": "use_mybookie",
}


def default_budget_state() -> dict[str, Any]:
    br = float(DEFAULT_TOTAL_BANKROLL)
    return {
        "total_bankroll": br,
        "card_budget": float(default_card_budget_usd(br, profile=UFC_PROFILE)),
        "betnow_balance": 25.0,
        "draftkings_balance": 25.0,
        "mybookie_balance": 25.0,
        "use_betnow": True,
        "use_draftkings": True,
        "use_mybookie": True,
    }


def normalize_budget_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Merge persisted budget with defaults and coerce types."""
    base = default_budget_state()
    if not raw:
        return base
    for key in base:
        if key not in raw:
            continue
        val = raw[key]
        if key.startswith("use_"):
            base[key] = bool(val)
        else:
            try:
                base[key] = float(val)
            except (TypeError, ValueError):
                pass
    base["use_betnow"] = bool(base["use_betnow"])
    base["use_draftkings"] = bool(base["use_draftkings"])
    base["use_mybookie"] = bool(base["use_mybookie"])
    base["total_bankroll"] = max(float(base["total_bankroll"]), 0.0)
    base["card_budget"] = max(float(base["card_budget"]), 0.0)
    for bal_key in ("betnow_balance", "draftkings_balance", "mybookie_balance"):
        base[bal_key] = max(float(base[bal_key]), 0.0)
    return base


def load_budget() -> dict[str, Any]:
    """Load budget from data/budget.json; fall back to defaults."""
    try:
        if BUDGET_JSON_PATH.is_file():
            import json

            raw = json.loads(BUDGET_JSON_PATH.read_text(encoding="utf-8"))
            return normalize_budget_state(raw if isinstance(raw, dict) else None)
    except Exception:
        pass
    state = default_budget_state()
    state["total_bankroll"] = float(INITIAL_BANKROLL)
    state["card_budget"] = float(CARD_BUDGET)
    return state


def _sync_env_var(env_path: Path, key: str, value: float | str) -> None:
    import re

    text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    line = f"{key}={value}"
    if re.search(rf"^{re.escape(key)}=", text, flags=re.MULTILINE):
        text = re.sub(rf"^{re.escape(key)}=.*$", line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + (f"\n{line}\n" if text else f"{line}\n")
    env_path.write_text(text, encoding="utf-8")


def upsert_env_vars(updates: dict[str, str], *, env_path: Path | None = None) -> Path:
    """
    Write key=value pairs into .env, update os.environ, and refresh runtime flags.

    Used by cookie capture so scrapers see BETNOW_COOKIE / MYBOOKIE_COOKIE immediately.
    """
    path = env_path or (ROOT_DIR / ".env")
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in updates.items():
        val = str(value).strip()
        _sync_env_var(path, key, val)
        os.environ[key] = val
    # Re-load so dotenv-backed aliases stay consistent, then refresh module globals
    try:
        load_dotenv(path, override=True)
    except Exception:
        pass
    refresh_runtime_env()
    return path


def _sync_env_budget(state: dict[str, Any]) -> None:
    """Mirror bankroll and card budget into .env when the file exists."""
    env_path = ROOT_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        _sync_env_var(env_path, "INITIAL_BANKROLL", state["total_bankroll"])
        _sync_env_var(env_path, "CARD_BUDGET", state["card_budget"])
    except Exception:
        pass


def enabled_books_from_budget(budget_state: dict[str, Any] | None) -> set[str]:
    """Books selected in Budget Manager (defaults to all)."""
    state = normalize_budget_state(budget_state)
    enabled = {
        book
        for book in BUDGET_BOOKS
        if state.get(BUDGET_USE_KEYS[book], True)
    }
    return enabled or set(BUDGET_BOOKS)


def save_budget(state: dict[str, Any]) -> dict[str, Any]:
    """Persist budget to data/budget.json and sync INITIAL_BANKROLL."""
    import json

    normalized = normalize_budget_state(state)
    BUDGET_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_JSON_PATH.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    apply_budget_state(normalized)
    _sync_env_budget(normalized)
    return normalized


def apply_budget_state(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply budget bankroll/card cap to module-level defaults."""
    global INITIAL_BANKROLL, CARD_BUDGET
    normalized = normalize_budget_state(state) if state else load_budget()
    INITIAL_BANKROLL = float(normalized["total_bankroll"])
    CARD_BUDGET = float(normalized["card_budget"])
    return normalized


def live_card_budget_cap_usd(bankroll: float | None = None) -> float:
    """Hard USD cap for card budget in Live profile (default $12)."""
    br = max(float(bankroll if bankroll is not None else INITIAL_BANKROLL), 1.0)
    if is_live_profile():
        return float(profile_settings().get("max_card_stake_usd") or LIVE_MAX_CARD_BUDGET_USD)
    return max_card_stake_cap(br)


def live_small_bankroll_warnings(bankroll: float | None = None) -> list[str]:
    """Live-mode warnings when bankroll is small relative to card caps."""
    if not is_live_profile():
        return []
    br = max(float(bankroll if bankroll is not None else INITIAL_BANKROLL), 0.0)
    if br <= 0:
        return ["Bankroll is $0 — set a bankroll before placing live bets."]
    cap = live_card_budget_cap_usd(br)
    pct = cap / br * 100.0
    warnings: list[str] = []
    if br <= LIVE_SMALL_BANKROLL_USD:
        warnings.append(
            f"LIVE + ${br:,.0f} bankroll: one max card (${cap:,.0f}) risks {pct:.0f}% of your roll. "
            "Use 1–2 small bets only."
        )
    if br <= 50:
        warnings.append(
            f"CRITICAL: ${br:,.0f} bankroll is extremely thin for Live — "
            f"${cap:,.0f}/card is {pct:.0f}% exposure. Paper mode recommended until roll grows."
        )
    return warnings

