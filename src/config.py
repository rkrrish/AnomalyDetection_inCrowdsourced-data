# src/config.py
"""Central Configuration for CatBoost Crowdsourced Data Validation System."""

import os

# ──────────────────────────────────────────────
# Base Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "submissions.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "catboost_model.cbm")
TRAIN_STATS_PATH = os.path.join(MODELS_DIR, "train_stats.json")
PEER_STATS_PATH = os.path.join(MODELS_DIR, "peer_stats.json")
MODEL_META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
DB_PATH = os.path.join(BASE_DIR, "data", "system_database.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# ──────────────────────────────────────────────
# Scoring & Decision Parameters
# ──────────────────────────────────────────────
DEFAULT_ANOMALY_THRESHOLD = 0.5072  # Calibrated on validation PR-AUC curve
TRUST_SCORE_BOOST = 0.03            # +3% user trust boost on 'Approve' / high-quality submissions
TRUST_SCORE_PENALTY_CRITICAL = 0.35 # Hard penalty for severe fake/impossible records
TRUST_SCORE_PENALTY_WARNING = 0.15  # Soft penalty for suspicious inconsistencies
INITIAL_USER_TRUST = 1.0            # Baseline trust for new accounts

# Scoring Weights for Final Submission Quality/Confidence Score (0-100)
WEIGHT_RULE_CONSISTENCY = 0.35
WEIGHT_ML_CONFIDENCE = 0.50
WEIGHT_COMPLETENESS = 0.15

# Decision Thresholds
DECISION_APPROVE_MIN_SCORE = 75.0
DECISION_REJECT_MAX_SCORE = 45.0

# ──────────────────────────────────────────────
# Required & Core Fields
# ──────────────────────────────────────────────
REQUIRED_SUBMISSION_FIELDS = [
    "company",
    "jobFamily",
    "level",
    "location",
    "baseSalary",
    "totalCompensation",
    "yearsOfExperience",
    "baseSalaryCurrency",
]

VALID_CURRENCY_CODES = {
    "INR", "USD", "EUR", "GBP", "CAD", "SGD", "AUD", "PLN", "SEK", "CHF", "BRL", "AED",
    "JPY", "CNY", "NZD", "HKD", "KRW", "MXN", "ZAR", "NOK", "DKK", "CZK", "HUF", "ILS",
    "THB", "IDR", "MYR", "PHP", "TWD", "TRY"
}

CURRENCY_ALIASES = {
    "INRR": "INR", "RS": "INR", "RS.": "INR", "RUPEE": "INR", "RUPEES": "INR",
    "US$": "USD", "US DOLLAR": "USD", "EURO": "EUR"
}

WORK_ARRANGEMENT_ALIASES = {
    "hybrid": "hybrid", "onsite": "onsite", "on-site": "onsite", "on site": "onsite",
    "office": "onsite", "in-office": "onsite", "in office": "onsite",
    "remote": "remote", "remoteee": "remote", "wfh": "remote",
    "work from home": "remote", "work-from-home": "remote", "fully remote": "remote",
}
EXPECTED_WORK_ARRANGEMENTS = {"hybrid", "onsite", "remote"}

# ──────────────────────────────────────────────
# Public Companies & Organizations
# ──────────────────────────────────────────────
PUBLIC_COMPANIES = {
    "google", "meta", "amazon", "apple", "microsoft", "netflix",
    "oracle", "salesforce", "servicenow", "snowflake", "adobe", "ibm", "intel",
    "uber", "lyft", "airbnb", "stripe", "linkedin", "twitter", "x",
    "jpmorgan", "goldman sachs", "morgan stanley", "bank of america", "wells fargo",
    "visa", "mastercard", "unitedhealth", "walmart", "target", "boeing",
    "accenture", "infosys", "wipro", "tcs", "capgemini", "cognizant", "hcl",
    "tech mahindra", "genpact", "ltimindtree", "paypal", "swiggy"
}

STARTUP_STAGES = {"seed", "series a", "series b", "series c", "series d", "early stage"}

# ──────────────────────────────────────────────
# Rule Columns used in Validation & Scoring
# ──────────────────────────────────────────────
RULE_COLUMNS = [
    "flag_missing_required",
    "flag_gibberish_company",
    "flag_zero_or_negative_base",
    "flag_tc_less_than_base",
    "flag_tc_arithmetic_mismatch",
    "flag_ftc_arithmetic_mismatch",
    "flag_yal_gt_yac",
    "flag_yac_gt_yoe",
    "flag_negative_experience",
    "flag_yoe_out_of_range",
    "flag_junior_level_high_yoe",
    "flag_offer_after_submit",
    "flag_offer_same_time_of_day",
    "flag_company_name_mismatch",
    "flag_currency_country_mismatch",
    "flag_vest_pct_not_100",
    "flag_vest_malformed",
    "flag_bonus_pct_mismatch",
    "flag_duplicate",
    "flag_burst_submission",
]