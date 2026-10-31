"""train_catboost.py - Production Training & Serialization Script for CatBoost Anomaly Detection.

Implements the end-to-end pipeline tested in CatBoostv2.ipynb:
1. Loads data/submissions.csv.
2. Computes causal (past-only) behaviour features (prior submission counts, hours since previous).
3. Constructs connected-component groups (userId, deviceHash, ipHash) to eliminate actor/ring leakage.
4. Performs 60/20/20 grouped & stratified train/val/test split.
5. Fits train statistics: USD conversion rate map, modal country currencies, and frequency tables.
6. Computes hierarchical peer statistics (cjll -> cjl -> jll -> cj -> jl -> j) for base & total compensation.
7. Trains CatBoostClassifier using the validated best hyperparameters.
8. Calibrates optimal decision threshold on the validation set.
9. Evaluates performance on train, validation, and holdout test sets.
10. Serializes all artifacts to models/:
    - models/catboost_model.cbm
    - models/train_stats.json
    - models/peer_stats.json
    - models/model_metadata.json

Usage:
    python train_catboost.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
SEED = 42
DATA_PATH = os.path.join("data", "submissions.csv")
MODELS_DIR = "models"
MIN_PEER_N = 5
USE_CALENDAR_YEAR_MONTH = False

# Exact best hyperparameters discovered in CatBoostv2.ipynb
BEST_CATBOOST_PARAMS: Dict[str, Any] = {
    "iterations": 4000,
    "learning_rate": 0.031707193242141093,
    "depth": 9,
    "l2_leaf_reg": 2.411289811529197,
    "random_strength": 2.113705944064573,
    "bagging_temperature": 1.5585553804470549,
    "border_count": 128,
    "one_hot_max_size": 2,
    "auto_class_weights": "Balanced",
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "early_stopping_rounds": 200,
    "random_seed": SEED,
    "thread_count": -1,
    "verbose": 200,
}

PRESENCE_COLS = [
    "yearsAtLevel", "salaryFormat", "employmentType", "salesComp", "negotiatedAmount", "gender", "ethnicity",
    "education", "otherDetails", "vestingSchedule", "tags", "additionalBonuses", "avgAnnualOptionsGranted",
    "totalOptionsGranted", "optionStrikePrice", "optionPreferredPrice", "totalOutstandingShares",
    "latestCompanyValuation", "companySize", "fundingStage", "annualTargetBonusValue",
    "annualTargetBonusPercentage", "userExchangeRate", "commentsAndRepliesCount", "threadId",
    "avgAnnualStockGrantValue", "avgAnnualBonusValue", "firstYearBonusValue", "totalStockGrantValue",
    "yearsAtCompany", "baseSalary", "totalCompensation",
]

KEEP_RAW_NUMERIC = [
    "yearsOfExperience", "yearsAtCompany", "yearsAtLevel", "salesComp", "negotiatedAmount",
    "avgAnnualOptionsGranted", "totalOptionsGranted", "optionStrikePrice", "optionPreferredPrice",
    "totalOutstandingShares", "latestCompanyValuation", "annualTargetBonusPercentage", "userExchangeRate",
    "commentsAndRepliesCount", "userAccountAgeDays", "userPriorSubmissions",
]

BEHAVIOUR_COLS = [
    "user_prior_submission_count", "ip_prior_submission_count", "device_prior_submission_count",
    "user_hours_since_prev", "ip_hours_since_prev", "device_hours_since_prev",
    "device_distinct_users", "ip_distinct_users", "user_distinct_devices", "user_distinct_ips",
    "user_distinct_companies", "device_distinct_companies",
]

RAW_CATEGORICAL = [
    "company", "jobFamily", "level", "focusTag", "location", "compPerspective", "gender", "education",
    "salaryFormat", "employmentType", "ethnicity", "companySize", "fundingStage", "stockType"
]

WORK_ARRANGEMENT_ALIASES = {
    "hybrid": "hybrid", "onsite": "onsite", "on-site": "onsite", "on site": "onsite",
    "office": "onsite", "in-office": "onsite", "in office": "onsite",
    "remote": "remote", "remoteee": "remote", "wfh": "remote",
    "work from home": "remote", "work-from-home": "remote", "fully remote": "remote",
}
EXPECTED_WORK_ARRANGEMENTS = {"hybrid", "onsite", "remote"}

VALID_CURRENCY_CODES = {"INR", "USD", "EUR", "GBP", "CAD", "SGD", "AUD", "PLN", "SEK", "CHF", "BRL", "AED"}
CURRENCY_ALIASES = {"INRR": "INR", "RS": "INR", "RS.": "INR", "RUPEE": "INR", "RUPEES": "INR", "US$": "USD"}
AMBIGUOUS_CURRENCY_ALIASES = {"RS", "RS."}
PLACEHOLDER_CURRENCY_CODES = {"XXX"}

PEER_LEVELS = {
    "cjll": ["company", "jobFamily", "level", "location"],
    "cjl":  ["company", "jobFamily", "level"],
    "jll":  ["jobFamily", "level", "location"],
    "cj":   ["company", "jobFamily"],
    "jl":   ["jobFamily", "level"],
    "j":    ["jobFamily"],
}
PEER_TARGETS = ["base_usd_log", "total_usd_log"]


# ---------------------------------------------------------------------------
# Utility & Preprocessing Functions
# ---------------------------------------------------------------------------
def _clean_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    token = " ".join(str(value).strip().split())
    return token if token else None


def normalize_work_arrangement(value: Any) -> Any:
    token = _clean_token(value)
    if token is None:
        return pd.NA
    token = token.lower()
    return WORK_ARRANGEMENT_ALIASES.get(token, token)


def normalize_currency(value: Any) -> Any:
    token = _clean_token(value)
    if token is None:
        return pd.NA
    token = token.upper()
    if token in PLACEHOLDER_CURRENCY_CODES:
        return pd.NA
    if token in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[token]
    if token in VALID_CURRENCY_CODES:
        return token
    return pd.NA


def classify_currency(value: Any) -> str:
    token = _clean_token(value)
    if token is None:
        return "missing"
    raw = token
    token = token.upper()
    if token in PLACEHOLDER_CURRENCY_CODES:
        return "placeholder_code_review"
    if token in AMBIGUOUS_CURRENCY_ALIASES:
        return "ambiguous_alias_review"
    if token in CURRENCY_ALIASES:
        return "alias_normalized"
    if token in VALID_CURRENCY_CODES:
        return "standard" if raw == token else "alias_normalized"
    return "unrecognized_review"


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")


def present(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    return t.notna() & ~t.isin(["", "[]", "nan", "None"])


def sdiv(a: pd.Series, b: pd.Series) -> pd.Series:
    r = a / b.where(b > 0)
    return r.replace([np.inf, -np.inf], np.nan)


def parse_offer_date(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    cleaned = s.str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    out = pd.to_datetime(cleaned, format="%a %b %d %Y %H:%M:%S GMT%z", errors="coerce", utc=True)
    miss = out.isna() & s.notna()
    if miss.any():
        out[miss] = pd.to_datetime(cleaned[miss], errors="coerce", utc=True, format="mixed")
    return out


def add_strict_prior_count(frame: pd.DataFrame, source_col: str, time_col: str, output_col: str) -> None:
    out = pd.Series(0, index=frame.index, dtype="int64")
    mask = frame[source_col].notna() & frame[time_col].notna()
    if mask.any():
        counts = (frame.loc[mask, [source_col, time_col]]
                  .groupby([source_col, time_col], dropna=True, sort=True)
                  .size().rename("event_count").reset_index()
                  .sort_values([source_col, time_col], kind="mergesort"))
        counts["prior_count"] = counts.groupby(source_col, sort=False)["event_count"].cumsum() - counts["event_count"]
        prior_map = counts.set_index([source_col, time_col])["prior_count"]
        row_keys = pd.MultiIndex.from_frame(frame.loc[mask, [source_col, time_col]])
        out.loc[mask] = prior_map.reindex(row_keys).to_numpy(dtype=np.int64)
    frame[output_col] = out


def add_behaviour_features(d: pd.DataFrame) -> pd.DataFrame:
    """Causal (past-only) behaviour features computed before train/test splitting."""
    d = d.copy()
    ents = [("userId", "user"), ("ipHash", "ip"), ("deviceHash", "device")]
    for key, name in ents:
        add_strict_prior_count(d, key, "submittedAt", f"{name}_prior_submission_count")
    s = d.sort_values("submittedAt", kind="mergesort", na_position="first")
    for key, name in ents:
        gap = s.groupby(key)["submittedAt"].diff().dt.total_seconds() / 3600.0
        d[f"{name}_hours_since_prev"] = gap.reindex(d.index)
    pairs = [
        ("deviceHash", "userId", "device_distinct_users"),
        ("ipHash", "userId", "ip_distinct_users"),
        ("userId", "deviceHash", "user_distinct_devices"),
        ("userId", "ipHash", "user_distinct_ips"),
        ("userId", "company", "user_distinct_companies"),
        ("deviceHash", "company", "device_distinct_companies")
    ]
    for key, val, name in pairs:
        m = s[key].notna() & s[val].notna()
        first = pd.Series(0, index=s.index)
        first.loc[m] = (~s.loc[m].duplicated([key, val])).astype(int)
        d[name] = first.groupby(s[key]).cumsum().reindex(d.index).fillna(0)
    return d


def _components(frame: pd.DataFrame, cols: List[str]) -> np.ndarray:
    n = len(frame)
    rows, cidx, offset = [], [], n
    for c in cols:
        codes, uniques = pd.factorize(frame[c])
        m = codes >= 0
        rows.append(np.flatnonzero(m))
        cidx.append(codes[m] + offset)
        offset += len(uniques)
    r, c_ = np.concatenate(rows), np.concatenate(cidx)
    g = coo_matrix((np.ones(len(r)), (r, c_)), shape=(offset, offset))
    _, labels = connected_components(g, directed=False)
    return labels[:n]


def build_groups(frame: pd.DataFrame) -> np.ndarray:
    for cols in (["userId", "deviceHash", "ipHash"], ["userId", "deviceHash"], ["userId"]):
        labels = _components(frame, cols)
        top = pd.Series(labels).value_counts(normalize=True).iloc[0]
        print(f"Grouping on {cols}: {labels.max() + 1} groups, largest group = {top:.1%} of rows")
        if top < 0.25:
            return labels
    return labels


def _key(X: pd.DataFrame, cols: List[str]) -> pd.Series:
    k = X[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + X[c].astype(str)
    return k


def fit_train_stats(X: pd.DataFrame) -> Dict[str, Any]:
    st: Dict[str, Any] = {}
    cur = X["baseSalaryCurrency"].map(normalize_currency)
    r = pd.DataFrame({"c": cur.astype("string"), "r": to_num(X["exchangeRate"])}).dropna()
    r = r[r["r"] > 0]
    rate_map = r.groupby("c")["r"].median().to_dict()
    rate_map["USD"] = 1.0
    st["rate_map"] = rate_map

    ctry = X["locationSlug"].astype("string").str.extract(r"-([a-z]{3})$")[0]
    tmp = pd.DataFrame({"ctry": ctry, "c": cur.astype("string")}).dropna()
    st["country_cur"] = tmp.groupby("ctry")["c"].agg(lambda s: s.mode().iloc[0]).to_dict()

    st["freq"] = {
        "company": X["company"].astype(str).value_counts().to_dict(),
        "location": X["location"].astype(str).value_counts().to_dict(),
        "company_level": _key(X, ["company", "level"]).value_counts().to_dict(),
        "company_family_level": _key(X, ["company", "jobFamily", "level"]).value_counts().to_dict(),
    }
    return st


def parse_vesting(v: Any) -> Tuple[float, float, float, int]:
    if pd.isna(v) or str(v).strip() in ("", "[]"):
        return (np.nan, np.nan, np.nan, 0)
    try:
        lst = json.loads(v)
        pct = [float(t.get("percent", np.nan)) for t in lst]
        occ = [float(t.get("occurrences", np.nan)) for t in lst]
        return (float(np.nansum(pct)), float(len(lst)), float(np.nansum(occ)), 0)
    except Exception:
        return (np.nan, np.nan, np.nan, 1)


def parse_addl_bonuses(v: Any, rate_map: Dict[str, float]) -> Tuple[float, float, int]:
    if pd.isna(v) or str(v).strip() in ("", "[]"):
        return (0.0, np.nan, 0)
    try:
        lst = json.loads(v)
        tot = 0.0
        for t in lst:
            rate = rate_map.get(str(t.get("currency", "")).upper(), np.nan)
            tot += float(t.get("value", 0)) / rate if rate == rate and rate > 0 else np.nan
        return (float(len(lst)), tot, 0)
    except Exception:
        return (np.nan, np.nan, 1)


def count_json_list(v: Any) -> float:
    if pd.isna(v) or str(v).strip() in ("", "[]"):
        return 0.0
    try:
        return float(len(json.loads(v)))
    except Exception:
        return np.nan


def base_features(X: pd.DataFrame, st: Dict[str, Any]) -> pd.DataFrame:
    F = pd.DataFrame(index=X.index)

    # Categoricals
    for c in RAW_CATEGORICAL:
        F[c] = X[c].astype("string").str.strip() if c in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
    F["workArrangement"] = X["workArrangement"].map(normalize_work_arrangement).astype("string") if "workArrangement" in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
    F["workArrangement_unexpected"] = (F["workArrangement"].notna() & ~F["workArrangement"].isin(EXPECTED_WORK_ARRANGEMENTS)).astype("int8")
    F["companyInfo_registered"] = X["companyInfo_registered"].astype("string") if "companyInfo_registered" in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
    
    for c in ["cityId", "dmaId", "countryId"]:
        F[c] = to_num(X[c]).round().astype("Int64").astype("string") if c in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
    
    F["country_token"] = X["locationSlug"].astype("string").str.extract(r"-([a-z]{3})$")[0] if "locationSlug" in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
    for c in ["baseSalaryCurrency", "stockGrantCurrency", "bonusCurrency", "userCurrency"]:
        if c in X.columns:
            F[c] = X[c].map(normalize_currency).astype("string")
            F[f"{c}_status"] = X[c].map(classify_currency)
        else:
            F[c] = pd.Series(pd.NA, index=X.index, dtype="string")
            F[f"{c}_status"] = "missing"

    # Numerics
    N = {c: to_num(X[c]) for c in
         ["baseSalary", "totalCompensation", "firstYearTotalCompensation", "avgAnnualStockGrantValue",
          "firstYearStockGrantValue", "totalStockGrantValue", "avgAnnualBonusValue", "firstYearBonusValue",
          "annualTargetBonusValue", "annualTargetBonusPercentage", "salesComp", "negotiatedAmount",
          "yearsOfExperience", "yearsAtCompany", "yearsAtLevel", "exchangeRate",
          "userPriorSubmissions", "userAccountAgeDays"] if c in X.columns}

    def rate_of(cur: pd.Series) -> pd.Series:
        return pd.to_numeric(cur.astype("string").map(st["rate_map"]), errors="coerce")

    r_base = rate_of(F["baseSalaryCurrency"])
    r_stock = rate_of(F["stockGrantCurrency"])
    r_bonus = rate_of(F["bonusCurrency"])

    # USD normalized
    usd = {
        "base": N["baseSalary"] / r_base,
        "total": N["totalCompensation"] / r_base,
        "first_total": N["firstYearTotalCompensation"] / r_base,
        "avg_stock": N["avgAnnualStockGrantValue"] / r_stock,
        "first_stock": N["firstYearStockGrantValue"] / r_stock,
        "total_stock": N["totalStockGrantValue"] / r_stock,
        "avg_bonus": N["avgAnnualBonusValue"] / r_bonus,
        "first_bonus": N["firstYearBonusValue"] / r_bonus,
        "target_bonus": N["annualTargetBonusValue"] / r_bonus,
        "sales": N["salesComp"] / r_base,
        "negotiated": N["negotiatedAmount"] / r_base,
    }
    for k, v in usd.items():
        F[f"{k}_usd_log"] = np.log1p(v.clip(lower=0))
    F["baseSalary_local"] = N["baseSalary"]
    F["totalCompensation_local"] = N["totalCompensation"]
    F["exchangeRate_dev"] = np.log(N["exchangeRate"].where(N["exchangeRate"] > 0) / r_base)
    F["currency_mismatch_base_stock"] = (F["baseSalaryCurrency"] != F["stockGrantCurrency"]).where(F["baseSalaryCurrency"].notna() & F["stockGrantCurrency"].notna()).astype(float)
    F["currency_mismatch_base_bonus"] = (F["baseSalaryCurrency"] != F["bonusCurrency"]).where(F["baseSalaryCurrency"].notna() & F["bonusCurrency"].notna()).astype(float)

    # Accounting identities
    sum_raw = N["baseSalary"].fillna(0) + N["avgAnnualStockGrantValue"].fillna(0) + N["avgAnnualBonusValue"].fillna(0)
    F["tc_resid_raw"] = N["totalCompensation"] - sum_raw
    F["tc_resid_raw_rel"] = sdiv(F["tc_resid_raw"], N["totalCompensation"])
    first_sum_raw = N["baseSalary"].fillna(0) + N["firstYearStockGrantValue"].fillna(0) + N["firstYearBonusValue"].fillna(0)
    F["ftc_resid_raw"] = N["firstYearTotalCompensation"] - first_sum_raw
    F["ftc_resid_raw_rel"] = sdiv(F["ftc_resid_raw"], N["firstYearTotalCompensation"])
    sum_usd = usd["base"].fillna(0) + usd["avg_stock"].fillna(0) + usd["avg_bonus"].fillna(0)
    F["tc_resid_usd_rel"] = sdiv(usd["total"] - sum_usd, usd["total"])
    F["stock_implied_vest_years"] = sdiv(N["totalStockGrantValue"], N["avgAnnualStockGrantValue"])
    F["bonus_pct_implied"] = sdiv(usd["target_bonus"], usd["base"]) * 100
    F["bonus_pct_resid"] = N["annualTargetBonusPercentage"] - F["bonus_pct_implied"]
    F["ftc_over_tc"] = sdiv(N["firstYearTotalCompensation"], N["totalCompensation"])
    F["stock_to_base"] = sdiv(usd["avg_stock"], usd["base"])
    F["bonus_to_base"] = sdiv(usd["avg_bonus"], usd["base"])
    F["total_to_base"] = sdiv(usd["total"], usd["base"])
    F["first_stock_vs_avg"] = sdiv(N["firstYearStockGrantValue"], N["avgAnnualStockGrantValue"])
    F["first_bonus_vs_avg"] = sdiv(N["firstYearBonusValue"], N["avgAnnualBonusValue"])
    F["base_negative_or_zero"] = (N["baseSalary"] <= 0).where(N["baseSalary"].notna()).astype(float)

    # Experience consistency
    yoe, yac, yal = N["yearsOfExperience"], N["yearsAtCompany"], N["yearsAtLevel"]
    F["yac_gt_yoe"] = (yac > yoe).where(yac.notna() & yoe.notna()).astype(float)
    F["yal_gt_yac"] = (yal > yac).where(yal.notna() & yac.notna()).astype(float)
    F["yoe_minus_yac"] = yoe - yac
    F["yoe_out_of_range"] = ((yoe < 0) | (yoe > 50)).where(yoe.notna()).astype(float)

    # Time features
    sub, off = X.get("submittedAt"), X.get("offerDate")
    if sub is not None and off is not None:
        both = sub.notna() & off.notna()
        F["days_since_offer"] = (sub - off).dt.total_seconds() / 86400.0
        F["offer_after_submit"] = (off > sub).where(both).astype(float)
        F["offer_same_time_of_day"] = ((sub.dt.hour * 60 + sub.dt.minute) == (off.dt.hour * 60 + off.dt.minute)).where(both).astype(float)
        F["submit_hour"] = sub.dt.hour
        F["submit_dayofweek"] = sub.dt.dayofweek
    else:
        for c in ["days_since_offer", "offer_after_submit", "offer_same_time_of_day", "submit_hour", "submit_dayofweek"]:
            F[c] = np.nan

    # Identity / naming consistency
    if "company" in X.columns and "companyInfo_name" in X.columns:
        F["company_name_mismatch"] = (X["company"].astype(str).str.lower().str.strip() != X["companyInfo_name"].astype(str).str.lower().str.strip()).where(X["company"].notna() & X["companyInfo_name"].notna()).astype(float)
        F["company_info_unknown"] = (X["companyInfo_name"].astype(str).str.lower() == "unknown").astype(float)
    else:
        F["company_name_mismatch"] = np.nan
        F["company_info_unknown"] = np.nan

    if "title" in X.columns and "jobFamily" in X.columns:
        F["title_eq_jobFamily"] = (X["title"].astype(str).str.lower().str.strip() == X["jobFamily"].astype(str).str.lower().str.strip()).where(X["title"].notna() & X["jobFamily"].notna()).astype(float)
        F["title_len"] = X["title"].astype("string").str.len()
    else:
        F["title_eq_jobFamily"] = np.nan
        F["title_len"] = np.nan

    F["otherDetails_len"] = X["otherDetails"].astype("string").str.len() if "otherDetails" in X.columns else np.nan
    exp_cur = F["country_token"].map(st["country_cur"])
    F["currency_country_mismatch"] = (F["baseSalaryCurrency"] != exp_cur).where(F["baseSalaryCurrency"].notna() & exp_cur.notna()).astype(float)

    # Frequency encodings
    F["company_freq"] = np.log1p(X["company"].astype(str).map(st["freq"]["company"]).fillna(0))
    F["location_freq"] = np.log1p(X["location"].astype(str).map(st["freq"]["location"]).fillna(0))
    F["company_level_freq"] = np.log1p(_key(X, ["company", "level"]).map(st["freq"]["company_level"]).fillna(0))
    F["company_family_level_freq"] = np.log1p(_key(X, ["company", "jobFamily", "level"]).map(st["freq"]["company_family_level"]).fillna(0))

    # Structured parsing
    v = pd.DataFrame(X["vestingSchedule"].map(parse_vesting).tolist(), index=X.index,
                     columns=["vest_pct_sum", "vest_n_tranches", "vest_total_occ", "vest_malformed"]) if "vestingSchedule" in X.columns else pd.DataFrame(np.nan, index=X.index, columns=["vest_pct_sum", "vest_n_tranches", "vest_total_occ", "vest_malformed"])
    F = F.join(v)
    F["vest_pct_not_100"] = ((F["vest_pct_sum"] - 100).abs() > 1).where(F["vest_pct_sum"].notna()).astype(float)

    a = pd.DataFrame(X["additionalBonuses"].map(lambda z: parse_addl_bonuses(z, st["rate_map"])).tolist(), index=X.index,
                     columns=["addl_bonus_n", "addl_bonus_usd", "addl_bonus_malformed"]) if "additionalBonuses" in X.columns else pd.DataFrame(np.nan, index=X.index, columns=["addl_bonus_n", "addl_bonus_usd", "addl_bonus_malformed"])
    F = F.join(a)
    F["tags_count"] = X["tags"].map(count_json_list) if "tags" in X.columns else 0.0

    # Presence flags
    for c in PRESENCE_COLS:
        if c in X.columns:
            F[f"has_{c}"] = present(X[c]).astype("int8")
    for c in KEEP_RAW_NUMERIC:
        if c in X.columns:
            F[c] = N[c] if c in N else to_num(X[c])

    # Behaviour features
    for c in BEHAVIOUR_COLS:
        if c in X.columns:
            F[c] = X[c]
    if "userPriorSubmissions" in N and "user_prior_submission_count" in X.columns:
        F["prior_declared_minus_observed"] = N["userPriorSubmissions"] - X["user_prior_submission_count"]

    return F


def fit_peer_stats(F: pd.DataFrame, valid_mask: np.ndarray) -> Dict[str, Any]:
    G = F.loc[valid_mask]
    stats: Dict[str, Any] = {}
    for name, keys in PEER_LEVELS.items():
        for t in PEER_TARGETS:
            sub = G.loc[G[t] > 0, keys + [t]].dropna()
            if sub.empty:
                continue
            g = sub.groupby(keys, observed=True)[t]
            tab = pd.DataFrame({
                "med": g.median(),
                "q25": g.quantile(0.25),
                "q75": g.quantile(0.75),
                "n": g.size()
            })
            tab["scale"] = ((tab["q75"] - tab["q25"]) / 1.349).clip(lower=0.05)
            # Convert to dictionary keyed by stringified tuple
            records = {}
            for idx_val, row in tab.iterrows():
                key_str = "|".join(str(x) for x in (idx_val if isinstance(idx_val, tuple) else [idx_val]))
                records[key_str] = {
                    "med": float(row["med"]),
                    "scale": float(row["scale"]),
                    "n": int(row["n"])
                }
            stats[f"{name}__{t}"] = records
    return stats


def _lookup_peer(F: pd.DataFrame, keys: List[str], peer_dict: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    meds = np.full(len(F), np.nan)
    scales = np.full(len(F), np.nan)
    ns = np.full(len(F), np.nan)
    key_series = F[keys[0]].astype(str)
    for c in keys[1:]:
        key_series = key_series + "|" + F[c].astype(str)
    for i, k in enumerate(key_series):
        if k in peer_dict:
            entry = peer_dict[k]
            if entry["n"] >= MIN_PEER_N:
                meds[i] = entry["med"]
                scales[i] = entry["scale"]
                ns[i] = entry["n"]
    return meds, scales, ns


def apply_peer_stats_dict(F: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=F.index)
    for t in PEER_TARGETS:
        actual = F[t].where(F[t] > 0).to_numpy(float)
        exp, scale, n, lvl = (np.full(len(F), np.nan) for _ in range(4))
        looked: Dict[str, np.ndarray] = {}
        for i, (name, keys) in enumerate(PEER_LEVELS.items()):
            p_dict = stats.get(f"{name}__{t}", {})
            med, sc, nn = _lookup_peer(F, keys, p_dict)
            looked[name] = med
            take = np.isnan(exp) & ~np.isnan(med)
            exp[take], scale[take], n[take], lvl[take] = med[take], sc[take], nn[take], i
        resid = actual - exp
        out[f"peer_exp_{t}"] = exp
        out[f"peer_resid_{t}"] = resid
        out[f"peer_z_{t}"] = resid / scale
        out[f"peer_n_{t}"] = n
        out[f"peer_lvl_{t}"] = lvl
        cjl_med = looked.get("cjl", np.full(len(F), np.nan))
        jll_med = looked.get("jll", np.full(len(F), np.nan))
        jl_med = looked.get("jl", np.full(len(F), np.nan))
        xloc = cjl_med + (jll_med - jl_med)
        out[f"peer_xloc_resid_{t}"] = actual - xloc
    return out


def peer_features_oof(F: pd.DataFrame, y_valid: pd.Series, groups: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    parts = []
    for fit_i, app_i in GroupKFold(n_splits=n_splits).split(F, groups=groups):
        stats = fit_peer_stats(F.iloc[fit_i], (y_valid.iloc[fit_i] == 1).to_numpy())
        parts.append(apply_peer_stats_dict(F.iloc[app_i], stats))
    return pd.concat(parts).reindex(F.index)


def pick_threshold(y: np.ndarray, p: np.ndarray, min_precision: float | None = None) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    if min_precision is not None:
        ok = prec >= min_precision
        if ok.any():
            return float(thr[ok][np.argmax(rec[ok])])
    return float(thr[np.argmax(f1)])


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("CatBoost Anomaly Detection Model Training")
    print(f"Data source: {DATA_PATH}")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Error: dataset {DATA_PATH} not found!")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Step 1: Reading and parsing dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["submittedAt"] = pd.to_datetime(df["submittedAt"], errors="coerce", utc=True, format="mixed")
    df["offerDate"] = parse_offer_date(df["offerDate"])
    print(f"Loaded {len(df):,} rows | submittedAt parsed: {df['submittedAt'].notna().mean():.1%}")

    print("Step 2: Computing causal behaviour features...")
    df = add_behaviour_features(df)
    df = df[df["is_valid"].notna()].copy()
    df["is_valid"] = df["is_valid"].astype(int)
    print(f"Rows with label: {len(df):,} | Anomaly rate: {(df['is_valid'] == 0).mean():.2%}")

    print("Step 3: Constructing leakage-free actor groups...")
    groups = build_groups(df)
    y_all = df["is_valid"]

    # 60 / 20 / 20 Grouped + Stratified splits
    sg1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tv_pos, te_pos = next(sg1.split(df, y_all, groups))
    sg2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    tr_rel, va_rel = next(sg2.split(df.iloc[tv_pos], y_all.iloc[tv_pos], groups[tv_pos]))
    tr_pos, va_pos = tv_pos[tr_rel], tv_pos[va_rel]

    X_tr_raw = df.iloc[tr_pos].drop(columns=["is_valid"])
    X_va_raw = df.iloc[va_pos].drop(columns=["is_valid"])
    X_te_raw = df.iloc[te_pos].drop(columns=["is_valid"])

    y_tr_valid = y_all.iloc[tr_pos]
    y_va_valid = y_all.iloc[va_pos]
    y_te_valid = y_all.iloc[te_pos]

    # Target: 1 = anomaly, 0 = valid
    y_tr = 1 - y_tr_valid
    y_va = 1 - y_va_valid
    y_te = 1 - y_te_valid
    groups_tr = groups[tr_pos]

    print(f"Train set: {len(X_tr_raw):,} | Val set: {len(X_va_raw):,} | Test set: {len(X_te_raw):,}")

    print("Step 4: Fitting train statistics (rate map, country currency, frequencies)...")
    st = fit_train_stats(X_tr_raw)

    print("Step 5: Extracting base features...")
    F_tr = base_features(X_tr_raw, st)
    F_va = base_features(X_va_raw, st)
    F_te = base_features(X_te_raw, st)

    print("Step 6: Computing hierarchical peer group statistics...")
    P_tr = peer_features_oof(F_tr, y_tr_valid, groups_tr)
    stats_full = fit_peer_stats(F_tr, (y_tr_valid == 1).to_numpy())
    P_va = apply_peer_stats_dict(F_va, stats_full)
    P_te = apply_peer_stats_dict(F_te, stats_full)

    X_tr = F_tr.join(P_tr)
    X_va = F_va.join(P_va)
    X_te = F_te.join(P_te)

    # Column alignment & drop constant columns
    X_va = X_va.reindex(columns=X_tr.columns)
    X_te = X_te.reindex(columns=X_tr.columns)
    const_cols = [c for c in X_tr.columns if X_tr[c].nunique(dropna=False) <= 1]
    print(f"Dropping {len(const_cols)} constant columns: {const_cols}")
    X_tr = X_tr.drop(columns=const_cols)
    X_va = X_va.drop(columns=const_cols)
    X_te = X_te.drop(columns=const_cols)

    cat_cols = X_tr.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    for x in (X_tr, X_va, X_te):
        for c in cat_cols:
            x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
        for c in num_cols:
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")

    print(f"Final feature matrix: {X_tr.shape[1]} features ({len(cat_cols)} categorical, {len(num_cols)} numeric)")

    pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols)
    pool_va = Pool(X_va, y_va, cat_features=cat_cols)
    pool_te = Pool(X_te, y_te, cat_features=cat_cols)

    print("\nStep 7: Training CatBoost model with optimal hyperparameters...")
    model = CatBoostClassifier(**BEST_CATBOOST_PARAMS)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    best_iter = model.get_best_iteration()
    print(f"Training completed! Best iteration: {best_iter}")

    print("\nStep 8: Calibrating decision threshold on validation set...")
    p_va = model.predict_proba(pool_va)[:, 1]
    p_te = model.predict_proba(pool_te)[:, 1]
    threshold = pick_threshold(y_va.to_numpy(), p_va)
    print(f"Calibrated anomaly probability threshold: {threshold:.4f}")

    # Evaluation
    pred_te_anom = (p_te >= threshold).astype(int)
    y_te_v = 1 - y_te.to_numpy()
    pred_te_v = 1 - pred_te_anom

    acc = accuracy_score(y_te_v, pred_te_v)
    f1_macro = f1_score(y_te_v, pred_te_v, average="macro")
    f1_anom = f1_score(y_te_v, pred_te_v, pos_label=0)
    pr_auc = average_precision_score(y_te, p_te)
    roc_auc = roc_auc_score(y_te, p_te)

    print("\n" + "=" * 30 + " HOLD-OUT TEST RESULTS " + "=" * 30)
    print(f"Accuracy:                {acc:.4f}")
    print(f"Macro F1:                {f1_macro:.4f}")
    print(f"F1 (Anomaly, is_valid=0): {f1_anom:.4f}")
    print(f"PR-AUC (Anomaly):        {pr_auc:.4f}")
    print(f"ROC-AUC:                 {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_te_v, pred_te_v, labels=[0, 1], target_names=["is_valid=0 (Anomaly)", "is_valid=1 (Valid)"], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_te_v, pred_te_v, labels=[0, 1]))

    print("\nStep 9: Serializing model and metadata artifacts...")
    model_path = os.path.join(MODELS_DIR, "catboost_model.cbm")
    model.save_model(model_path)
    print(f"Saved CatBoost model to: {model_path}")

    train_stats_path = os.path.join(MODELS_DIR, "train_stats.json")
    with open(train_stats_path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    print(f"Saved train stats to: {train_stats_path}")

    peer_stats_path = os.path.join(MODELS_DIR, "peer_stats.json")
    with open(peer_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_full, f, indent=2)
    print(f"Saved peer stats to: {peer_stats_path}")

    # Feature metadata
    feature_meta = {
        "all_columns": list(X_tr.columns),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "constant_columns_dropped": const_cols,
        "optimal_threshold": float(threshold),
        "metrics": {
            "test_accuracy": float(acc),
            "test_macro_f1": float(f1_macro),
            "test_anomaly_f1": float(f1_anom),
            "test_pr_auc": float(pr_auc),
            "test_roc_auc": float(roc_auc),
            "best_iteration": int(best_iter)
        }
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE & READY FOR INFERENCE")
    print("=" * 60)


if __name__ == "__main__":
    main()
