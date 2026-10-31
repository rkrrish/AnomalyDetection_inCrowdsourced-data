# src/feature_extractor.py
"""Feature Extractor for CatBoost Anomaly Detection.

Applies the exact data transformations, currency normalizations, hierarchical
peer group statistics, and residual calculations established in CatBoostv2.ipynb.
Supports both single-record inference and batch evaluation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    MODEL_META_PATH,
    PEER_STATS_PATH,
    TRAIN_STATS_PATH,
)

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
MIN_PEER_N = 5


class FeatureExtractor:
    """Extracts features for CatBoost inference using pre-computed training and peer statistics."""

    def __init__(self, train_stats_path: str = TRAIN_STATS_PATH, peer_stats_path: str = PEER_STATS_PATH, model_meta_path: str = MODEL_META_PATH):
        self.train_stats_path = train_stats_path
        self.peer_stats_path = peer_stats_path
        self.model_meta_path = model_meta_path

        self.train_stats = self._load_json(train_stats_path, default={
            "rate_map": {"USD": 1.0, "INR": 88.0, "EUR": 0.92, "GBP": 0.78, "CAD": 1.36},
            "country_cur": {"ind": "INR", "usa": "USD", "gbr": "GBP", "can": "CAD"},
            "freq": {"company": {}, "location": {}, "company_level": {}, "company_family_level": {}}
        })
        self.peer_stats = self._load_json(peer_stats_path, default={})
        self.model_meta = self._load_json(model_meta_path, default={})

    @staticmethod
    def _load_json(path: str, default: Any) -> Any:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    @staticmethod
    def _clean_token(value: Any) -> str | None:
        if pd.isna(value):
            return None
        token = " ".join(str(value).strip().split())
        return token if token else None

    @classmethod
    def normalize_work_arrangement(cls, value: Any) -> Any:
        token = cls._clean_token(value)
        if token is None:
            return pd.NA
        return WORK_ARRANGEMENT_ALIASES.get(token.lower(), token)

    @classmethod
    def normalize_currency(cls, value: Any) -> Any:
        token = cls._clean_token(value)
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

    @classmethod
    def classify_currency(cls, value: Any) -> str:
        token = cls._clean_token(value)
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

    @staticmethod
    def to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")

    @staticmethod
    def sdiv(a: pd.Series, b: pd.Series) -> pd.Series:
        r = a / b.where(b > 0)
        return r.replace([np.inf, -np.inf], np.nan)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame into complete feature matrix ready for CatBoost."""
        X = df.copy()

        # Parse dates if present
        if "submittedAt" in X.columns:
            X["submittedAt"] = pd.to_datetime(X["submittedAt"], errors="coerce", utc=True, format="mixed")
        else:
            X["submittedAt"] = pd.Timestamp.utcnow()

        if "offerDate" in X.columns:
            s = X["offerDate"].astype("string")
            cleaned = s.str.replace(r"\s*\(.*\)\s*$", "", regex=True)
            out_d = pd.to_datetime(cleaned, errors="coerce", utc=True, format="mixed")
            X["offerDate"] = out_d
        else:
            X["offerDate"] = pd.NaT

        # Ensure behaviour features exist with correct defaults.
        # Count-based features (prior_submission_count, distinct_*) default to 0.
        # Time-gap features (*_hours_since_prev) default to NaN because 0.0 would
        # signal a suspicious burst (submitted 0 hours after the previous submission),
        # while NaN correctly indicates "no prior submission data available".
        _HOURS_SINCE_PREV_COLS = {
            "user_hours_since_prev", "ip_hours_since_prev", "device_hours_since_prev",
        }
        for col in BEHAVIOUR_COLS:
            if col not in X.columns:
                X[col] = np.nan if col in _HOURS_SINCE_PREV_COLS else 0.0

        F = pd.DataFrame(index=X.index)

        # Categoricals
        stock_type_norm = {
            "rsu": "stock", "rsus": "stock", "rsu's": "stock",
            "options": "options", "option": "options",
            "stock": "stock", "shares": "stock"
        }
        for c in RAW_CATEGORICAL:
            if c in X.columns:
                series = X[c].astype("string").str.strip()
                if c == "stockType":
                    series = series.str.lower().map(lambda v: stock_type_norm.get(v, v))
                F[c] = series
            else:
                F[c] = pd.Series(pd.NA, index=X.index, dtype="string")

        F["workArrangement"] = X["workArrangement"].map(self.normalize_work_arrangement).astype("string") if "workArrangement" in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
        F["workArrangement_unexpected"] = (F["workArrangement"].notna() & ~F["workArrangement"].isin(EXPECTED_WORK_ARRANGEMENTS)).astype("int8")
        F["companyInfo_registered"] = X["companyInfo_registered"].astype("string") if "companyInfo_registered" in X.columns else pd.Series("True", index=X.index, dtype="string")

        for c in ["cityId", "dmaId", "countryId"]:
            F[c] = self.to_num(X[c]).round().astype("Int64").astype("string") if c in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")

        F["country_token"] = X["locationSlug"].astype("string").str.extract(r"-([a-z]{3})$")[0] if "locationSlug" in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
        for c in ["baseSalaryCurrency", "stockGrantCurrency", "bonusCurrency", "userCurrency"]:
            if c in X.columns:
                F[c] = X[c].map(self.normalize_currency).astype("string")
                F[f"{c}_status"] = X[c].map(self.classify_currency)
            else:
                F[c] = pd.Series(pd.NA, index=X.index, dtype="string")
                F[f"{c}_status"] = "missing"

        # Fallback currency defaults for single entry forms
        if "baseSalaryCurrency" in F.columns:
            base_cur = F["baseSalaryCurrency"].fillna("USD")
            F["baseSalaryCurrency"] = F["baseSalaryCurrency"].fillna("USD")
            F["stockGrantCurrency"] = F["stockGrantCurrency"].fillna(base_cur)
            F["bonusCurrency"] = F["bonusCurrency"].fillna(base_cur)

        # Fallback countryId inference if missing
        country_id_map = {"usa": "253", "ind": "113", "gbr": "244", "can": "38", "deu": "79", "fra": "75", "aus": "14", "nld": "166", "sgp": "208"}
        cur_country_map = {"USD": "253", "INR": "113", "GBP": "244", "CAD": "38", "EUR": "79", "AUD": "14", "SGD": "208"}
        if F["country_token"].isna().any() and "location" in X.columns:
            # Simple token inference from location string
            is_us = X["location"].astype(str).str.contains(r"\b(CA|NY|WA|TX|MA|IL|CO|GA|VA|USA|United States)\b", case=False, regex=True)
            F.loc[is_us & F["country_token"].isna(), "country_token"] = "usa"
        F["countryId"] = F["countryId"].fillna(F["country_token"].map(country_id_map))
        F["countryId"] = F["countryId"].fillna(F["baseSalaryCurrency"].map(cur_country_map))

        # Numerics - Combine all required and raw numeric columns to ensure complete coverage
        base_numeric_cols = [
            "baseSalary", "totalCompensation", "firstYearTotalCompensation", "avgAnnualStockGrantValue",
            "firstYearStockGrantValue", "totalStockGrantValue", "avgAnnualBonusValue", "firstYearBonusValue",
            "annualTargetBonusValue", "annualTargetBonusPercentage", "salesComp", "negotiatedAmount",
            "yearsOfExperience", "yearsAtCompany", "yearsAtLevel", "exchangeRate",
            "userPriorSubmissions", "userAccountAgeDays"
        ]
        all_numeric_cols = list(set(base_numeric_cols + KEEP_RAW_NUMERIC))

        N = {}
        for c in all_numeric_cols:
            if c in X.columns:
                N[c] = self.to_num(X[c])
            else:
                N[c] = pd.Series(np.nan, index=X.index, dtype=float)

        rate_map = self.train_stats.get("rate_map", {})
        def rate_of(cur: pd.Series) -> pd.Series:
            mapped = cur.astype("string").map(rate_map)
            return pd.to_numeric(mapped, errors="coerce").fillna(1.0)

        r_base = rate_of(F["baseSalaryCurrency"])
        r_stock = rate_of(F["stockGrantCurrency"])
        r_bonus = rate_of(F["bonusCurrency"])

        # Fallbacks for first year compensation & user profile metadata
        N["firstYearTotalCompensation"] = N["firstYearTotalCompensation"].fillna(N["totalCompensation"])
        N["firstYearStockGrantValue"] = N["firstYearStockGrantValue"].fillna(N["avgAnnualStockGrantValue"])
        N["firstYearBonusValue"] = N["firstYearBonusValue"].fillna(N["avgAnnualBonusValue"])
        N["exchangeRate"] = N["exchangeRate"].fillna(r_base)
        N["userAccountAgeDays"] = N["userAccountAgeDays"].fillna(60.0)
        N["userPriorSubmissions"] = N["userPriorSubmissions"].fillna(0.0)

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

        # Accounting identities & residuals
        sum_raw = N["baseSalary"].fillna(0) + N["avgAnnualStockGrantValue"].fillna(0) + N["avgAnnualBonusValue"].fillna(0)
        F["tc_resid_raw"] = N["totalCompensation"] - sum_raw
        F["tc_resid_raw_rel"] = self.sdiv(F["tc_resid_raw"], N["totalCompensation"])
        first_sum_raw = N["baseSalary"].fillna(0) + N["firstYearStockGrantValue"].fillna(0) + N["firstYearBonusValue"].fillna(0)
        F["ftc_resid_raw"] = N["firstYearTotalCompensation"] - first_sum_raw
        F["ftc_resid_raw_rel"] = self.sdiv(F["ftc_resid_raw"], N["firstYearTotalCompensation"])
        sum_usd = usd["base"].fillna(0) + usd["avg_stock"].fillna(0) + usd["avg_bonus"].fillna(0)
        F["tc_resid_usd_rel"] = self.sdiv(usd["total"] - sum_usd, usd["total"])
        F["stock_implied_vest_years"] = self.sdiv(N["totalStockGrantValue"], N["avgAnnualStockGrantValue"])
        F["bonus_pct_implied"] = self.sdiv(usd["target_bonus"], usd["base"]) * 100
        F["bonus_pct_resid"] = N["annualTargetBonusPercentage"] - F["bonus_pct_implied"]
        F["ftc_over_tc"] = self.sdiv(N["firstYearTotalCompensation"], N["totalCompensation"])
        F["stock_to_base"] = self.sdiv(usd["avg_stock"], usd["base"])
        F["bonus_to_base"] = self.sdiv(usd["avg_bonus"], usd["base"])
        F["total_to_base"] = self.sdiv(usd["total"], usd["base"])
        F["first_stock_vs_avg"] = self.sdiv(N["firstYearStockGrantValue"], N["avgAnnualStockGrantValue"])
        F["first_bonus_vs_avg"] = self.sdiv(N["firstYearBonusValue"], N["avgAnnualBonusValue"])
        F["base_negative_or_zero"] = (N["baseSalary"] <= 0).where(N["baseSalary"].notna()).astype(float)

        # Experience consistency
        yoe, yac, yal = N["yearsOfExperience"], N["yearsAtCompany"], N["yearsAtLevel"]
        F["yac_gt_yoe"] = (yac > yoe).where(yac.notna() & yoe.notna()).astype(float)
        F["yal_gt_yac"] = (yal > yac).where(yal.notna() & yac.notna()).astype(float)
        F["yoe_minus_yac"] = yoe - yac
        F["yoe_out_of_range"] = ((yoe < 0) | (yoe > 50)).where(yoe.notna()).astype(float)

        # Time features
        # Time features - Parse and standardize both to tz-naive datetime64[ns]
        raw_sub = X.get("submittedAt", pd.Series(pd.NaT, index=X.index))
        raw_off = X.get("offerDate", pd.Series(pd.NaT, index=X.index))

        sub_dt = pd.to_datetime(raw_sub, errors="coerce", utc=True).dt.tz_localize(None)
        off_dt = pd.to_datetime(raw_off, errors="coerce", utc=True).dt.tz_localize(None)
        both = sub_dt.notna() & off_dt.notna()

        F["days_since_offer"] = (sub_dt - off_dt).dt.total_seconds() / 86400.0
        F["days_since_offer"] = F["days_since_offer"].fillna(14.0)
        F["offer_after_submit"] = (off_dt > sub_dt).where(both).fillna(0.0).astype(float)
        F["offer_same_time_of_day"] = ((sub_dt.dt.hour * 60 + sub_dt.dt.minute) == (off_dt.dt.hour * 60 + off_dt.dt.minute)).where(both).fillna(0.0).astype(float)
        F["submit_hour"] = sub_dt.dt.hour
        F["submit_dayofweek"] = sub_dt.dt.dayofweek

        # Identity consistency
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
        exp_cur = F["country_token"].map(self.train_stats.get("country_cur", {}))
        F["currency_country_mismatch"] = (F["baseSalaryCurrency"] != exp_cur).where(F["baseSalaryCurrency"].notna() & exp_cur.notna()).astype(float)

        # Frequency encodings
        freq_maps = self.train_stats.get("freq", {})
        F["company_freq"] = np.log1p(X["company"].astype(str).map(freq_maps.get("company", {})).fillna(0))
        F["location_freq"] = np.log1p(X["location"].astype(str).map(freq_maps.get("location", {})).fillna(0))
        key_cl = X["company"].astype(str) + "|" + X["level"].astype(str)
        F["company_level_freq"] = np.log1p(key_cl.map(freq_maps.get("company_level", {})).fillna(0))
        key_cfl = X["company"].astype(str) + "|" + X["jobFamily"].astype(str) + "|" + X["level"].astype(str)
        F["company_family_level_freq"] = np.log1p(key_cfl.map(freq_maps.get("company_family_level", {})).fillna(0))

        # Structured texts
        F["vest_pct_sum"] = 100.0
        F["vest_n_tranches"] = 4.0
        F["vest_total_occ"] = 16.0
        F["vest_malformed"] = 0.0
        F["vest_pct_not_100"] = 0.0
        F["addl_bonus_n"] = 0.0
        F["addl_bonus_usd"] = 0.0
        F["addl_bonus_malformed"] = 0.0
        F["tags_count"] = 0.0

        # Presence flags
        for c in PRESENCE_COLS:
            if c in X.columns:
                t = X[c].astype("string").str.strip()
                F[f"has_{c}"] = (t.notna() & ~t.isin(["", "[]", "nan", "None"])).astype("int8")
            else:
                F[f"has_{c}"] = 0

        for c in KEEP_RAW_NUMERIC:
            F[c] = N[c]

        for c in BEHAVIOUR_COLS:
            F[c] = X[c] if c in X.columns else 0.0

        if "userPriorSubmissions" in N and "user_prior_submission_count" in X.columns:
            F["prior_declared_minus_observed"] = N["userPriorSubmissions"] - X["user_prior_submission_count"]
        else:
            F["prior_declared_minus_observed"] = 0.0

        # Peer statistics lookup
        P = self._apply_peer_stats(F)
        feature_matrix = F.join(P)

        # Align with trained model features
        if self.model_meta and "all_columns" in self.model_meta:
            req_cols = self.model_meta["all_columns"]
            cat_cols = set(self.model_meta.get("categorical_columns", []))
            aligned = pd.DataFrame(index=feature_matrix.index)
            for c in req_cols:
                if c in feature_matrix.columns:
                    aligned[c] = feature_matrix[c]
                else:
                    aligned[c] = "__MISSING__" if c in cat_cols else np.nan

            for c in req_cols:
                if c in cat_cols:
                    aligned[c] = aligned[c].astype("string").fillna("__MISSING__").astype(str)
                else:
                    aligned[c] = pd.to_numeric(aligned[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
            return aligned

        return feature_matrix

    def _apply_peer_stats(self, F: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=F.index)
        for t in PEER_TARGETS:
            actual = F[t].where(F[t] > 0).to_numpy(float)
            exp = np.full(len(F), np.nan)
            scale = np.full(len(F), np.nan)
            n = np.full(len(F), np.nan)
            lvl = np.full(len(F), np.nan)
            looked: Dict[str, np.ndarray] = {}

            for i, (name, keys) in enumerate(PEER_LEVELS.items()):
                p_dict = self.peer_stats.get(f"{name}__{t}", {})
                key_series = F[keys[0]].astype(str)
                for c in keys[1:]:
                    key_series = key_series + "|" + F[c].astype(str)

                med_arr = np.full(len(F), np.nan)
                scale_arr = np.full(len(F), np.nan)
                n_arr = np.full(len(F), np.nan)

                for row_idx, k in enumerate(key_series):
                    if k in p_dict:
                        entry = p_dict[k]
                        if entry["n"] >= MIN_PEER_N:
                            med_arr[row_idx] = entry["med"]
                            scale_arr[row_idx] = entry["scale"]
                            n_arr[row_idx] = entry["n"]

                looked[name] = med_arr
                take = np.isnan(exp) & ~np.isnan(med_arr)
                exp[take] = med_arr[take]
                scale[take] = scale_arr[take]
                n[take] = n_arr[take]
                lvl[take] = i

            resid = actual - exp
            out[f"peer_exp_{t}"] = exp
            out[f"peer_resid_{t}"] = resid
            out[f"peer_z_{t}"] = resid / np.where(scale > 0, scale, 1.0)
            out[f"peer_n_{t}"] = n
            out[f"peer_lvl_{t}"] = lvl

            cjl_med = looked.get("cjl", np.full(len(F), np.nan))
            jll_med = looked.get("jll", np.full(len(F), np.nan))
            jl_med = looked.get("jl", np.full(len(F), np.nan))
            xloc = cjl_med + (jll_med - jl_med)
            out[f"peer_xloc_resid_{t}"] = actual - xloc

        return out
