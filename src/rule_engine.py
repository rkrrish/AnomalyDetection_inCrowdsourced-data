# src/rule_engine.py
"""Validation Rule Engine for Crowdsourced Submissions.

Implements structural validation, completeness checking, logical sanity checks,
and format verification as specified in the SRS Document and Assignment Architecture.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    EXPECTED_WORK_ARRANGEMENTS,
    PUBLIC_COMPANIES,
    REQUIRED_SUBMISSION_FIELDS,
    RULE_COLUMNS,
    STARTUP_STAGES,
    VALID_CURRENCY_CODES,
)


def _is_gibberish(name: Any) -> bool:
    """Return True if company name is numeric-only, symbol-only, or absurdly short."""
    if not isinstance(name, str) or len(name.strip()) < 2:
        return True
    cleaned = name.strip().lower()
    if cleaned in ("xyz", "asdf", "test", "fake", "none", "na", "null", "unknown"):
        return True
    return bool(re.fullmatch(r"[\d\W]+", name.strip()))


class RuleEngine:
    """Deterministic validation rule engine evaluating logical consistency and completeness."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def apply_rules(self) -> pd.DataFrame:
        """Apply all deterministic consistency and sanity rules.

        Returns a boolean DataFrame where True indicates a rule violation / defect.
        """
        flags = pd.DataFrame(False, index=self.df.index, columns=RULE_COLUMNS)

        # 1. Required fields presence check
        missing_mask = pd.Series(False, index=self.df.index)
        for col in REQUIRED_SUBMISSION_FIELDS:
            if col not in self.df.columns:
                missing_mask = True
            else:
                s = self.df[col].astype("string").str.strip()
                missing_mask |= (s.isna() | s.isin(["", "nan", "None", "null", "[]"]))
        flags["flag_missing_required"] = missing_mask

        # 2. Company name quality
        if "company" in self.df.columns:
            flags["flag_gibberish_company"] = self.df["company"].apply(_is_gibberish)

        # 3. Numeric conversions
        base = pd.to_numeric(self.df.get("baseSalary", pd.Series(np.nan, index=self.df.index)), errors="coerce")
        tc = pd.to_numeric(self.df.get("totalCompensation", pd.Series(np.nan, index=self.df.index)), errors="coerce")
        ftc = pd.to_numeric(self.df.get("firstYearTotalCompensation", pd.Series(np.nan, index=self.df.index)), errors="coerce")
        stock = pd.to_numeric(self.df.get("avgAnnualStockGrantValue", pd.Series(0, index=self.df.index)), errors="coerce").fillna(0)
        bonus = pd.to_numeric(self.df.get("avgAnnualBonusValue", pd.Series(0, index=self.df.index)), errors="coerce").fillna(0)
        first_stock = pd.to_numeric(self.df.get("firstYearStockGrantValue", pd.Series(0, index=self.df.index)), errors="coerce").fillna(0)
        first_bonus = pd.to_numeric(self.df.get("firstYearBonusValue", pd.Series(0, index=self.df.index)), errors="coerce").fillna(0)

        # Base and TC sanity
        flags["flag_zero_or_negative_base"] = (base <= 0).where(base.notna(), False)
        flags["flag_tc_less_than_base"] = (tc < (base - 50.0)).where(tc.notna() & base.notna(), False)

        # TC arithmetic consistency
        comp_sum = base.fillna(0) + stock + bonus
        has_tc_data = tc.notna() & (tc > 0)
        tc_diff = (tc - comp_sum).abs()
        flags["flag_tc_arithmetic_mismatch"] = ((tc_diff / tc) > 0.08).where(has_tc_data, False)

        # First year TC arithmetic consistency
        if ftc.notna().any():
            ftc_sum = base.fillna(0) + first_stock + first_bonus
            ftc_diff = (ftc - ftc_sum).abs()
            flags["flag_ftc_arithmetic_mismatch"] = ((ftc_diff / ftc) > 0.08).where(ftc.notna() & (ftc > 0), False)

        # 4. Experience & tenure consistency
        yoe = pd.to_numeric(self.df.get("yearsOfExperience", pd.Series(np.nan, index=self.df.index)), errors="coerce")
        yac = pd.to_numeric(self.df.get("yearsAtCompany", pd.Series(np.nan, index=self.df.index)), errors="coerce")
        yal = pd.to_numeric(self.df.get("yearsAtLevel", pd.Series(np.nan, index=self.df.index)), errors="coerce")

        flags["flag_yal_gt_yac"] = (yal > (yac + 0.1)).where(yal.notna() & yac.notna(), False)
        flags["flag_yac_gt_yoe"] = (yac > (yoe + 0.1)).where(yac.notna() & yoe.notna(), False)
        flags["flag_negative_experience"] = ((yoe < 0) | (yac < 0) | (yal < 0)).where(yoe.notna() | yac.notna() | yal.notna(), False)
        flags["flag_yoe_out_of_range"] = ((yoe < 0) | (yoe > 50)).where(yoe.notna(), False)

        # Junior level with unreasonable YoE (> 8 years)
        level_norm = self.df.get("level", pd.Series("", index=self.df.index, dtype=str)).str.lower().str.strip().fillna("")
        is_entry = level_norm.isin(["l1", "ic1", "e1", "junior", "swe i", "entry", "intern", "associate", "analyst"])
        flags["flag_junior_level_high_yoe"] = (is_entry & (yoe > 8)).where(yoe.notna(), False)

        # 5. Chronological checks (Offer Date vs Submitted At)
        if "offerDate" in self.df.columns and "submittedAt" in self.df.columns:
            off = pd.to_datetime(self.df["offerDate"], errors="coerce", utc=True, format="mixed")
            sub = pd.to_datetime(self.df["submittedAt"], errors="coerce", utc=True, format="mixed")
            both = off.notna() & sub.notna()
            flags["flag_offer_after_submit"] = (off > (sub + pd.Timedelta(days=1))).where(both, False)
            flags["flag_offer_same_time_of_day"] = (
                (off.dt.hour * 60 + off.dt.minute) == (sub.dt.hour * 60 + sub.dt.minute)
            ).where(both, False)

        # 6. Entity & Geo Mismatch
        if "company" in self.df.columns and "companyInfo_name" in self.df.columns:
            mismatch = (
                self.df["company"].astype(str).str.lower().str.strip()
                != self.df["companyInfo_name"].astype(str).str.lower().str.strip()
            )
            both_co = self.df["company"].notna() & self.df["companyInfo_name"].notna()
            flags["flag_company_name_mismatch"] = mismatch.where(both_co, False)

        # Currency Country mismatch
        if "locationSlug" in self.df.columns and "baseSalaryCurrency" in self.df.columns:
            ctry = self.df["locationSlug"].astype("string").str.extract(r"-([a-z]{3})$")[0]
            cur = self.df["baseSalaryCurrency"].astype("string").str.upper()
            country_currency_map = {"ind": "INR", "usa": "USD", "gbr": "GBP", "can": "CAD", "sgp": "SGD", "aus": "AUD"}
            exp_cur = ctry.map(country_currency_map)
            mismatch_cur = (cur != exp_cur).where(exp_cur.notna() & cur.notna(), False)
            flags["flag_currency_country_mismatch"] = mismatch_cur

        # 7. Vesting schedule structured check
        if "vestingSchedule" in self.df.columns:
            def _check_vest(v):
                if pd.isna(v) or str(v).strip() in ("", "[]"):
                    return False, False
                try:
                    lst = json.loads(v)
                    pct_sum = sum(float(t.get("percent", 0)) for t in lst)
                    return abs(pct_sum - 100.0) > 1.0, False
                except Exception:
                    return False, True

            vest_res = self.df["vestingSchedule"].apply(_check_vest)
            flags["flag_vest_pct_not_100"] = [r[0] for r in vest_res]
            flags["flag_vest_malformed"] = [r[1] for r in vest_res]

        # 8. Bonus percentage vs declared value mismatch (> 25% deviation)
        target_pct = pd.to_numeric(self.df.get("annualTargetBonusPercentage", pd.Series(np.nan, index=self.df.index)), errors="coerce") / 100.0
        target_val = pd.to_numeric(self.df.get("annualTargetBonusValue", pd.Series(np.nan, index=self.df.index)), errors="coerce")
        has_bonus_pair = (base > 0) & (target_pct > 0) & (target_val > 0)
        expected_bonus = base * target_pct
        bonus_ratio = (target_val / expected_bonus.replace(0, np.nan)).fillna(1.0)
        flags["flag_bonus_pct_mismatch"] = ((bonus_ratio < 0.70) | (bonus_ratio > 1.30)).where(has_bonus_pair, False)

        # 9. Duplicates & rapid bursts
        if "flag_duplicate" in self.df.columns:
            flags["flag_duplicate"] = self.df["flag_duplicate"].fillna(False).astype(bool)
        if "flag_burst_submission" in self.df.columns:
            flags["flag_burst_submission"] = self.df["flag_burst_submission"].fillna(False).astype(bool)

        return flags.fillna(False)

    @staticmethod
    def explain_row_flags(row_flags: Dict[str, bool]) -> List[str]:
        """Convert a dict of boolean flags for one submission into clear, human-readable reasons."""
        explanations = []
        if row_flags.get("flag_missing_required"):
            explanations.append("Missing mandatory fields (company, title, salary, or experience)")
        if row_flags.get("flag_gibberish_company"):
            explanations.append("Suspicious or placeholder company name (gibberish/test entry)")
        if row_flags.get("flag_zero_or_negative_base"):
            explanations.append("Base salary is zero or negative")
        if row_flags.get("flag_tc_less_than_base"):
            explanations.append("Total compensation is strictly less than base salary")
        if row_flags.get("flag_tc_arithmetic_mismatch"):
            explanations.append("Total compensation differs significantly from Base + Stock + Bonus sum")
        if row_flags.get("flag_ftc_arithmetic_mismatch"):
            explanations.append("First-year total compensation does not agree with first-year components")
        if row_flags.get("flag_yal_gt_yac"):
            explanations.append("Years at current level exceeds total years at company")
        if row_flags.get("flag_yac_gt_yoe"):
            explanations.append("Years at company exceeds total lifetime years of experience")
        if row_flags.get("flag_negative_experience"):
            explanations.append("Experience or tenure field contains negative value")
        if row_flags.get("flag_yoe_out_of_range"):
            explanations.append("Total experience is outside realistic range (0 - 50 years)")
        if row_flags.get("flag_junior_level_high_yoe"):
            explanations.append("Entry-level job ladder (IC1/L1) with excessive years of experience (>8 years)")
        if row_flags.get("flag_offer_after_submit"):
            explanations.append("Offer date is set in the future relative to submission date")
        if row_flags.get("flag_offer_same_time_of_day"):
            explanations.append("Offer timestamp exactly matches submission timestamp (automated script artifact)")
        if row_flags.get("flag_company_name_mismatch"):
            explanations.append("Stated company name does not match verified organization record")
        if row_flags.get("flag_currency_country_mismatch"):
            explanations.append("Stated currency does not match standard domestic currency for location")
        if row_flags.get("flag_vest_pct_not_100"):
            explanations.append("Equity vesting schedule percentages do not sum to 100%")
        if row_flags.get("flag_vest_malformed"):
            explanations.append("Malformed or unparseable vesting schedule data structure")
        if row_flags.get("flag_bonus_pct_mismatch"):
            explanations.append("Target bonus percentage contradicts stated target bonus amount")
        if row_flags.get("flag_duplicate"):
            explanations.append("Duplicate submission detected (identical role, company, comp, and user)")
        if row_flags.get("flag_burst_submission"):
            explanations.append("Suspicious burst activity from the same IP/Device within minutes")

        return explanations