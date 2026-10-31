# src/decision.py
"""Decision Engine for Submissions.

Implements the 3-tier classification hierarchy:
  - 'Approve' (High Quality)
  - 'Review' (Review Recommended)
  - 'Reject' (Potentially Problematic / Anomalous)
Generates human-readable actionable explanations for all flagged issues.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.config import DECISION_APPROVE_MIN_SCORE, DECISION_REJECT_MAX_SCORE
from src.rule_engine import RuleEngine


class DecisionEngine:
    """Classifies submissions into 3 actionable tiers and generates clear explanations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def classify_single_submission(
        quality_score: float,
        p_anomaly: float,
        rule_flags: Dict[str, bool],
        ml_factors: List[str] | None = None,
    ) -> Tuple[str, List[str]]:
        """Classify a single submission and generate actionable explanation reasons.

        Returns:
            decision: 'Approve' | 'Review' | 'Reject'
            reasons: list of actionable bulleted reason strings
        """
        # Critical failure checks
        critical_flags = [
            "flag_zero_or_negative_base",
            "flag_tc_less_than_base",
            "flag_negative_experience",
            "flag_yac_gt_yoe",
            "flag_yal_gt_yac",
            "flag_gibberish_company",
            "flag_duplicate",
        ]
        has_critical = any(rule_flags.get(f, False) for f in critical_flags)

        # Warning checks
        warning_flags = [
            "flag_tc_arithmetic_mismatch",
            "flag_ftc_arithmetic_mismatch",
            "flag_junior_level_high_yoe",
            "flag_offer_after_submit",
            "flag_company_name_mismatch",
            "flag_currency_country_mismatch",
            "flag_vest_pct_not_100",
            "flag_bonus_pct_mismatch",
            "flag_burst_submission",
        ]
        has_warning = any(rule_flags.get(f, False) for f in warning_flags)

        # 3-Tier Decision Logic
        if has_critical or quality_score < DECISION_REJECT_MAX_SCORE or p_anomaly >= 0.70:
            decision = "Reject"
        elif has_warning or (p_anomaly >= 0.45) or (quality_score < DECISION_APPROVE_MIN_SCORE):
            decision = "Review"
        else:
            decision = "Approve"

        # Actionable reasons
        reasons = RuleEngine.explain_row_flags(rule_flags)

        if p_anomaly >= 0.50:
            reasons.append(f"CatBoost ML flagged this compensation profile as anomalous (anomaly score: {p_anomaly:.1%})")
            if ml_factors:
                for f in ml_factors:
                    if f not in reasons and not f.startswith("ML evaluation"):
                        reasons.append(f)
        elif p_anomaly >= 0.35:
            reasons.append(f"CatBoost ML noted borderline compensation pattern (anomaly score: {p_anomaly:.1%})")

        if not reasons:
            reasons = ["All logical consistency checks passed; compensation is well-aligned with peer distributions."]

        return decision, reasons

    def make_decisions(self) -> pd.DataFrame:
        """Classify all submissions in the DataFrame and produce explanations."""
        decisions = []
        explanations = []

        for _, row in self.df.iterrows():
            q_score = float(row.get("quality_score", 50.0))
            p_anom = float(row.get("p_anomaly", 0.0))
            flags_dict = {c: bool(row[c]) for c in row.index if str(c).startswith("flag_")}
            ml_factors = row.get("ml_factors", [])
            if isinstance(ml_factors, str):
                try:
                    import json
                    ml_factors = json.loads(ml_factors)
                except Exception:
                    ml_factors = [ml_factors]

            dec, reasons = self.classify_single_submission(
                quality_score=q_score,
                p_anomaly=p_anom,
                rule_flags=flags_dict,
                ml_factors=ml_factors,
            )
            decisions.append(dec)
            explanations.append("; ".join(reasons))

        self.df["decision"] = decisions
        self.df["explanation"] = explanations
        return self.df
