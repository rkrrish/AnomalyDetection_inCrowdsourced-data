# src/trust.py
"""Trust Scoring Engine for Crowdsourced Platform Users.

Upgraded from legacy LOF trust module to integrate CatBoost ML anomaly detection,
deterministic rule violation severity, and historical user credibility tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_USER_TRUST,
    TRUST_SCORE_BOOST,
    TRUST_SCORE_PENALTY_CRITICAL,
    TRUST_SCORE_PENALTY_WARNING,
)


class TrustScorer:
    """Calculates and maintains user trust scores based on submission quality and historical credibility."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df.copy() if df is not None else pd.DataFrame()

    @staticmethod
    def compute_single_trust(
        initial_trust: float = INITIAL_USER_TRUST,
        rule_flags: Dict[str, bool] | None = None,
        p_anomaly: float = 0.0,
        decision: str = "Review",
    ) -> float:
        """Compute user trust score for a single record.

        Penalties:
          - High CatBoost anomaly probability (p_anomaly >= 0.5): penalty up to -0.35
          - Critical violations (TC < Base, negative experience, gibberish, duplicate): -0.30
          - Soft warnings (arithmetic drift, mild timing quirks): -0.15
        Reward:
          - +0.03 (+3%) boost if decision == 'Approve' (High Quality)
        """
        trust = float(initial_trust)
        rule_flags = rule_flags or {}

        # 1. Critical rule violations penalty
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
        if has_critical:
            trust -= TRUST_SCORE_PENALTY_CRITICAL

        # 2. CatBoost ML anomaly penalty
        if p_anomaly >= 0.50:
            ml_penalty = 0.20 + 0.15 * ((p_anomaly - 0.50) / 0.50)
            trust -= ml_penalty
        elif p_anomaly >= 0.35:
            trust -= 0.10

        # 3. Soft warnings penalty
        warning_flags = [
            "flag_tc_arithmetic_mismatch",
            "flag_ftc_arithmetic_mismatch",
            "flag_junior_level_high_yoe",
            "flag_burst_submission",
            "flag_currency_country_mismatch",
            "flag_vest_pct_not_100",
        ]
        has_warning = any(rule_flags.get(f, False) for f in warning_flags)
        if has_warning and not has_critical:
            trust -= TRUST_SCORE_PENALTY_WARNING

        # 4. Reward boost on genuine high quality submission
        if decision in ("Approve", "High Quality"):
            trust = min(1.0, trust + TRUST_SCORE_BOOST)

        return float(np.clip(round(trust, 2), 0.0, 1.0))

    def compute_trust_batch(self, initial_trust: float = INITIAL_USER_TRUST) -> pd.DataFrame:
        """Compute user trust scores across a batch DataFrame."""
        trust_scores = []
        for _, row in self.df.iterrows():
            flags_dict = {c: bool(row[c]) for c in row.index if str(c).startswith("flag_")}
            p_anom = float(row.get("p_anomaly", 0.0))
            dec = str(row.get("decision", "Review"))
            u_trust = row.get("user_trust", initial_trust)

            score = self.compute_single_trust(
                initial_trust=float(u_trust),
                rule_flags=flags_dict,
                p_anomaly=p_anom,
                decision=dec,
            )
            trust_scores.append(score)

        self.df["user_trust"] = trust_scores
        return self.df

    def apply_trust_boost(self) -> pd.DataFrame:
        """Boost trust score by TRUST_SCORE_BOOST if the final decision is 'Approve'."""
        if "decision" not in self.df.columns or "user_trust" not in self.df.columns:
            return self.df

        def _boost(row):
            t = float(row["user_trust"])
            if row["decision"] in ("Approve", "High Quality"):
                t = min(1.0, t + TRUST_SCORE_BOOST)
            return round(t, 2)

        self.df["user_trust"] = self.df.apply(_boost, axis=1)
        return self.df
