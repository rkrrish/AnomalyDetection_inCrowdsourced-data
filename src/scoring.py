# src/scoring.py
"""Consistency & Quality Scoring Engine for Crowdsourced Submissions.

Combines rule consistency, CatBoost ML confidence, and field completeness into
an overall quality assessment score (0 - 100) and confidence score.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    REQUIRED_SUBMISSION_FIELDS,
    RULE_COLUMNS,
    WEIGHT_COMPLETENESS,
    WEIGHT_ML_CONFIDENCE,
    WEIGHT_RULE_CONSISTENCY,
)


class ConsistencyScorer:
    """Computes multidimensional quality and consistency scores for submissions."""

    def __init__(self, df: pd.DataFrame, flags: pd.DataFrame, p_anomaly: np.ndarray | None = None):
        self.df = df.copy()
        self.flags = flags.copy()
        self.p_anomaly = p_anomaly if p_anomaly is not None else np.zeros(len(df))

    @staticmethod
    def compute_single_score(
        row_dict: Dict[str, Any],
        row_flags: Dict[str, bool],
        p_anomaly: float = 0.0,
    ) -> Dict[str, float]:
        """Compute detailed score components for a single submission record.

        Returns dictionary with:
            - rule_score: 0.0 - 1.0
            - ml_confidence: 0.0 - 1.0
            - completeness_score: 0.0 - 1.0
            - quality_score: 0.0 - 100.0
            - confidence_level: 0.0 - 100.0
        """
        # 1. Rule consistency score
        # Weight critical rules heavier than warnings
        critical_flags = [
            "flag_zero_or_negative_base",
            "flag_tc_less_than_base",
            "flag_negative_experience",
            "flag_yac_gt_yoe",
            "flag_yal_gt_yac",
            "flag_gibberish_company",
            "flag_duplicate",
        ]
        num_critical = sum(1 for f in critical_flags if row_flags.get(f, False))
        num_warnings = sum(1 for f, v in row_flags.items() if v and f not in critical_flags)

        rule_penalty = (num_critical * 0.35) + (num_warnings * 0.12)
        rule_score = float(np.clip(1.0 - rule_penalty, 0.0, 1.0))

        # 2. ML Anomaly Confidence (Higher is cleaner/valid)
        ml_confidence = float(np.clip(1.0 - p_anomaly, 0.0, 1.0))

        # 3. Completeness score
        req_present = sum(
            1 for c in REQUIRED_SUBMISSION_FIELDS
            if c in row_dict and pd.notna(row_dict[c]) and str(row_dict[c]).strip() not in ("", "nan", "None", "[]")
        )
        completeness = float(req_present / len(REQUIRED_SUBMISSION_FIELDS))

        # 4. Overall Quality / Trust Score (0 - 100 scale)
        composite = (
            WEIGHT_RULE_CONSISTENCY * rule_score
            + WEIGHT_ML_CONFIDENCE * ml_confidence
            + WEIGHT_COMPLETENESS * completeness
        )

        # Severe penalty if impossible arithmetic or negative salary
        if row_flags.get("flag_tc_less_than_base") or row_flags.get("flag_zero_or_negative_base"):
            composite = min(composite, 0.35)

        quality_score = float(round(composite * 100.0, 1))

        # 5. Confidence Level of the validation system
        # Distance from the decision threshold represents classification margin/certainty
        dist = abs(p_anomaly - 0.50) * 2.0  # 0 at boundary, 1 at 0 or 1
        conf_level = float(round((0.60 + 0.40 * dist) * 100.0, 1))

        return {
            "rule_score": round(rule_score, 3),
            "ml_confidence": round(ml_confidence, 3),
            "completeness_score": round(completeness, 3),
            "quality_score": quality_score,
            "confidence_level": conf_level,
        }

    def compute_scores(self) -> pd.DataFrame:
        """Calculate quality scores across the entire batch DataFrame."""
        rule_scores = []
        ml_confidences = []
        completeness_scores = []
        quality_scores = []
        confidence_levels = []

        for i in range(len(self.df)):
            row_dict = self.df.iloc[i].to_dict()
            flags_dict = self.flags.iloc[i].to_dict()
            p_anom = float(self.p_anomaly[i])

            res = self.compute_single_score(row_dict, flags_dict, p_anom)
            rule_scores.append(res["rule_score"])
            ml_confidences.append(res["ml_confidence"])
            completeness_scores.append(res["completeness_score"])
            quality_scores.append(res["quality_score"])
            confidence_levels.append(res["confidence_level"])

        self.df["rule_score"] = rule_scores
        self.df["ml_confidence"] = ml_confidences
        self.df["completeness_score"] = completeness_scores
        self.df["quality_score"] = quality_scores
        self.df["confidence_level"] = confidence_levels

        # Merge flags into main df
        for col in RULE_COLUMNS:
            if col in self.flags.columns:
                self.df[col] = self.flags[col].values

        return self.df