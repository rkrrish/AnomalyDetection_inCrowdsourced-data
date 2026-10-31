# src/pipeline.py
"""Central Pipeline for Validation, Anomaly Detection, and Quality Scoring.

Orchestrates the entire sequence from raw submission ingestion to final decision
and actionable explanation reporting, as specified in the system sequence diagram.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.catboost_engine import CatBoostEngine
from src.config import INITIAL_USER_TRUST
from src.database import SystemDatabase
from src.decision import DecisionEngine
from src.duplicate_detection import DuplicateDetector
from src.feature_extractor import FeatureExtractor
from src.rule_engine import RuleEngine
from src.scoring import ConsistencyScorer
from src.trust import TrustScorer


class ValidationPipeline:
    """End-to-end crowdsourced data validation pipeline."""

    def __init__(self, db: Optional[SystemDatabase] = None):
        self.db = db or SystemDatabase()
        self.feature_extractor = FeatureExtractor()
        self.catboost_engine = CatBoostEngine()
        self.duplicate_detector = DuplicateDetector(self.db.get_historical_df())

    def evaluate_submission(
        self,
        submission: Dict[str, Any],
        initial_trust: Optional[float] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Process a single incoming submission end-to-end.

        Follows the sequence diagram:
        1. Store raw submission
        2. Validate required fields, data types, completeness, logical consistency
        3. Check duplicate/suspicious patterns
        4. Extract features & query peer benchmarks
        5. Predict CatBoost ML anomaly score
        6. Calculate trust & consistency scores
        7. Classify submission (Approve / Review / Reject)
        8. Update user trust score & persist result
        """
        sub_id = submission.get("submissionId") or submission.get("uuid") or str(uuid.uuid4())
        u_id = submission.get("userId", f"u_{sub_id[:8]}")
        submission["submissionId"] = sub_id
        submission["userId"] = u_id

        # 1. Look up user trust if not provided
        if initial_trust is None:
            user_prof = self.db.get_user(u_id)
            initial_trust = float(user_prof.get("trustScore", INITIAL_USER_TRUST))

        # 2. Duplicate & Burst check
        is_dup, is_burst, dup_reason = self.duplicate_detector.check_single(submission)
        submission["flag_duplicate"] = is_dup
        submission["flag_burst_submission"] = is_burst

        # 3. Rule validation
        df_single = pd.DataFrame([submission])
        rule_engine = RuleEngine(df_single)
        flags_df = rule_engine.apply_rules()
        flags_dict = flags_df.iloc[0].to_dict()

        # 4. Feature Extraction & Peer statistics
        feat_df = self.feature_extractor.extract_features(df_single)

        # 5. ML Anomaly Prediction
        p_anom, preds, ml_exps = self.catboost_engine.predict_anomalies(feat_df)
        anomaly_prob = float(p_anom[0])
        ml_factors = ml_exps[0].get("top_factors", [])

        # 6. Consistency & Quality Scoring
        scores = ConsistencyScorer.compute_single_score(
            row_dict=submission,
            row_flags=flags_dict,
            p_anomaly=anomaly_prob,
        )

        # 7. Decision Classification & Actionable Explanations
        decision, reasons = DecisionEngine.classify_single_submission(
            quality_score=scores["quality_score"],
            p_anomaly=anomaly_prob,
            rule_flags=flags_dict,
            ml_factors=ml_factors,
        )
        if dup_reason and dup_reason not in reasons:
            reasons.append(dup_reason)

        # 8. User Trust Score calculation
        user_trust = TrustScorer.compute_single_trust(
            initial_trust=initial_trust,
            rule_flags=flags_dict,
            p_anomaly=anomaly_prob,
            decision=decision,
        )

        # 9. Extract Peer Benchmarks for Display
        peer_median = feat_df.get("peer_exp_base_usd_log", pd.Series([np.nan])).iloc[0]
        peer_z = feat_df.get("peer_z_base_usd_log", pd.Series([0.0])).iloc[0]
        peer_median_usd = float(np.expm1(peer_median)) if pd.notna(peer_median) and peer_median > 0 else None

        # Build full validation result
        result: Dict[str, Any] = {
            "submissionId": sub_id,
            "userId": u_id,
            "company": submission.get("company", "Unknown"),
            "jobRole": submission.get("title", submission.get("jobFamily", "Software Engineer")),
            "jobFamily": submission.get("jobFamily", "Software Engineer"),
            "level": submission.get("level", "IC1"),
            "location": submission.get("location", "Unknown"),
            "baseSalary": float(submission.get("baseSalary", 0) or 0),
            "baseSalaryCurrency": submission.get("baseSalaryCurrency", "USD"),
            "totalCompensation": float(submission.get("totalCompensation", 0) or 0),
            "yearsOfExperience": float(submission.get("yearsOfExperience", 0) or 0),
            "submittedAt": str(submission.get("submittedAt", pd.Timestamp.utcnow().isoformat())),
            "qualityScore": scores["quality_score"],
            "confidenceScore": scores["confidence_level"],
            "classification": decision,
            "p_anomaly": round(anomaly_prob, 4),
            "user_trust": user_trust,
            "rule_score": scores["rule_score"],
            "completeness_score": scores["completeness_score"],
            "peer_median_usd": peer_median_usd,
            "peer_z_score": round(float(peer_z), 2) if pd.notna(peer_z) else 0.0,
            "flags": {k: bool(v) for k, v in flags_dict.items() if v},
            "explanation": "; ".join(reasons),
            "reasons": reasons,
            "status": "Validated",
        }

        # Save to database
        if persist:
            self.db.save_submission(result)

        return result

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch process a DataFrame of submissions."""
        out = df.copy()

        # 1. Duplicate & Burst detection
        out = self.duplicate_detector.detect_batch_duplicates(out)

        # 2. Rule evaluation
        rule_engine = RuleEngine(out)
        flags_df = rule_engine.apply_rules()

        # 3. Feature Extraction
        feat_df = self.feature_extractor.extract_features(out)

        # 4. ML Anomaly Prediction
        p_anom, preds, ml_exps = self.catboost_engine.predict_anomalies(feat_df)
        out["p_anomaly"] = p_anom

        # 5. Scoring
        scorer = ConsistencyScorer(out, flags_df, p_anom)
        out = scorer.compute_scores()

        # 6. Decisions & Explanations
        dec_engine = DecisionEngine(out)
        out = dec_engine.make_decisions()

        # 7. User Trust calculation
        trust_scorer = TrustScorer(out)
        out = trust_scorer.compute_trust_batch()

        return out


def run_pipeline(input_data: pd.DataFrame | Dict[str, Any]) -> pd.DataFrame | Dict[str, Any]:
    """Helper entrypoint for pipeline invocation."""
    pipeline = ValidationPipeline()
    if isinstance(input_data, dict):
        return pipeline.evaluate_submission(input_data)
    elif isinstance(input_data, pd.DataFrame):
        return pipeline.evaluate_batch(input_data)
    else:
        raise ValueError("Input must be a dictionary or a pandas DataFrame")