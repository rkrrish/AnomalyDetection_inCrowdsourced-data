# src/catboost_engine.py
"""CatBoost Anomaly Detector Engine.

Loads the serialized CatBoostClassifier and predicts anomaly probabilities,
calibrated anomaly classifications, and key driver explanations.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import DEFAULT_ANOMALY_THRESHOLD, MODEL_META_PATH, MODEL_PATH
from src.feature_extractor import FeatureExtractor


class CatBoostEngine:
    """Wrapper around trained CatBoost model for fast anomaly probability estimation."""

    def __init__(self, model_path: str = MODEL_PATH, meta_path: str = MODEL_META_PATH):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model: CatBoostClassifier | None = None
        self.threshold: float = DEFAULT_ANOMALY_THRESHOLD
        self.meta: Dict[str, Any] = {}
        self.is_loaded: bool = False

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load trained CatBoost model and metadata."""
        if os.path.exists(self.model_path):
            try:
                self.model = CatBoostClassifier()
                self.model.load_model(self.model_path)
                self.is_loaded = True
            except Exception as e:
                print(f"[CatBoostEngine] Warning: Failed to load model from {self.model_path}: {e}")
                self.model = None
                self.is_loaded = False

        if os.path.exists(self.meta_path):
            try:
                import json
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
                    self.threshold = float(self.meta.get("optimal_threshold", DEFAULT_ANOMALY_THRESHOLD))
            except Exception:
                pass

    def predict_anomalies(self, X_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Generate anomaly probabilities, binary anomaly flags, and top feature drivers.

        Returns:
            probabilities: 1D array of p_anomaly in [0, 1]
            predictions: 1D array of binary anomaly flags (1 = anomaly, 0 = valid)
            explanations: list of dictionaries detailing the top driving feature factors
        """
        n_rows = len(X_features)

        if not self.is_loaded or self.model is None:
            # Fallback heuristic if model is not yet trained
            p_anom = np.zeros(n_rows, dtype=float)
            if "peer_z_base_usd_log" in X_features.columns:
                z = X_features["peer_z_base_usd_log"].abs().fillna(0).to_numpy()
                p_anom = np.clip(z / 4.0, 0.05, 0.95)
            preds = (p_anom >= self.threshold).astype(int)
            explanations = [{"top_factors": ["Heuristic fallback (Run train_catboost.py to train model)"]}] * n_rows
            return p_anom, preds, explanations

        cat_cols = self.meta.get("categorical_columns", [])
        pool = Pool(X_features, cat_features=cat_cols)

        # Class 1 = Anomaly, Class 0 = Valid
        proba = self.model.predict_proba(pool)[:, 1]
        preds = (proba >= self.threshold).astype(int)

        # Feature importance drivers
        feature_names = list(X_features.columns)
        importances = self.model.get_feature_importance()
        top_feat_indices = np.argsort(importances)[::-1][:5]
        top_global_feats = [feature_names[i] for i in top_feat_indices]

        explanations = []
        for i in range(n_rows):
            factors = []
            row = X_features.iloc[i]
            # Check for peer z deviation
            z_base = row.get("peer_z_base_usd_log", 0)
            if pd.notna(z_base) and abs(z_base) > 2.0:
                dir_str = "above" if z_base > 0 else "below"
                factors.append(f"Base compensation is {abs(z_base):.1f} std deviations {dir_str} peer median")

            tc_resid = row.get("tc_resid_raw_rel", 0)
            if pd.notna(tc_resid) and abs(tc_resid) > 0.10:
                factors.append(f"Total comp deviates by {abs(tc_resid):.1%} from component sum")

            if not factors:
                factors.append(f"ML evaluation influenced by {top_global_feats[0]} and {top_global_feats[1]}")

            explanations.append({
                "top_factors": factors,
                "global_drivers": top_global_feats[:3]
            })

        return proba, preds, explanations
