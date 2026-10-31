# src/duplicate_detection.py
"""Duplicate & Coordinated Burst Detection Engine.

Detects repeated/jittered compensation submissions and coordinated bursts
from identical IP/Device/User actors as described in Levels.pdf and DATASET_DOCUMENTATION.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class DuplicateDetector:
    """Detects duplicate submissions and rapid coordinated bursts."""

    def __init__(self, historical_df: pd.DataFrame | None = None):
        self.history = historical_df.copy() if historical_df is not None else pd.DataFrame()

    def check_single(self, new_record: Dict[str, Any]) -> Tuple[bool, bool, str]:
        """Check a single incoming record against historical submissions.

        Returns:
            is_duplicate: True if matching role, company, and near-identical comp found
            is_burst: True if same IP or device made multiple submissions within 15 minutes
            reason: explanatory text
        """
        sub_id = str(new_record.get("submissionId") or new_record.get("uuid") or "")
        u_id = str(new_record.get("userId", ""))
        co = str(new_record.get("company", "")).strip().lower()
        fam = str(new_record.get("jobFamily", "")).strip().lower()
        lvl = str(new_record.get("level", "")).strip().lower()
        base = float(new_record.get("baseSalary", 0) or 0)
        tc = float(new_record.get("totalCompensation", 0) or 0)
        yoe = float(new_record.get("yearsOfExperience", 0) or 0)
        loc = str(new_record.get("location", "")).strip().lower()
        ip = str(new_record.get("ipHash", ""))
        dev = str(new_record.get("deviceHash", ""))
        sub_time = pd.to_datetime(new_record.get("submittedAt", pd.Timestamp.utcnow()), utc=True)

        is_dup = False
        is_burst = False
        reasons = []

        # Exclude self if already present in history
        history_pool = self.history
        if sub_id and "submissionId" in history_pool.columns:
            history_pool = history_pool[history_pool["submissionId"].astype(str) != sub_id]
        if sub_id and "uuid" in history_pool.columns:
            history_pool = history_pool[history_pool["uuid"].astype(str) != sub_id]

        if not history_pool.empty and co and fam and base > 0:
            # Mask for candidate match on core role
            match_mask = (
                (history_pool["company"].astype(str).str.lower().str.strip() == co) &
                (history_pool["jobFamily"].astype(str).str.lower().str.strip() == fam) &
                (history_pool["level"].astype(str).str.lower().str.strip() == lvl)
            )
            candidates = history_pool.loc[match_mask]
            if not candidates.empty:
                # 1. Same user submitting duplicate for same role
                if u_id and "userId" in candidates.columns:
                    same_user_match = (candidates["userId"].astype(str) == u_id)
                    if same_user_match.any():
                        is_dup = True
                        reasons.append("Duplicate: Same user already submitted this role and level")

                # 2. Same IP/Device submitting duplicate with near-identical comp (within 3%)
                if not is_dup and (ip or dev):
                    actor_cond = pd.Series(False, index=candidates.index)
                    if ip and "ipHash" in candidates.columns:
                        actor_cond |= (candidates["ipHash"].astype(str) == ip)
                    if dev and "deviceHash" in candidates.columns:
                        actor_cond |= (candidates["deviceHash"].astype(str) == dev)
                    actor_candidates = candidates.loc[actor_cond]
                    if not actor_candidates.empty:
                        hist_bases = pd.to_numeric(actor_candidates["baseSalary"], errors="coerce")
                        diffs = (hist_bases - base).abs() / base
                        if (diffs <= 0.03).any():
                            is_dup = True
                            reasons.append("Duplicate: Repeated submission from identical network/device")

                # 3. Exact full clone across company, role, level, location, base, totalComp, and YOE
                if not is_dup and tc > 0 and "totalCompensation" in candidates.columns:
                    cand_bases = pd.to_numeric(candidates["baseSalary"], errors="coerce")
                    cand_tcs = pd.to_numeric(candidates["totalCompensation"], errors="coerce")
                    cand_yoes = pd.to_numeric(candidates.get("yearsOfExperience", pd.Series(np.nan, index=candidates.index)), errors="coerce")
                    clone_match = (
                        (cand_bases == base) &
                        (cand_tcs == tc) &
                        ((cand_yoes == yoe) | cand_yoes.isna())
                    )
                    if "location" in candidates.columns and loc:
                        clone_match &= (candidates["location"].astype(str).str.lower().str.strip() == loc)

                    if clone_match.any():
                        is_dup = True
                        reasons.append("Duplicate: Exact duplicate clone detected across all compensation fields")

        # Check for rapid burst from IP/Device
        if (ip or dev) and "submittedAt" in self.history.columns:
            actor_mask = pd.Series(False, index=self.history.index)
            if ip and "ipHash" in self.history.columns:
                actor_mask |= (self.history["ipHash"].astype(str) == ip)
            if dev and "deviceHash" in self.history.columns:
                actor_mask |= (self.history["deviceHash"].astype(str) == dev)

            actor_rows = self.history.loc[actor_mask]
            if not actor_rows.empty:
                hist_times = pd.to_datetime(actor_rows["submittedAt"], errors="coerce", utc=True)
                time_diffs_mins = (sub_time - hist_times).abs().dt.total_seconds() / 60.0
                recent_count = (time_diffs_mins <= 15.0).sum()
                if recent_count >= 2:
                    is_burst = True
                    reasons.append(f"Rapid Burst: {recent_count} submissions from this network/device within 15 minutes")

        return is_dup, is_burst, "; ".join(reasons)

    def detect_batch_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch duplicate and burst detection on a DataFrame."""
        out = df.copy()
        n = len(out)
        flag_dup = [False] * n
        flag_burst = [False] * n

        if "submittedAt" in out.columns:
            times = pd.to_datetime(out["submittedAt"], errors="coerce", utc=True)
        else:
            times = pd.Series(pd.Timestamp.utcnow(), index=out.index)

        # Content duplicate check: Requires actor match OR exact clone across all comp fields
        key_cols = ["company", "jobFamily", "level"]
        present_keys = [c for c in key_cols if c in out.columns]
        if len(present_keys) == len(key_cols) and "baseSalary" in out.columns and "totalCompensation" in out.columns:
            clone_keys = out["company"].astype(str).str.lower() + "|" + \
                         out["jobFamily"].astype(str).str.lower() + "|" + \
                         out["level"].astype(str).str.lower() + "|" + \
                         out["baseSalary"].astype(str) + "|" + \
                         out["totalCompensation"].astype(str)
            if "userId" in out.columns:
                user_role_keys = out["userId"].astype(str) + "|" + out["company"].astype(str).str.lower() + "|" + out["jobFamily"].astype(str).str.lower()
            else:
                user_role_keys = pd.Series("", index=out.index)

            seen_clones = set()
            seen_user_roles = set()
            for i in range(n):
                ck = clone_keys.iloc[i]
                urk = user_role_keys.iloc[i]
                if ck in seen_clones:
                    flag_dup[i] = True
                else:
                    seen_clones.add(ck)

                if urk and urk.split("|")[0]:
                    if urk in seen_user_roles:
                        flag_dup[i] = True
                    else:
                        seen_user_roles.add(urk)

        # Burst check
        if "ipHash" in out.columns or "deviceHash" in out.columns:
            for i in range(n):
                ip_i = out["ipHash"].iloc[i] if "ipHash" in out.columns else None
                t_i = times.iloc[i]
                if pd.isna(t_i):
                    continue
                count = 0
                for j in range(max(0, i - 50), i):
                    ip_j = out["ipHash"].iloc[j] if "ipHash" in out.columns else None
                    t_j = times.iloc[j]
                    if ip_i and ip_i == ip_j and pd.notna(t_j):
                        if abs((t_i - t_j).total_seconds()) < 900:  # 15 minutes
                            count += 1
                if count >= 2:
                    flag_burst[i] = True

        out["flag_duplicate"] = flag_dup
        out["flag_burst_submission"] = flag_burst
        return out