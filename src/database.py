# src/database.py
"""Lightweight Database & Persistence Layer for Submissions and Audit Records.

Supports CRUD operations for submissions, user trust records, moderation queues,
and user bug/issue reports as detailed in the sequence and class diagrams.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import DATA_PATH, DB_PATH


class SystemDatabase:
    """Thread-safe persistent database for crowdsourced validation system."""

    _lock = threading.Lock()

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.submissions: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.reports: List[Dict[str, Any]] = []
        self.audit_logs: List[Dict[str, Any]] = []

        self._load()
        if not self.submissions:
            self._seed_initial_data()

    def _load(self) -> None:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.submissions = data.get("submissions", {})
                    self.users = data.get("users", {})
                    self.reports = data.get("reports", [])
                    self.audit_logs = data.get("audit_logs", [])
            except Exception as e:
                print(f"[SystemDatabase] Warning: Failed to load {self.db_path}: {e}")

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump({
                "submissions": self.submissions,
                "users": self.users,
                "reports": self.reports,
                "audit_logs": self.audit_logs,
            }, f, indent=2)

    def _seed_initial_data(self) -> None:
        """Seed a small initial set of clean sample submissions from data/submissions.csv."""
        if os.path.exists(DATA_PATH):
            try:
                sample_df = pd.read_csv(DATA_PATH, nrows=500, low_memory=False)
                if "is_valid" in sample_df.columns:
                    sample_df = sample_df[sample_df["is_valid"] == 1].head(30)
                else:
                    sample_df = sample_df.head(30)
                for _, row in sample_df.iterrows():
                    sub_id = str(row.get("uuid", uuid.uuid4()))
                    u_id = str(row.get("userId", f"u_{sub_id[:8]}"))
                    is_valid = 1

                    sub_data = {
                        "submissionId": sub_id,
                        "userId": u_id,
                        "company": str(row.get("company", "Unknown")),
                        "jobRole": str(row.get("title", row.get("jobFamily", "Software Engineer"))),
                        "jobFamily": str(row.get("jobFamily", "Software Engineer")),
                        "level": str(row.get("level", "IC1")),
                        "location": str(row.get("location", "US")),
                        "baseSalary": float(row.get("baseSalary", 100000) or 100000),
                        "baseSalaryCurrency": str(row.get("baseSalaryCurrency", "USD")),
                        "totalCompensation": float(row.get("totalCompensation", 120000) or 120000),
                        "avgAnnualStockGrantValue": float(row.get("avgAnnualStockGrantValue", 0) or 0),
                        "avgAnnualBonusValue": float(row.get("avgAnnualBonusValue", 0) or 0),
                        "yearsOfExperience": float(row.get("yearsOfExperience", 2) or 2),
                        "yearsAtCompany": float(row.get("yearsAtCompany", 1) or 1),
                        "yearsAtLevel": float(row.get("yearsAtLevel", 1) or 1),
                        "submittedAt": str(row.get("submittedAt", pd.Timestamp.utcnow().isoformat())),
                        "qualityScore": 92.0 if is_valid == 1 else 35.0,
                        "confidenceScore": 95.0,
                        "classification": "Approve" if is_valid == 1 else "Reject",
                        "p_anomaly": 0.08 if is_valid == 1 else 0.85,
                        "status": "Validated",
                        "explanation": "Verified clean entry from initial reference dataset." if is_valid == 1 else "Flagged defect in initial reference dataset.",
                    }
                    self.submissions[sub_id] = sub_data
                    if u_id not in self.users:
                        self.users[u_id] = {
                            "userId": u_id,
                            "username": f"user_{u_id[-4:]}",
                            "role": "User",
                            "trustScore": 1.0 if is_valid == 1 else 0.65,
                            "submissionCount": 1
                        }

                self._save()
            except Exception as e:
                print(f"[SystemDatabase] Warning during seeding: {e}")

    def save_submission(self, sub_data: Dict[str, Any]) -> str:
        with self._lock:
            sub_id = sub_data.get("submissionId") or str(uuid.uuid4())
            sub_data["submissionId"] = sub_id
            self.submissions[sub_id] = sub_data

            # Update user profile
            u_id = sub_data.get("userId", "guest")
            if u_id not in self.users:
                self.users[u_id] = {
                    "userId": u_id,
                    "username": f"user_{u_id[-4:]}",
                    "role": "User",
                    "trustScore": 1.0,
                    "submissionCount": 0
                }
            self.users[u_id]["submissionCount"] = self.users[u_id].get("submissionCount", 0) + 1
            if "user_trust" in sub_data:
                self.users[u_id]["trustScore"] = float(sub_data["user_trust"])

            self._save()
            return sub_id

    def get_submission(self, sub_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.submissions.get(sub_id)

    def list_submissions(self, filter_status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            subs = list(self.submissions.values())
            if filter_status:
                subs = [s for s in subs if s.get("classification") == filter_status or s.get("status") == filter_status]
            # Most recent first
            subs.sort(key=lambda x: str(x.get("submittedAt", "")), reverse=True)
            return subs[:limit]

    def get_user(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.users.get(user_id, {
                "userId": user_id,
                "username": f"user_{user_id[-4:]}",
                "role": "User",
                "trustScore": 1.0,
                "submissionCount": 0
            })

    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [s for s in self.submissions.values() if s.get("userId") == user_id]

    def update_admin_decision(self, sub_id: str, new_decision: str, comments: str, admin_id: str = "admin") -> bool:
        with self._lock:
            if sub_id not in self.submissions:
                return False
            sub = self.submissions[sub_id]
            old_dec = sub.get("classification")
            sub["classification"] = new_decision
            sub["status"] = f"Admin_{new_decision}"
            sub["adminComments"] = comments

            # Adjust user trust
            u_id = sub.get("userId")
            if u_id and u_id in self.users:
                if new_decision in ("Approve", "High Quality"):
                    self.users[u_id]["trustScore"] = min(1.0, self.users[u_id]["trustScore"] + 0.05)
                elif new_decision in ("Reject", "Anomalous"):
                    self.users[u_id]["trustScore"] = max(0.0, self.users[u_id]["trustScore"] - 0.20)

            # Audit log
            self.audit_logs.append({
                "auditId": str(uuid.uuid4()),
                "submissionId": sub_id,
                "adminId": admin_id,
                "oldDecision": old_dec,
                "newDecision": new_decision,
                "comments": comments,
                "timestamp": pd.Timestamp.utcnow().isoformat()
            })
            self._save()
            return True

    def create_report(self, report_data: Dict[str, Any]) -> str:
        with self._lock:
            rep_id = str(uuid.uuid4())
            report_data["reportId"] = rep_id
            report_data["createdAt"] = pd.Timestamp.utcnow().isoformat()
            report_data["status"] = "Open"
            self.reports.append(report_data)
            self._save()
            return rep_id

    def list_reports(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.reports)

    def get_historical_df(self) -> pd.DataFrame:
        with self._lock:
            if not self.submissions:
                return pd.DataFrame()
            return pd.DataFrame(list(self.submissions.values()))
