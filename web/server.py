"""web/server.py - Lightweight REST API & Simulation Server.

Zero-dependency HTTP server built with Python standard library (http.server)
running in the py310 conda environment. Implements all endpoints specified
in the sequence diagram and use case diagram.

Usage:
    python web/server.py --port 5000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database import SystemDatabase
from src.pipeline import ValidationPipeline

PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
db = SystemDatabase()
pipeline = ValidationPipeline(db=db)

# Pre-loaded simulation scenarios directly inspired by Levels.pdf & DATASET_DOCUMENTATION.md
DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "genuine_senior": {
        "title": "Genuine SWE Compensation (Clean)",
        "badge": "Genuine",
        "description": "Standard authentic compensation profile at a Tier-1 tech company.",
        "data": {
            "company": "Microsoft",
            "title": "Software Engineer",
            "jobFamily": "Software Engineer",
            "jobFamilySlug": "software-engineer",
            "level": "60",
            "focusTag": "Software Engineer",
            "yearsOfExperience": 3,
            "yearsAtCompany": 0,
            "offerDate": "Mon Feb 06 2023 08:58:00 GMT+0000 (Coordinated Universal Time)",
            "location": "New York, NY",
            "locationSlug": "new-york-usa",
            "workArrangement": "hybrid",
            "compPerspective": "offer",
            "cityId": 70032,
            "dmaId": 18651,
            "countryId": 253,
            "exchangeRate": 1.0,
            "baseSalary": 158600,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 235300.0,
            "firstYearTotalCompensation": 233100,
            "avgAnnualStockGrantValue": 55500.0,
            "firstYearStockGrantValue": 55500.0,
            "totalStockGrantValue": 222000.0,
            "stockGrantCurrency": "USD",
            "avgAnnualBonusValue": 21200.0,
            "firstYearBonusValue": 19000,
            "bonusCurrency": "USD",
            "stockType": "stock",
            "education": "Master's Degree",
            "submittedAt": "2023-02-11T11:35:03Z",
            "userId": "u_genuine_01"
        }
    },
    "absurd_entry": {
        "title": "Absurd Entry-Level Compensation",
        "badge": "Defect A1",
        "description": "Problem Statement Scenario: An entry-level applicant claiming $850,000 base salary.",
        "data": {
            "company": "Google",
            "jobFamily": "Software Engineer",
            "title": "Software Engineer",
            "level": "L3",
            "location": "Mountain View, CA",
            "locationSlug": "mountain-view-usa",
            "baseSalary": 850000,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 1200000,
            "avgAnnualStockGrantValue": 250000,
            "avgAnnualBonusValue": 100000,
            "yearsOfExperience": 0,
            "yearsAtCompany": 0,
            "yearsAtLevel": 0,
            "workArrangement": "remote",
            "userId": "u_bad_actor_9"
        }
    },
    "tc_less_than_base": {
        "title": "Total Comp Less Than Base (Arithmetic Inversion)",
        "badge": "Defect B1",
        "description": "Contradictory data: Total Compensation ($90,000) is strictly lower than Base Salary ($140,000).",
        "data": {
            "company": "Microsoft",
            "jobFamily": "Software Engineer",
            "title": "Software Engineer",
            "level": "62",
            "location": "Redmond, WA",
            "locationSlug": "redmond-usa",
            "baseSalary": 140000,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 90000,
            "avgAnnualStockGrantValue": 30000,
            "avgAnnualBonusValue": 15000,
            "yearsOfExperience": 5,
            "yearsAtCompany": 2,
            "yearsAtLevel": 1,
            "workArrangement": "onsite",
            "userId": "u_confused_user"
        }
    },
    "experience_inversion": {
        "title": "Tenure Exceeds Lifetime Experience",
        "badge": "Defect B2",
        "description": "Logically impossible career progression: 8 years at company but only 2 total years of experience.",
        "data": {
            "company": "Amazon",
            "jobFamily": "Software Engineer",
            "title": "Software Engineer",
            "level": "L5",
            "location": "Seattle, WA",
            "locationSlug": "seattle-usa",
            "baseSalary": 160000,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 240000,
            "avgAnnualStockGrantValue": 60000,
            "avgAnnualBonusValue": 20000,
            "yearsOfExperience": 2,
            "yearsAtCompany": 8,
            "yearsAtLevel": 4,
            "workArrangement": "hybrid",
            "userId": "u_error_user"
        }
    },
    "currency_confusion": {
        "title": "Currency / Unit Confusion",
        "badge": "Defect A3",
        "description": "Indian Rupee compensation (1,800,000) incorrectly entered with currency code USD.",
        "data": {
            "company": "Infosys",
            "jobFamily": "Software Engineer",
            "title": "Software Engineer",
            "level": "Senior Systems Engineer",
            "location": "Bengaluru, KA, India",
            "locationSlug": "bengaluru-ind",
            "baseSalary": 1800000,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 2100000,
            "avgAnnualStockGrantValue": 0,
            "avgAnnualBonusValue": 300000,
            "yearsOfExperience": 4,
            "yearsAtCompany": 2,
            "yearsAtLevel": 1,
            "workArrangement": "hybrid",
            "userId": "u_rupee_usd_actor"
        }
    },
    "rapid_burst": {
        "title": "Coordinated Rapid Burst (Spam)",
        "badge": "Defect A6",
        "description": "Repeated submission from identical IP hash within 2 minutes.",
        "data": {
            "company": "Meta",
            "jobFamily": "Software Engineer",
            "title": "Software Engineer",
            "level": "E4",
            "location": "Menlo Park, CA",
            "locationSlug": "menlo-park-usa",
            "baseSalary": 175000,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 270000,
            "avgAnnualStockGrantValue": 75000,
            "avgAnnualBonusValue": 20000,
            "yearsOfExperience": 4,
            "yearsAtCompany": 1,
            "yearsAtLevel": 1,
            "ipHash": "repeat_ip_999",
            "deviceHash": "repeat_dev_888",
            "userId": "u_bot_spammer"
        }
    }
}


class ValidationServerHandler(BaseHTTPRequestHandler):
    """Custom request handler implementing REST API and static asset serving."""

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # API Endpoints
        if path == "/api/scenarios":
            self._send_json(DEMO_SCENARIOS)
            return

        elif path == "/api/submissions":
            status_filter = query.get("status", [None])[0]
            limit = int(query.get("limit", [100])[0])
            subs = db.list_submissions(filter_status=status_filter, limit=limit)
            self._send_json(subs)
            return

        elif path.startswith("/api/submissions/"):
            sub_id = path[len("/api/submissions/"):]
            sub = db.get_submission(sub_id)
            if sub:
                self._send_json(sub)
            else:
                self._send_json({"error": "Submission not found"}, status=404)
            return

        elif path == "/api/user/profile":
            user_id = query.get("userId", ["u_genuine_01"])[0]
            user = db.get_user(user_id)
            history = db.get_user_history(user_id)
            self._send_json({"user": user, "history": history})
            return

        elif path == "/api/reports":
            reports = db.list_reports()
            self._send_json(reports)
            return

        elif path == "/api/stats":
            subs = db.list_submissions(limit=1000)
            total = len(subs)
            approved = sum(1 for s in subs if s.get("classification") in ("Approve", "High Quality"))
            review = sum(1 for s in subs if s.get("classification") in ("Review", "Review Recommended"))
            rejected = sum(1 for s in subs if s.get("classification") in ("Reject", "Anomalous", "Potentially Problematic"))

            scores = [float(s.get("qualityScore", 0)) for s in subs if "qualityScore" in s]
            avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

            stats = {
                "totalSubmissions": total,
                "approvedCount": approved,
                "reviewCount": review,
                "rejectedCount": rejected,
                "averageQualityScore": avg_score,
                "approvalRate": round(approved / total * 100, 1) if total > 0 else 0.0,
                "anomalyRate": round(rejected / total * 100, 1) if total > 0 else 0.0,
                "modelLoaded": pipeline.catboost_engine.is_loaded,
                "modelThreshold": pipeline.catboost_engine.threshold
            }
            self._send_json(stats)
            return

        # Static Asset Serving
        if path == "/" or path == "":
            path = "/index.html"

        safe_path = os.path.normpath(path.lstrip("/"))
        file_path = os.path.join(PUBLIC_DIR, safe_path)

        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                self._send_json({"error": f"Failed to read file: {e}"}, status=500)
                return

        self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {e}"}, status=400)
            return

        if path == "/api/submissions":
            # New Submission Validation
            result = pipeline.evaluate_submission(body)
            self._send_json(result)
            return

        elif path == "/api/admin/review":
            # Admin Review & Override
            sub_id = body.get("submissionId")
            decision = body.get("decision")
            comments = body.get("comments", "")
            admin_id = body.get("adminId", "admin")

            if not sub_id or not decision:
                self._send_json({"error": "Missing submissionId or decision"}, status=400)
                return

            ok = db.update_admin_decision(sub_id, decision, comments, admin_id)
            if ok:
                self._send_json({"status": "success", "submissionId": sub_id, "newDecision": decision})
            else:
                self._send_json({"error": "Submission not found"}, status=404)
            return

        elif path == "/api/reports":
            # User Reports Issue / Anomalous Data
            rep_id = db.create_report(body)
            self._send_json({"status": "success", "reportId": rep_id})
            return

        self._send_json({"error": "Not Found"}, status=404)


def run_server(port: int = 5000) -> None:
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, ValidationServerHandler)
    print("=" * 60)
    print("CROWDSOURCED DATA VALIDATION SIMULATION SERVER")
    print(f"Running locally at: http://localhost:{port}")
    print(f"Serving UI from:    {PUBLIC_DIR}")
    print(f"CatBoost Model:     {'LOADED' if pipeline.catboost_engine.is_loaded else 'FALLBACK (Run train_catboost.py)'}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Validation Simulation Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind server (default 5000)")
    args = parser.parse_args()
    run_server(port=args.port)
