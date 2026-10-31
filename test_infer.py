import os
import sys

from src.pipeline import ValidationPipeline
from infer import print_single_report

test_cases = [
    {
        "name": "Case 1: Genuine SWE (Clean)",
        "expected": "Approve",
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
        }
    },
    {
        "name": "Case 2: Absurd Entry-Level Compensation ($850k Base)",
        "expected": "Reject",
        "data": {
            "company": "Google",
            "jobFamily": "Software Engineer",
            "title": "Software Engineer",
            "level": "L3",
            "baseSalary": 850000,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 1200000,
            "yearsOfExperience": 0,
            "yearsAtCompany": 0,
            "yearsAtLevel": 0,
            "location": "Mountain View, CA"
        }
    },
    {
        "name": "Case 3: Arithmetic Contradiction (TC < Base)",
        "expected": "Reject",
        "data": {
            "company": "Microsoft",
            "jobFamily": "Software Engineer",
            "title": "Software Engineer",
            "level": "62",
            "baseSalary": 150000,
            "baseSalaryCurrency": "USD",
            "totalCompensation": 90000,
            "yearsOfExperience": 5,
            "yearsAtCompany": 2,
            "yearsAtLevel": 1,
            "location": "Redmond, WA"
        }
    }
]

print("=" * 65)
print("TESTING CATBOOST CROWDSOURCED VALIDATION PIPELINE")
print("=" * 65)

pipeline = ValidationPipeline()
all_passed = True

for tc in test_cases:
    print(f"\n--- Running {tc['name']} ---")
    res = pipeline.evaluate_submission(tc["data"], persist=False)
    print_single_report(res)
    actual = res["classification"]
    expected = tc["expected"]
    match = actual == expected
    status_str = "PASS" if match else "FAIL"
    print(f"Result: {status_str} (Expected: {expected}, Got: {actual})")
    if not match:
        all_passed = False

print("\n" + "=" * 65)
if all_passed:
    print("ALL TEST CASES PASSED SUCCESSFULLY!")
else:
    print("SOME TEST CASES FAILED!")
print("=" * 65)
