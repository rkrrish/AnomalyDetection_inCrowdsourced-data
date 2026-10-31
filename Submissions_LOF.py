"""Fixed preprocessing + Local Outlier Factor (LOF) evaluation for submissions.csv.

Usage:
    python Submissions_LOF_Fixed.py --data data/submissions.csv

Assumptions:
- is_valid is binary: 1 = valid submission, 0 = invalid submission.
- LOF is unsupervised. is_valid is used ONLY for post-hoc evaluation, not fitting.
- A chronological holdout is used whenever submittedAt is available. Rows with no
  submittedAt are assigned to training because their temporal position is unknown.
- LOF marks outliers as invalid (0) and inliers as valid (1); this is an evaluation
  mapping, not a guarantee that every invalid submission is an outlier.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
MISSINGNESS_THRESHOLD = 0.80
MAX_LOF_FIT_ROWS = 20_000  # Set to None to fit LOF on the complete training split.
LOF_NEIGHBORS = 20
SVD_MAX_COMPONENTS = 50

CURRENCY_COLUMNS = [
    "baseSalaryCurrency", "stockGrantCurrency", "bonusCurrency", "userCurrency"
]
VALID_CURRENCY_CODES = {
    "INR", "USD", "EUR", "GBP", "CAD", "SGD", "AUD", "PLN", "SEK", "CHF",
    "BRL", "AED", "JPY", "CNY", "NZD", "HKD", "KRW", "MXN", "ZAR", "NOK",
    "DKK", "CZK", "HUF", "ILS", "THB", "IDR", "MYR", "PHP", "TWD", "TRY"
}
CURRENCY_ALIASES = {
    "INRR": "INR", "RS": "INR", "RS.": "INR", "RUPEE": "INR",
    "RUPEES": "INR", "US$": "USD", "US DOLLAR": "USD", "EURO": "EUR"
}
AMBIGUOUS_CURRENCY_ALIASES = {"RS", "RS."}
PLACEHOLDER_CURRENCY_CODES = {"XXX", "N/A", "NA", "NONE", "UNKNOWN"}

WORK_ARRANGEMENT_ALIASES = {
    "hybrid": "hybrid",
    "onsite": "onsite", "on-site": "onsite", "on site": "onsite",
    "office": "onsite", "in-office": "onsite", "in office": "onsite",
    "remote": "remote", "remoteee": "remote", "wfh": "remote",
    "work from home": "remote", "work-from-home": "remote", "fully remote": "remote",
}
EXPECTED_WORK_ARRANGEMENTS = {"hybrid", "onsite", "remote"}

# The notebook's stated policy is to exclude these free-text/structured-text fields.
MANUAL_DROP_COLUMNS = [
    "uuid", "userId", "ipHash", "deviceHash", "threadId",
    "submittedAt", "offerDate",
    "title", "otherDetails", "comments", "tags", "additionalBonuses",
    "vestingSchedule", "companyInfo_icon", "companyInfo_name", "companyInfo_slug",
    "jobFamilySlug", "locationSlug",
]

NUMERIC_CANDIDATES = [
    "baseSalary", "yearsOfExperience", "yearsAtCompany", "yearsAtLevel",
    "exchangeRate", "totalCompensation", "firstYearTotalCompensation",
    "avgAnnualStockGrantValue", "firstYearStockGrantValue", "totalStockGrantValue",
    "negotiatedAmount", "avgAnnualBonusValue", "firstYearBonusValue",
    "avgAnnualOptionsGranted", "totalOptionsGranted", "optionStrikePrice",
    "optionPreferredPrice", "totalOutstandingShares", "latestCompanyValuation",
    "annualTargetBonusValue", "annualTargetBonusPercentage", "userAccountAgeDays",
    "userPriorSubmissions", "commentsAndRepliesCount", "cityId", "dmaId", "countryId",
]
IDENTIFIER_CATEGORICAL_COLUMNS = ["cityId", "dmaId", "countryId"]


def clean_token(value):
    """Return a trimmed, whitespace-normalized string or pd.NA."""
    if pd.isna(value):
        return pd.NA
    token = " ".join(str(value).strip().split())
    return token if token else pd.NA


def normalize_work_arrangement(value):
    token = clean_token(value)
    if pd.isna(token):
        return pd.NA
    token = str(token).lower()
    return WORK_ARRANGEMENT_ALIASES.get(token, token)


def normalize_currency(value):
    """Normalize known aliases, preserve unfamiliar tokens, and return audit status."""
    token = clean_token(value)
    if pd.isna(token):
        return pd.NA, "missing"

    raw = str(token).upper()
    if raw in PLACEHOLDER_CURRENCY_CODES:
        return pd.NA, "placeholder_code_review"
    if raw in AMBIGUOUS_CURRENCY_ALIASES:
        return CURRENCY_ALIASES[raw], "ambiguous_alias_review"
    if raw in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[raw], "alias_normalized"
    if raw in VALID_CURRENCY_CODES:
        return raw, "standard"

    # Keep unexpected values visible as categorical features rather than silently
    # converting every unknown code to missing. Review statuses identify them.
    return raw, "unrecognized_review"


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse mixed ISO timestamps and browser-style date strings consistently."""
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:  # Compatibility with older pandas versions.
        return pd.to_datetime(series, errors="coerce", utc=True)


def add_strict_prior_count(frame: pd.DataFrame, source_col: str, time_col: str,
                           output_col: str) -> None:
    """Count records for a key with strictly earlier timestamps.

    Records sharing an identical timestamp do not count one another as prior. Rows
    with a missing key or timestamp receive 0 because their temporal order is unknown.
    """
    out = pd.Series(0, index=frame.index, dtype="int64")
    valid = frame[source_col].notna() & frame[time_col].notna()
    if not valid.any():
        frame[output_col] = out
        return

    event_counts = (
        frame.loc[valid, [source_col, time_col]]
        .groupby([source_col, time_col], dropna=True, sort=True)
        .size()
        .rename("event_count")
        .reset_index()
        .sort_values([source_col, time_col], kind="mergesort")
    )
    event_counts["prior_count"] = (
        event_counts.groupby(source_col, sort=False)["event_count"].cumsum()
        - event_counts["event_count"]
    )
    prior_map = event_counts.set_index([source_col, time_col])["prior_count"]
    row_keys = pd.MultiIndex.from_frame(frame.loc[valid, [source_col, time_col]])
    out.loc[valid] = prior_map.reindex(row_keys).to_numpy(dtype=np.int64)
    frame[output_col] = out


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply row-wise normalization and historical feature creation to all rows."""
    data = frame.copy()

    # Parse dates once, before chronological splitting and historical feature creation.
    if "submittedAt" in data.columns:
        data["submittedAt"] = parse_datetime(data["submittedAt"])
    if "offerDate" in data.columns:
        data["offerDate"] = parse_datetime(data["offerDate"])

    # Normalize work arrangement without erasing novel categories.
    if "workArrangement" in data.columns:
        normalized = data["workArrangement"].map(normalize_work_arrangement)
        data["workArrangement_unexpected"] = (
            normalized.notna() & ~normalized.isin(EXPECTED_WORK_ARRANGEMENTS)
        ).astype("int8")
        data["workArrangement"] = normalized

    # Normalize every currency feature consistently, not just baseSalaryCurrency.
    for col in CURRENCY_COLUMNS:
        if col in data.columns:
            normalized_pairs = data[col].map(normalize_currency)
            data[col] = normalized_pairs.map(lambda pair: pair[0])
            data[f"{col}_status"] = normalized_pairs.map(lambda pair: pair[1])
            data[f"{col}_review_required"] = data[f"{col}_status"].isin(
                {"ambiguous_alias_review", "placeholder_code_review", "unrecognized_review"}
            ).astype("int8")

    # Historical counts are created across the full timestamp-ordered source, before
    # splitting, and count only strictly earlier events. They are not split-local ranks.
    if "submittedAt" in data.columns:
        for source_col, feature_col in [
            ("userId", "user_prior_submission_count"),
            ("ipHash", "ip_prior_submission_count"),
            ("deviceHash", "device_prior_submission_count"),
        ]:
            if source_col in data.columns:
                add_strict_prior_count(data, source_col, "submittedAt", feature_col)

        data["submit_year"] = data["submittedAt"].dt.year
        data["submit_month"] = data["submittedAt"].dt.month
        data["submit_dayofweek"] = data["submittedAt"].dt.dayofweek
        data["submit_hour"] = data["submittedAt"].dt.hour

    if "submittedAt" in data.columns and "offerDate" in data.columns:
        data["days_since_offer"] = (data["submittedAt"] - data["offerDate"]).dt.total_seconds() / 86400.0

    for col in NUMERIC_CANDIDATES:
        if col in data.columns:
            cleaned = data[col].astype("string").str.replace(",", "", regex=False).str.strip()
            data[col] = pd.to_numeric(cleaned, errors="coerce")

    # Numeric-looking IDs are nominal categories, not ordered quantities.
    for col in IDENTIFIER_CATEGORICAL_COLUMNS:
        if col in data.columns:
            data[col] = data[col].map(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() else pd.NA
            )

    return data


def chronological_split(data: pd.DataFrame, target_col: str, test_size: float = TEST_SIZE):
    """Chronological holdout; undated rows go to train because their order is unknown."""
    if "submittedAt" not in data.columns:
        raise ValueError("A 'submittedAt' column is required for the chronological split.")

    dated = data.loc[data["submittedAt"].notna()].sort_values("submittedAt", kind="mergesort")
    undated = data.loc[data["submittedAt"].isna()]

    # Fall back only if there are too few dated rows to form a sensible holdout.
    if len(dated) < 5:
        from sklearn.model_selection import train_test_split

        stratify = data[target_col] if data[target_col].nunique() > 1 else None
        tr, te = train_test_split(
            data, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
        )
        warnings.warn(
            "Fewer than five timestamped rows; using a random split as a fallback.",
            RuntimeWarning,
        )
        return tr.copy(), te.copy()

    cut = int(np.floor(len(dated) * (1 - test_size)))
    cut = min(max(cut, 1), len(dated) - 1)
    train = pd.concat([dated.iloc[:cut], undated], axis=0).sort_index(kind="stable")
    test = dated.iloc[cut:].copy()
    if len(test) == 0:
        raise ValueError("The chronological test split is empty; check dataset size.")
    return train, test


def print_metrics(name: str, y_true, y_pred) -> None:
    print(f"\n{name} evaluation")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification report (post-hoc; labels were not used to fit LOF):")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["invalid (0)", "valid (1)"],
        zero_division=0, digits=4
    ))
    print("Confusion matrix (rows=true [0, 1], columns=predicted [0, 1]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


def main(data_path: str) -> None:
    df = pd.read_csv(data_path, low_memory=False)
    if "is_valid" not in df.columns:
        raise KeyError("Target column 'is_valid' is missing from the CSV.")
    if "submittedAt" not in df.columns:
        raise KeyError("Column 'submittedAt' is required for the chronological split.")

    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    target = pd.to_numeric(df["is_valid"], errors="coerce")
    labelled = target.isin([0, 1])
    n_unlabelled = int((~labelled).sum())
    if n_unlabelled:
        print(f"Excluding {n_unlabelled:,} rows with missing/non-binary is_valid labels from model evaluation.")
    if target.loc[labelled].nunique() < 2:
        raise ValueError("is_valid must contain both 0 and 1 to calculate classification metrics.")

    # Engineer history over all source rows first, including records whose target is unknown.
    # No label is used by this feature engineering; is_valid is retained only for later evaluation.
    df["is_valid"] = target
    df = engineer_features(df)
    df = df.loc[labelled].copy()
    df["is_valid"] = target.loc[labelled].astype("int8")
    train_df, test_df = chronological_split(df, "is_valid", TEST_SIZE)

    y_train = train_df["is_valid"].astype("int8").copy()
    y_test = test_df["is_valid"].astype("int8").copy()
    X_train = train_df.drop(columns=["is_valid"], errors="ignore").copy()
    X_test = test_df.drop(columns=["is_valid"], errors="ignore").copy()

    drop_cols = [col for col in MANUAL_DROP_COLUMNS if col in X_train.columns]
    X_train.drop(columns=drop_cols, inplace=True)
    X_test.drop(columns=[col for col in drop_cols if col in X_test.columns], inplace=True)

    # The split boundary and all data-dependent column filtering are fitted using train only.
    sparse_cols = X_train.columns[X_train.isna().mean() > MISSINGNESS_THRESHOLD].tolist()
    X_train.drop(columns=sparse_cols, inplace=True)
    X_test.drop(columns=[col for col in sparse_cols if col in X_test.columns], inplace=True)

    constant_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1]
    X_train.drop(columns=constant_cols, inplace=True)
    X_test.drop(columns=[col for col in constant_cols if col in X_test.columns], inplace=True)

    # Guarantee identical columns and order even when a feature is absent in the test split.
    X_test = X_test.reindex(columns=X_train.columns)
    if X_train.shape[1] == 0:
        raise ValueError("No features remain after dropping identifiers/sparse/constant columns.")

    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

# Fix mixed data types in categorical columns before OneHotEncoder.
# Convert all non-missing categorical values to strings and preserve
# missing values as np.nan for SimpleImputer.

    def category_to_string(value):
        if value is None or value is pd.NA:
            return np.nan

        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return np.nan
        except (TypeError, ValueError):
            pass

        return str(value)


    for col in categorical_cols:
        X_train[col] = X_train[col].map(category_to_string).astype(object)
        X_test[col] = X_test[col].map(category_to_string).astype(object)

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    transformers = []
    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=10,
                    max_categories=100, sparse_output=True,
                )),
            ]),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.3
    )
    Z_train_full = preprocessor.fit_transform(X_train)
    Z_test = preprocessor.transform(X_test)

    rng = np.random.default_rng(RANDOM_STATE)
    if MAX_LOF_FIT_ROWS is not None and len(X_train) > MAX_LOF_FIT_ROWS:
        fit_idx = np.sort(rng.choice(len(X_train), size=MAX_LOF_FIT_ROWS, replace=False))
        print(
            f"LOF reference sample: {len(fit_idx):,} of {len(X_train):,} training rows. "
            "Training metrics below refer to this fitted subset."
        )
    else:
        fit_idx = np.arange(len(X_train))
        print(f"LOF reference sample: all {len(fit_idx):,} training rows.")

    Z_fit_raw = Z_train_full[fit_idx]
    y_fit = y_train.iloc[fit_idx]

    # Reduce wide one-hot feature spaces before distance-based modeling.
    svd = None
    n_components = min(
        SVD_MAX_COMPONENTS,
        max(0, Z_fit_raw.shape[1] - 1),
        max(0, Z_fit_raw.shape[0] - 1),
    )
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        Z_fit = svd.fit_transform(Z_fit_raw)
        Z_train_eval = svd.transform(Z_train_full[fit_idx])
        Z_test_eval = svd.transform(Z_test)
        print(f"TruncatedSVD reduced {Z_fit_raw.shape[1]:,} encoded features to {n_components} components.")
    else:
        Z_fit = Z_fit_raw
        Z_train_eval = Z_fit_raw
        Z_test_eval = Z_test
        print("Skipping TruncatedSVD because fewer than three rows/features are available.")

    if len(fit_idx) < 3:
        raise ValueError("At least three training rows are required to fit LOF.")
    n_neighbors = min(LOF_NEIGHBORS, len(fit_idx) - 1)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination="auto",
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(Z_fit)

    # For the fitted reference rows, use LOF's training-set outlier factors/offset.
    # This is the correct fit-time thresholding path for novelty=True.
    fit_lof_labels = np.where(lof.negative_outlier_factor_ < lof.offset_, -1, 1)
    y_pred_train = np.where(fit_lof_labels == -1, 0, 1).astype("int8")

    # Test predictions use the learned novelty detector. Labels do not affect fitting.
    test_lof_labels = lof.predict(Z_test_eval)
    y_pred_test = np.where(test_lof_labels == -1, 0, 1).astype("int8")

    print("\nLOF setup")
    print(f"Train split rows: {len(X_train):,} | Test split rows: {len(X_test):,}")
    print(f"Features after train-only filtering: {X_train.shape[1]:,}")
    print(f"Numeric features: {len(numeric_cols)} | Categorical features: {len(categorical_cols)}")
    print(f"n_neighbors={n_neighbors}, contamination='auto', novelty=True")
    print(f"Train label distribution (full train split): {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test label distribution: {y_test.value_counts().sort_index().to_dict()}")
    if svd is not None:
        print(f"SVD cumulative explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    print_metrics("TRAIN FIT SUBSET", y_fit, y_pred_train)
    print_metrics("TEST (CHRONOLOGICAL HOLDOUT)", y_test, y_pred_test)

    print("\nPreprocessing audit")
    print(f"Dropped >{MISSINGNESS_THRESHOLD:.0%} missing in train: {sparse_cols}")
    print(f"Dropped train-constant columns: {constant_cols}")
    print("Historical counts use only records with strictly earlier submittedAt timestamps;")
    print("equal timestamps do not count one another, and undated rows receive zero history counts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/submissions.csv",
        help="Path to submissions.csv (default: data/submissions.csv)",
    )
    args = parser.parse_args()
    main(args.data)
