# Anomaly Detection in Crowdsourced Compensation Data

This project builds an anomaly detection pipeline to identify fake, invalid, or anomalous entries in crowdsourced salary data. Platforms like [Levels.fyi](https://www.levels.fyi/) rely on user-submitted compensation data to provide market insights — but crowdsourced data is inherently noisy. Malicious submissions, copy-paste errors, currency mix-ups, and coordinated manipulation rings can all degrade data quality. This pipeline is designed to catch those problems automatically, classifying each submission as valid or invalid before it enters any downstream analytics.

---

## Repository Structure

```
SELab/
├── data/                        # Dataset files (submissions.csv, labels sidecar, etc.)
├── models/                      # Saved model artifacts
├── src/                         # Core pipeline modules
├── web/                         # Dashboard web application
├── CatBoostv2.ipynb             # CatBoost exploration notebook
├── dbscan_umap.ipynb            # DBSCAN + UMAP exploration notebook
├── deep_autoencoder.ipynb       # Deep autoencoder exploration notebook
├── Submissions_LOF.py           # LOF baseline model script
├── train_catboost.py            # CatBoost training script (production)
├── infer.py                     # Single-submission inference script
├── test_infer.py                # Inference test harness
├── generate_dataset.py          # Synthetic dataset generator
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── DATASET_DOCUMENTATION.md     # Detailed dataset documentation
└── README.md                    # This file
```

---

## Dataset

The project uses a **synthetic dataset of 50,000 compensation submissions** (39,000 valid / 11,000 invalid, 78/22% split) generated to resemble real Levels.fyi-style entries. The data was modeled using real anchor points — ServiceNow compensation data from Levels.fyi, the Kaggle *Data Science Job Salaries* dataset, and the *India Job Market* dataset — and then extended with synthetic company tiers, job families, geographic pay factors, and user/network metadata.

Invalid rows are injected with **14 distinct defect types** across three modules:
- **Module A** — Statistical anomalies: extreme/moderate pay outliers, currency confusion, coordinated submission rings, near-duplicates, and scattershot bursts.
- **Module B** — Consistency violations: arithmetic mismatches, experience/date contradictions, geo-currency errors, company-level inconsistencies, and vesting/bonus structure problems.
- **Module C** — Data quality issues: missing required fields, format errors, and spam/placeholder content.

The dataset is deliberately challenging: valid rows include genuine outliers (2% exceptional pay, cross-border remote hires), and many invalid rows have only subtle defects (mild 1.4× inflation, 6–15% arithmetic drift, near-duplicate jitter).

---

## Methods Explored

Several approaches were explored to classify submissions as valid or invalid. The table below summarizes the performance of each model on the held-out test set.

| # | Model | Type | Accuracy | F1 (Anomaly) | Notes |
|---|-------|------|----------|--------------|-------|
| 1 | **CatBoost** | Supervised (gradient boosting) | **0.9611** | **0.9045** | Best overall; handles categorical features natively; ROC-AUC 0.9608 |
| 2 | **LOF (baseline)** | Unsupervised (local outlier factor) | 0.7707 | 0.0482 | Novelty detection; high precision on anomalies but extremely low recall |
| 3 | **DBSCAN + UMAP** | Unsupervised (density clustering) | 0.7125 | 0.1655 | UMAP for dimensionality reduction, DBSCAN for clustering, KNN for test inference |
| 4 | **Deep Autoencoder** | Semi-supervised (reconstruction error) | 0.6794 | 0.2481 | Neural network trained on valid data; threshold on reconstruction error |

### 1. CatBoost (Supervised Baseline)

A gradient-boosted decision tree model using [CatBoost](https://catboost.ai/), trained with native categorical feature support and a time-aware train/test split. This is the current best-performing model with strong precision and recall across both classes.

> See [`CatBoostv2.ipynb`](file:///CatBoostv2.ipynb) and [`train_catboost.py`](file:///train_catboost.py) for the training pipeline.

### 2. Local Outlier Factor (LOF) — Baseline

An unsupervised novelty detection model using scikit-learn's `LocalOutlierFactor`. Features are preprocessed with one-hot encoding and reduced via `TruncatedSVD` (50 components, ~90% variance). LOF is fitted in novelty mode on the training split, and anomaly scores are thresholded on the test set. While precision on flagged anomalies is high (~0.84), recall is extremely low (~0.025), meaning the model misses the vast majority of invalid submissions.

> See [`Submissions_LOF.py`](file:///Submissions_LOF.py).

### 3. DBSCAN + UMAP (Unsupervised Clustering)

UMAP is used to project the feature space into a lower-dimensional embedding, and DBSCAN identifies density-based clusters. Submissions falling into noise clusters or sparse regions are flagged as anomalous. A KNN classifier is trained on the DBSCAN-assigned labels to enable test-set inference. Performance is limited because density-based clustering struggles with the overlapping distributions of valid outliers and mildly invalid rows.

> See [`dbscan_umap.ipynb`](file:///dbscan_umap.ipynb).

### 4. Deep Autoencoder (Semi-Supervised)

A deep autoencoder is trained on valid-only data to learn normal submission patterns. At inference time, submissions with high reconstruction error are flagged as anomalous. The approach captures some structural anomalies but underperforms on subtle defects where the feature distribution closely resembles valid data.

> See [`deep_autoencoder.ipynb`](file:///deep_autoencoder.ipynb).

---

## Inference Pipeline

A standalone inference script ([`infer.py`](file:///infer.py)) allows evaluation of new, incoming single submissions. It routes the entry through the full feature-engineering and preprocessing pipeline, scores it against the trained CatBoost model, and returns a classification along with a confidence score. The pipeline also generates an isolated scatter plot visualizing exactly where the new entry lands relative to the historical dataset for its peer group (job family, level, and location).

The inference flow:
1. Parse and validate input fields.
2. Engineer features (compensation ratios, user history, temporal signals).
3. Score through the trained model.
4. Return verdict (`valid` / `invalid`) with probability and peer-group context.

A test harness ([`test_infer.py`](file:///test_infer.py)) provides sample entries covering common edge cases.

---


## Dashboard

The project includes a web-based dashboard ([`web/`](file:///web)) for interactive exploration of submissions and model results. The dashboard provides:

- **Submission browser** — View, search, and filter submissions with model predictions.
- **Anomaly visualizations** — Scatter plots and distribution charts for compensation data across peer groups.
- **Model metrics** — Summary statistics and performance metrics for the active model.

The dashboard is served via a Python backend ([`web/server.py`](file:///web/server.py)) with a static frontend.

---

## Current Work

- **CatBoost** is the current production model, delivering the best accuracy (96.1%) and anomaly F1 (0.90) among all tested approaches.
- The rule-based validation layer (in [`src/rule_engine.py`](file:///src/rule_engine.py)) runs independently and catches deterministic violations (negative values, arithmetic mismatches, impossible experience/tenure combinations).
- Feature engineering incorporates compensation ratios, user behavioral signals (account age, prior submission count), temporal patterns, and network metadata (IP/device hash clustering).

---

## Limitations

- The dataset is **synthetic** — compensation anchors outside ServiceNow SWE and data-science roles are assumptions, not real market data. Model performance on genuine Levels.fyi submissions may differ.
- Defect types and the 22% invalid rate are modelling choices; real-world fraud patterns will vary.
- Unsupervised methods (LOF, DBSCAN) struggle significantly with this dataset because many invalid rows have only subtle defects that overlap with the valid distribution.
- The CatBoost model is supervised and therefore requires labelled data; it cannot detect novel defect types absent from the training set.
- FX rates, city/DMA IDs (except Pune), and T3 company names are synthetic placeholders.

---

## Future Work

I am currently working on this project and actively experimenting with models. Future updates may introduce new algorithms and improved validation techniques.
