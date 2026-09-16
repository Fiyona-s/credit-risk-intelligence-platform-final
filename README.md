# Credit Risk Intelligence Platform

An end-to-end, explainable and auditable credit-risk decision-support platform: batch data engineering on cloud object storage, imbalance-aware gradient boosting, probability calibration, cost-sensitive decision optimization, applicant-level SHAP explanations, fairness auditing, natural-language portfolio analytics, and model/data monitoring — built on the Kaggle Home Credit Default Risk dataset.

> **This is a technical credit-risk decision-support platform built for a learning/portfolio project. It is not an actual lender approval system, and none of its risk bands, thresholds, or decisions represent real banking or regulatory policy.**

## Why this exists

A model that only outputs a probability of default from a black box creates two problems: credit analysts can't defend a decision to a regulator or a customer, and the risk team can't audit whether the model is tracking sensible signals or spurious correlations, whether its probabilities are actually reliable, or whether it treats applicant groups equitably. This platform pairs a calibrated LightGBM default-risk model with **per-applicant SHAP explanations**, **cost-sensitive decision optimization**, a **fairness audit**, **drift monitoring**, and a **talk-to-data chatbot** — turning a raw prediction into something a credit policy team could actually inspect, question, and defend, backed by a small but genuine batch data-engineering layer (S3 → validation → curated Parquet → PostgreSQL) rather than a notebook wrapped in a UI.

## Key capabilities

**Data Engineering**
Batch ingestion · schema/data-quality validation (reused across the whole platform, not duplicated) · ETL onto the existing feature-engineering pipeline · S3 raw/curated/metadata zones · curated Parquet · PostgreSQL analytical table · deterministic Python orchestration · run-ID lineage · idempotent full-refresh loads · bounded retry on transient cloud/DB failures

**Machine Learning**
Logistic Regression / XGBoost / LightGBM comparison (LightGBM deployed) · bureau/previous-application/POS/credit-card/installment feature engineering with a feature registry · automated leakage/data validation · class-imbalance handling · monotonic constraints · sigmoid probability calibration · calibrated risk scoring and project-defined risk bands · cost-sensitive decision optimization

**Explainability / Responsible AI**
Applicant-level TreeSHAP explanations (raw-model-margin scale, with business-readable feature-registry meanings) · fairness audit (demographic parity / equal opportunity) · Evidently-based drift monitoring · an auto-generated model card PDF

**Application**
FastAPI REST endpoint · Streamlit multi-tab UI · natural-language-to-SQL chatbot on DuckDB · Docker / Docker Compose · pytest test suite (145 tests as of the last full run — see § Testing)

## Architecture

```mermaid
flowchart TD
    S3[AWS S3<br/>raw / curated / metadata zones<br/>optional — DATA_SOURCE=s3]
    RAW[data/*.csv<br/>Home Credit application + 6 auxiliary tables]
    INGEST[Batch Ingestion<br/>src/pipeline/ingest.py]
    VALIDATE[Data Validation<br/>src/data/validation.py — reused everywhere]
    ETL[ETL / Feature Engineering<br/>src/data/loader.py + src/data/preprocessor.py<br/>SAME functions used by training, inference, and the pipeline]
    CURATED[Curated Parquet<br/>curated_data/applicant_features/run_id=.../part-000.parquet]
    PG[(PostgreSQL<br/>applicant_features table — optional, DATABASE_URL)]
    LGBM[LightGBM<br/>scale_pos_weight + monotonic constraints]
    RAWP[Raw Probability]
    SHAP[SHAP Explanation<br/>raw model margin]
    CAL[Sigmoid Calibration]
    CPD[Calibrated PD]
    RISK[Risk Scoring<br/>Low / Moderate / High / Very High]
    DECOPT[Cost-Sensitive Decision Optimization]
    DECISION[Approve / Decline]
    APIUI[FastAPI + Streamlit]

    S3 -.optional.-> RAW
    RAW --> INGEST --> VALIDATE --> ETL --> CURATED --> PG
    ETL --> LGBM
    LGBM --> RAWP
    RAWP --> SHAP
    RAWP --> CAL --> CPD
    CPD --> RISK
    CPD --> DECOPT --> DECISION
    RISK --> APIUI
    DECISION --> APIUI
    SHAP --> APIUI
    PG -.analytics only, not read by ML pipeline.-> APIUI
```

**Two independent decision surfaces, deliberately not merged**: `calibrated PD → risk score/band` describes the model's assessed risk level; `calibrated PD → cost-sensitive threshold → Approve/Decline` is a separate, configurable business-policy layer. A "Very High" risk band does not automatically mean "Decline" — the mapping from PD to a decision is entirely parameterized by `FP_COST`/`FN_COST`, never hardcoded. See § Risk scoring and § Cost-sensitive decision optimization below.

**Data pipeline vs. ML pipeline**: the data pipeline (`src/pipeline/pipeline.py`) does not sit "in front of" model training/inference — both are independent consumers of the same `src.data.loader.build_joined_dataset()` / `src.data.preprocessor.engineer_features()` functions. Training and prediction always re-derive features fresh from the raw CSVs; they never read the curated Parquet/Postgres output. This means there is no version-skew risk between "what the pipeline curated for analytics" and "what the model was actually trained/predicted on" — the curated table is a lineage/analytics artifact, not a dependency of the model.

## Project structure

```
src/
├── data/            loader, preprocessor (feature engineering), feature_registry,
│                    validation (leakage/quality checks), cloud_storage (S3)
├── ml/              train, evaluate, predict, calibration, risk_scoring,
│                    decision_optimization, model_comparison
├── explainability/  shap_explainer (applicant-level SHAP)
├── fairness/        audit (demographic parity / equal opportunity)
├── monitoring/       drift_report (Evidently data/target drift)
├── rules/           rule_derivation (surrogate decision-tree rules)
├── pipeline/        ingest, curate, postgres_loader, metadata, pipeline (orchestrator)
├── talk_to_data/    nl_to_sql, query_runner, prompt_templates (chatbot)
└── utils/           config, helpers, logger

tests/               pytest suite — feature engineering, validation, model
                     comparison, calibration, risk scoring, decision
                     optimization, SHAP, data pipeline
reports/             generated artifacts (model comparison, calibration,
                     risk scoring, decision optimization, SHAP, data
                     engineering, data validation reports)
models/              credit_risk_model.joblib (committed), plus run-generated
                     optimal_threshold.json / probability_calibrator.joblib /
                     decision_threshold.json / calibration_metadata.json
scripts/             generate_synthetic_data, load_dataset_to_postgres,
                     generate_model_card

app.py               Streamlit UI (10 tabs — see § Streamlit below)
api.py               FastAPI REST endpoint
docker-compose.yml   single-container app (Streamlit + FastAPI via supervisord)
requirements.txt
.env.example         placeholder configuration — see § Configuration
```

## Setup & run

The minimum path to a fully working app needs **no Kaggle dataset, no AWS account, and no local Python install**:

1. `cp .env.example .env` and fill in `GROQ_API_KEY` (recommended, free tier at [console.groq.com](https://console.groq.com)) or `GEMINI_API_KEY` as a fallback.
2. ```
   docker-compose up --build
   ```
3. Open http://localhost:8501.

**EDA, Risk Prediction, Model Evaluation, Explainability, Business Rules, Fairness Audit, and Monitoring** all work immediately off the pretrained model artifact already committed at `models/credit_risk_model.joblib` — no training step needed. **Chatbot** works immediately too: if `data/application_train.csv` isn't present, `src/talk_to_data/query_runner.py` falls back to a small synthetic demo dataset (`data/sample_applicants.csv`, committed) with no on-screen indication of the substitution (removed by explicit request during development — see git history if this matters for your use of the platform). No paid services or credit card required anywhere in the stack. Cloud (S3/PostgreSQL) configuration is entirely optional and unrelated to the app's own startup — see § Data engineering below.

### Using the real dataset / retraining (optional)

1. Download the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/data) from Kaggle and place the CSVs in `data/` (`application_train.csv`, `application_test.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`).
2. Retrain **inside the container** — no local Python/pandas/LightGBM install needed:
   ```
   docker-compose run --rm --entrypoint python app -m src.ml.train
   ```
   This overwrites `models/credit_risk_model.joblib` using the exact scikit-learn/LightGBM versions pinned in `requirements.txt`, and also runs the Milestone 2 data-validation suite and the Milestone 1 cost-sensitive threshold sweep as part of the same run.
3. Regenerate the calibrator (Milestone 4) and decision-optimization artifact (Milestone 6) if you retrained:
   ```
   docker-compose run --rm --entrypoint python app -m src.ml.calibration
   docker-compose run --rm --entrypoint python app -m src.ml.decision_optimization
   ```
4. Restart the app (`docker-compose restart`) — the Chatbot tab automatically switches from the synthetic fallback to the real dataset once `application_train.csv` is present, no code changes needed.

### Serving the chatbot from Postgres (optional, deployment only)

On a deployed instance (Render, Streamlit Cloud, etc.) the real CSV is never present, so a deployed chatbot falls back to synthetic data by default. As a fully optional third path, `query_runner.py` can have DuckDB **attach a Postgres database live**:

1. Provision a free Postgres instance and note its connection string.
2. From a machine with the real Kaggle CSVs in `data/`, run once:
   ```
   python scripts/load_dataset_to_postgres.py --connection-string "postgresql://user:pass@host:5432/dbname"
   ```
3. Set `POSTGRES_URL` as a secret on your deployment platform and redeploy.

This `POSTGRES_URL`/`applicants` table is **only** used by the chatbot — it is deliberately independent of the Milestone 8 data pipeline's own `DATABASE_URL`/`applicant_features` table (see § Data engineering). The two Postgres paths can point at the same physical database if convenient, but neither feature depends on the other. Any connection failure falls through cleanly to the synthetic fallback rather than crashing.

## Data engineering (batch pipeline + cloud storage)

```
AWS S3 (optional)          data/*.csv
  raw/ curated/ metadata/       │
        │                  Batch Ingestion (src/pipeline/ingest.py)
        │                       │
        └───────────────► Data Validation (src/data/validation.py — REUSED, not a second framework)
                                 │
                        ETL / Feature Engineering (the SAME src.data.loader /
                                 │                  src.data.preprocessor functions
                                 │                  training and inference use)
                        Curated Parquet (run_id-partitioned)
                                 │
                        PostgreSQL applicant_features table (optional, DATABASE_URL)
```

Run it: `python -m src.pipeline.pipeline --source local` (or `--source s3` with S3 configured; `--dry-run` verifies S3 configuration without touching any data; `--skip-postgres` skips the LOAD stage).

**S3 zones**:
- `raw/` — original CSVs staged verbatim per run, immutable.
- `validated/` — not a data copy; the pass/fail validation outcome is recorded in the run manifest instead of duplicating the raw CSVs a second time just to represent a boolean.
- `curated/` — the transformed applicant-level Parquet dataset, partitioned as `curated/applicant_features/run_id=<RUN_ID>/part-000.parquet`. Partitioned by run_id (not by any applicant attribute) — the dataset is small enough (a single ~70MB file) that there's no query-pruning benefit to further partitioning, but per-run isolation means a new run can never silently corrupt a previous run's output.
- `features/` — reserved, intentionally unpopulated at this project's scale (`curated/` already is the applicant-level feature table; a separate export would only be justified by a genuine second, diverging consumer).
- `metadata/` — one JSON run-manifest per pipeline run.

**Why S3**: durable, versioned, immutable object storage for raw/curated snapshots, independent of the pipeline compute layer. **Why Parquet**: columnar, typed, compressed — the curated dataset is 71.7MB as Parquet vs. several times that as untyped CSV, and it round-trips exactly with the pandas/pyarrow stack the ML pipeline already depends on. **Why PostgreSQL as well**: different consumer — Parquet is for pipeline-to-pipeline handoff and archival, a relational table is what an analyst or BI tool actually wants to point SQL at. **Why both, not one replacing the other**: object storage without a queryable table means every consumer needs a pandas/pyarrow runtime; a database without object storage means no cheap, durable, versioned raw/curated archive independent of the database's own backup story.

**Why Python orchestration, not Airflow**: this is a single deterministic batch job with no cross-job dependencies, backfills, or SLAs — `cron + python -m src.pipeline.pipeline` is the entire "orchestration platform" this workload needs; Airflow's scheduler/metadata-DB/web-UI would be overhead with no corresponding benefit at this scale.

**Idempotency**: every output path is a deterministic function of `run_id` (curated Parquet, metadata JSON) — rerunning the same `run_id` overwrites, never duplicates. PostgreSQL uses `CREATE OR REPLACE TABLE` (full-refresh), independent of `run_id` — two loads never combine rows. This is a full-refresh design, not incremental upsert, because the source data is a single static Kaggle snapshot with no "what changed since last run" signal to track.

**Retry**: S3 and Postgres I/O get a bounded retry (3 attempts, linear backoff) for transient failures. A missing `S3_BUCKET`, a genuine S3 404, or a data-validation `FAIL` are **not** retried — they're deterministic outcomes that retrying would only delay, not fix.

**Real, measured pipeline results** (local mode, actual Home Credit dataset):
```
Input files: 8 (all expected tables present)
Input rows:  58,489,893 (sum across all 8 raw files)
Output (curated) rows: 307,511  ==  unique SK_ID_CURR in curated output: 307,511
Engineered columns: 202 (203 on read-back — see note below)
Validation status: WARNING (12 checks passed, 2 warnings, 0 failures)
Curated Parquet size: 71.69 MB
Runtime: ~78 seconds end-to-end
```
Note: reading the curated Parquet back via `pyarrow.parquet.read_table()` auto-injects a `run_id` column from the Hive-style partition path (`run_id=<RUN_ID>/...`) — a free lineage signal, not a schema mismatch.

**Validation scope**: the pipeline's local mode above was run end-to-end against the real 307,511-row dataset. The S3 and PostgreSQL integration paths (`src/data/cloud_storage.py`, `src/pipeline/postgres_loader.py`) are exercised by unit tests against mocked boto3/DuckDB clients (`tests/test_data_pipeline.py`) and by `--dry-run --source s3`'s configuration-error handling; they haven't been run against a live AWS account or Postgres instance in this environment. The code paths are the same either way — pointing `DATA_SOURCE=s3` and `DATABASE_URL` at real infrastructure exercises them for real with no code change required.

## Exploratory data analysis

`notebooks/eda.ipynb` (exported to `notebooks/eda.py`) covers dataset summary, feature categorization, missing-value analysis, `TARGET` class balance, and six business insights with charts:

1. Lower income brackets → higher default rate
2. Shorter employment tenure → higher default rate
3. Lower education level → higher default rate
4. Credit amount alone is non-monotonic; ratio to income matters more
5. Younger applicants default more often than older applicants
6. **Applicants with no bureau history default more often (10.12%) than those with bureau history (7.73%)** — a 2.39-percentage-point gap computed directly from `application_train.csv` joined against `bureau.csv`. This is the empirical basis for the bureau/previous-application feature engineering below.

## Feature engineering & data validation

`src/data/preprocessor.py` builds credit-behavior features from the six auxiliary Home Credit tables (`build_bureau_features`, `build_previous_application_features`, `build_pos_cash_features`, `build_credit_card_features`, `build_installment_features`), joined via `build_aggregate_features()`, plus ratio features (`engineer_features()`) — the single shared code path used by training, inference, SHAP, rule derivation, and the data pipeline, so features can never drift between contexts.

**80 registered engineered features** (`src/data/feature_registry.py`) — each with a recorded source table, business meaning, actionable/leakage classification. Raw application-table passthrough columns (e.g. `AMT_INCOME_TOTAL`, `EXT_SOURCE_1/2/3`) are intentionally not registered — the registry covers only *derived* features. The joined dataset has **194 columns** at this stage (up from 166 before the Milestone 1 feature-engineering upgrade); after one-hot encoding, the model sees **319 encoded features**.

**Automated data validation** (`src/data/validation.py`) checks TARGET leakage, applicant-ID uniqueness, row-count preservation across joins, infinite/constant/high-cardinality features, feature-registry consistency, non-negative business fields, and a measured (not assumed) form of train/validation statistical contamination — reused verbatim by both `src.ml.train` and the Milestone 8 data pipeline, not reimplemented. **Scope note, stated honestly**: this catches known, mechanically-detectable failure modes; it does not and cannot guarantee the complete absence of semantic data leakage, which requires domain review no automated check can fully replace.

## Model development & selection

Three models were compared under **identical** data, split, preprocessing, target, and evaluation (`src/ml/model_comparison.py`), not a "LightGBM was obviously right" assumption:

| Model | ROC-AUC | PR-AUC | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7711 | 0.2557 | 0.1747 | 0.6983 | 0.2795 |
| XGBoost | 0.7823 | 0.2800 | 0.1940 | 0.6693 | 0.3008 |
| **LightGBM (deployed)** | **0.7854** | **0.2851** | 0.1881 | 0.6971 | 0.2963 |

**5-fold cross-validation** (stability check): LightGBM ROC-AUC 0.7827 ± 0.0032, PR-AUC 0.2746 ± 0.0043.

LightGBM leads on both PR-AUC and ROC-AUC in both the single-split and CV evaluation — but the margin over XGBoost is modest (CV PR-AUC gap of 0.0028, smaller than either model's own CV standard deviation) and should not be oversold as a decisive win. XGBoost would have been a defensible alternative choice. LightGBM was selected primarily because it never underperforms on the metrics that matter most, and its monotonic constraints (`AMT_INCOME_TOTAL`, `EXT_SOURCE_1/2/3` — verified to cost effectively zero measured performance) and existing SHAP TreeExplainer integration give it an interpretability/fair-lending edge XGBoost would need equivalent work to match.

**Imbalance handling**: `TARGET` is ~92%/8%. LightGBM's `scale_pos_weight = negative_count / positive_count` is the primary strategy (SMOTE available via `--smote` for comparison, but not used by default — at this scale and dimensionality it mainly adds training cost without improving PR-AUC/ROC-AUC over a correctly weighted loss).

**Monotonic constraints**: `AMT_INCOME_TOTAL` and `EXT_SOURCE_1/2/3` get `monotone_constraints=-1` — higher income or a higher external credit-bureau score can never *increase* predicted risk, regardless of local noise a tree might otherwise learn. Deliberately not applied to every feature (e.g. `CNT_CHILDREN`, `DAYS_EMPLOYED` have plausible non-monotonic real-world relationships with risk).

## Probability calibration

**Raw LightGBM probabilities are severely overconfident** — a direct, measured consequence of `scale_pos_weight≈11.39` reshaping the probability distribution for ranking at the cost of reliability. Verified empirically: at every decile, raw mean predicted probability sat well above the observed default frequency (e.g. top decile: 0.78 predicted vs. 0.30 observed).

**Methodology (leakage-free)**: 5-fold out-of-fold raw probabilities on the *training* population (never the held-out validation split) — each fold's OOF prediction comes from a model that never saw that row — then sigmoid and isotonic calibrators are fit on those OOF predictions. Evaluation happens entirely on the validation split, untouched by calibration fitting.

| Model Output | Brier Score | Log Loss | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Raw LightGBM | 0.17679 | 0.52595 | 0.7854 | 0.2851 |
| **Sigmoid (selected)** | 0.06581 | 0.23732 | 0.7854 | 0.2851 |
| Isotonic | 0.06577 | 0.23711 | 0.7852 | 0.2781 |

Sigmoid and isotonic are a **statistical tie** on Brier score (a 0.00004 difference — noise, not signal). The tie was broken on ranking preservation: isotonic's step-function measurably degraded PR-AUC (−0.0070, a real, non-noise drop caused by ties among distinct raw probabilities collapsing onto the same isotonic output), while sigmoid — a strictly monotonic transform — preserved PR-AUC/ROC-AUC exactly. Sigmoid was **not** picked by default "because it's simpler"; isotonic was the code's first automatic choice by raw Brier score, caught and corrected by inspecting the ranking-metric side effect.

`src/ml/predict.py::predict_applicant()` exposes both `raw_probability` and `calibrated_probability` explicitly (never silently substituting one for the other), plus `calibration_method` reporting which calibrator (or `"none"` if unavailable) produced the calibrated value.

## Risk scoring

Calibrated PD (not raw probability) feeds risk scoring:
```
risk_score = round(clip(calibrated_pd, 0, 1) × 1000)   — bounded 0-1000, deliberately not
                                                           a 300-850 scale, to avoid resembling
                                                           FICO/VantageScore
```

**Risk bands** — a hybrid of fixed thresholds informed by real decile analysis of calibrated PD on the validation population (overall observed default rate 8.07%):

| Band | Calibrated PD range | Share of population | Observed default rate |
|---|---|---:|---:|
| Low | < 0.05 | 49.2% | 2.4% |
| Moderate | 0.05 – 0.10 | 23.6% | 6.3% |
| High | 0.10 – 0.20 | 17.7% | 14.1% |
| Very High | ≥ 0.20 | 9.6% | 30.6% |

Monotonicity confirmed directly (mean PD and observed default rate both strictly increase across bands). **These are project-defined categories for this project, evaluated on one validation snapshot — not official banking, FICO, or credit-bureau risk grades.**

**A large, honestly-reported shift from the old (pre-calibration) bands**: the previous system used fixed raw-probability thresholds (`RISK_THRESHOLD_LOW=0.10`, `RISK_THRESHOLD_MEDIUM=0.30`, three bands: Low/Medium/High). Because raw probabilities were inflated population-wide, 90.6% of applicants previously labeled "Medium" or part of "High" fall into the "Low" band under calibrated PD. Ranking (ROC-AUC/PR-AUC) is unchanged by calibration — this is a scale correction, not a change in the model's underlying risk assessment.

## Cost-sensitive decision optimization

```
Expected Cost = FP_COST × false_positives + FN_COST × false_negatives
```
where "positive" = predicted high-risk/default (calibrated PD ≥ threshold); FP = a good applicant wrongly declined, FN = a bad applicant wrongly approved.

Optimized on **calibrated PD** (not raw probability, unlike the legacy Milestone-1 sweep — both now coexist, see below), over a configurable threshold sweep (default 0.01–0.99):

| Scenario (illustrative only — not real bank policy) | FP/FN Cost | Optimal threshold | Expected cost | Precision | Recall | Acceptance rate |
|---|---|---:|---:|---:|---:|---:|
| Balanced | 1 / 1 | 0.500 | 4,911 | 0.6164 | 0.0288 | 99.6% |
| **Conservative (baseline)** | 1 / 5 | **0.170** | 19,622 | 0.2756 | 0.4419 | 87.1% |
| Very Conservative | 1 / 10 | 0.100 | 29,743 | 0.1988 | 0.6717 | 72.7% |

**Old (raw scale, threshold 0.67) vs. new (calibrated scale, threshold 0.17) at the same FP=1/FN=5 cost assumption**: the two numeric thresholds are **not directly comparable** — they're optimized on different probability scales, and "the threshold improved from 0.67 to 0.17" would be a meaningless claim. What *is* comparable is the resulting expected cost under the same cost ratio: old = 19,616, new = 19,622 — a **statistical tie** (0.03% difference). This makes sense: sigmoid calibration is a strictly monotonic transform, so cost-optimal threshold selection (which depends only on rank order) converges on nearly the same underlying applicant partition regardless of scale — the huge numeric difference between 0.67 and 0.17 reflects the calibration curve's shape, not a change in which applicants get flagged.

`predict_applicant()` now exposes, alongside the legacy raw-probability `cost_optimal_decision`/`optimal_threshold` (kept unchanged for backward compatibility): `decision`, `decision_probability_source` (explicitly `"calibrated"`, `"raw"`, or `"unavailable"` — never implicit), `decision_threshold`, `decision_scenario`, `fp_cost`/`fn_cost`. Both decision surfaces coexist; neither silently replaces the other.

## SHAP explainability

`shap.TreeExplainer` with its **default** `model_output="raw"` — not reconfigured to explain probability. Verified empirically against a real applicant: `base_value + Σ SHAP contributions == model.predict(X, raw_score=True)` (the raw margin) to full floating-point precision, and `sigmoid(that sum) == pipeline.predict_proba()` exactly.

**SHAP explains the model's raw margin — it does not directly explain calibrated PD, the risk band, or the Approve/Decline decision**, which are separate downstream transformations. `shap_explanation_target = "raw_model_margin"` is set explicitly on every structured explanation so this is never implied otherwise.

`src/explainability/shap_explainer.py::explain_applicant_full()` returns an `ApplicantExplanation` with the raw/calibrated probability, risk band, decision, and SHAP contributions split into risk-increasing (SHAP > 0) vs. risk-reducing (SHAP < 0) drivers, ranked by `|SHAP value|`. Each contribution is mapped to a business-readable description via the feature registry (falling back to the plain feature name — never a fabricated interpretation — if unregistered). One-hot categorical features are split back into `(source_feature, category)` pairs (e.g. `NAME_CONTRACT_TYPE_Cash loans` → source `NAME_CONTRACT_TYPE`, category `Cash loans`) — a real naming bug (generic `x0_`/`x1_`-style sklearn defaults) was found and fixed to make this possible.

Real example (SK_ID_CURR=100002): raw probability 0.8356, calibrated PD 0.3518, risk band Very High, decision Decline. Top risk-increasing drivers: `EXT_SOURCE_3`, `EXT_SOURCE_1`, `EXT_SOURCE_2`. Top risk-reducing: `DAYS_BIRTH`, `INSTALL_LATE_RATIO`, `POS_CNT_INSTALMENT_FUTURE_MEAN`.

## Fairness audit

`src/fairness/audit.py` computes demographic parity difference and equal opportunity difference across `CODE_GENDER` and `NAME_EDUCATION_TYPE`, on the same validation split, using the deployed model's raw probability against `RISK_THRESHOLD_MEDIUM` (an intentionally independent system from the calibrated risk bands above — see its own module docstring).

**Real findings, not smoothed over**: `NAME_EDUCATION_TYPE` shows a demographic parity difference of **0.2865** and equal opportunity difference of **0.1992** (42.5%–71.6% predicted-high-risk range across education levels); `CODE_GENDER` similarly shows a substantial disparity. Both are well past the 5-point flag threshold — **this model, as-is, would not pass a real fair-lending review.** No mitigation (post-hoc threshold adjustment, fairness-constrained training, proxy-feature removal) has been implemented — this audit's job is to **surface** disparity accurately, not resolve it. Do not interpret these metrics as proof of legal compliance or its absence; that determination requires domain/legal expertise this audit does not provide.

## Model monitoring

`src/monitoring/drift_report.py` uses Evidently to compare a saved training-time reference snapshot against a current batch, covering **data drift** (feature distributions) and **target drift** (TARGET distribution) — the two forms of drift currently implemented. Prediction drift and live performance monitoring (which need labeled outcomes over time) are **not** implemented — this is honestly a batch/on-demand data-quality check, not a full enterprise observability platform. A real bug (conflating p-value and distance-based drift-test thresholds) was found and fixed during development — see `src/monitoring/drift_report.py`'s docstring.

## Model card

`scripts/generate_model_card.py` produces `reports/model_card.pdf` — intended use, out-of-scope uses, training data, performance metrics, per-group fairness metrics, monotonic constraints, and known limitations, every number computed fresh from the current model artifact.

## REST API

`api.py` exposes `GET /health` and `POST /predict` on port 8000, alongside Streamlit (port 8501) in the same container via `supervisord`. It is a thin wrapper around `src.ml.predict.predict_applicant()` and the SHAP explainer — the same inference path the UI uses.

**Documented accurately, not aspirationally**: `PredictResponse.risk_score` returns the **raw** LightGBM probability (0–1) — not the calibrated probability, and not the separate 0–1000 integer risk score `src.ml.risk_scoring` computes internally. This naming collision is a known, deliberate backward-compatibility tradeoff, not an oversight — documented in the field's own description. `risk_band` reflects the current Low/Moderate/High/Very High calibrated-PD bands. The API does **not** currently expose `calibrated_probability`, the Milestone 6 cost-sensitive `decision`, or the Milestone 5 integer `risk_score` — only what's shown above, plus top-3 SHAP drivers.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 406597.5, "AMT_ANNUITY": 24700.5,
  "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Secondary / secondary special",
  "NAME_CONTRACT_TYPE": "Cash loans", "DAYS_BIRTH": -9461, "DAYS_EMPLOYED": -637
}'
```
Interactive OpenAPI docs at `http://localhost:8000/docs`.

## Streamlit application

10 tabs, reflecting what's actually implemented (not an aspirational list):

1. **EDA** — dataset summary, class balance, the 6 business insights.
2. **Risk Prediction** — score one applicant: raw + calibrated probability, risk band, the new calibrated-PD business decision, the legacy raw-probability cost-optimal decision, and an interactive threshold-vs-cost curve.
3. **Model Evaluation** — live ROC/PR curves, confusion matrix, and the **Download Model Card** button.
4. **Explainability** — per-applicant SHAP (plain-language) plus a global SHAP summary/beeswarm plot.
5. **Business Rules** — surrogate decision-tree rules.
6. **Fairness Audit** — per-group metrics, demographic parity/equal opportunity, a bar chart, with an explicit disclaimer that disparity metrics are not proof of legal compliance.
7. **Monitoring** — live data/target drift vs. a saved reference snapshot.
8. **Chatbot** — natural-language-to-SQL portfolio analytics.

(Tabs are grouped here by function; see `app.py`'s sidebar for the exact current list and order.)

## Prompt engineering & chatbot

- **Provider**: [Groq](https://groq.com) free tier tried first, [Gemini](https://ai.google.dev) free tier as automatic fallback.
- **Guardrails**: `sqlparse`-based validation rejects anything but a single `SELECT` against the known `applicants` table/columns; one retry on validation failure; a graceful "couldn't answer" message if both attempts fail; `LIMIT` enforced on every query.
- Renamed conceptually to **natural-language risk analytics** — it answers portfolio questions ("default rate by education level," "high-risk applicants with no bureau history"), not general chit-chat.

## Testing

```
pytest suite: 145 tests across feature engineering, data validation, model
comparison, calibration, risk scoring, decision optimization, SHAP
explainability, and the data pipeline — 0 failures on the last full run.
```
Run: `docker-compose run --rm --entrypoint python app -m pytest tests/ -v`

## Configuration

All configuration lives in `.env` (gitignored; `.env.example` has placeholders only — never commit real credentials). Groups:
- **LLM** (chatbot): `GROQ_API_KEY`, `GEMINI_API_KEY` and model names.
- **Risk/decision thresholds**: `RISK_THRESHOLD_LOW`/`MEDIUM` (legacy raw-probability bands, still used by the fairness audit and legacy `cost_optimal_decision`), `FP_COST`/`FN_COST` (used by both the legacy Milestone 1 sweep and the Milestone 6 calibrated-PD optimizer).
- **Chatbot Postgres** (optional): `POSTGRES_URL`.
- **Data pipeline / cloud** (optional, Milestone 8): `DATA_SOURCE`, `AWS_REGION`, `S3_BUCKET`, `S3_PREFIX`, `DATABASE_URL` — a deliberately separate setting from `POSTGRES_URL` above; the two Postgres integrations don't depend on each other.
- **Paths**: `DATA_DIR`, `MODELS_DIR`.

AWS credentials themselves are never read from `.env` — they come entirely from boto3's standard credential chain (environment variables, `~/.aws/credentials`, an IAM role).

## Docker

`docker-compose up --build` runs one container with Streamlit (8501) and FastAPI (8000) via `supervisord`. Volume mounts: `data/`, `models/`, `reports/`, `curated_data/`, `metadata/` (the last two added in Milestone 8 — without them, data-pipeline output only existed inside the container and was lost on rebuild, a real bug found and fixed). The app starts and serves correctly with zero cloud configuration present.

## Known limitations

- Risk bands and the illustrative cost scenarios are **project-defined, not official banking/FICO/regulatory categories** — stated throughout, not just here.
- Fairness disparities are measured and surfaced, not mitigated — this model, unmodified, would not pass a real fair-lending review.
- No temporal validation anywhere in the platform — every split is random 80/20; credit data has real temporal structure a random split doesn't test.
- The SQL chatbot's validator is a token-scan safeguard, not a full parser.
- Monitoring covers data/target drift only — no prediction drift, no live performance monitoring (both need labeled outcomes over time this project's static dataset can't provide).
- The S3/PostgreSQL integration paths are covered by mocked-client tests, not a live AWS/Postgres run — see § Data engineering above for validation scope.
- The data pipeline's PostgreSQL table (`applicant_features`) is analytical storage sized for this project, not a substitute for an enterprise data warehouse.
- Calibration, risk bands, and the cost-optimal threshold were all computed once against the current model/dataset snapshot; none of them re-validate themselves automatically after a retrain — rerunning `src.ml.calibration`/`src.ml.risk_scoring`/`src.ml.decision_optimization` after any retrain is a manual step.
- The legacy raw-probability `cost_optimal_decision`/`optimal_threshold` fields (Milestone 1) remain in `predict_applicant()`'s output for backward compatibility, alongside the new calibrated-PD decision fields (Milestone 6) — two coexisting, uncoordinated decision surfaces, by design, not an oversight.
