# Credit Risk Intelligence Platform — Interview Prep / Project Explainer

**Update note (final synchronization pass)**: this document was originally written for the initial NeoStats assignment submission (§1-16 below). The project has since grown through 8 further milestones — model comparison, probability calibration, calibrated risk scoring/bands, cost-sensitive decision optimization, upgraded SHAP explainability, and a batch data-engineering/cloud layer (S3/Parquet/PostgreSQL). Some numbers below (feature/column counts, PR-AUC/ROC-AUC, risk-band names) reflect the *original* submission and have been superseded — they've been corrected inline where found, but for the definitive, currently-accurate architecture and metrics, see **`README.md`** and **`reports/final_architecture.md`** first. Treat this file as a record of the original reasoning (still valid for the "why" questions) layered under numbers that have since improved.

This document is meant to be **self-contained**: everything needed to explain this project in an interview, including *why* every major choice was made, what the alternatives were, why they were rejected, what actually broke in production, and what was learned. If you feed only one file to an LLM to prep from, feed `README.md` + `reports/final_architecture.md` first, then this one for historical "why" context. (If you want exact command syntax or the full raw metrics tables, `README.md` and `DESIGN_DECISIONS.md` in the repo root are the primary-source backups — see §16 for guidance on when those are worth pulling in too.)

---

## 1. The assignment (what was asked)

This was a NeoStats AI Engineer internship assignment: build a **Dockerized, end-to-end credit risk platform** on the Kaggle **Home Credit Default Risk** dataset, using **only free-tier tools** (no paid APIs, no paid hosting, no credit card anywhere). The deliverable needed to demonstrate five things working together, not just a model:

1. **A trained ML model** predicting probability of default, with imbalance handled properly (the dataset is ~92% repaid / 8% defaulted).
2. **Explainability** (SHAP) — not just a score, but *why* that score.
3. **Human-readable business rules** derived from the model — something a credit policy analyst without ML background could read.
4. **A "talk to your data" chatbot** — natural language → SQL → plain-English answer, safely, over the applicant data.
5. **A Streamlit UI** tying all of it together, **shippable as a single Docker container**, evaluable by a stranger who just clones the repo and runs `docker-compose up`.

The evaluation criteria explicitly rewarded **code quality, correctness, and honest reasoning about tradeoffs** — not just a high accuracy number.

---

## 2. Business framing (the "why", for the interview)

A bank that only outputs a raw probability from a black-box model has two problems:
- **Credit analysts can't defend a rejection** to a regulator or a customer with just a number.
- **The risk team can't audit** whether the model is picking up on sensible signals or spurious correlations.

That's the reason the platform has four layers on top of the raw prediction: SHAP explanations (why this applicant), business rules (what pattern, in general), a chatbot (so risk teams can ask ad-hoc questions without writing SQL), and — added later — monotonic constraints (so certain features are *provably* well-behaved, which matters for fair-lending defensibility).

---

## 3. Tech stack and why each piece was chosen (all free-tier)

| Need | Choice | Why (short version — full alternatives comparison in §10) |
|---|---|---|
| LLM for chatbot | **Groq free tier** (`openai/gpt-oss-20b`), Gemini free tier as fallback | Fast, generous free rate limits, no credit card. |
| SQL-over-CSV | **DuckDB** | Embedded, zero infrastructure, and its `postgres` extension lets it attach a real Postgres database live later. |
| Model | **LightGBM** | Handles missing values natively, mixed numeric/categorical after one-hot encoding, strong out-of-the-box tabular performance. |
| Imbalance handling | `scale_pos_weight` (not SMOTE) | At ~300k rows and 100+ one-hot-expanded features, SMOTE's synthetic samples land in unrealistic regions and mostly add training cost. |
| Explainability | **SHAP** (TreeExplainer) | Exact (not approximate) Shapley values for tree models specifically. |
| Rule extraction | **Decision-tree surrogate** | See §7 — a full section, because "why this step at all" deserves more than one line. |
| UI | **Streamlit** | Fast multi-page apps, runs inside a single Docker container. |
| Deployment | **Docker + docker-compose**, live on **Render's free tier** | See §10.6 for why Render specifically, not Streamlit Community Cloud. |

---

## 4. Repository structure (know what every folder does)

```
credit_risk_platform/
├── app.py                        # Streamlit app: 6 tabs, described in §8
├── Dockerfile, docker-compose.yml, packages.txt
├── requirements.txt              # pinned versions — all free/open-source
├── .env.example                  # documents every config option (no secrets committed)
├── README.md                     # full technical documentation
├── DESIGN_DECISIONS.md           # condensed "why we chose X over Y" for every major decision
├── data/                         # Kaggle CSVs (gitignored) + committed synthetic fallback + EDA chart PNGs
├── documents/                    # presentation PDF, UI screenshots
├── models/credit_risk_model.joblib   # the ONE trained artifact everything else loads
├── notebooks/eda.ipynb, eda.py   # EDA notebook + its exported script twin
├── sql/schema.sql                # reference copy of the chatbot's DuckDB schema
├── scripts/
│   ├── generate_synthetic_data.py    # builds the committed synthetic fallback dataset
│   └── load_dataset_to_postgres.py   # one-time, manual: loads real data into Postgres
└── src/
    ├── data/
    │   ├── loader.py              # reads raw CSVs; build_joined_dataset() = the one entry point
    │   └── preprocessor.py        # build_aggregate_features(), engineer_features(), the sklearn pipeline builder
    ├── ml/
    │   ├── train.py                # train(), train_baseline(), cross_validate(), tune()
    │   ├── predict.py              # predict_applicant() — used by the UI's Risk Prediction tab
    │   └── evaluate.py             # evaluate_model(), compute_roc_pr_curves(), top_feature_importances()
    ├── explainability/shap_explainer.py   # explain_applicant() (per-row), compute_global_shap_values() (dataset-wide)
    ├── rules/rule_derivation.py    # derive_rules() — surrogate decision tree → if/then sentences
    ├── talk_to_data/
    │   ├── prompt_templates.py     # SCHEMA_DESCRIPTION, SYSTEM_PROMPT_V1, FEW_SHOT_EXAMPLES
    │   ├── nl_to_sql.py             # ask() — the whole NL→SQL→answer loop, with retry-on-error
    │   └── query_runner.py         # DuckDB connection + the SQL safety allowlist (ALLOWED_COLUMNS, validate_sql)
    └── utils/                       # config.py (Settings dataclass from .env), logger.py, helpers.py
```

**The one rule that makes this all hang together**: `build_aggregate_features()` in `preprocessor.py` is called by *training*, *prediction*, and *explainability* code paths alike — so the features the model was trained on are guaranteed to be the exact same features used at inference time. This was a deliberate design choice to prevent train/serve skew.

---

## 5. The dataset — Home Credit Default Risk, in depth

**Know this section cold — "tell me about your dataset" is close to a guaranteed question.**

### 5.1 What it is and why it's hard
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) is a Kaggle competition dataset from Home Credit Group, a lender that specifically serves people with **little or no traditional credit history** — the population banks usually can't score at all. That's the real-world hard part this dataset represents: you can't lean on a FICO-style score, because for a large share of applicants one doesn't meaningfully exist yet. Instead, the signal has to come from alternative sources: the application itself, external bureau records (even thin ones), prior interactions with this same lender, and month-by-month repayment behavior on whatever credit the applicant already has.

### 5.2 The seven tables and how they relate

| Table | Rows | Cols | Grain | What it captures |
|---|---|---|---|---|
| `application_train.csv` | 307,511 | 122 | 1 row per applicant | The current loan application — demographics, income, requested credit amount, and `TARGET` (1 = defaulted, 0 = repaid). This is the base table everything else joins onto. |
| `application_test.csv` | ~48,744 | 121 | 1 row per applicant | Same schema, no `TARGET` — Kaggle's holdout for competition scoring. Not used for evaluation in this project (this isn't a Kaggle submission), only referenced by `build_joined_dataset(is_train=False)` for symmetry. |
| `bureau.csv` | 1,716,428 | 17 | many rows per applicant | Every credit line the applicant has with *other* lenders, as reported to a credit bureau — one row per external credit line (`SK_ID_CURR` → many `SK_ID_BUREAU`). |
| `bureau_balance.csv` | 27,299,925 | 3 | many rows per bureau credit | Month-by-month status history for each bureau credit line. `STATUS` is ordinal: `0` = current, `1`–`5` = escalating days-past-due severity, `C` = closed, `X` = unknown. This is the largest table in the whole dataset by row count. |
| `previous_application.csv` | 1,670,214 | 37 | many rows per applicant | The applicant's own history of *previous* loan applications with Home Credit itself (not other lenders) — approved, refused, or cancelled. |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | many rows per applicant | Month-by-month balance snapshots of the applicant's previous point-of-sale/cash loans. |
| `credit_card_balance.csv` | 3,840,312 | 23 | many rows per applicant | Month-by-month balance snapshots of the applicant's previous Home Credit credit cards. |
| `installments_payments.csv` | 13,605,401 | 8 | many rows per applicant | Actual repayment history on previous credits — due date/amount vs. actual paid date/amount, at the individual-installment level. |

All six auxiliary tables get **aggregated to one row per `SK_ID_CURR`** (see §6.1) before joining onto the base table — none of them are used at their raw many-rows-per-applicant grain in the model.

### 5.3 Feature categorization (of the 122 raw `application_train.csv` columns)
- **1 ID column**: `SK_ID_CURR`.
- **5 date-like columns** (anything containing `DAYS_`): `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`. Important quirk: these are **negative day-offsets relative to the application date**, not real calendar dates — e.g. age in years = `-DAYS_BIRTH / 365`.
- **16 categorical columns**: things like `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, plus several building/apartment "MODE" columns.
- **99 numeric feature columns**: income, credit amount, annuity, the three `EXT_SOURCE_*` normalized external credit-bureau scores, building/apartment area statistics, document-provided flags, social-circle default counts, and more.
- **`TARGET`**: binary, 1 = defaulted, 0 = repaid.

### 5.4 Missing-value analysis
**67 of the 122 raw columns** have at least one missing value. The worst offenders are all **self-reported building/apartment attributes** — `COMMONAREA_AVG`/`_MODE`/`_MEDI` and `NONLIVINGAPARTMENTS_*` are missing for roughly **69–70%** of applicants, `FONDKAPREMONT_MODE` and `LIVINGAPARTMENTS_*` around **68%**. This makes intuitive sense: these are details about the applicant's residence that many simply don't have or didn't report, not a data-quality bug. Handling: numeric columns get median imputation, categorical columns get most-frequent imputation (`SimpleImputer` inside the sklearn `ColumnTransformer`) — applied explicitly in the pipeline rather than relying only on LightGBM's native NaN handling, because the logistic-regression baseline (§6.6) needs fully imputed input too.

### 5.5 Class balance — the central challenge
```
TARGET = 0 (repaid):    91.93%
TARGET = 1 (defaulted):  8.07%
```
This ~11:1 imbalance is *the* thing that drives most of the modeling decisions downstream: plain accuracy is meaningless (a model that predicts "never defaults" for every applicant scores ~92% while being useless), which is why `scale_pos_weight` was used instead of leaving the loss unweighted, and why PR-AUC — not ROC-AUC — was chosen as the primary evaluation metric (§6.7).

### 5.6 The six business insights (from `notebooks/eda.ipynb` / `eda.py`, live in the app's EDA tab)

| # | Insight | Default rate by group | Shape |
|---|---|---|---|
| 1 | Income bracket (quintiles) | Lowest 8.21% · Low 8.59% · Mid 8.68% · High 8.06% · **Highest 6.52%** | Mostly flat, only the top quintile clearly stands out — raw income alone is a *weak* signal on its own. |
| 2 | Employment length band | **<1yr 10.97%** · 1–3yr 11.07% · 3–7yr 8.97% · 7–15yr 6.38% · **15yr+ 4.49%** | Cleanly monotonic — longer job tenure, lower risk. One of the strongest single categorical splits. |
| 3 | Education level | **Lower secondary 10.93%** · Secondary/secondary special 8.94% · Incomplete higher 8.48% · Higher education 5.36% · **Academic degree 1.83%** | Strongly monotonic, and the widest spread of any single categorical (~6× between extremes). |
| 4 | Credit amount band (quintiles) | Lowest 7.24% · Low 9.17% · **Mid 10.05%** · High 7.85% · Highest 6.08% | **Not** monotonic — risk peaks in the *middle*, not at the largest loans. |
| 5 | Age band | **20–30 11.46%** · 30–40 9.58% · 40–50 7.65% · 50–60 6.13% · **60–70 4.92%** | Cleanly monotonic, younger applicants default more. Also the most ethically sensitive predictor here — flagged for fair-lending review in §13. |
| 6 | Bureau history (yes/no) | **No bureau history 10.12%** (n=44,020) · **Has bureau history 7.73%** (n=263,491) | A 2.4-point gap that directly justified joining `bureau.csv` into the training pipeline — see §6.2. |

**Two of these insights fed directly into concrete engineering decisions, not just narrative charts:**
- **Insight 4 (credit amount is non-monotonic)** is exactly why raw `AMT_CREDIT` was never trusted alone — it's the reason `CREDIT_INCOME_RATIO` and `CREDIT_TERM` (engineered ratios, §6.3) were prioritized: the *ratio* of credit to what someone can actually afford is a cleaner signal than the raw amount.
- **Insight 6 (bureau history gap)** is the empirical justification for the single biggest feature-engineering decision in the whole project (§6.2) — the join was proposed *because* this number was measured first, not the other way around.

### 5.7 How the notebook is structured
`notebooks/eda.ipynb` (with `eda.py` as its exported, script-runnable twin — useful for headless verification without a running notebook server) walks through, in order: load data → dataset summary → feature categorization (§5.3) → missing-value analysis + chart (§5.4) → `TARGET` class balance + chart (§5.5) → the six business insights, each with its own chart (§5.6) → a plain-language summary list at the end. The same six insights (plus the live applicant/feature/default-rate counters) are reproduced as **live, recomputed** Streamlit charts in the app's EDA tab — not static images pasted in — so they reflect whatever data source is actually active (real CSV or synthetic fallback) at runtime.

---

## 6. The ML pipeline, step by step

### 6.1 Data joining
- `application_train.csv` is the base table (one row per applicant).
- The six auxiliary tables from §5.2 get aggregated to one row per `SK_ID_CURR` and left-joined on.
- Result at the time of the original submission: 307,511 rows × 166 columns. **Since superseded** by the Milestone 1 feature-engineering upgrade (80 registered engineered features, credit-behavior aggregates with recency windows) — the joined dataset is now 307,511 rows × 194 columns, 319 after encoding. See `README.md` § Feature engineering.
- **No leakage check performed and passed**: every `DAYS_*`/`MONTHS_BALANCE` column in the auxiliary tables is ≤ 0 relative to the *current* application — i.e. genuinely historical, not looking into the future.

### 6.2 Why the bureau/previous-application join specifically (empirical, not assumed)
Before writing the join code, EDA insight #6 (§5.6) measured: applicants with **no** bureau history default at **10.12%**, applicants **with** bureau history default at **7.73%** — a real, measured 2.4-point gap in the raw data. That finding is what justified the join, not "credit datasets usually benefit from this."

### 6.3 Feature engineering
`engineer_features()` adds domain ratios known to matter for this dataset: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `DAYS_EMPLOYED_RATIO`, `BUREAU_DEBT_CREDIT_RATIO`. Also fixes a known Kaggle data quirk: `DAYS_EMPLOYED` uses the sentinel value `365243` to mean "not employed" — that gets replaced with `NaN`.

### 6.4 Imbalance handling
`scale_pos_weight = negative_count / positive_count` fed into LightGBM. SMOTE is available as an alternative (`--smote`) but not the default — full alternatives comparison in §10.2.

### 6.5 Monotonic constraints (a later, deliberate addition)
`AMT_INCOME_TOTAL` and `EXT_SOURCE_1/2/3` get `monotone_constraints = -1` — meaning higher income or a higher external credit-bureau score can **never** increase the model's predicted risk, no matter what a tree happens to learn locally from noise. This is standard practice in credit risk scoring specifically: a fair-lending reviewer can reasonably ask "does raising this applicant's income ever make them look riskier?", and for these four features the answer needs to provably be no.

- The `-1` direction was **verified, not assumed** — checked against this codebase's own derived rules output (`EXT_SOURCE_3 > 0.55 → Low Risk`, `≤ 0.55 → High Risk`, confirming higher = safer).
- Deliberately **not** applied to every feature. `CNT_CHILDREN` and `DAYS_EMPLOYED` have plausible *non-monotonic* real-world relationships with default risk — forcing a direction on them would be scientifically dishonest, not more rigorous. Only 4 of 291 trained features were constrained at original submission (319 after the Milestone 1 feature upgrade) — still just these same 4.
- **Measured cost of the constraint**: essentially zero (PR-AUC moved by −0.0001, ROC-AUC by +0.0003). Be ready to say: *"we checked this, we didn't assume it — on a different dataset it might cost real performance, and if it did, we'd have reported that honestly."*

### 6.6 Model comparison, cross-validation, and tuning (rigor added on review)
A code review surfaced four real gaps against standard ML practice, all fixed:

1. **Unsupported baseline claim.** The README said "LightGBM chosen over logistic regression" with no logistic regression ever actually trained. Fixed: `python -m src.ml.train --baseline` trains a real `LogisticRegression(class_weight="balanced")` on the same split (features scaled with `StandardScaler` — logistic regression needs scaling, LightGBM doesn't). **Honest finding**: LightGBM wins, but the gap is *smaller* on ROC-AUC alone (0.782 vs 0.765) than the confident tone implied — the gap is clearer on PR-AUC (0.276 vs 0.245), which is the metric that actually matters here.
2. **No cross-validation.** A single 80/20 split's number has no confidence interval. Fixed: `--cv` runs 5-fold `StratifiedKFold`, reporting mean ± std (ROC-AUC 0.7790 ± 0.0039, PR-AUC 0.2679 ± 0.0060) — confirms the single-split number wasn't a lucky draw.
3. **No hyperparameter tuning.** Fixed: `--tune` runs a bounded `RandomizedSearchCV` (n_iter=15, cv=3, scored on PR-AUC). Found only **+0.0020 ROC-AUC** improvement — smaller than the CV's own fold-to-fold noise (±0.0039) — so current defaults were **kept**, not chased.
4. **No monotonic constraints** — covered in §6.5.

All three extra paths (`--baseline`, `--cv`, `--tune`) are **opt-in flags**, not part of the default `python -m src.ml.train` — the default stays fast (~50s) so a fresh clone isn't punished with a multi-minute training run it didn't ask for.

### 6.7 Final numbers (original submission — since superseded, see below)

| Metric | Value (original) |
|---|---|
| PR-AUC (primary metric) | 0.2760 |
| ROC-AUC | 0.7817 |
| Precision @ 0.5 | 0.1845 |
| Recall @ 0.5 | 0.6959 |
| F1 @ 0.5 | 0.2916 |

**Current deployed numbers** (post Milestone 1 feature upgrade + Milestone 3 model comparison — LightGBM re-confirmed as the selected model against Logistic Regression and XGBoost under identical conditions): **PR-AUC 0.2851, ROC-AUC 0.7854** (5-fold CV: ROC-AUC 0.7827±0.0032, PR-AUC 0.2746±0.0043). See `README.md` § Model development & selection and `reports/model_comparison.md` for the full comparison.

**Why PR-AUC is the headline, not ROC-AUC**: with ~92/8 imbalance, ROC-AUC can look deceptively strong while the model still performs poorly on the minority (default) class that a screening tool actually cares about. PR-AUC is computed entirely from precision/recall on the positive class, so it's the more honest number here.

Confusion matrix @ threshold 0.5:
```
              Pred: No Default   Pred: Default
Actual: No Default   41,265          15,273
Actual: Default       1,510           3,455
```
At 0.5 the model favors **recall** (~70% of actual defaulters caught) over precision — the right tradeoff for a screening tool, where missing a defaulter costs more than a false flag that goes to manual review anyway. The threshold is adjustable via `.env` (`RISK_THRESHOLD_LOW`/`RISK_THRESHOLD_MEDIUM`).

---

## 7. Business rules — the full "why this step" explanation

**This is worth its own section because "why did you build this at all" is a different, deeper question than "how does it work."**

### 7.1 The problem it solves
SHAP explains *one applicant at a time*. A credit policy analyst doesn't want to read a SHAP plot for every applicant — they want a small number of general, memorable, auditable statements like *"if EXT_SOURCE_3 is below X and the applicant is under Y years old, flag as high risk"* that can be socialized into actual bank policy, printed in a compliance document, or challenged by a human. That's a fundamentally different deliverable from either the raw model or a per-applicant explanation, and it's explicitly what the assignment asked for: *"extract simple, human-readable if/then rules... something a credit policy analyst could read without any ML background."*

### 7.2 How it works, precisely
`derive_rules()` in `src/rules/rule_derivation.py`:
1. Takes a sample of applicants (default up to 20,000, for speed) and runs them through the **already-trained LightGBM model** to get its predictions.
2. Trains a **shallow `DecisionTreeClassifier`** (`max_depth=4`, `class_weight="balanced"`) where the **target is the LightGBM model's own predicted class — not the real `TARGET` column.**
3. Exports that shallow tree as text (`sklearn.tree.export_text`) and converts each leaf's path into an `IF ... THEN predicted outcome = ...` sentence.

Real sample output from this codebase:
```
IF EXT_SOURCE_3 <= 0.54 AND EXT_SOURCE_2 <= 0.54 AND EXT_SOURCE_2 <= 0.38 AND DAYS_BIRTH <= -24078.50
   THEN predicted outcome = Low Risk (likely repay)

IF EXT_SOURCE_3 <= 0.54 AND EXT_SOURCE_2 <= 0.54 AND EXT_SOURCE_2 <= 0.38 AND DAYS_BIRTH > -24078.50
   THEN predicted outcome = High Risk (likely default)

IF EXT_SOURCE_3 <= 0.54 AND EXT_SOURCE_2 > 0.54 AND EXT_SOURCE_3 <= 0.31 AND EXT_SOURCE_1 <= 0.53
   THEN predicted outcome = High Risk (likely default)
```
(16 rules total per run, from a depth-4 tree — `2^4 = 16` leaves.)

### 7.3 Why *predict on the model's predictions*, not the real `TARGET`
This is the single most important design decision in this whole feature, and worth stating precisely if asked: **the goal is not to build a second, independent model of default risk — it's to explain the first one.** If the surrogate tree were trained on the real `TARGET`, it would just be a separate, worse, shallow model that happens to sometimes agree with LightGBM — its rules wouldn't actually describe *what LightGBM does*, they'd describe *what a shallow tree thinks about the raw data*, which is a different (less accurate) question. Training on the LightGBM model's own output means the tree is doing **model distillation** / **surrogate modeling** — a well-known interpretability technique — and its rules are a genuine (if imperfect) map of the actual deployed model's decision boundary.

### 7.4 Why a decision tree specifically, and why depth 4
- A shallow decision tree is **inherently if/then-readable** — that's literally its structure, no translation step needed, unlike e.g. a linear model's coefficients (which need a reader to understand log-odds) or a shallow neural net (opaque regardless of size).
- **Depth 4** is a deliberate, explicit tradeoff: deep enough to capture real structure (16 distinct rules, using the model's actual top features), shallow enough that a non-technical reader can hold one path in their head. Depth 2 would be too coarse (4 rules, likely missing real structure); depth 8+ (256 leaves) would defeat the entire purpose — nobody reads 256 rules.
- `class_weight="balanced"` on the surrogate tree matters too: without it, a shallow tree trained on a skewed prediction distribution could just learn to always predict the majority class and still look "accurate," producing degenerate one-sided rules.

### 7.5 Honest limitation, stated up front
Documented explicitly, not hidden: **at rule boundaries, the surrogate tree's decision can disagree with the actual LightGBM prediction.** A depth-4 tree is a genuine simplification of a model built from hundreds of boosted trees — it approximates the boundary, it doesn't reproduce it exactly. This is stated in the README/app as: *"treat rules as an explanatory aid, not a replacement scoring engine."* Be ready to say this proactively if asked "how accurate are these rules" — the honest answer is "they're an approximation by design, and here's why that's still useful anyway" (§7.1).

### 7.6 Alternatives that were considered and why this one won
See §10.7 for the full comparison against LIME/Anchors/direct feature-importance lists.

---

## 8. The Streamlit app — six tabs

1. **EDA** — dataset summary, live applicant/feature/default-rate counters, and the 6 business insights with live charts (§5.6).
2. **Risk Prediction** — pick a sample applicant or fill a manual form → raw + calibrated probability, risk band (now Low/Moderate/High/Very High, calibrated-PD based as of Milestone 5 — originally Low/Medium/High on raw probability), and the cost-sensitive decision.
3. **Model Evaluation** — *added on review* because evaluation metrics existed only in the README/deck, never in the live app. Shows live ROC curve, PR curve, confusion-matrix heatmap, and precision/recall/F1 — reconstructing `train.py`'s exact validation split so the numbers match documentation. Cached on the model file's mtime so it doesn't recompute every rerun.
4. **Explainability** — two sections, clearly separated: *"This applicant's explanation"* (per-row SHAP, plain language) and *"Global feature importance across the dataset"* (a SHAP beeswarm summary plot over 300 sampled applicants) — the second was also added on review, since a single-applicant view alone doesn't tell a regulator or underwriter what the model leans on in general.
5. **Business Rules** — see §7 in full.
6. **Chatbot** — described fully in §9.

---

## 9. The chatbot — natural language to SQL, safely (the most iterated-on piece)

### 9.1 The pipeline
User question → `nl_to_sql.py::ask()` → Groq (or Gemini fallback) generates SQL against a compact schema description → `query_runner.py::validate_sql()` checks it → DuckDB executes it → the LLM is asked again for a 1–3 sentence plain-business-language summary of the result.

### 9.2 The safety layer (be ready to explain this in detail — it's a strong interview talking point)
`query_runner.py` has a **token-scan allowlist**, not a full SQL parser, built with `sqlparse`:
- Only a single `SELECT` statement is allowed — `INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`CREATE`/etc. are all rejected outright.
- Only the `applicants` table and a fixed `ALLOWED_COLUMNS` set can be referenced — any other column name found in the parsed token stream (that isn't a known SQL keyword/function or an alias) is treated as hallucinated and rejected.
- A `LIMIT` is always enforced (`MAX_QUERY_ROWS`, default 200).
- If validation fails, `nl_to_sql.py` retries **once** with the specific error fed back to the LLM (a different prompt, `RETRY_PROMPT_TEMPLATE`) — if that also fails, it returns a graceful "couldn't answer" message rather than ever executing unvalidated SQL.

### 9.3 Few-shot examples (a small, honestly-costed accuracy improvement)
`SYSTEM_PROMPT_V1` includes 4 example question→SQL pairs (matching the documented "Example queries" in the README, so the examples and the docs never drift apart) covering a simple aggregate, a group-by, a filter+count, and a boolean-column query. This costs roughly **~190 tokens per call** — measured, not guessed — traded for demonstrably more consistent SQL on ambiguous phrasing. Concrete before/after: the question *"compare genders"* without few-shot produced an inconsistent, unrequested-column SQL; with few-shot it exactly mirrored the `ROUND(...) AS default_rate_pct ... ORDER BY ... DESC` convention from the examples.

### 9.4 The "LIMIT 200 confuses the LLM" bug
Early on, the LLM's plain-language answers sometimes said things like *"this is based on the 200 defaulters in the sample"* — it was misreading the safety `LIMIT 200` clause as if it capped the underlying aggregation, when in reality `LIMIT` only caps the *displayed rows*, applied *after* `AVG`/`COUNT`/`GROUP BY` already ran over the full table. Fixed by adding an explicit instruction to `ANSWER_SUMMARY_PROMPT_TEMPLATE` telling the LLM exactly what `LIMIT` does and does not mean, and to never mention the number 200.

### 9.5 Data source, and *why synthetic data exists at all* (read this if asked "why synthetic data")
This is a direct answer to "why did you use synthetic data — why couldn't you just deploy the real thing?":

**The constraint**: the assignment explicitly forbids committing the Kaggle dataset to git (a completely standard rule — Kaggle's terms of use don't allow redistributing the dataset, and it's also just bad practice to put ~2GB of CSVs in a git repo). So the real data can only ever live in: (a) Kaggle itself, (b) a contributor's own local `data/` folder, gitignored. **A deployed instance (Render) starts from a fresh git clone — it never has the real CSVs, by design, on every single cold start.** That's not a bug, it's the correct behavior of the "never commit the dataset" rule — but it means a deployed instance with *nothing else* would have a permanently broken/empty chatbot and EDA tab.

Three tiers were built to solve this, in priority order:
1. **Real local CSV** — works if you `git clone` and manually download the Kaggle CSVs into `data/` yourself (exactly what local development does).
2. **Live Postgres** (chatbot-only) — a real Postgres database, provisioned separately, pre-loaded *once* from a machine that has the real CSVs, via `scripts/load_dataset_to_postgres.py`. The deployed app then queries it live via DuckDB's `postgres` extension — real data, but the dataset itself still never touches git or the Docker image.
3. **Synthetic fallback** — a 4,000-row dataset with fabricated values, generated with a fixed seed, calibrated to match this project's own measured real-data statistics (8% base default rate, the same bureau-history gap direction) so it's at least *directionally* honest.

**Why the synthetic tier is still necessary even with Postgres available**: Postgres is an *external dependency the app doesn't control* — it can be slow to provision, can expire (see §11.2 — this is exactly what happened), can have network issues, or a reviewer might just not want to set one up at all. The synthetic fallback is the guarantee that **the app is never in a broken state** — `docker-compose up` with zero configuration always produces a fully working demo of every tab, even if every real-data path is unavailable. That guarantee (never a broken page, worst case a fabricated-but-plausible one) was judged more valuable than letting the app show an error screen when Postgres happens to be down.

**Important nuance**: the *other* pages (EDA, Risk Prediction, Model Evaluation, Explainability) were briefly extended to also try Postgres, then **deliberately reverted back to chatbot-only**. Reasoning: the Postgres table only has the chatbot-facing columns, not the 166 the model needs — so those other pages got a *degraded* experience from Postgres anyway (missing `EXT_SOURCE_1/2/3` etc.), while adding real connection-attempt overhead on every cold start for no real benefit. Good interview point: **recognizing and reverting your own over-extension** of a feature once you see the actual tradeoff is a mark of good engineering judgment, not indecision.

### 9.6 The synthetic-data disclosure banner — and its removal
Originally, every page showed a banner (🧪 "Using synthetic demo data...") whenever real data wasn't available, specifically so an evaluator wouldn't be misled by numbers that don't match the documented real-Kaggle statistics. **This banner was later explicitly removed on request** — be honest and direct about this if asked: the deployed site can now show synthetic numbers with no on-screen indication they aren't real. The underlying fallback logic and server-side logging are unchanged; only the user-visible disclosure was removed.

---

## 10. Why this, not that — alternatives considered for every major choice

**This section exists specifically to answer "why did you use X instead of Y" and "how do you know X was actually the right call."** For each one: what else was on the table, and where the confidence that this was the right call actually comes from — general ML/engineering consensus, this specific dataset's known behavior, or a number we actually measured ourselves.

### 10.1 LightGBM vs. XGBoost vs. CatBoost vs. Random Forest vs. plain logistic regression
- **XGBoost**: the closest competitor. Both are gradient-boosted tree ensembles with similar accuracy ceilings on tabular data; LightGBM is generally faster to train (histogram-based leaf-wise growth) at this row count, and its Python API's native categorical/missing-value handling matched this project's needs slightly more directly. This is a "either would have worked, LightGBM was the pragmatic pick" answer, not a "XGBoost is worse" answer — be honest about that if pushed.
- **CatBoost**: strong for datasets with *many high-cardinality* categorical columns; this dataset's categoricals (16 columns, mostly low-cardinality like education/gender) didn't play as strongly to CatBoost's specific advantage.
- **Random Forest**: no boosting, so typically a lower accuracy ceiling on this kind of structured, moderately-sized tabular data than gradient boosting; also doesn't support `monotone_constraints` in scikit-learn's implementation, which mattered once fair-lending defensibility (§6.5) entered the picture.
- **Logistic regression**: not rejected — actually trained and measured (§6.6, point 1). The honest finding was the gap is real but smaller than expected on ROC-AUC. **Source of confidence**: this specific dataset (Home Credit Default Risk) is a well-known Kaggle competition where gradient-boosted trees (LightGBM/XGBoost) dominated the public leaderboard — that's community consensus *for this exact dataset*, not a generic "trees are always better" claim, and it was backed up with an actual measured baseline rather than taken purely on faith.

### 10.2 `scale_pos_weight` vs. SMOTE vs. random undersampling
- **SMOTE** (oversampling via synthetic minority interpolation): available behind `--smote`, not the default. At ~250k training rows and 150+ one-hot-expanded features, SMOTE's synthetic points are interpolated in a very high-dimensional space where "nearest neighbors" stop being a meaningful notion — the synthetic minority samples tend to land in unrealistic regions, adding training time without a measured accuracy benefit here.
- **Random undersampling** of the majority class: would throw away real data (at 92/8, undersampling to balance means discarding the large majority of the ~226k "repaid" training rows) — strictly worse use of available signal than reweighting the loss function, which keeps every row.
- **`scale_pos_weight`**: keeps 100% of the data, just tells the loss function "a missed default costs `neg/pos` times more than a false flag" — the standard, cheapest-to-reason-about first choice for boosted trees specifically, and it's what was actually deployed.

### 10.3 SHAP vs. LIME for explainability
- **LIME** approximates any model locally with a simple linear surrogate around one prediction — model-agnostic, but an *approximation*, and known to be less stable (small input perturbations can shift which features it highlights).
- **SHAP's `TreeExplainer`** computes **exact** Shapley values for tree ensembles specifically (not an approximation, for this model family) — since the model is LightGBM, this specific mode of SHAP is strictly more precise than LIME would be for the same job, at comparable or better speed. This is why SHAP was assumed from the assignment brief itself ("Explainability (SHAP)") — but the *reason* it's the right tool here, not just the assigned one, is this model-family-specific exactness argument.

### 10.4 DuckDB vs. SQLite vs. running everything through pandas directly
- **SQLite**: would have worked for the basic "query a CSV via SQL" need, but has no equivalent to DuckDB's `postgres` extension — the live-Postgres-attach capability (§9.5) that made the "real data without committing the dataset" deployment story possible **did not exist as an option** with SQLite. This was the actual deciding factor, not just "DuckDB is trendy."
- **Raw pandas** (no SQL layer at all): would have meant either exposing pandas query syntax to an LLM (worse-understood by most models than SQL, which they're extensively trained on) or hand-writing a much larger allowlist/validation layer for arbitrary pandas expressions — SQL's `SELECT`-only shape is a much smaller, easier-to-validate surface (§9.2) than general-purpose pandas code would be.

### 10.5 Streamlit vs. Flask/Gradio/Dash for the UI
- **Flask** would need hand-built HTML/CSS/JS for anything beyond a form — far more code for the same six-tab multi-page result.
- **Gradio** is closer to Streamlit but leans more toward single-model-demo UIs (one input → one output) than multi-page dashboards with charts, tables, and a chat interface side by side.
- **Dash** is more powerful for complex custom interactivity but has a steeper learning curve and more boilerplate for something this shape.
- Streamlit's multi-page-app pattern (a sidebar `radio` switching between page functions) matched this project's actual shape — six largely independent views sharing a few common data-loading functions — with the least code.

### 10.6 Render (Docker) vs. Streamlit Community Cloud vs. Railway/Heroku for deployment
**This is worth knowing precisely, because it's the choice most likely to get a "why not the obvious free option" follow-up:** Streamlit Community Cloud is free and literally built for Streamlit apps — so why not use it? Because **Streamlit Community Cloud does not run a custom Docker image** — it runs your `app.py` directly against a `requirements.txt` in its own managed Python environment. The assignment's core deployment requirement was explicitly **"shippable as a single Docker container"** — Streamlit Cloud structurally can't satisfy that requirement, since you don't control the container at all. Render (and comparable platforms like Railway or Fly.io) supports deploying an arbitrary `Dockerfile`, meaning **the exact same image that runs locally via `docker-compose up` is what's actually deployed** — no "works differently in the cloud" gap, no separate deployment-specific code path to maintain. That parity (one image, same behavior everywhere except environment variables) was the deciding factor, not brand preference.

### 10.7 Decision-tree surrogate vs. LIME/Anchors vs. a plain feature-importance list, for business rules
- **A plain global feature-importance list** (e.g. "top 10 features by gain") tells you *what the model leans on* but not *what decision it actually makes* — it's not an if/then rule a policy team could apply to a specific applicant.
- **LIME/Anchors**-style rule extraction produces *per-instance* explanations (an "anchor" rule that holds for one applicant's neighborhood), not a small set of *global* rules covering the whole applicant population the way a policy document needs.
- **A shallow decision-tree surrogate** is the one option that produces a *small, finite, global* set of if/then statements in one pass — exactly the shape of thing a policy document already looks like. Full reasoning for the "surrogate on predictions, not ground truth" choice is in §7.3.

---

## 11. Production incidents and what they taught (strong interview material — shows real debugging, not just "it worked")

### 11.1 The dependency-version mismatch
The committed model artifact (`models/credit_risk_model.joblib`) was at one point retrained on a local machine with a different scikit-learn version (1.6.1) than the one pinned in `requirements.txt` / the Docker image (1.5.1). This produced `InconsistentVersionWarning` at runtime — silent risk of subtly wrong behavior. **Fix**: always retrain *inside* the container (`docker-compose run --rm --entrypoint python app -m src.ml.train`), never on a local host with a different environment. This also surfaced a second, smaller gotcha: the Dockerfile's `ENTRYPOINT` (which runs Streamlit) has to be explicitly overridden for this to work — `docker-compose run app python -m ...` alone does *not* work as most people expect, since the entrypoint still runs first and just gets weird extra arguments.

### 11.2 The Postgres hang → Render 502 → the ThreadPoolExecutor bug (the best "tell me about a hard bug" story here)
Sequence of events, in order:
1. Deployed the Postgres-fallback feature. Worked in local testing.
2. The Postgres database (Render free tier) went down/expired.
3. On the live site, every page hung indefinitely trying to connect, and Render's proxy eventually returned a 502 Bad Gateway — the *entire app* went down because of one unreachable dependency.
4. **First fix attempt**: add `connect_timeout=5` to the Postgres connection string. This works for some failure modes (active refusal, DNS failure) but Render's specific network path still hung *past* it — the DuckDB postgres extension didn't honor `connect_timeout` the way a bare libpq client would in that environment.
5. **Second fix attempt**: wrap the connection attempt in a background thread with a hard wall-clock timeout via `concurrent.futures.ThreadPoolExecutor`, using `future.result(timeout=8)`. Deployed — **still hung**.
6. **Root cause found**: `with ThreadPoolExecutor() as pool:` — the context manager calls `pool.shutdown(wait=True)` on exit **unconditionally**, even when the `TimeoutError` fires and propagates out of the `with` block. So the fix silently re-blocked waiting for the still-hung worker thread anyway, defeating its own purpose. **Real fix**: create the pool manually (no `with`), and call `pool.shutdown(wait=False)` explicitly after catching the timeout — the abandoned thread leaks in the background (a real, accepted tradeoff) but the calling code returns immediately.
7. This is a genuinely subtle Python gotcha worth being able to explain clearly: *context managers can have exit-time side effects that aren't obvious from the "happy path" reading of the code, and a naive `with`-block timeout pattern is a common but broken idiom.*

### 11.3 The memory-limit crashes (separate root cause, same week)
Render kept restarting the service for exceeding its memory limit — *even after* the Postgres hang was fixed. Investigation: `matplotlib`, `seaborn`, and `shap` (a heavy library — brings in its own C extensions) were imported at the **top of `app.py`**, meaning *every single session* paid their import memory cost regardless of which of the six tabs it actually visited. Measured locally with `docker stats`:
- **Before**: idle baseline ~1.3–2.2GB.
- **After** moving those imports to be lazy (imported inside `page_model_evaluation()`/`page_explainability()` and their cached helper functions only, not at module level): idle/EDA/Chatbot-only sessions ~220MB; even visiting every heavy tab peaks around 380MB.

Interview point: **a memory-limit crash is not always the exotic case (a true leak) — sometimes it's simply "we import expensive things unconditionally that most sessions never need."** Profiling with `docker stats` before guessing at causes was the right first move, not immediately reaching for a bigger instance type.

### 11.4 Free-tier LLM model deprecation (an easy-to-underestimate operational headache)
Both Groq and Gemini rotate/rename their free-tier model identifiers more often than expected — a model name that worked at the start of the project (e.g. an older Groq or Gemini model string) stopped resolving partway through, with no advance warning, requiring the `.env`/config defaults to be updated to currently-valid model names. Operational lesson: **hardcoding a specific free-tier model name is inherently a maintenance liability**, not a one-time setup decision — worth checking against the provider's current model list periodically, and the code deliberately falls back to a second provider (Gemini) if the primary (Groq) fails for *any* reason, including a dead model name.

### 11.5 Git history hygiene
Also handled: a request to strip `Co-Authored-By: Claude` trailers from git commit history entirely (not just future commits). Done via `git filter-branch --msg-filter`, verified with three independent checks (local `git log`, direct query against `origin/main`, and GitHub's own REST API) that zero references remained anywhere in any commit message or author/committer field, before force-pushing with `--force-with-lease` (which refuses to push if someone else pushed in the meantime — safer than a bare `--force`). A backup tag was created first in case anything needed to be recovered.

---

## 12. What this project actually taught (the reflective section — good for "what did you learn")

- **"Works on my machine" and "works in production" are genuinely different claims.** Three separate production incidents (§11.1–11.3) all passed local testing first. The gap wasn't sloppiness — it was environment differences (library versions), network conditions (Render's path to Postgres vs. a local test), and load patterns (every session importing the same heavy libraries) that simply don't show up until the exact deployment environment is exercised.
- **A "fix" that isn't tested under the actual failure condition isn't verified — it's hoped.** The first two attempts at the Postgres timeout (§11.2, steps 4–5) both looked correct on inspection and both failed in production. The bug (`shutdown(wait=True)` on context-manager exit) was only found by reasoning carefully about Python's actual exit-time semantics, not by re-reading the "happy path."
- **Empirical justification beats domain-knowledge assumption, and it's worth the extra step.** The bureau-history join, the LightGBM-vs-logistic-regression comparison, and the monotonic-constraint direction were all things that *could* have been asserted from general knowledge — and all three were instead measured first. In at least one case (the logistic-regression baseline) the measured result was more nuanced than the assumption would have suggested.
- **Reporting a negative or marginal result honestly is more valuable than hiding it.** The hyperparameter search finding only a noise-level improvement, and the monotonic constraints costing essentially nothing, were both reported as measured rather than framed to look more impressive than they were.
- **Recognizing you've over-extended a feature, and reverting it, is a normal and good part of iterating** — not a failure. Extending Postgres to every tab, then walking it back to chatbot-only once the real tradeoff was visible, is a concrete example of that.
- **Free-tier infrastructure has real, ongoing maintenance cost** (§11.4) — "free" doesn't mean "static," and a system depending on free-tier LLM APIs or a free-tier database needs graceful degradation built in from the start, not bolted on after the first outage.

---

## 13. Known limitations (say these proactively — it signals maturity, not weakness)

- Surrogate-tree business rules approximate the model's boundary but can disagree with it near split edges — an explanatory aid, not a replacement scoring engine (§7.5).
- `EXT_SOURCE_*` and `DAYS_BIRTH` (age) are top predictors; age-correlated features need fair-lending / disparate-impact review before any real production use.
- The chatbot's SQL validator is a token-scan safeguard, not a formal SQL parser — robust for the required query patterns, not proven against every conceivable malformed statement.
- Auxiliary tables are joined only as simple aggregates (count/sum/mean/max) — recency-weighted features ("delinquency in the last 6 months" vs. all-time) are a natural next step and would likely close more of the gap to heavily-tuned public Kaggle solutions.
- The bureau/previous-app lift (+0.008 ROC-AUC) was measured on a random split; credit data has real temporal structure, so a time-based split would be a more rigorous validation before trusting this for a materially different risk appetite.
- Only 4 of 319 features (post Milestone 1) have monotonic constraints — a deliberately conservative starting set, not an exhaustive fair-lending review.
- Categorical SHAP feature names currently render as generic sklearn names (e.g. `x1_F`) instead of `CODE_GENDER_F` — a known, unfixed cosmetic issue in `get_output_feature_names()`.
- The synthetic-data disclosure banner was explicitly removed — a deployed instance without real data now shows fabricated numbers with no on-screen indication (§9.6).
- Free-tier LLM model names need periodic re-checking against provider changes (§11.4) — this is an ongoing operational cost, not a one-time fix.

---

## 14. Likely interview questions and how to answer them

**"Tell me about your dataset."** → §5, start with §5.1 (why this dataset is hard — thin/no credit history) and §5.5 (the imbalance), then walk through the seven tables (§5.2) and pick 2–3 of the six insights (§5.6) that show clean signal (education, employment length) plus the one that's *not* monotonic (credit amount) to show you actually looked at the data rather than assuming linear relationships everywhere.

**"Walk me through your architecture."** → Use §4's tree, then narrate the data flow: CSVs → loader/preprocessor → LightGBM → joblib artifact → loaded by three independent consumers (predict, explain, rules) → surfaced in a 6-tab Streamlit app, with the chatbot as a parallel DuckDB-backed path.

**"Why did you build business rules at all — isn't SHAP enough?"** → §7.1: SHAP is per-applicant, a policy document needs a small global set of statements. Then walk through §7.3 (why trained on the model's predictions, not the real label) since that's the part people usually get wrong when explaining surrogate models.

**"Why LightGBM over XGBoost/CatBoost/Random Forest/logistic regression?"** → §10.1. Give the real measured logistic-regression numbers, and be honest that LightGBM vs. XGBoost is a "either would work" call, not a hard technical win.

**"Why SHAP over LIME?"** → §10.3 — the exactness argument for tree models specifically, not just "it was in the assignment."

**"Why DuckDB instead of just SQLite / Postgres from the start?"** → §10.4 — the `postgres`-attach extension is the actual deciding factor, not preference.

**"Why didn't you just use Streamlit Community Cloud, it's free and built for this?"** → §10.6 — Streamlit Cloud can't run a custom Docker image, and the assignment explicitly required Docker deployability.

**"Why synthetic data — isn't that misleading?"** → §9.5 in full, then be ready to discuss §9.6 (the banner removal) honestly if pushed further.

**"How do you know your model isn't overfit to one lucky split?"** → 5-fold CV, §6.6 point 2. Know the actual mean±std numbers.

**"How do you handle fairness/regulatory concerns?"** → Monotonic constraints (§6.5), and the honest limitation that only 4 features are covered so far — this is a start, not a finished fair-lending audit.

**"Tell me about a bug you had to debug."** → §11.2, the ThreadPoolExecutor story. It's specific, has a clear root cause, and shows you understand Python context-manager semantics, not just "I added a try/except."

**"What was the biggest challenge?"** → Pick from §11 (production incidents) or §11.4 (free-tier LLM churn) — both are concrete, neither is "the dataset was big" (true but generic).

**"What did you learn from this project?"** → §12 — lead with "works on my machine vs. works in production," it's the single most defensible, specific takeaway.

**"How would you scale this / what would you do with more time?"** → §13's limitations list, plus: recency-weighted aggregate features, a time-based validation split, extending monotonic constraints after a fuller fair-lending review, a full SQL parser instead of the token-scan validator.

**"What would you do differently?"** → The synthetic-banner removal (§9.6) is a real tension between "always deployable and demoable" and "never show fabricated data without saying so" — you made the call the way you were asked to, but you'd flag that tension earlier next time rather than build the transparency feature first and remove it later.

---

## 15. Free-tier constraints — a quick reference (in case asked directly)

- LLM: Groq free tier primary, Gemini free tier fallback. No OpenAI, no paid API anywhere.
- Database: DuckDB (embedded) + an optional free-tier Postgres (Render), never a paid database service.
- Hosting: Docker locally, or Render's free web-service tier for the live deployment (not Streamlit Community Cloud — see §10.6 for why).
- No credit card required anywhere in the stack, by design, from the very first line of the original assignment prompt.

---

## 16. How to use this document

This file is written to be self-contained — every number, every "why X not Y," and every incident story that's likely to come up is in here, with section references so you (or an LLM you feed this to) can jump straight to the relevant depth on follow-up questions.

If you want to go one level deeper than this document on any specific point:
- **`README.md`** — the primary technical documentation; exact CLI commands/flags, the full architecture diagram, and the complete evaluation tables in their original form.
- **`DESIGN_DECISIONS.md`** — a shorter, more clipped version of §10 above, useful if you want the "why" statements in a more compressed, skimmable form.
- The actual source files named throughout this document (e.g. `src/rules/rule_derivation.py`, `src/talk_to_data/query_runner.py`) — if you're asked to show code, not just describe it.

You don't need to feed all of those to an LLM alongside this file for a normal interview-prep pass — this document was written specifically to make that unnecessary. Pull in `README.md` only if you want exact command syntax memorized, or the source files only if you expect a live code-reading portion of the interview.
