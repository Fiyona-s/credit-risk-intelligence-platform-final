# Credit Risk Intelligence Platform

## Why this exists

A bank that only outputs a probability of default from a black-box model creates two problems: credit analysts can't defend a rejection to a regulator or a customer, and the risk team can't audit whether the model is picking up on sensible signals or spurious correlations. This platform pairs a LightGBM default-risk model with **per-applicant SHAP explanations**, **plain-English business rules** distilled from the model, and a **talk-to-data chatbot** so risk analysts can interrogate the applicant portfolio in natural language without writing SQL — turning a prediction into something a credit policy team can actually act on and defend.

## Architecture

```
                        ┌─────────────────────────┐
                        │   Kaggle Home Credit     │
                        │   CSVs (data/)           │
                        │  application, bureau,    │
                        │  bureau_balance,         │
                        │  previous_application,   │
                        │  POS_CASH_balance,       │
                        │  credit_card_balance,    │
                        │  installments_payments   │
                        └────────────┬─────────────┘
                                     │
                              src/data/loader.py
                          (load raw application +
                           six auxiliary tables)
                                     │
                    src/data/preprocessor.py
              build_aggregate_features(df)
        (bureau/prev-app/POS/credit-card/installments ->
         one row per SK_ID_CURR, left-joined onto df;
         shared by train.py, predict.py, shap_explainer.py
         so train-time and inference-time features match exactly)
                                     │
                     ┌───────────────┼───────────────────┐
                     ▼               ▼                   ▼
        src/data/preprocessor.py   DuckDB table      notebooks/eda.ipynb
        (clean/impute/encode/      ("applicants")     (EDA + business
         engineer ratios)                ▲             insights)
                     │                   │
                     │      ┌────────────┴────────────┐
                     │      │  query_runner.py tries,  │
                     │      │  in order:               │
                     │      │  1. real join above      │
                     │      │  2. POSTGRES_URL, if set:│
                     │      │     DuckDB ATTACHes a    │
                     │      │     Postgres db live     │
                     │      │     (pre-loaded, once,   │
                     │      │     by scripts/load_     │
                     │      │     dataset_to_postgres  │
                     │      │     .py — deployment     │
                     │      │     only, optional)      │
                     │      │  3. data/sample_         │
                     │      │     applicants.csv       │
                     │      │     (committed,          │
                     │      │     SYNTHETIC — from     │
                     │      │     scripts/generate_    │
                     │      │     synthetic_data.py)   │
                     │      │  Any failure at step 2   │
                     │      │  falls through to step 3.│
                     │      └────────────┬────────────┘
                     │                   │
                     ▼                   ▼
             src/ml/train.py      src/talk_to_data/
        (LightGBM + scale_pos_    nl_to_sql.py + query_runner.py
         weight + monotonic         (NL -> SQL via Groq/Gemini,
         constraints on income/     few-shot examples + schema +
         EXT_SOURCE_*, saved to     rules in the prompt; validated,
         models/*.joblib.           executed on DuckDB; app.py shows
         --baseline/--cv/--tune     a banner when running on
         are opt-in reporting-      synthetic data)
         only side paths, never
         change the saved model)
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
  src/ml/predict.py  src/explainability/  src/rules/
  (risk score +       shap_explainer.py    rule_derivation.py
   band)              (per-prediction      (surrogate decision
                       SHAP explanation)    tree -> if/then rules)
        │            │             │
        └────────────┴─────────────┘
                     ▼
              app.py (Streamlit)
   EDA | Risk Prediction | Model Evaluation |
   Explainability | Business Rules | Chatbot
  (Model Evaluation: live ROC/PR curves + confusion
   matrix heatmap, reconstructing train.py's split.
   Explainability: per-applicant SHAP + a global
   SHAP summary/beeswarm plot across sampled rows.)
                     │
                     ▼
           Docker + docker-compose
    (single container; runs locally via
     `docker-compose up`, or deployed as a
     web service on Render — same image
     either way, POSTGRES_URL is the only
     deployment-specific config)
```

## Setup & run

The minimum path to a fully working app needs **no Kaggle dataset and no local Python install**:

1. `cp .env.example .env` and fill in `GROQ_API_KEY` (recommended, free tier at [console.groq.com](https://console.groq.com)) or `GEMINI_API_KEY` as a fallback.
2. ```
   docker-compose up --build
   ```
3. Open http://localhost:8501.

That's it. **EDA, Risk Prediction, Model Evaluation, Explainability, and Business Rules** all work immediately — they run off the pretrained model artifact already committed at `models/credit_risk_model.joblib`, no training step needed. **Chatbot** works immediately too: if `data/application_train.csv` isn't present, `src/talk_to_data/query_runner.py` automatically falls back to a small synthetic demo dataset (`data/sample_applicants.csv`, generated by `scripts/generate_synthetic_data.py`, committed to the repo) and the Chatbot tab shows a clear on-screen banner saying so — the numbers are illustrative, not real Kaggle statistics, but the interface and query patterns are fully demonstrable without downloading anything.

No paid services, no credit card, required anywhere in the stack.

### Using the real dataset / retraining (optional)

To train on the real data and get accurate chatbot answers instead of the synthetic fallback:

1. Download the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/data) from Kaggle and place the CSVs in `data/` (at minimum `application_train.csv`, `application_test.csv`; also `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv` for the richer joined features described below).
2. Retrain **inside the container** — no local Python/pandas/LightGBM install needed. The image's `ENTRYPOINT` runs Streamlit, so override it to run the training script instead:
   ```
   docker-compose run --rm --entrypoint python app -m src.ml.train
   ```
   This overwrites `models/credit_risk_model.joblib` with a model trained on the real data, using the exact scikit-learn/LightGBM versions pinned in `requirements.txt` — training locally with a different local Python environment can silently produce a model pickled with a mismatched scikit-learn version (you'll see `InconsistentVersionWarning` in the container logs if this happens).
3. Restart the app (`docker-compose restart`) or just reopen the Chatbot tab — `query_runner.py` detects `application_train.csv` is now present and automatically switches from the synthetic fallback to the real dataset, with no code changes needed. The synthetic-data banner disappears.

### Serving the chatbot from Postgres (optional, deployment only)

On a deployed instance (Render, Streamlit Cloud, etc.) the real CSV is never present — it's correctly gitignored, per the "never commit the dataset" rule — so a deployed chatbot always falls back to synthetic data by default. As a third, fully optional path, `query_runner.py` can instead have DuckDB **attach a Postgres database live** and query it directly, with no CSV file, no download step at cold start, and the dataset never touching git at any point:

1. Provision a free Postgres instance (e.g. Render's free Postgres add-on) and note its connection string.
2. From a machine that has the real Kaggle CSVs in `data/`, run the one-time load script locally (not in Docker, not automated — this is a manual, once-per-database step):
   ```
   python scripts/load_dataset_to_postgres.py --connection-string "postgresql://user:pass@host:5432/dbname"
   ```
   This reuses `build_joined_dataset()`, filters down to just the chatbot-facing columns (`query_runner.ALLOWED_COLUMNS`), and writes them into `pg_db.public.applicants` — no new Python dependency, DuckDB's `postgres` extension does the write.
3. Set `POSTGRES_URL` as a secret on your deployment platform (not in `.env` committed anywhere) and redeploy.

This is entirely optional — the app works correctly with `POSTGRES_URL` unset (real local CSV if present, else the synthetic fallback), and the real dataset never lives anywhere except Kaggle, your local `data/` folder, and the Postgres database you provision yourself. Any failure to connect (unset/wrong `POSTGRES_URL`, unreachable database, missing table) is caught and logged as a warning — the app always falls through cleanly to the synthetic fallback rather than crashing.

## Exploratory data analysis

`notebooks/eda.ipynb` (exported to `notebooks/eda.py`) covers dataset summary, feature categorization, missing-value analysis, `TARGET` class balance, and six business insights with charts:

1. Lower income brackets → higher default rate
2. Shorter employment tenure → higher default rate
3. Lower education level → higher default rate
4. Credit amount alone is non-monotonic; ratio to income matters more
5. Younger applicants default more often than older applicants
6. **Applicants with no bureau history default more often (10.12%) than those with bureau history (7.73%)** — a 2.39-percentage-point gap computed directly from `application_train.csv` joined against `bureau.csv`'s `SK_ID_CURR` values (`data/_eda_insight6_bureau_history.png`). This is the empirical basis for joining `bureau.csv` into the ML pipeline (see below) — not just an assumption from general domain knowledge about the dataset.

## Model selection & imbalance handling

- **Model**: LightGBM gradient-boosted trees. Chosen over logistic regression for native handling of missing values, mixed numeric/categorical signal after one-hot encoding, and strong out-of-the-box performance on tabular data at this scale (~300k rows, 166 columns after joining) without heavy tuning. This used to be an unsupported assertion — no baseline was ever actually trained. It's now backed by a real number: run `python -m src.ml.train --baseline` to train a `LogisticRegression(class_weight="balanced")` (features scaled with `StandardScaler`, since unlike LightGBM, logistic regression needs scaled inputs to converge cleanly) on the identical validation split. See the comparison table below — **the honest finding is that the gap is real but smaller than the confident tone of "chosen over logistic regression" might imply**, especially on ROC-AUC alone.
- **Imbalance**: `TARGET` is ~92% / 8% (see `notebooks/eda.ipynb`). Plain accuracy is meaningless here — a model that predicts "no default" for every applicant scores ~92%. We use LightGBM's `scale_pos_weight = negative_count / positive_count` as the primary strategy rather than SMOTE oversampling: with this many rows and a high-dimensional one-hot-encoded feature space, SMOTE's synthetic minority samples tend to land in unrealistic regions of feature space and mainly add training cost without improving ROC-AUC/PR-AUC over a correctly weighted loss. SMOTE remains available via `python -m src.ml.train --smote` for comparison. The logistic regression baseline above uses `class_weight="balanced"` for the same role.
- **Why bureau/previous-application features were added (empirical, not assumed)**: EDA insight #6 above shows applicants with no bureau history default at nearly 1.3× the rate of those with bureau history (10.12% vs 7.73%) — a real, measured gap in the raw data, not a general assumption about credit datasets. That gap motivated joining `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, and `installments_payments.csv` into the training pipeline (see "Bureau & previous-application history" below), which measurably improved ROC-AUC from 0.7735 to 0.7817.
- **Bureau & previous-application history**: beyond the single application-time snapshot, `src/data/preprocessor.py::build_aggregate_features` rolls up `bureau.csv` (+ `bureau_balance.csv`), `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, and `installments_payments.csv` — one row per `SK_ID_CURR` — and left-joins them onto the applicant table (44 new columns, prefixed by source table: `BUREAU_*`, `PREV_APP_*`, `POS_*`, `CC_*`, `INSTALL_*`). This is the single biggest lever for ROC-AUC on this dataset, since an applicant's actual repayment history is a much stronger signal than a one-time snapshot. Counts default to 0 and ratios/amounts default to their column median for applicants with no history at all, distinguished from genuine zero-risk history via `HAS_BUREAU_HISTORY`/`HAS_PREV_APPLICATION` boolean flags. The exact same function is called by `train.py`, `predict.py`, and `shap_explainer.py` so inference-time features always match training-time features. All `DAYS_*`/`MONTHS_BALANCE` columns in these auxiliary tables are ≤ 0 relative to the current application (i.e. historical by construction), so this introduces no target leakage. Disable with `python -m src.ml.train --no-bureau-features` for an ablation comparison.
- **Cross-validation (stability check)**: a single 80/20 split's ROC-AUC has no confidence interval on its own — it could be a lucky or unlucky split. `python -m src.ml.train --cv` runs 5-fold `StratifiedKFold` and reports mean ± std instead. See the evaluation section below; the std is tight enough that the single-split number isn't an outlier.
- **Hyperparameter tuning**: `python -m src.ml.train --tune` runs a bounded `RandomizedSearchCV` (`n_iter=15, cv=3`, scored on PR-AUC given the imbalance) over `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`. It found a `+0.0020` ROC-AUC improvement (`n_estimators=500, min_child_samples=100` vs. the current `n_estimators=300` default) — **smaller than the 5-fold CV's own fold-to-fold standard deviation (±0.0039)**, i.e. within noise. Current defaults (`n_estimators=300, learning_rate=0.05, num_leaves=31`) were kept rather than chasing a difference smaller than the model's own run-to-run variance. This is opt-in (`--tune`) and not part of the default training run, since a 15×3 search takes several minutes.
- **Monotonic constraints**: `AMT_INCOME_TOTAL` and `EXT_SOURCE_1`/`EXT_SOURCE_2`/`EXT_SOURCE_3` get a `monotone_constraints=-1` in the default LightGBM training run — higher income or a higher external credit-bureau score can never *increase* the model's predicted risk, regardless of what a tree happens to learn locally from noise in some region of feature space. This is standard, expected practice in credit risk scoring specifically: a regulator or an internal fair-lending review can reasonably ask "does raising this applicant's income ever make them look riskier to the model?", and for these four features the honest, defensible answer needs to be no. The `EXT_SOURCE_*` direction (-1 = higher is safer) was confirmed against this codebase's own derived rules before applying it (`src/rules/rule_derivation.py` output shows `EXT_SOURCE_3 > 0.55 → Low Risk`, `≤ 0.55 → High Risk`, i.e. higher is consistently safer). Deliberately **not** applied to every feature — e.g. `CNT_CHILDREN` and `DAYS_EMPLOYED` have plausible non-monotonic real-world relationships with default risk (very long tenure at a failing employer, very high dependents load at high income, etc.) and forcing a direction on them would be scientifically dishonest, not more rigorous. Measured cost of applying the four constraints: **essentially none** — see the before/after table below.

## Evaluation metrics & results

**PR-AUC is the primary metric for this problem**, not ROC-AUC — with ~92%/8% class imbalance, ROC-AUC can look deceptively strong (a model can rank well overall while still doing poorly on the minority default class that actually matters for a credit-risk screening tool), while PR-AUC is computed entirely from precision and recall on the positive (default) class and better reflects real screening performance. Both numbers are reported throughout; PR-AUC is just presented first.

**Model comparison** (same stratified 80/20 validation split, `random_state=42`):

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **LightGBM** (deployed, with monotonic constraints) | **0.2760** | **0.7817** |
| Logistic Regression (baseline, `--baseline`) | 0.2445 | 0.7648 |

LightGBM wins on both metrics, more clearly so on PR-AUC (+0.0315) than on ROC-AUC alone (+0.0169) — a properly-scaled logistic regression is a closer baseline than the "chosen over logistic regression" framing might suggest, and PR-AUC (the metric that matters most given the imbalance) is where the gap is more meaningful. Report this honestly rather than only the flattering framing.

**5-fold cross-validation (stability check)**, `python -m src.ml.train --cv`, application+bureau features, LightGBM defaults, no monotonic constraints applied in this path:

| Metric | Mean ± std across 5 folds |
|---|---|
| PR-AUC | 0.2679 ± 0.0060 |
| ROC-AUC | 0.7790 ± 0.0039 |

The single 80/20 split's numbers (below) fall within about one fold-to-fold standard deviation of these means — the split wasn't a lucky or unlucky draw.

**Before/after monotonic constraints** (application + bureau/previous-app features, same 80/20 split), before vs. after adding the bureau/previous-app aggregates, then before vs. after adding monotonic constraints on top:

| Metric | Application-level only | + Bureau & previous-app history | + Monotonic constraints (deployed) |
|---|---|---|---|
| PR-AUC | 0.2667 | 0.2761 | **0.2760** (−0.0001) |
| ROC-AUC | 0.7735 | 0.7814 | **0.7817** (+0.0003) |
| Precision @ 0.5 | 0.178 | 0.184 | 0.1845 |
| Recall @ 0.5 | 0.684 | 0.695 | 0.6959 |
| F1 @ 0.5 | 0.283 | 0.291 | 0.2916 |

Monotonic constraints typically cost a small amount of raw performance in exchange for interpretability/regulatory defensibility — here the cost is effectively zero (within noise, PR-AUC moved by -0.0001 and ROC-AUC by +0.0003), so this is a case where the fair-lending defensibility came free rather than requiring a real tradeoff. That won't always be true on a different dataset or feature set; it should still be checked, not assumed, each time constraints are added or changed. Run `python -m src.ml.evaluate` after training to print the full top-20 feature-importance list for the current model.

Confusion matrix @ threshold 0.5, current deployed model (rows = actual, cols = predicted):

```
              Pred: No Default   Pred: Default
Actual: No Default   41,265          15,273
Actual: Default       1,510           3,455
```

At the default 0.5 threshold the model favors recall (catching ~70% of actual defaulters) at the cost of precision — appropriate for a screening tool where missing a defaulter is costlier than a false flag that goes to manual review. The threshold is adjustable per business risk appetite via `RISK_THRESHOLD_LOW`/`RISK_THRESHOLD_MEDIUM` in `.env`, which control the Low/Medium/High risk-band cutoffs used in `src/ml/predict.py`.

## Prompt engineering & token optimization

- **Provider**: [Groq](https://groq.com) free tier (`openai/gpt-oss-20b` by default) is tried first for its fast inference and generous free rate limits, ideal for an interactive chat UI. [Google Gemini](https://ai.google.dev) free tier (`gemini-flash-latest`) is the automatic fallback if `GROQ_API_KEY` is unset or the Groq call fails — see `src/talk_to_data/nl_to_sql.py::call_llm`. Both providers rotate/rename their free-tier models fairly often; if either fails with a "model not found" or "no longer available" error, check `GROQ_MODEL`/`GEMINI_MODEL` in `.env` against the provider's current model list.
- **Token optimization**: the system prompt (`src/talk_to_data/prompt_templates.py`) sends only a compact column-level schema description (a curated ~20 columns, one line each — including a handful of the new bureau/previous-application aggregates) rather than sample rows or the full 166-column joined dataset — this keeps the prompt short and cheap while still giving the model everything needed to write correct SQL. Conversation memory is capped to the last 3 turns to bound context growth on long chat sessions. `SYSTEM_PROMPT_V1` also includes 4 few-shot question→SQL examples (`FEW_SHOT_EXAMPLES`, matching the "Example queries" section above), adding roughly ~190 tokens to every call — a deliberate small, fixed cost traded for meaningfully more consistent SQL on ambiguous phrasing (e.g. "risky applicants," "compare genders") that the schema and rules alone don't fully disambiguate.
- **Guardrails against hallucination**: `src/talk_to_data/query_runner.py` parses the generated SQL with `sqlparse`, rejects anything that isn't a single `SELECT`, rejects any referenced table other than `applicants`, and rejects any referenced column not in the known schema. If validation fails, `nl_to_sql.py` retries once with the error fed back to the LLM; if the retry also fails, it returns a graceful "couldn't answer" message instead of ever executing unvalidated SQL. A `LIMIT` is enforced on every query (`MAX_QUERY_ROWS` in `.env`, default 200).

### Example queries

Five representative query patterns the chatbot handles end-to-end (natural language → validated SQL → plain-language answer). Exact figures depend on which data source is loaded — the real Kaggle dataset if present, otherwise the committed synthetic fallback (`data/sample_applicants.csv`) — see "Setup & run" above.

1. **Simple aggregate** — *"What is the average income of applicants who defaulted?"*
   ```sql
   SELECT AVG(AMT_INCOME_TOTAL) AS avg_income FROM applicants WHERE TARGET = 1
   ```
   → *"Applicants who defaulted had an average annual income of approximately ₹X, compared to the overall applicant base."*

2. **Group-by rate comparison** — *"What's the default rate by education level?"*
   ```sql
   SELECT NAME_EDUCATION_TYPE, ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
   FROM applicants
   GROUP BY NAME_EDUCATION_TYPE
   ORDER BY default_rate_pct DESC
   ```
   → *"Applicants with only secondary education default at the highest rate, while those with an academic degree default least often — consistent with EDA insight #3."*

3. **Filter + count** — *"How many applicants have more than 2 children and a credit amount over 1,000,000?"*
   ```sql
   SELECT COUNT(*) AS applicant_count
   FROM applicants
   WHERE CNT_CHILDREN > 2 AND AMT_CREDIT > 1000000
   ```
   → *"There are N applicants matching both conditions."*

4. **Boolean-flag / engineered-feature query** — *"Compare the default rate for applicants with and without bureau history."*
   ```sql
   SELECT HAS_BUREAU_HISTORY, ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
   FROM applicants
   GROUP BY HAS_BUREAU_HISTORY
   ```
   → *"Applicants with no bureau history default more often than those with a bureau history — the same directional gap documented in EDA insight #6."*

5. **Conversational follow-up** (tests 3-turn memory) — Turn 1: *"What's the average credit amount by gender?"* Turn 2 (follow-up, no repeated context): *"And just for those with children?"*
   ```sql
   -- Turn 2 SQL, resolved using conversation history from Turn 1
   SELECT CODE_GENDER, AVG(AMT_CREDIT) AS avg_credit
   FROM applicants
   WHERE CNT_CHILDREN > 0
   GROUP BY CODE_GENDER
   ```
   → *"Demonstrates that `nl_to_sql.py`'s conversation history (last 3 turns) lets the model correctly resolve "those" back to applicants, without the user having to restate the full question."*

## Rule derivation logic & sample output

`src/rules/rule_derivation.py` trains a shallow (`max_depth=4`) `DecisionTreeClassifier` **surrogate** on the trained LightGBM model's own predictions (not the ground-truth `TARGET`) — this approximates the model's decision boundary in a form readable without any ML background, rather than trying to summarize an ensemble of hundreds of trees directly.

Sample derived rules:

```
IF EXT_SOURCE_3 <= 0.55 AND EXT_SOURCE_2 <= 0.52 AND EXT_SOURCE_1 <= 0.62 AND EXT_SOURCE_2 <= 0.35
   THEN predicted outcome = High Risk (likely default)

IF EXT_SOURCE_3 > 0.55 AND EXT_SOURCE_2 <= 0.42 AND EXT_SOURCE_2 > 0.15 AND DAYS_EMPLOYED <= -1624.50
   THEN predicted outcome = Low Risk (likely repay)

IF EXT_SOURCE_3 > 0.55 AND EXT_SOURCE_2 > 0.42 AND EXT_SOURCE_1 <= 0.28 AND ANNUITY_INCOME_RATIO > 0.15
   THEN predicted outcome = High Risk (likely default)
```

(`EXT_SOURCE_1/2/3` are the Home Credit dataset's normalized external credit-bureau scores — consistently the strongest predictors in this dataset, which the surrogate correctly surfaces as the top splits.)

## Known limitations & future improvements

- The surrogate-tree rules approximate the model's boundary but are not the model itself — at rule boundaries the tree's decision can disagree with the underlying LightGBM prediction; treat rules as an explanatory aid, not a replacement scoring engine.
- `EXT_SOURCE_*` and `DAYS_BIRTH` (age) are top predictors; age-correlated features should be reviewed for fair-lending / disparate-impact compliance before any production use.
- The chatbot's SQL validator is a token-scan safeguard (via `sqlparse`), not a full SQL parser — it covers the required query patterns robustly but is not guaranteed to catch every conceivable malformed statement.
- All six auxiliary Home Credit tables are now joined (`bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments`), but only as simple per-applicant rollups (count/sum/mean/max). Recency-weighted aggregates (e.g. "delinquency in the last 6 months" vs. all-time) and second-order features (trend of utilization over time, not just its mean) are the natural next layer and would likely close more of the gap to heavily-tuned public Kaggle solutions.
- The bureau/previous-application lift measured here (+0.008 ROC-AUC) is real but modest; before relying on it for a materially different risk appetite, re-validate on a held-out time-based split rather than a random split, since credit data has strong temporal structure that a random 80/20 split doesn't fully test for.
- A bounded hyperparameter search has now been run (`python -m src.ml.train --tune`, 15-iteration `RandomizedSearchCV`) — it found only a +0.0020 ROC-AUC improvement, smaller than the model's own 5-fold cross-validation noise (±0.0039), so the original defaults were kept rather than chasing that difference. A wider or Bayesian search (e.g. `optuna`) over more iterations/a larger parameter space might still find a real improvement past what this bounded search covered.
- Monotonic constraints are applied to only 4 of 291 trained features (`AMT_INCOME_TOTAL`, `EXT_SOURCE_1/2/3`) — a deliberately conservative starting set. Other features with a plausible case for constraining (e.g. `CREDIT_TERM`, `ANNUITY_INCOME_RATIO`) were left unconstrained pending a closer domain review, rather than extending the constraint set speculatively.
