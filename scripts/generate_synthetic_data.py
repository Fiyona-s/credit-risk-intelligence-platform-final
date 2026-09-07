"""Generates data/sample_applicants.csv — a SYNTHETIC demo dataset for the chatbot.

*** THIS FILE IS 100% SYNTHETIC / RANDOMLY GENERATED. ***
It is NOT extracted, sampled, or derived from the real Kaggle Home Credit
Default Risk dataset in any way — every value is drawn from a numpy random
generator with a fixed seed. Its only purpose is to let the Talk-to-Data
chatbot answer sensible-looking demo questions when an evaluator runs
`docker-compose up` without first downloading and placing the real Kaggle
CSVs in data/ (see src/talk_to_data/query_runner.py's fallback logic).

Column schema matches src.talk_to_data.query_runner.ALLOWED_COLUMNS exactly.
Distributions are calibrated to be *directionally* consistent with this
project's own documented real-dataset findings (README.md):
  - overall TARGET default rate ~8% (matches the real dataset's base rate)
  - applicants with bureau history default LESS than those without
    (7.73% vs 10.12%, per notebooks/eda.ipynb business insight #6)
  - income is right-skewed (lognormal), not derived from any real applicant

Run: python -m scripts.generate_synthetic_data
"""
import os

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 4000
OUTPUT_PATH = os.path.join("data", "sample_applicants.csv")

# Category proportions are plausible, round numbers chosen by inspection of
# the real dataset's public column descriptions — not copied real rows.
NAME_CONTRACT_TYPE_CHOICES = ["Cash loans", "Revolving loans"]
NAME_CONTRACT_TYPE_WEIGHTS = [0.90, 0.10]

CODE_GENDER_CHOICES = ["F", "M"]
CODE_GENDER_WEIGHTS = [0.66, 0.34]

NAME_EDUCATION_TYPE_CHOICES = [
    "Secondary / secondary special", "Higher education", "Incomplete higher",
    "Lower secondary", "Academic degree",
]
NAME_EDUCATION_TYPE_WEIGHTS = [0.71, 0.24, 0.03, 0.015, 0.005]

NAME_FAMILY_STATUS_CHOICES = [
    "Married", "Single / not married", "Civil marriage", "Separated", "Widow",
]
NAME_FAMILY_STATUS_WEIGHTS = [0.64, 0.15, 0.10, 0.06, 0.05]

OCCUPATION_TYPE_CHOICES = [
    None, "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
    "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
    "Cooking staff", "Cleaning staff", "Low-skill Laborers",
]
OCCUPATION_TYPE_WEIGHTS = [0.31, 0.18, 0.10, 0.09, 0.07, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.03]


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sk_id_curr = np.arange(900001, 900001 + n_rows)  # 9000xx range, disjoint from real SK_ID_CURR

    # --- Bureau / previous-application history flags, generated first ---
    # so TARGET can be made to depend on them in the direction our own EDA
    # (notebooks/eda.ipynb, insight #6) found in the real data.
    has_bureau_history = rng.random(n_rows) < 0.857  # matches real dataset's ~85.7% share
    has_prev_application = rng.random(n_rows) < 0.90

    # --- TARGET: depends on HAS_BUREAU_HISTORY, calibrated to the real
    # per-group default rates from notebooks/eda.ipynb insight #6 ---
    default_prob = np.where(has_bureau_history, 0.0773, 0.1012)
    target = (rng.random(n_rows) < default_prob).astype(int)

    # --- Demographics ---
    age_years = rng.uniform(21, 69, n_rows)
    days_birth = -(age_years * 365).astype(int)

    years_employed = np.clip(rng.exponential(6, n_rows), 0, age_years - 18)
    days_employed = -(years_employed * 365).astype(int)

    cnt_children = rng.poisson(0.4, n_rows)
    cnt_children = np.clip(cnt_children, 0, 8)

    code_gender = rng.choice(CODE_GENDER_CHOICES, n_rows, p=CODE_GENDER_WEIGHTS)
    name_contract_type = rng.choice(NAME_CONTRACT_TYPE_CHOICES, n_rows, p=NAME_CONTRACT_TYPE_WEIGHTS)
    name_education_type = rng.choice(NAME_EDUCATION_TYPE_CHOICES, n_rows, p=NAME_EDUCATION_TYPE_WEIGHTS)
    name_family_status = rng.choice(NAME_FAMILY_STATUS_CHOICES, n_rows, p=NAME_FAMILY_STATUS_WEIGHTS)
    occupation_type = rng.choice(OCCUPATION_TYPE_CHOICES, n_rows, p=OCCUPATION_TYPE_WEIGHTS)

    # --- Financials: right-skewed (lognormal), not sampled from real applicants ---
    amt_income_total = np.round(rng.lognormal(mean=np.log(140000), sigma=0.5, size=n_rows), -2)
    amt_income_total = np.clip(amt_income_total, 25650, 4_000_000)

    credit_to_income = rng.uniform(1.5, 6.0, n_rows)
    amt_credit = np.round(amt_income_total * credit_to_income, -2)

    loan_term_months = rng.uniform(10, 40, n_rows)
    amt_annuity = np.round(amt_credit / loan_term_months, 1)

    # --- Bureau-history aggregates (0 for applicants with no bureau history) ---
    bureau_count = np.where(has_bureau_history, rng.poisson(3, n_rows) + 1, 0)
    bureau_active_count = rng.binomial(bureau_count, 0.4)
    bureau_debt_sum_sum = np.where(
        has_bureau_history,
        np.round(rng.lognormal(mean=np.log(150000), sigma=1.1, size=n_rows), -2),
        0.0,
    )

    # --- Previous-application aggregates (0 for applicants with no prior applications) ---
    prev_app_count = np.where(has_prev_application, rng.poisson(2, n_rows) + 1, 0)
    prev_app_approval_rate = np.where(
        has_prev_application, np.round(rng.beta(5, 2, n_rows), 3), 0.0
    )
    prev_app_refused_count = rng.binomial(prev_app_count, 0.15)

    # Late-installment count: mildly higher for defaulters, directionally
    # consistent with a real credit-risk dataset (not calibrated to an exact
    # real statistic, unlike the bureau-history rates above).
    install_late_count = np.where(
        target == 1, rng.poisson(1.4, n_rows), rng.poisson(0.4, n_rows)
    )

    df = pd.DataFrame({
        "SK_ID_CURR": sk_id_curr,
        "TARGET": target,
        "NAME_CONTRACT_TYPE": name_contract_type,
        "CODE_GENDER": code_gender,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "NAME_EDUCATION_TYPE": name_education_type,
        "NAME_FAMILY_STATUS": name_family_status,
        "OCCUPATION_TYPE": occupation_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "BUREAU_COUNT": bureau_count,
        "BUREAU_ACTIVE_COUNT": bureau_active_count,
        "BUREAU_DEBT_SUM_SUM": bureau_debt_sum_sum,
        "HAS_BUREAU_HISTORY": has_bureau_history,
        "PREV_APP_COUNT": prev_app_count,
        "PREV_APP_APPROVAL_RATE": prev_app_approval_rate,
        "PREV_APP_REFUSED_COUNT": prev_app_refused_count,
        "HAS_PREV_APPLICATION": has_prev_application,
        "INSTALL_LATE_COUNT": install_late_count,
    })
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()

    with open(OUTPUT_PATH, "w") as f:
        f.write("# SYNTHETIC DEMO DATA - randomly generated, NOT from the real Kaggle dataset.\n")
        f.write("# Generated by scripts/generate_synthetic_data.py (fixed seed=42). See that file's\n")
        f.write("# header for details and calibration notes.\n")
        df.to_csv(f, index=False)

    print(f"Wrote {len(df)} synthetic rows to {OUTPUT_PATH}")
    print(f"\nOverall TARGET rate: {df['TARGET'].mean() * 100:.2f}%")
    print(df.groupby("HAS_BUREAU_HISTORY")["TARGET"].agg(default_rate_pct=lambda s: s.mean() * 100, count="count"))
