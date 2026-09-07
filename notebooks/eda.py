# %% [markdown]
# # Home Credit Default Risk — Exploratory Data Analysis
#
# Covers: dataset summary, feature categorization, missing-value analysis,
# TARGET class balance, and 5+ business insights with charts.

# %%
import os
import sys

# Works whether run as a script (cwd = project root) or as a notebook (cwd = notebooks/).
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
PROJECT_ROOT = _here if os.path.isdir(os.path.join(_here, "src")) else os.path.abspath(os.path.join(_here, ".."))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # so relative paths (data/, settings.data_dir) resolve the same as running from repo root

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import build_joined_dataset

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

ASSETS_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(ASSETS_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load data

# %%
df = build_joined_dataset(is_train=True)
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Dataset summary

# %%
print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
df.describe(include="all").T.head(20)

# %% [markdown]
# ## 3. Feature categorization
#
# Split columns into numeric, categorical, date-like, and id columns so
# downstream preprocessing knows how to treat each one.

# %%
id_cols = [c for c in df.columns if c.upper().endswith("_ID_CURR") or c.upper().endswith("_ID_PREV") or c.upper().endswith("_ID_BUREAU")]
date_like_cols = [c for c in df.columns if "DAYS_" in c.upper()]
categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns]
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in id_cols and c not in date_like_cols and c != "TARGET"]

print(f"ID columns: {len(id_cols)}")
print(f"Date-like (DAYS_*) columns: {len(date_like_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numeric feature columns: {len(numeric_cols)}")

# %% [markdown]
# ## 4. Missing-value analysis

# %%
missing = df.isna().mean().sort_values(ascending=False) * 100
missing = missing[missing > 0]
print(f"{len(missing)} columns have missing values")
missing.head(20)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
missing.head(20).plot(kind="barh", ax=ax)
ax.set_xlabel("% missing")
ax.set_title("Top 20 columns by missing-value rate")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_missing_values.png", dpi=100)
plt.show()

# %% [markdown]
# ## 5. TARGET class balance
#
# TARGET=1 means the applicant defaulted. This dataset is heavily imbalanced
# (~8% positive class), which drives the imbalance-handling choices in
# `src/ml/train.py` (scale_pos_weight / SMOTE) — plain accuracy would be
# misleading here (a model predicting "no default" for everyone scores ~92%).

# %%
target_counts = df["TARGET"].value_counts(normalize=True) * 100
print(target_counts)

fig, ax = plt.subplots(figsize=(5, 4))
df["TARGET"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["No Default (0)", "Default (1)"], rotation=0)
ax.set_title("TARGET class balance")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_target_balance.png", dpi=100)
plt.show()

# %% [markdown]
# ## 6. Business Insight 1 — Default rate by income bracket

# %%
df["INCOME_BRACKET"] = pd.qcut(df["AMT_INCOME_TOTAL"], q=5,
                                labels=["Lowest", "Low", "Mid", "High", "Highest"])
insight1 = df.groupby("INCOME_BRACKET", observed=True)["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
insight1.plot(kind="bar", ax=ax, color="#55A868")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by income bracket")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_insight1_income.png", dpi=100)
plt.show()
print(insight1)
# Insight: lower-income brackets show a meaningfully higher default rate,
# supporting income-to-credit ratio as a strong risk feature.

# %% [markdown]
# ## 7. Business Insight 2 — Default rate by employment length

# %%
df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365
df.loc[df["YEARS_EMPLOYED"] < 0, "YEARS_EMPLOYED"] = np.nan  # DAYS_EMPLOYED has a known 365243 anomaly for pensioners
df["EMPLOYMENT_BRACKET"] = pd.cut(
    df["YEARS_EMPLOYED"], bins=[-1, 1, 3, 7, 15, 100],
    labels=["<1yr", "1-3yr", "3-7yr", "7-15yr", "15yr+"]
)
insight2 = df.groupby("EMPLOYMENT_BRACKET", observed=True)["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
insight2.plot(kind="bar", ax=ax, color="#C44E52")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by employment length")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_insight2_employment.png", dpi=100)
plt.show()
print(insight2)
# Insight: shorter employment tenure correlates with higher default risk —
# job stability is a meaningful predictor.

# %% [markdown]
# ## 8. Business Insight 3 — Default rate by education level

# %%
insight3 = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100

fig, ax = plt.subplots(figsize=(8, 4))
insight3.plot(kind="barh", ax=ax, color="#8172B2")
ax.set_xlabel("Default rate (%)")
ax.set_title("Default rate by education level")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_insight3_education.png", dpi=100)
plt.show()
print(insight3)
# Insight: applicants with only lower secondary education default at roughly
# double the rate of those with higher education.

# %% [markdown]
# ## 9. Business Insight 4 — Default rate by credit amount band

# %%
df["CREDIT_BRACKET"] = pd.qcut(df["AMT_CREDIT"], q=5,
                                labels=["Lowest", "Low", "Mid", "High", "Highest"])
insight4 = df.groupby("CREDIT_BRACKET", observed=True)["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
insight4.plot(kind="bar", ax=ax, color="#CCB974")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by credit amount band")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_insight4_credit.png", dpi=100)
plt.show()
print(insight4)
# Insight: default rate is not monotonic with loan size alone — it interacts
# strongly with income, which is why credit-to-income ratio (engineered in
# preprocessor.py) is a stronger signal than either raw feature.

# %% [markdown]
# ## 10. Business Insight 5 — Default rate by age band

# %%
df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
df["AGE_BAND"] = pd.cut(df["AGE_YEARS"], bins=[20, 30, 40, 50, 60, 70],
                         labels=["20-30", "30-40", "40-50", "50-60", "60-70"])
insight5 = df.groupby("AGE_BAND", observed=True)["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
insight5.plot(kind="bar", ax=ax, color="#64B5CD")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by age band")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_insight5_age.png", dpi=100)
plt.show()
print(insight5)
# Insight: younger applicants (20-30) default noticeably more often than
# older applicants — age is a useful, if ethically sensitive, risk signal
# and should be reviewed for fair-lending compliance before production use.

# %% [markdown]
# ## 11. Business Insight 6 — Default rate by bureau-history status
#
# This insight is computed directly from the raw source CSVs (not the
# feature-engineered `df` used above) specifically to test, empirically,
# whether joining `bureau.csv` into the ML pipeline is justified by an actual
# difference in default behavior — rather than assuming it from general
# domain knowledge about the dataset.

# %%
app_raw = pd.read_csv(os.path.join(ASSETS_DIR, "application_train.csv"), usecols=["SK_ID_CURR", "TARGET"])
bureau_ids = pd.read_csv(os.path.join(ASSETS_DIR, "bureau.csv"), usecols=["SK_ID_CURR"])["SK_ID_CURR"].unique()

app_raw["HAS_BUREAU_HISTORY"] = app_raw["SK_ID_CURR"].isin(bureau_ids).map(
    {True: "Has bureau history", False: "No bureau history"}
)

insight6_grouped = app_raw.groupby("HAS_BUREAU_HISTORY")["TARGET"].agg(default_rate="mean", count="count")
insight6_grouped["default_rate_pct"] = insight6_grouped["default_rate"] * 100
insight6 = insight6_grouped["default_rate_pct"].reindex(["No bureau history", "Has bureau history"])

print(insight6_grouped[["count", "default_rate_pct"]])

fig, ax = plt.subplots(figsize=(6, 4))
insight6.plot(kind="bar", ax=ax, color=["#C44E52", "#4C72B0"])
ax.set_ylabel("Default rate (%)")
ax.set_xticklabels(insight6.index, rotation=0)
ax.set_title("Default rate: applicants with vs without bureau history")
plt.tight_layout()
plt.savefig(ASSETS_DIR + "/_eda_insight6_bureau_history.png", dpi=150)
plt.show()

no_bureau_rate = insight6["No bureau history"]
has_bureau_rate = insight6["Has bureau history"]
gap = no_bureau_rate - has_bureau_rate
no_bureau_n = insight6_grouped.loc["No bureau history", "count"]
has_bureau_n = insight6_grouped.loc["Has bureau history", "count"]

print(f"\nNo bureau history:  {no_bureau_rate:.2f}% default rate (n={no_bureau_n:,})")
print(f"Has bureau history: {has_bureau_rate:.2f}% default rate (n={has_bureau_n:,})")
print(f"Gap: {gap:.2f} percentage points")

# %% [markdown]
# **Insight:** Applicants with no external bureau history default at 10.12%
# vs 7.73% for those with bureau history — a gap of 2.39 percentage points,
# which is why bureau-derived features were added to the ML pipeline.

# %% [markdown]
# ## Summary of insights for the Streamlit EDA tab
# 1. Lower income brackets -> higher default rate
# 2. Shorter employment tenure -> higher default rate
# 3. Lower education level -> higher default rate
# 4. Credit amount alone is non-monotonic; ratio to income matters more
# 5. Younger applicants default more often than older applicants
# 6. Applicants with no bureau history default more (10.12%) than those with
#    bureau history (7.73%) — a 2.39pp gap justifying the bureau.csv join
