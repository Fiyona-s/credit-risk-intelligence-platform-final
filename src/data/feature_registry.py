"""Feature metadata registry for the credit-behavior feature layer (src/data/preprocessor.py).

Every entry here corresponds to a feature actually produced by
build_bureau_features / build_previous_application_features / build_pos_cash_features
/ build_credit_card_features / build_installment_features / engineer_features, or a
history-presence flag those functions set. This file does not invent or aspire to
features that don't exist in code — see test_feature_registry_matches_generated_columns
in tests/test_feature_engineering.py, which fails if the registry and the actual
generated columns drift apart.

Raw application-table columns passed through unchanged (e.g. AMT_INCOME_TOTAL,
CODE_GENDER) are NOT registered here — this registry covers only features this
project's own code derives/engineers, which is the part future work (leakage
validation, counterfactual "actionable feature" selection) actually needs metadata
for.

Fields per entry:
  name             - column name as it appears in the joined/engineered DataFrame
  group            - credit_history | previous_application_behavior | pos_cash_behavior
                     | credit_card_behavior | installment_repayment_behavior | ratio
  source           - raw CSV(s) the feature is derived from
  calculation      - one-line description of the aggregation/formula
  business_meaning - what this tells you about credit risk
  dtype            - int | float | bool
  historical       - True if derived purely from historical (pre-application) records
                     (all of these are, by construction — see preprocessor.py's module
                     docstring on DAYS_*/MONTHS_BALANCE columns)
  actionable       - could an applicant realistically change this before a re-application
                     (used by src/explainability/counterfactual.py in a later milestone
                     to restrict which features a counterfactual is allowed to vary)
  leakage_risk     - Low | Medium | High — Low for every entry here, since all source
                     columns are confirmed historical/pre-application; recorded
                     explicitly rather than assumed, for src/data/feature_validation.py
                     (a later milestone) to check against.
"""
from typing import List

FEATURE_REGISTRY: List[dict] = [
    # ---- credit_history (bureau.csv + bureau_balance.csv) ----
    {
        "name": "BUREAU_COUNT", "group": "credit_history", "source": "bureau",
        "calculation": "count of bureau credit records per applicant",
        "business_meaning": "Breadth of external credit history",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_ACTIVE_COUNT", "group": "credit_history", "source": "bureau",
        "calculation": "count of bureau records with CREDIT_ACTIVE == 'Active'",
        "business_meaning": "Number of currently open external credit lines",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_CLOSED_COUNT", "group": "credit_history", "source": "bureau",
        "calculation": "count of bureau records with CREDIT_ACTIVE == 'Closed'",
        "business_meaning": "Number of fully repaid/closed external credit lines",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_OVERDUE_ACCOUNT_COUNT", "group": "credit_history", "source": "bureau",
        "calculation": "count of bureau records with AMT_CREDIT_SUM_OVERDUE > 0",
        "business_meaning": "Number of historical credit accounts showing delinquency",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_ACTIVE_RATIO", "group": "credit_history", "source": "bureau",
        "calculation": "BUREAU_ACTIVE_COUNT / BUREAU_COUNT",
        "business_meaning": "Share of external credit lines still open vs. fully repaid",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DELINQUENCY_RATE", "group": "credit_history", "source": "bureau",
        "calculation": "BUREAU_OVERDUE_ACCOUNT_COUNT / BUREAU_COUNT",
        "business_meaning": "Share of external credit lines with any overdue balance",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_CREDIT_SUM_SUM", "group": "credit_history", "source": "bureau",
        "calculation": "sum of AMT_CREDIT_SUM across bureau records",
        "business_meaning": "Total credit ever extended by other lenders",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_CREDIT_SUM_MEAN", "group": "credit_history", "source": "bureau",
        "calculation": "mean of AMT_CREDIT_SUM across bureau records",
        "business_meaning": "Average size of external credit lines",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_CREDIT_SUM_MAX", "group": "credit_history", "source": "bureau",
        "calculation": "max of AMT_CREDIT_SUM across bureau records",
        "business_meaning": "Largest single external credit line",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DEBT_SUM_SUM", "group": "credit_history", "source": "bureau",
        "calculation": "sum of AMT_CREDIT_SUM_DEBT across bureau records",
        "business_meaning": "Total outstanding debt across all external credit lines",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DEBT_SUM_MEAN", "group": "credit_history", "source": "bureau",
        "calculation": "mean of AMT_CREDIT_SUM_DEBT across bureau records",
        "business_meaning": "Average outstanding debt per external credit line",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DEBT_SUM_MAX", "group": "credit_history", "source": "bureau",
        "calculation": "max of AMT_CREDIT_SUM_DEBT across bureau records",
        "business_meaning": "Largest single outstanding external debt",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_OVERDUE_SUM", "group": "credit_history", "source": "bureau",
        "calculation": "sum of AMT_CREDIT_SUM_OVERDUE across bureau records",
        "business_meaning": "Total overdue amount across all external credit lines",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_OVERDUE_MEAN", "group": "credit_history", "source": "bureau",
        "calculation": "mean of AMT_CREDIT_SUM_OVERDUE across bureau records",
        "business_meaning": "Average overdue amount per external credit line",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_OVERDUE_MAX", "group": "credit_history", "source": "bureau",
        "calculation": "max of AMT_CREDIT_SUM_OVERDUE across bureau records",
        "business_meaning": "Worst single overdue amount on record",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DAY_OVERDUE_MEAN", "group": "credit_history", "source": "bureau",
        "calculation": "mean of CREDIT_DAY_OVERDUE across bureau records",
        "business_meaning": "Average days-past-due across external credit lines",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DAY_OVERDUE_MAX", "group": "credit_history", "source": "bureau",
        "calculation": "max of CREDIT_DAY_OVERDUE across bureau records",
        "business_meaning": "Worst single days-past-due on record",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_UTILIZATION_MEAN", "group": "credit_history", "source": "bureau",
        "calculation": "mean of (AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM) across bureau records",
        "business_meaning": "Average historical utilization of external credit lines",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_CREDIT_DURATION_MEAN", "group": "credit_history", "source": "bureau",
        "calculation": "mean of -DAYS_CREDIT (days since each bureau credit was opened)",
        "business_meaning": "Average age of external credit lines — longer history is typically lower risk",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_BAL_WORST_STATUS", "group": "credit_history", "source": "bureau_balance",
        "calculation": "max of ordinal-encoded monthly STATUS (0=current .. 5=worst DPD bucket)",
        "business_meaning": "Worst monthly delinquency bucket ever reached on any external credit line",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_BAL_DPD_MONTHS", "group": "credit_history", "source": "bureau_balance",
        "calculation": "sum of months flagged as any-DPD status across all bureau records, all-time",
        "business_meaning": "Total months of delinquency across external credit lines' full history",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_BAL_DPD_MONTHS_RECENT_6M", "group": "credit_history", "source": "bureau_balance",
        "calculation": "sum of months flagged as any-DPD status, restricted to MONTHS_BALANCE >= -6",
        "business_meaning": "Recent (last ~6 months) delinquency months — distinguishes recent from historical behavior",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "HAS_BUREAU_HISTORY", "group": "credit_history", "source": "bureau",
        "calculation": "True if the applicant has any bureau record at all (before fill)",
        "business_meaning": "Distinguishes 'no external credit history' from 'history exists with zero-valued fields'",
        "dtype": "bool", "historical": True, "actionable": False, "leakage_risk": "Low",
    },

    # ---- previous_application_behavior (previous_application.csv) ----
    {
        "name": "PREV_APP_COUNT", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "count of previous applications with this lender",
        "business_meaning": "Depth of prior relationship with this lender",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_APPROVED_COUNT", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "count of previous applications with NAME_CONTRACT_STATUS == 'Approved'",
        "business_meaning": "Number of previously approved applications",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_APPROVAL_RATE", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "mean of (NAME_CONTRACT_STATUS == 'Approved') across previous applications",
        "business_meaning": "Historical approval rate with this lender",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_REFUSED_COUNT", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "count of previous applications with NAME_CONTRACT_STATUS == 'Refused'",
        "business_meaning": "Number of previously refused applications",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_REJECTION_RATE", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "PREV_APP_REFUSED_COUNT / PREV_APP_COUNT",
        "business_meaning": "Historical rejection rate with this lender",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_AMT_APPLICATION_MEAN", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "mean of AMT_APPLICATION across previous applications",
        "business_meaning": "Average amount previously requested",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_AMT_APPLICATION_MAX", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "max of AMT_APPLICATION across previous applications",
        "business_meaning": "Largest amount previously requested",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_AMT_CREDIT_MEAN", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "mean of AMT_CREDIT across previous applications",
        "business_meaning": "Average amount previously extended",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_AMT_CREDIT_MAX", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "max of AMT_CREDIT across previous applications",
        "business_meaning": "Largest amount previously extended",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_AMT_ANNUITY_MEAN", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "mean of AMT_ANNUITY across previous applications",
        "business_meaning": "Average previous loan annuity",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_AMT_ANNUITY_MAX", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "max of AMT_ANNUITY across previous applications",
        "business_meaning": "Largest previous loan annuity",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APP_CNT_PAYMENT_MEAN", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "mean of CNT_PAYMENT (planned installment count) across previous applications",
        "business_meaning": "Average previous loan term length",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "HAS_PREV_APPLICATION", "group": "previous_application_behavior", "source": "previous_application",
        "calculation": "True if the applicant has any previous_application record at all (before fill)",
        "business_meaning": "Distinguishes 'no prior relationship with this lender' from 'history exists with zero-valued fields'",
        "dtype": "bool", "historical": True, "actionable": False, "leakage_risk": "Low",
    },

    # ---- pos_cash_behavior (POS_CASH_balance.csv) ----
    {
        "name": "POS_COUNT", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "nunique SK_ID_PREV across POS/cash balance records",
        "business_meaning": "Number of previous POS/cash loans",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_ACTIVE_COUNT", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "count of POS/cash balance rows with NAME_CONTRACT_STATUS == 'Active'",
        "business_meaning": "Number of currently active POS/cash loan months on record",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_COMPLETED_COUNT", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "count of POS/cash balance rows with NAME_CONTRACT_STATUS == 'Completed'",
        "business_meaning": "Number of POS/cash loan months that reached completion",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_OVERDUE_COUNT", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "count of POS/cash balance rows with SK_DPD > 0",
        "business_meaning": "Number of months a previous POS/cash loan was overdue",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_CNT_INSTALMENT_FUTURE_MEAN", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "mean of CNT_INSTALMENT_FUTURE across POS/cash balance records",
        "business_meaning": "Average remaining installments on previous POS/cash loans",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_SK_DPD_MEAN", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "mean of SK_DPD across POS/cash balance records",
        "business_meaning": "Average days-past-due on previous POS/cash loans",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_SK_DPD_MAX", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "max of SK_DPD across POS/cash balance records",
        "business_meaning": "Worst days-past-due on previous POS/cash loans",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_SK_DPD_DEF_MEAN", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "mean of SK_DPD_DEF (DPD with tolerance) across POS/cash balance records",
        "business_meaning": "Average tolerance-adjusted days-past-due on previous POS/cash loans",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "POS_SK_DPD_DEF_MAX", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "max of SK_DPD_DEF across POS/cash balance records",
        "business_meaning": "Worst tolerance-adjusted days-past-due on previous POS/cash loans",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "HAS_POS_HISTORY", "group": "pos_cash_behavior", "source": "POS_CASH_balance",
        "calculation": "True if the applicant has any POS_CASH_balance record at all (before fill)",
        "business_meaning": "Distinguishes 'no POS/cash loan history' from 'history exists with zero-valued fields'",
        "dtype": "bool", "historical": True, "actionable": False, "leakage_risk": "Low",
    },

    # ---- credit_card_behavior (credit_card_balance.csv) ----
    {
        "name": "CC_COUNT", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "nunique SK_ID_PREV across credit card balance records",
        "business_meaning": "Number of previous credit cards",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_BALANCE_MEAN", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "mean of AMT_BALANCE across credit card balance records",
        "business_meaning": "Average outstanding credit card balance",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_BALANCE_MAX", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "max of AMT_BALANCE across credit card balance records",
        "business_meaning": "Highest outstanding credit card balance on record",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_LIMIT_MEAN", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "mean of AMT_CREDIT_LIMIT_ACTUAL across credit card balance records",
        "business_meaning": "Average credit limit extended",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_LIMIT_MAX", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "max of AMT_CREDIT_LIMIT_ACTUAL across credit card balance records",
        "business_meaning": "Highest credit limit ever extended",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_UTILIZATION_MEAN", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "mean of (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL) across credit card balance records",
        "business_meaning": "Average balance-to-limit ratio (utilization) — a well-known credit-risk signal",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_UTILIZATION_MAX", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "max of (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL) across credit card balance records",
        "business_meaning": "Highest historical utilization reached",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_UTILIZATION_MEAN_RECENT_6M", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "mean utilization restricted to MONTHS_BALANCE >= -6",
        "business_meaning": "Recent (last ~6 months) credit card utilization vs. the all-time average",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_PAYMENT_BALANCE_RATIO_MEAN", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "mean of (AMT_PAYMENT_TOTAL_CURRENT / AMT_BALANCE) across credit card balance records",
        "business_meaning": "How much of the outstanding balance is typically paid off each period",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_OVERDUE_COUNT", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "count of credit card balance rows with SK_DPD > 0",
        "business_meaning": "Number of months a previous credit card was overdue",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_SK_DPD_MEAN", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "mean of SK_DPD across credit card balance records",
        "business_meaning": "Average days-past-due on previous credit cards",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_SK_DPD_MAX", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "max of SK_DPD across credit card balance records",
        "business_meaning": "Worst days-past-due on previous credit cards",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_SK_DPD_DEF_MEAN", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "mean of SK_DPD_DEF across credit card balance records",
        "business_meaning": "Average tolerance-adjusted days-past-due on previous credit cards",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "CC_SK_DPD_DEF_MAX", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "max of SK_DPD_DEF across credit card balance records",
        "business_meaning": "Worst tolerance-adjusted days-past-due on previous credit cards",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "HAS_CC_HISTORY", "group": "credit_card_behavior", "source": "credit_card_balance",
        "calculation": "True if the applicant has any credit_card_balance record at all (before fill)",
        "business_meaning": "Distinguishes 'no credit card history' from 'history exists with zero-valued fields'",
        "dtype": "bool", "historical": True, "actionable": False, "leakage_risk": "Low",
    },

    # ---- installment_repayment_behavior (installments_payments.csv) ----
    {
        "name": "INSTALL_COUNT", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "count of installment records",
        "business_meaning": "Number of historical installments on record",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_DAYS_LATE_MEAN", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "mean of (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) across installment records",
        "business_meaning": "Average payment delay in days (negative = paid early)",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_DAYS_LATE_MAX", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "max of (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) across installment records",
        "business_meaning": "Worst single payment delay in days",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_UNDERPAY_RATIO_MEAN", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "mean of (AMT_PAYMENT / AMT_INSTALMENT) across installment records",
        "business_meaning": "Average share of the scheduled installment actually paid",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_UNDERPAY_RATIO_MIN", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "min of (AMT_PAYMENT / AMT_INSTALMENT) across installment records",
        "business_meaning": "Worst single underpayment share",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_LATE_COUNT", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "count of installments paid after their due date, all-time",
        "business_meaning": "Total number of late payments on record",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_LATE_RATIO", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "INSTALL_LATE_COUNT / INSTALL_COUNT",
        "business_meaning": "Share of historical installments paid late",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_LATE_COUNT_RECENT_6M", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "count of late installments restricted to DAYS_INSTALMENT >= -180",
        "business_meaning": "Recent (last ~6 months) late-payment count vs. the all-time count",
        "dtype": "int", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_COMPLETION_RATIO", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "share of installment records with a non-null AMT_PAYMENT",
        "business_meaning": "Proxy for how consistently scheduled installments were actually paid",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "INSTALL_PAYMENT_SHORTFALL_MEAN", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "mean of (AMT_INSTALMENT - AMT_PAYMENT) across installment records",
        "business_meaning": "Average shortfall between what was owed and what was paid per installment",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "HAS_INSTALLMENT_HISTORY", "group": "installment_repayment_behavior", "source": "installments_payments",
        "calculation": "True if the applicant has any installments_payments record at all (before fill)",
        "business_meaning": "Distinguishes 'no installment history' from 'history exists with zero-valued fields'",
        "dtype": "bool", "historical": True, "actionable": False, "leakage_risk": "Low",
    },

    # ---- ratio (engineer_features — second-order, need both applicant + aggregate columns) ----
    {
        "name": "CREDIT_INCOME_RATIO", "group": "ratio", "source": "application (AMT_CREDIT, AMT_INCOME_TOTAL)",
        "calculation": "AMT_CREDIT / AMT_INCOME_TOTAL",
        "business_meaning": "Requested credit relative to income — affordability signal",
        "dtype": "float", "historical": False, "actionable": True, "leakage_risk": "Low",
    },
    {
        "name": "ANNUITY_INCOME_RATIO", "group": "ratio", "source": "application (AMT_ANNUITY, AMT_INCOME_TOTAL)",
        "calculation": "AMT_ANNUITY / AMT_INCOME_TOTAL",
        "business_meaning": "Loan repayment burden relative to income",
        "dtype": "float", "historical": False, "actionable": True, "leakage_risk": "Low",
    },
    {
        "name": "CREDIT_TERM", "group": "ratio", "source": "application (AMT_ANNUITY, AMT_CREDIT)",
        "calculation": "AMT_ANNUITY / AMT_CREDIT",
        "business_meaning": "Implied loan repayment rate (inverse of loan term)",
        "dtype": "float", "historical": False, "actionable": True, "leakage_risk": "Low",
    },
    {
        "name": "DAYS_EMPLOYED_RATIO", "group": "ratio", "source": "application (DAYS_EMPLOYED, DAYS_BIRTH)",
        "calculation": "DAYS_EMPLOYED / DAYS_BIRTH",
        "business_meaning": "Employment tenure relative to age — job stability signal",
        "dtype": "float", "historical": False, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DEBT_CREDIT_RATIO", "group": "ratio", "source": "bureau (applicant-level sums)",
        "calculation": "BUREAU_DEBT_SUM_SUM / BUREAU_CREDIT_SUM_SUM",
        "business_meaning": "Overall external debt-to-credit ratio across all bureau lines combined",
        "dtype": "float", "historical": True, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "BUREAU_DEBT_INCOME_RATIO", "group": "ratio", "source": "bureau + application",
        "calculation": "BUREAU_DEBT_SUM_SUM / AMT_INCOME_TOTAL",
        "business_meaning": "Total external debt relative to income",
        "dtype": "float", "historical": False, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_CREDIT_INCOME_RATIO", "group": "ratio", "source": "previous_application + application",
        "calculation": "PREV_APP_AMT_CREDIT_MEAN / AMT_INCOME_TOTAL",
        "business_meaning": "Average previous credit amount relative to current income",
        "dtype": "float", "historical": False, "actionable": False, "leakage_risk": "Low",
    },
    {
        "name": "PREV_APPLICATION_INCOME_RATIO", "group": "ratio", "source": "previous_application + application",
        "calculation": "PREV_APP_AMT_APPLICATION_MEAN / AMT_INCOME_TOTAL",
        "business_meaning": "Average previous requested amount relative to current income",
        "dtype": "float", "historical": False, "actionable": False, "leakage_risk": "Low",
    },
]


def get_registered_feature_names() -> List[str]:
    return [entry["name"] for entry in FEATURE_REGISTRY]


def get_features_by_group(group: str) -> List[dict]:
    return [entry for entry in FEATURE_REGISTRY if entry["group"] == group]


def get_actionable_features() -> List[dict]:
    """Features flagged actionable=True — the only ones a counterfactual explanation
    (a later milestone) is allowed to vary."""
    return [entry for entry in FEATURE_REGISTRY if entry["actionable"]]


if __name__ == "__main__":
    from collections import Counter

    groups = Counter(entry["group"] for entry in FEATURE_REGISTRY)
    print(f"Feature registry: {len(FEATURE_REGISTRY)} entries across {len(groups)} groups")
    for group, count in groups.items():
        print(f"  {group}: {count}")
