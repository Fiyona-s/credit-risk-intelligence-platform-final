-- DuckDB schema for the talk-to-data chatbot.
-- This mirrors what src/talk_to_data/query_runner.py creates at runtime
-- (it loads and creates this table from the joined + aggregated CSVs
-- in-memory on startup); kept here as a readable reference for the LLM
-- prompt and for documentation. Only the columns the chatbot is allowed to
-- reference (ALLOWED_COLUMNS in query_runner.py) are listed — the full
-- joined dataset used for model training has 166 columns.

CREATE TABLE applicants (
    SK_ID_CURR              INTEGER,     -- unique applicant id
    TARGET                  INTEGER,     -- 1 = defaulted, 0 = repaid
    NAME_CONTRACT_TYPE      VARCHAR,     -- 'Cash loans' or 'Revolving loans'
    CODE_GENDER              VARCHAR,     -- 'M' or 'F'
    CNT_CHILDREN            INTEGER,
    AMT_INCOME_TOTAL        DOUBLE,      -- annual income
    AMT_CREDIT              DOUBLE,      -- loan amount
    AMT_ANNUITY              DOUBLE,      -- loan annuity
    NAME_EDUCATION_TYPE     VARCHAR,
    NAME_FAMILY_STATUS      VARCHAR,
    OCCUPATION_TYPE          VARCHAR,
    DAYS_BIRTH               INTEGER,     -- negative; age = -DAYS_BIRTH / 365
    DAYS_EMPLOYED            INTEGER,     -- negative; anomalous value 365243 means not employed

    -- Rolled up from bureau.csv (+ bureau_balance.csv), one row per SK_ID_CURR
    BUREAU_COUNT             INTEGER,     -- number of credit bureau records, 0 if none
    BUREAU_ACTIVE_COUNT      INTEGER,     -- number of currently active bureau credits
    BUREAU_DEBT_SUM_SUM      DOUBLE,      -- total outstanding debt across all bureau credits
    HAS_BUREAU_HISTORY       BOOLEAN,     -- whether this applicant has any bureau history at all

    -- Rolled up from previous_application.csv, one row per SK_ID_CURR
    PREV_APP_COUNT            INTEGER,     -- number of previous applications with this lender, 0 if none
    PREV_APP_APPROVAL_RATE    DOUBLE,      -- share of previous applications approved, 0-1
    PREV_APP_REFUSED_COUNT    INTEGER,     -- number of previously refused applications
    HAS_PREV_APPLICATION      BOOLEAN,     -- whether this applicant has any previous application

    -- Rolled up from installments_payments.csv, one row per SK_ID_CURR
    INSTALL_LATE_COUNT        INTEGER      -- number of previous installments paid after the due date
);
