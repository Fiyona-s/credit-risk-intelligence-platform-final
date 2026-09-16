"""Streamlit multi-page app: EDA, Risk Prediction, Model Evaluation, Explainability, Business Rules, Fairness Audit, Monitoring, Chatbot."""
import os

import pandas as pd
import streamlit as st

from src.data.loader import build_joined_dataset
from src.data.preprocessor import engineer_features
from src.ml.predict import predict_applicant, load_model
from src.rules.rule_derivation import derive_rules
from src.utils.config import settings
from src.utils.logger import get_logger

# matplotlib/seaborn/shap (and src.explainability.shap_explainer, which imports shap
# at its own module level) are imported lazily inside the Model Evaluation and
# Explainability pages (and their cached helpers) rather than at module level here —
# they're heavy (each pulls in its own C/Fortran extensions) and a session that
# never visits those two tabs shouldn't pay their import memory cost. This
# matters on memory-constrained deployments (e.g. Render's free tier, 512MB).

log = get_logger(__name__)

st.set_page_config(page_title="Credit Risk Intelligence Platform", layout="wide")

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")


SYNTHETIC_DATA_PATH = os.path.join(settings.data_dir, "sample_applicants.csv")


@st.cache_data
def get_sample_data():
    """The real joined+aggregated dataset, or the synthetic fallback if application_train.csv
    is missing. Postgres is intentionally NOT a fallback here — only the chatbot
    (src/talk_to_data/query_runner.py) uses Postgres, since that table only has the ~21
    chatbot-facing columns, not the 166 these other pages need for full model feature parity;
    routing every page through an extra Postgres attempt added connection overhead for a
    degraded result. Keep this two-tier."""
    try:
        return build_joined_dataset(is_train=True)
    except FileNotFoundError:
        return pd.read_csv(SYNTHETIC_DATA_PATH, comment="#")


@st.cache_data
def get_data_source_label() -> str:
    """"real_local" or "synthetic" — for accurate banners across pages (EDA, Risk Prediction,
    Model Evaluation, Explainability, Business Rules). Deliberately does not check Postgres —
    see get_sample_data()'s docstring."""
    if os.path.exists(os.path.join(settings.data_dir, "application_train.csv")):
        return "real_local"
    return "synthetic"


def synthetic_data_banner():
    """No-op: the on-screen synthetic-data notice was removed on request. The underlying
    fallback logic and server-side logging (see get_sample_data(), query_runner.py) are
    unchanged — only this UI disclosure was taken out. Kept as a function (rather than
    removing every call site) so this stays a single place to reintroduce it if needed."""
    pass


def _model_mtime():
    """Model file's mtime, passed into cached functions so a retrain invalidates the cache."""
    return os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_data(show_spinner=False)
def _compute_evaluation(model_mtime):
    """Reconstructs the same stratified 80/20 validation split as src.ml.train.train(), live.

    Cached on model_mtime so this only recomputes when the model is retrained, not on
    every Streamlit rerun. Uses the model bundle's own numeric_cols/categorical_cols
    (which is exactly what get_feature_columns() produced at train time) rather than
    recomputing them from the currently loaded df, so this degrades gracefully to the
    synthetic fallback dataset's much smaller column set instead of crashing.
    """
    import joblib
    from sklearn.model_selection import train_test_split

    from src.ml.evaluate import compute_roc_pr_curves, evaluate_model

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    df = get_sample_data()
    df = engineer_features(df)
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None

    y = df["TARGET"]
    X = df[numeric_cols + categorical_cols]
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    metrics = evaluate_model(pipeline, X_val, y_val)
    curves = compute_roc_pr_curves(pipeline, X_val, y_val)
    return metrics, curves


@st.cache_resource(show_spinner=False)
def _get_global_shap_figure(model_mtime, sample_size: int):
    """Global SHAP summary (beeswarm) plot across a sample of applicants.

    st.cache_resource (not cache_data) because the return value is a live matplotlib
    Figure, not something that needs to round-trip through pickling. Cached on
    (model_mtime, sample_size) so a retrain or a different sample size recomputes it.
    """
    import matplotlib.pyplot as plt
    import shap

    from src.explainability.shap_explainer import compute_global_shap_values

    df = get_sample_data()
    shap_values, X_transformed, feature_names = compute_global_shap_values(df, max_rows=sample_size)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _compute_fairness_audit(model_mtime, attribute: str):
    """Cached per (model_mtime, attribute) — reruns only on retrain or a different
    sensitive attribute, not on every Streamlit rerun. Passes get_sample_data()'s
    already-loaded (real-or-synthetic) dataframe into the audit module so this tab
    degrades the same way every other page does when the real dataset is absent.
    """
    from src.fairness.audit import run_audit

    df = get_sample_data()
    result = run_audit(attribute, df=df)
    # group_metrics is a DataFrame — cache_data needs to hash/serialize the return
    # value, which DataFrames support natively, so no extra handling needed here.
    return result


def page_eda():
    st.header("Exploratory Data Analysis")
    st.markdown(
        "Key insights from `notebooks/eda.ipynb`, computed live against the data in `data/`."
    )

    synthetic_data_banner()

    df = get_sample_data()
    df = engineer_features(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Applicants", f"{df.shape[0]:,}")
    col2.metric("Features", df.shape[1])
    col3.metric("Default Rate", f"{df['TARGET'].mean() * 100:.2f}%")

    st.subheader("1. Default rate by income bracket")
    df["INCOME_BRACKET"] = pd.qcut(df["AMT_INCOME_TOTAL"], q=5, labels=["Lowest", "Low", "Mid", "High", "Highest"])
    st.bar_chart(df.groupby("INCOME_BRACKET", observed=True)["TARGET"].mean() * 100)

    st.subheader("2. Default rate by education level")
    st.bar_chart(df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100)

    st.subheader("3. Default rate by credit amount band")
    df["CREDIT_BRACKET"] = pd.qcut(df["AMT_CREDIT"], q=5, labels=["Lowest", "Low", "Mid", "High", "Highest"])
    st.bar_chart(df.groupby("CREDIT_BRACKET", observed=True)["TARGET"].mean() * 100)

    st.subheader("4. Default rate by age band")
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["AGE_BAND"] = pd.cut(df["AGE_YEARS"], bins=[20, 30, 40, 50, 60, 70],
                             labels=["20-30", "30-40", "40-50", "50-60", "60-70"])
    st.bar_chart(df.groupby("AGE_BAND", observed=True)["TARGET"].mean() * 100)

    st.subheader("5. Missing values (top 15 columns)")
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    st.bar_chart(missing_pct[missing_pct > 0].head(15))

    st.subheader("6. Default rate: with vs without bureau history")
    if "HAS_BUREAU_HISTORY" in df.columns:
        bureau_label = df["HAS_BUREAU_HISTORY"].map({True: "Has bureau history", False: "No bureau history"})
        st.bar_chart(df.groupby(bureau_label)["TARGET"].mean() * 100)
        st.caption(
            "Empirical justification for joining bureau.csv into the ML pipeline — "
            "applicants with no bureau history default more often than those with history."
        )
    else:
        st.info("bureau.csv not found — this chart needs bureau-derived features.")


def page_risk_prediction():
    st.header("Risk Prediction")

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Run `python -m src.ml.train` first.")
        return

    synthetic_data_banner()

    df = get_sample_data()
    use_sample = st.checkbox("Use a sample applicant from the dataset instead of manual entry", value=True)

    if use_sample:
        idx = st.number_input("Applicant row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
        applicant = df.iloc[int(idx)].to_dict()
        st.dataframe(pd.DataFrame([applicant]).T.rename(columns={0: "value"}).head(20))
    else:
        applicant = {
            "AMT_INCOME_TOTAL": st.number_input("Annual income", value=150000.0),
            "AMT_CREDIT": st.number_input("Credit amount", value=500000.0),
            "AMT_ANNUITY": st.number_input("Annuity", value=25000.0),
            "DAYS_BIRTH": -st.number_input("Age (years)", value=35, min_value=18, max_value=100) * 365,
            "DAYS_EMPLOYED": -st.number_input("Years employed", value=5, min_value=0, max_value=50) * 365,
            "CNT_CHILDREN": st.number_input("Number of children", value=0, min_value=0),
            "NAME_EDUCATION_TYPE": st.selectbox(
                "Education", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"]
            ),
            "CODE_GENDER": st.selectbox("Gender", ["M", "F"]),
            "NAME_CONTRACT_TYPE": st.selectbox("Contract type", ["Cash loans", "Revolving loans"]),
            "NAME_FAMILY_STATUS": st.selectbox("Family status", ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"]),
        }

    if st.button("Predict risk"):
        result = predict_applicant(applicant)
        st.session_state["last_applicant"] = applicant
        st.session_state["last_prediction"] = result

        band = result["risk_band"]
        # Milestone 5 changed risk_band to Low/Moderate/High/Very High (calibrated-PD
        # based) — this map previously only had the old Low/Medium/High keys and
        # raised a KeyError for any Moderate/Very High applicant. Fixed here.
        color = {"Low": "green", "Moderate": "gold", "High": "orange", "Very High": "red"}[band]
        st.markdown(f"### Risk Band: :{color}[{band}]")
        st.caption("Project-defined risk category based on calibrated PD — not an official banking/credit-bureau grade.")
        col_a, col_b = st.columns(2)
        col_a.metric("Raw Model Probability", f"{result['raw_probability'] * 100:.2f}%")
        if result["calibrated_probability"] is not None:
            col_b.metric("Calibrated Probability of Default", f"{result['calibrated_probability'] * 100:.2f}%")
        else:
            col_b.metric("Calibrated Probability of Default", "n/a (no calibrator found)")

        if result.get("decision") is not None:
            decision_color = "red" if result["decision"] == "Decline" else "green"
            st.markdown(
                f"**Business decision** (calibrated PD vs. cost-sensitive threshold): "
                f":{decision_color}[{result['decision']}] "
                f"(threshold {result['decision_threshold']:.2f} on the calibrated scale, "
                f"scenario: {result['decision_scenario']})"
            )

        if result["cost_optimal_decision"] is not None:
            decision_color = "red" if result["cost_optimal_decision"] == "Decline" else "green"
            st.markdown(
                f"**Legacy cost-optimal decision** (raw probability vs. `models/optimal_threshold.json`, "
                f"kept for backward compatibility): :{decision_color}[{result['cost_optimal_decision']}] "
                f"(threshold {result['optimal_threshold']:.2f}, from the sweep below)"
            )

    st.divider()
    _render_cost_sensitive_threshold_section()


def _render_cost_sensitive_threshold_section():
    """Threshold-vs-cost curve + an interactive cost-ratio slider, using the
    false_positives/false_negatives arrays cached in models/optimal_threshold.json
    by src.ml.train.sweep_cost_sensitive_threshold() — recomputing total_cost for a
    different ratio is pure arithmetic on those arrays, no model/data reload needed.
    """
    from src.ml.predict import load_optimal_threshold

    st.subheader("Cost-sensitive threshold selection")

    saved = load_optimal_threshold()
    if saved is None:
        st.info(
            "No cost-sensitive threshold sweep found. Run `python -m src.ml.train` "
            "(it runs automatically as part of training) to generate "
            "`models/optimal_threshold.json`."
        )
        return

    st.markdown(
        "At training time (`FP_COST`/`FN_COST` in `.env`), the threshold that minimizes "
        "`total_cost = FP_COST × false_positives + FN_COST × false_negatives` on the "
        "validation set was found to be "
        f"**{saved['optimal_threshold']:.2f}** (vs. the fixed `RISK_THRESHOLD_MEDIUM` "
        f"of **{settings.risk_threshold_medium:.2f}** used for the risk band above) — "
        "adjust the cost ratio below to see how that optimal point shifts, without retraining."
    )

    import matplotlib.pyplot as plt
    import numpy as np

    thresholds = np.array(saved["thresholds"])
    false_positives = np.array(saved["false_positives"])
    false_negatives = np.array(saved["false_negatives"])

    default_ratio = saved["fn_cost"] / saved["fp_cost"] if saved["fp_cost"] else saved["fn_cost"]
    ratio = st.slider(
        "Cost ratio: FN_COST / FP_COST (how many times costlier is missing a defaulter "
        "than wrongly flagging a good applicant?)",
        min_value=1.0, max_value=20.0, value=float(min(max(default_ratio, 1.0), 20.0)), step=0.5,
    )

    # FP_COST held at 1 for this exploration — only the *ratio* to FN_COST matters
    # for where the minimum falls, not the absolute scale of both costs.
    total_cost = false_positives + ratio * false_negatives
    best_idx = int(np.argmin(total_cost))
    live_optimal_threshold = float(thresholds[best_idx])

    col1, col2 = st.columns(2)
    col1.metric("Optimal threshold at this ratio", f"{live_optimal_threshold:.2f}")
    col2.metric(
        "Optimal threshold saved at training time",
        f"{saved['optimal_threshold']:.2f}",
        delta=f"{live_optimal_threshold - saved['optimal_threshold']:+.2f}",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, total_cost, color="#4C72B0", label=f"Total cost (ratio={ratio:g})")
    ax.axvline(live_optimal_threshold, color="#C44E52", linestyle="--", label=f"Optimal @ ratio={ratio:g}")
    if abs(live_optimal_threshold - saved["optimal_threshold"]) > 1e-9:
        ax.axvline(saved["optimal_threshold"], color="gray", linestyle=":", label="Optimal @ training-time ratio")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Total cost (FP_COST=1 × FP + ratio × FN)")
    ax.set_title("Threshold vs. cost")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Raising the cost ratio (missing a defaulter gets relatively more expensive) pulls the "
        "optimal threshold down — the model is pushed to flag more applicants as high-risk to "
        "catch more actual defaulters, accepting more false positives in exchange."
    )


def page_model_evaluation():
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("Model Evaluation")
    st.markdown(
        "Live validation metrics for the trained model — reconstructs the same stratified "
        "80/20 split used in `src/ml/train.py` (`random_state=42`), so these numbers match "
        "what's documented in the README when run against the real dataset."
    )

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Run `python -m src.ml.train` first.")
        return

    synthetic_data_banner()

    with st.spinner("Computing validation metrics (cached after first load)..."):
        metrics, curves = _compute_evaluation(_model_mtime())

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col2.metric("PR-AUC (average precision)", f"{metrics['pr_auc']:.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(curves["fpr"], curves["tpr"], color="#4C72B0", label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Precision-Recall Curve")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.plot(curves["recall_curve"], curves["precision_curve"], color="#C44E52",
                  label=f"PR (AUC = {metrics['pr_auc']:.3f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend(loc="upper right")
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader(f"Precision / Recall / F1 @ threshold {metrics['threshold']}")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1"],
            "Value": [metrics["precision"], metrics["recall"], metrics["f1"]],
        }),
        hide_index=True,
    )

    st.subheader("Confusion Matrix")
    fig3, ax3 = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax3,
        xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"],
    )
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()
    st.subheader("Model Card")
    st.caption(
        "A PDF summary for sharing outside this app: intended use, out-of-scope uses, "
        "training data, these performance metrics, per-group fairness metrics, monotonic "
        "constraints, and known limitations — generated fresh from the current model, "
        "not a static document."
    )
    if st.button("Generate model card PDF"):
        with st.spinner("Generating model card (recomputes fairness metrics, ~30-60s)..."):
            from scripts.generate_model_card import generate_model_card

            pdf_bytes = generate_model_card()
        st.session_state["model_card_pdf"] = pdf_bytes

    if "model_card_pdf" in st.session_state:
        st.download_button(
            "Download Model Card",
            data=st.session_state["model_card_pdf"],
            file_name="model_card.pdf",
            mime="application/pdf",
        )


def page_explainability():
    from src.explainability.shap_explainer import explain_applicant

    st.header("Explainability (SHAP)")

    st.subheader("This applicant's explanation")
    if "last_applicant" not in st.session_state:
        st.info("Run a prediction on the Risk Prediction tab first.")
    else:
        st.markdown("Top features driving the most recent prediction, in plain language:")
        explanations = explain_applicant(st.session_state["last_applicant"])
        for line in explanations:
            st.markdown(f"- {line}")

    st.divider()
    st.subheader("Global feature importance across the dataset")

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Run `python -m src.ml.train` first.")
        return

    synthetic_data_banner()

    sample_size = 300
    st.caption(
        f"SHAP summary (beeswarm) plot: which features push predictions up or down across "
        f"{sample_size} sampled applicants — distinct from the single-applicant view above."
    )
    with st.spinner("Computing global SHAP summary (cached after first load)..."):
        fig = _get_global_shap_figure(_model_mtime(), sample_size)
    st.pyplot(fig)


def page_business_rules():
    st.header("Business Rules")
    st.markdown(
        "Simple if/then rules extracted from a shallow decision-tree surrogate "
        "trained to mimic the model's predictions — readable without any ML background."
    )

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Run `python -m src.ml.train` first.")
        return

    synthetic_data_banner()

    if st.button("Derive rules") or "rules" in st.session_state:
        if "rules" not in st.session_state:
            df = get_sample_data()
            df = engineer_features(df)
            st.session_state["rules"] = derive_rules(df)
        for rule in st.session_state["rules"]:
            st.markdown(f"- {rule}")


def page_fairness_audit():
    import matplotlib.pyplot as plt

    from src.fairness.audit import (
        SENSITIVE_ATTRIBUTES, DISPARITY_FLAG_THRESHOLD, MIN_RELIABLE_GROUP_SIZE, plot_approval_rate_by_group,
    )

    st.header("Fairness Audit")
    st.markdown(
        "Demographic parity and equal opportunity, computed on the same stratified 80/20 "
        "validation split as **Model Evaluation** — checks whether the model's predictions "
        "are evenly distributed across sensitive-attribute groups, not just accurate overall."
    )

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Run `python -m src.ml.train` first.")
        return

    synthetic_data_banner()

    attribute = st.selectbox("Sensitive attribute", SENSITIVE_ATTRIBUTES)

    with st.spinner("Computing fairness audit (cached after first load)..."):
        result = _compute_fairness_audit(_model_mtime(), attribute)

    group_metrics = result["group_metrics"]

    col1, col2 = st.columns(2)
    col1.metric("Demographic parity difference", f"{result['demographic_parity_difference']:.3f}")
    col2.metric("Equal opportunity difference", f"{result['equal_opportunity_difference']:.3f}")
    st.caption(
        "Demographic parity difference: max − min predicted-high-risk rate across groups "
        "(0 = identical rates). Equal opportunity difference: max − min true-positive rate "
        "(recall on actual defaulters) across groups — does the model catch defaulters "
        "equally well regardless of group, not just flag \"high risk\" at the same overall rate."
    )

    if result["flagged_groups"]:
        st.warning(
            f"⚠️ Flagged: {', '.join(str(g) for g in result['flagged_groups'])} — predicted "
            f"high-risk rate differs from the overall validation-set rate by more than "
            f"{DISPARITY_FLAG_THRESHOLD:.0%}."
        )
    else:
        st.success(f"No group's predicted high-risk rate differs from the overall rate by more than {DISPARITY_FLAG_THRESHOLD:.0%}.")

    small_groups = group_metrics.loc[group_metrics["n"] < MIN_RELIABLE_GROUP_SIZE, "group"].tolist()
    if small_groups:
        st.caption(
            f"⚠️ {', '.join(str(g) for g in small_groups)} — fewer than {MIN_RELIABLE_GROUP_SIZE} "
            "validation-set rows. Disparity figures for these groups are more likely sample-size "
            "noise than a reliable finding; treat with caution rather than at face value."
        )

    st.subheader("Per-group metrics")
    display = group_metrics.copy()
    for col in ["actual_default_rate", "mean_predicted_risk_score", "predicted_high_risk_rate",
                "approval_rate", "true_positive_rate", "disparity_vs_overall"]:
        display[col] = (display[col] * 100).round(2)
    display = display.rename(columns={
        "group": "Group", "n": "N",
        "actual_default_rate": "Actual default rate (%)",
        "mean_predicted_risk_score": "Mean predicted risk score (%)",
        "predicted_high_risk_rate": "Predicted high-risk rate (%)",
        "approval_rate": "Approval rate (%)",
        "true_positive_rate": "True positive rate (%)",
        "disparity_vs_overall": "Disparity vs. overall (pp)",
        "flagged": "Flagged",
    })
    st.dataframe(display, hide_index=True)

    st.subheader("Approval rate by group")
    fig = plot_approval_rate_by_group(group_metrics, attribute)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Red bars mark groups flagged above (>5% disparity in predicted high-risk rate vs. the overall validation-set rate).")


@st.cache_data(show_spinner=False)
def _compute_drift_monitoring(reference_mtime, data_source_label: str, sample_size: int = 5000):
    """Cached on (reference file mtime, current data source label) so a retrain
    (--monitor re-saves the reference snapshot) or a real-vs-synthetic data switch
    recomputes, but a plain Streamlit rerun doesn't. Both sides are capped to
    sample_size rows for UI responsiveness — Evidently's per-column K-S/Z-tests on
    the full ~300k-row dataset across ~150+ columns can take well over a minute;
    the standalone CLI (python -m src.monitoring.drift_report) uses the full data
    since it isn't rendered in a live UI waiting on it.
    """
    from src.monitoring.drift_report import load_reference_snapshot, run_drift_monitoring

    reference = load_reference_snapshot()
    if reference is None:
        return None

    current = get_sample_data()
    current = engineer_features(current)

    if len(reference) > sample_size:
        reference = reference.sample(sample_size, random_state=42)
    if len(current) > sample_size:
        current = current.sample(sample_size, random_state=42)

    return run_drift_monitoring(reference, current)


def page_monitoring():
    from src.monitoring.drift_report import REFERENCE_DATA_PATH

    st.header("Monitoring")
    st.markdown(
        "Data drift and target drift between the training-time reference snapshot "
        "(saved by `python -m src.ml.train --monitor`) and the currently loaded "
        "applicant data — flags features whose distribution has shifted since training."
    )

    if not os.path.exists(REFERENCE_DATA_PATH):
        st.warning(
            f"No reference snapshot found at `{REFERENCE_DATA_PATH}`. Run "
            "`python -m src.ml.train --monitor` to create one, then reload this tab."
        )
        return

    synthetic_data_banner()

    reference_mtime = os.path.getmtime(REFERENCE_DATA_PATH)
    with st.spinner("Computing drift report (cached after first load; both sides sampled to 5,000 rows)..."):
        result = _compute_drift_monitoring(reference_mtime, get_data_source_label())

    if result is None:
        st.warning(f"No reference snapshot found at `{REFERENCE_DATA_PATH}`.")
        return

    col1, col2 = st.columns(2)
    n_drifted = int(result["top_drifted_features"]["drifted"].sum()) if len(result["top_drifted_features"]) else 0
    col1.metric("Features with detected drift", n_drifted)
    target_status = "Unknown" if result["target_drifted"] is None else ("Yes" if result["target_drifted"] else "No")
    col2.metric("Target (TARGET) drifted", target_status)

    st.subheader("Most-drifted features")
    st.caption(
        "Evidently auto-selects a drift test per column (a p-value test for some, a "
        "distance/divergence test like Jensen-Shannon or PSI for others) — \"drifted\" "
        "already accounts for each method's own comparison direction (below threshold "
        "for p-value tests, above threshold for distance tests), not a single flat rule."
    )
    st.dataframe(result["top_drifted_features"], hide_index=True)

    st.subheader("Full Evidently reports")
    for label, path_key in [("Data drift report", "data_drift_report_path"), ("Target drift report", "target_drift_report_path")]:
        report_path = result[path_key]
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(f"Download {label} (HTML)", f.read(), file_name=os.path.basename(report_path), mime="text/html")


def page_chatbot():
    st.header("Talk to Your Data")
    st.markdown(
        "Ask questions in plain English about the applicant data, e.g. "
        "*'average income of defaulters'* or *'default rate by education level'*."
    )

    if not settings.has_groq and not settings.has_gemini:
        st.error(
            "No LLM API key configured. Set GROQ_API_KEY or GEMINI_API_KEY in your .env file "
            "(copy from .env.example) to enable the chatbot."
        )
        return

    from src.talk_to_data.query_runner import get_data_source

    if get_data_source() == "synthetic":
        synthetic_data_banner()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of (question, answer_summary)
    if "chat_display" not in st.session_state:
        st.session_state["chat_display"] = []  # list of (role, content) for rendering

    for role, content in st.session_state["chat_display"]:
        with st.chat_message(role):
            st.markdown(content.replace("$", "\\$"))

    question = st.chat_input("Ask a question about the applicant data...")
    if question:
        st.session_state["chat_display"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        from src.talk_to_data.nl_to_sql import ask

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(question, conversation_history=st.session_state["chat_history"])
            if result["success"]:
                st.markdown(result["message"].replace("$", "\\$"))
                st.code(result["sql"], language="sql")
                st.dataframe(result["result"])
                st.session_state["chat_history"].append((question, result["message"]))
                st.session_state["chat_display"].append(("assistant", result["message"]))
            else:
                st.warning(result["message"])
                st.session_state["chat_display"].append(("assistant", result["message"]))


def main():
    st.sidebar.title("Credit Risk Intelligence Platform")
    page = st.sidebar.radio(
        "Navigate",
        ["EDA", "Risk Prediction", "Model Evaluation", "Explainability", "Business Rules",
         "Fairness Audit", "Monitoring", "Chatbot"],
    )

    if page == "EDA":
        page_eda()
    elif page == "Risk Prediction":
        page_risk_prediction()
    elif page == "Model Evaluation":
        page_model_evaluation()
    elif page == "Explainability":
        page_explainability()
    elif page == "Business Rules":
        page_business_rules()
    elif page == "Fairness Audit":
        page_fairness_audit()
    elif page == "Monitoring":
        page_monitoring()
    elif page == "Chatbot":
        page_chatbot()


if __name__ == "__main__":
    main()
