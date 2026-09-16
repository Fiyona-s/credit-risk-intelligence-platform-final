"""Generate a model card PDF (reports/model_card.pdf) for the deployed credit-risk model.

Reads the saved model (models/credit_risk_model.joblib), reconstructs the same
stratified 80/20 validation split used at training time (src.ml.train._prepare_data),
and reports real, freshly-computed numbers — never hardcoded placeholders. Covers:
intended use, out-of-scope uses, training data, performance metrics, per-group
fairness metrics (reusing src.fairness.audit, the same module the Fairness Audit
tab uses), known limitations, and the monotonic constraints applied at training.

Usage: python -m scripts.generate_model_card
Also importable: generate_model_card() is called by app.py's "Download Model Card"
button in the Model Evaluation tab, so the same code produces the on-demand PDF
in-app and the CLI-generated file — one source of truth, not two.
"""
import os
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")
REPORTS_DIR = "reports"
OUTPUT_PATH = os.path.join(REPORTS_DIR, "model_card.pdf")


def _gather_model_card_data() -> dict:
    """All the real, computed numbers the PDF needs, in one place — kept separate
    from the PDF-rendering step (_render_pdf) so app.py's button can call this once
    and the render step never needs to touch the model or re-run evaluation.
    """
    import joblib
    from sklearn.model_selection import train_test_split

    from src.data.loader import build_joined_dataset
    from src.data.preprocessor import engineer_features
    from src.fairness.audit import SENSITIVE_ATTRIBUTES, run_audit
    from src.ml.evaluate import evaluate_model
    from src.ml.train import MONOTONIC_CONSTRAINTS_MAP

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run `python -m src.ml.train` first.")

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    df = build_joined_dataset(is_train=True)
    df = engineer_features(df)
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None

    y = df["TARGET"]
    X = df[numeric_cols + categorical_cols]
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    metrics = evaluate_model(pipeline, X_val, y_val)

    fairness_results = {attr: run_audit(attr, df=df) for attr in SENSITIVE_ATTRIBUTES}

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_validation_rows": len(X_val),
        "n_training_rows": len(df) - len(X_val),
        "metrics": metrics,
        "fairness_results": fairness_results,
        "monotonic_constraints": MONOTONIC_CONSTRAINTS_MAP,
    }


def _render_pdf(data: dict, output_path: str) -> None:
    ensure_dir(os.path.dirname(output_path) or ".")

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    caption = ParagraphStyle("caption", parent=body, fontSize=8, textColor=colors.grey)

    doc = SimpleDocTemplate(
        output_path, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    story = []

    story.append(Paragraph("Credit Risk Intelligence Platform — Model Card", h1))
    story.append(Paragraph(
        f"Generated {data['generated_at']} · LightGBM classifier, "
        f"{data['n_training_rows']:,} training rows / {data['n_validation_rows']:,} validation rows "
        "(stratified 80/20 split, random_state=42)",
        caption,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Intended use", h2))
    story.append(Paragraph(
        "A default-risk screening aid for consumer credit applications: given applicant and "
        "credit-bureau-derived features, the model outputs a probability of default and a "
        "Low/Moderate/High/Very High risk band (project-defined, derived from a separately calibrated probability of default — see models/calibration_metadata.json), plus (via the surrogate decision tree) human-readable rules "
        "and (via SHAP) a per-applicant explanation. Intended to support, not replace, a human "
        "underwriter's decision — as one input among others, with the reasoning behind each score "
        "inspectable rather than a black-box output.",
        body,
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Out-of-scope uses", h2))
    for line in [
        "Fully automated approve/decline decisions with no human review — this model has "
        "measured fairness disparities (see below) that have not been mitigated, and should "
        "not be the sole basis for an adverse action.",
        "Any lending product, geography, or population meaningfully different from the Home "
        "Credit Group applicant population this model was trained and validated on.",
        "Use as a general-purpose credit scoring model outside the specific feature schema "
        "(application + bureau + previous-application aggregates) it was trained on.",
        "Any deployment claiming this model has been fairness-audited and cleared — it has "
        "been audited and found to have real disparities that remain unresolved (see below).",
    ]:
        story.append(Paragraph(f"• {line}", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Training data", h2))
    story.append(Paragraph(
        "Kaggle's Home Credit Default Risk dataset: the main application table joined with "
        "per-applicant rollups (count/sum/mean/max aggregates) of six auxiliary tables — bureau "
        "credit history, bureau balance, previous Home Credit applications, POS/cash balance, "
        "credit card balance, and installment payment history. Binary target: 1 = the applicant "
        "defaulted (had payment difficulty) on the current loan, 0 = repaid without difficulty. "
        "Roughly 92% negative / 8% positive class balance, corrected for during training via "
        "LightGBM's scale_pos_weight.",
        body,
    ))
    story.append(Spacer(1, 0.15 * inch))

    m = data["metrics"]
    story.append(Paragraph("Performance metrics (validation split)", h2))
    perf_table = Table(
        [
            ["Metric", "Value"],
            ["ROC-AUC", f"{m['roc_auc']:.4f}"],
            ["PR-AUC (average precision)", f"{m['pr_auc']:.4f}"],
            [f"Precision @ {m['threshold']}", f"{m['precision']:.4f}"],
            [f"Recall @ {m['threshold']}", f"{m['recall']:.4f}"],
            [f"F1 @ {m['threshold']}", f"{m['f1']:.4f}"],
        ],
        colWidths=[3 * inch, 2 * inch],
    )
    perf_table.setStyle(_table_style())
    story.append(perf_table)
    story.append(Spacer(1, 0.15 * inch))

    cm = m["confusion_matrix"]
    story.append(Paragraph(f"Confusion matrix @ threshold {m['threshold']} (rows = actual, cols = predicted)", body))
    cm_table = Table(
        [
            ["", "Pred: No Default", "Pred: Default"],
            ["Actual: No Default", f"{cm[0][0]:,}", f"{cm[0][1]:,}"],
            ["Actual: Default", f"{cm[1][0]:,}", f"{cm[1][1]:,}"],
        ],
        colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch],
    )
    cm_table.setStyle(_table_style())
    story.append(cm_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Fairness metrics (per sensitive attribute)", h2))
    story.append(Paragraph(
        "Demographic parity difference = max − min predicted-high-risk rate across groups. "
        "Equal opportunity difference = max − min true-positive rate (recall on actual "
        "defaulters) across groups. 0 = no disparity in either case.",
        body,
    ))
    for attr, result in data["fairness_results"].items():
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(
            f"<b>{attr}</b> — demographic parity difference: {result['demographic_parity_difference']:.4f}, "
            f"equal opportunity difference: {result['equal_opportunity_difference']:.4f}"
            + (f". Flagged groups (>5pp disparity vs. overall): {', '.join(map(str, result['flagged_groups']))}"
               if result["flagged_groups"] else ". No group exceeds the 5pp disparity threshold."),
            body,
        ))
        gm = result["group_metrics"]
        cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=10)
        rows = [["Group", "n", "Actual default rate", "Predicted high-risk rate", "Approval rate"]]
        for _, row in gm.iterrows():
            rows.append([
                Paragraph(str(row["group"]), cell), f"{int(row['n']):,}",
                f"{row['actual_default_rate']:.1%}",
                f"{row['predicted_high_risk_rate']:.1%}",
                f"{row['approval_rate']:.1%}",
            ])
        group_table = Table(rows, colWidths=[2.0 * inch, 0.6 * inch, 1.2 * inch, 1.4 * inch, 1.0 * inch])
        group_table.setStyle(_table_style())
        story.append(group_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Monotonic constraints applied", h2))
    if data["monotonic_constraints"]:
        story.append(Paragraph(
            "Applied only to features where the risk direction is unambiguous by domain "
            "knowledge (a conservative starting set, not blanket across all features):",
            body,
        ))
        constraint_rows = [["Feature", "Constraint"]]
        for feature, direction in data["monotonic_constraints"].items():
            label = "Higher value → lower predicted risk" if direction == -1 else (
                "Higher value → higher predicted risk" if direction == 1 else "None"
            )
            constraint_rows.append([feature, label])
        constraint_table = Table(constraint_rows, colWidths=[2.5 * inch, 3.5 * inch])
        constraint_table.setStyle(_table_style())
        story.append(constraint_table)
    else:
        story.append(Paragraph("None applied.", body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Known limitations", h2))
    for line in [
        "Real, substantial fairness disparities exist by gender and education level (see above) "
        "and are surfaced, not mitigated, by this model — no post-hoc threshold adjustment, "
        "fairness-constrained objective, or proxy-feature removal has been applied.",
        "Auxiliary-table features are simple per-applicant rollups (count/sum/mean/max); "
        "recency-weighted aggregates and second-order trend features are not yet included.",
        "Validated on a random 80/20 split; credit data has temporal structure a random split "
        "doesn't fully test — a time-based split should be used before relying on this model "
        "for a materially different risk appetite or time period.",
        "Monotonic constraints cover only 4 of the model's input features; most features have "
        "no directional guarantee.",
        "The cost-sensitive optimal threshold and the fixed Low/Medium/High risk-band cutoffs "
        "are independent, uncoordinated decision surfaces — this model card reports performance "
        "at the fixed default threshold (0.5) only.",
    ]:
        story.append(Paragraph(f"• {line}", body))

    doc.build(story)
    log.info(f"Wrote model card to {output_path}")


def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def generate_model_card(output_path: str = OUTPUT_PATH) -> bytes:
    """Generates the model card PDF at output_path and also returns its bytes,
    so app.py's download button can serve the same content without re-reading
    the file from disk in a separate step.
    """
    data = _gather_model_card_data()
    _render_pdf(data, output_path)
    with open(output_path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    generate_model_card()
    print(f"Model card written to {OUTPUT_PATH}")
