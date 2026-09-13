"""SHAP explanation + retention action-plan utilities for EmployeeEdge.

Turns a black-box probability into something an HR manager can actually act on:

  - WHY is this employee at risk?   (SHAP per-employee explanation)
  - WHAT should we do about it?     (rule-based retention action plan)

Used by the serving app AND the notebooks, so training and serving share
exactly the same explanation code.
"""

import os

import joblib

import numpy as np

import shap

ACTION_RULES = {
    "OverTime": ("Overtime", "Offer flexible scheduling, overtime-pay review, or extra headcount."),
    "BusinessTravel": ("Business travel overload", "Move to remote meetings / redistribute travel load."),
    "JobSatisfaction": ("Job satisfaction", "Run an engagement interview and address job-fit blockers."),
    "MonthlyIncome": ("Compensation", "Benchmark salary against market and consider an adjustment."),
    "WorkLifeBalance": ("Work-life balance", "Introduce flexible hours or workload redistribution."),
    "YearsWithCurrManager": ("Manager relationship", "Strengthen manager-employee alignment with 1:1 check-ins."),
    "JobRole": ("Role fit", "Review responsibilities and growth path for this role."),
    "StockOptionLevel": ("Equity upside", "Consider a stock-option or bonus grant."),
    "DistanceFromHome": ("Commute burden", "Offer hybrid or remote-friendly scheduling."),
    "YearsInCurrentRole": ("Role stagnation", "Open a promotion or new-stretch-projects path."),
    "TotalWorkingYears": ("Tenure-fulfillment gap", "Align role scope with the employee's career seniority."),
    "JobInvolvement": ("Low involvement", "Delegate meaningful ownership and visibility."),
    "EnvironmentSatisfaction": ("Work environment", "Address facility/team dynamics concerns."),
    "Age": ("Early-career stage", "Provide mentorship and a clear growth plan."),
}

NUMERIC_LOW = ["MonthlyIncome", "WorkLifeBalance", "JobSatisfaction",
               "EnvironmentSatisfaction", "JobInvolvement"]
NUMERIC_HIGH = ["DistanceFromHome", "NumCompaniesWorked"]


def load_pipeline(path):
    """Load the fitted sklearn Pipeline (preprocessor + model) from disk."""
    pipe = joblib.load(path)
    print(f"Model pipeline loaded: {type(pipe.named_steps['model']).__name__}")
    return pipe


def get_background(pipeline, X_train, sample_size=50):
    """Build a SHAP background sample from the preprocessed training matrix."""
    pre = pipeline.named_steps["preprocessor"]
    X_encoded = pre.transform(X_train)
    rng = np.random.default_rng(0)
    idx = rng.choice(X_encoded.shape[0],
                     size=min(sample_size, X_encoded.shape[0]), replace=False)
    return X_encoded[idx]


def create_explainer(pipeline, background):
    """Create a SHAP explainer for the model, anchored to the background sample."""
    model = pipeline.named_steps["model"]
    return shap.Explainer(model, background)


def predict_proba(pipeline, raw_df):
    """Probability that each RAW row leaves (1 = attrition), 1D array."""
    return pipeline.predict_proba(raw_df)[:, 1]


def explain_raw_rows(pipeline, explainer, raw_df):
    """SHAP explanation for RAW (unencoded) employee rows.

    Encodes with the pipeline preprocessor, explains, and returns
    ``(shap_values, feature_names)`` where ``feature_names`` are the encoded
    ColumnTransformer output names (e.g. ``cat__OverTime_Yes``).
    """
    pre = pipeline.named_steps["preprocessor"]
    X = pre.transform(raw_df)
    feature_names = list(pre.get_feature_names_out())
    shap_values = explainer(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values.feature_names = feature_names
    return shap_values, feature_names


def top_pushing_factors(shap_values, feature_names, n=3):
    """Top-n features PUSHING toward leaving (highest positive SHAP).

    Returns a list of dicts: ``[{"factor": <label>, "shap": <float>}, ...]``.
    """
    vals = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    names = list(getattr(shap_values, "feature_names", None) or feature_names)
    if vals.ndim == 2:
        vals = vals[0]
    order = np.argsort(vals)[::-1]
    factors = []
    for i in order:
        base = names[i].split("__")[-1].replace("_", " ")
        factors.append({"factor": base, "shap": round(float(vals[i]), 2)})
        if len(factors) >= n:
            break
    return factors


def format_factor(f):
    """Human-readable line for one pushing factor dict."""
    sign = "+" if f["shap"] >= 0 else ""
    return f"{f['factor']} ({sign}{f['shap']:.2f})"


def build_action_plan(shap_values, top_index, raw_row, feature_names, top_n=3):
    """Rule-based retention action plan for ONE employee.

    Returns a list of human-readable action strings (non-empty by construction —
    if no rule fires we default to a general engagement check-in).
    """
    vals = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    names = list(getattr(shap_values, "feature_names", None) or feature_names)
    if vals.ndim == 2:
        vals = vals[top_index]

    order = np.argsort(vals)[::-1]
    plan = []
    for i in order:
        base = names[i].split("__")[-1]
        col = base.split("_")[0]
        rule = ACTION_RULES.get(col)
        if rule is None:
            continue
        label, advice = rule
        shap_val = round(float(vals[i]), 2)
        plan.append(f"{label}: {advice}  (SHAP {shap_val:+.2f})")
        if len(plan) >= top_n:
            break
    if not plan:
        plan = ["General engagement: run a 1:1 check-in and a stay interview."]
    return plan


def get_feature_names(pipeline):
    """Encoded feature names (e.g. ``num__Age``, ``cat__OverTime_Yes``).

    Uses every preprocessor in the pipeline's ``preprocessor`` step, so the
    app rendering SHAP labels always matches training-time features exactly.
    """
    pre = pipeline.named_steps["preprocessor"]
    return list(pre.get_feature_names_out())
