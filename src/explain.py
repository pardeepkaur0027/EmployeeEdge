"""SHAP explanation and retention action-plan utilities for EmployeeEdge.

Used by the interactive app to answer two questions:
1. Why is THIS employee predicted to leave? (per-employee SHAP explanation)
2. What should HR do about it? (rule-based retention action plan)
"""

import numpy as np
import pandas as pd
import shap
import joblib

MODEL_PATH = "../outputs/models/model.joblib"

RISK_FEATURE_RULES = {
    "MonthlyIncome": "Offer a salary review or compensation adjustment.",
    "OverTime": "Review overtime workload; consider flexible hours or extra headcount.",
    "JobSatisfaction": "Schedule an engagement interview and address job-fit.",
    "WorkLifeBalance": "Promote work-life balance initiatives (remote days, flexible schedule).",
    "StockOptionLevel": "Consider equity or bonus incentives for retention.",
    "YearsAtCompany": "Strengthen early-career support with a mentorship program.",
    "YearsInCurrentRole": "Create a clear growth path or new responsibilities.",
    "JobLevel": "Align role level and title with experience and expectations.",
    "Age": "Focus retention on early-career growth paths.",
    "TotalWorkingYears": "Provide mentorship linked to career seniority.",
    "BusinessTravel": "Reduce travel load or allow remote participation.",
    "DistanceFromHome": "Offer flexible/hybrid scheduling for long commutes.",
    "NumCompaniesWorked": "Understand reasons behind job-hopping; improve culture fit.",
    "PercentSalaryHike": "Review pay competitiveness against market.",
    "JobInvolvement": "Increase involvement in meaningful projects and decisions.",
    "MaritalStatus": "Consider stability factors if personal situations are relevant.",
    "EnvironmentSatisfaction": "Improve workplace environment; address reported issues.",
    "Department": "Review department management style and team load.",
    "JobRole": "Check role-specific factors (commissions, pressure, recognition).",
}

CATEGORIES_WITH_LOW_VALUES = [
    "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
    "WorkLifeBalance", "JobInvolvement",
]


def load_pipeline(path=MODEL_PATH):
    return joblib.load(path)


def get_background(pipeline, X_train, sample_size=50):
    """Processed sample of the training set, used by the SHAP explainer."""
    pre = pipeline.named_steps["preprocessor"]
    return pre.transform(X_train)[:sample_size]


def create_explainer(pipeline, background):
    return shap.Explainer(pipeline.named_steps["model"], background)


def explain_raw_rows(pipeline, explainer, raw_df):
    """Compute SHAP values for one or more raw (untransformed) rows."""
    pre = pipeline.named_steps["preprocessor"]
    feature_names = pre.get_feature_names_out()
    processed = pre.transform(raw_df)

    shap_values = explainer(processed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if getattr(shap_values, "ndim", 1) == 3:
        shap_values = shap_values[:, :, 1]
    shap_values.feature_names = list(feature_names)
    return shap_values, feature_names


def predict_proba(pipeline, raw_df):
    return pipeline.predict_proba(raw_df)[:, 1]


def top_pushing_factors(shap_values, feature_names, n=3):
    """Top features pushing the prediction TOWARD attrition (positive SHAP)."""
    vals = shap_values.values
    if vals.ndim == 2:
        vals = vals[0]
    order = np.argsort(vals)[::-1]
    results = []
    for idx in order:
        if len(results) >= n:
            break
        if vals[idx] <= 0.01:
            break
        full = feature_names[idx]
        label = full.split("__")[-1]
        results.append({"feature": label, "feature_full": full, "shap": vals[idx]})
    return results


def _raw_feature_value(raw_dict, feature_full):
    """Recover the human-readable value behind an encoded feature."""
    prefix, name = feature_full.split("__", 1)
    if prefix == "num":
        return raw_dict.get(name, "?")
    col, value = name.split("_", 1)
    return value


def choose_action(feature, raw_dict):
    value = raw_dict.get(feature, None)
    rule = RISK_FEATURE_RULES.get(feature)
    if rule is None:
        return None
    if feature in CATEGORIES_WITH_LOW_VALUES and value is not None:
        try:
            if int(value) >= 3:
                return None
        except (TypeError, ValueError):
            pass
    return rule


def build_action_plan(shap_values, index, raw_df, feature_names, top_n=3):
    """Turn SHAP values for one employee into a concrete action plan."""
    vals = shap_values.values
    row_shap = vals[index] if index < vals.shape[0] else vals[0]
    row_raw = raw_df.iloc[index if index < len(raw_df) else 0].to_dict()

    order = np.argsort(row_shap)[::-1]
    plan = []
    for idx in order:
        if len(plan) >= top_n:
            break
        if row_shap[idx] <= 0.02:
            break
        feature_full = feature_names[idx]
        feature = feature_full.split("__")[-1]
        action = choose_action(feature, row_raw)
        if action is None:
            continue
        plan.append(action)
    return plan


def format_factor(line):
    return f"  - {line['feature']} (+{line['shap']:.2f} risk contribution)"