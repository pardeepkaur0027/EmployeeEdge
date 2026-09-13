"""EmployeeEdge -- Flask web app.

Routes:
  GET  /         -> input form
  POST /predict  -> risk prediction + SHAP explanation + action plan + what-if sims
"""

import sys
import os
import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import shap

from flask import Flask, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import preprocess as pre
from src import explain as ex

MODEL_PATH = os.path.join(ROOT, "outputs", "models", "model.joblib")
DATA_PATH = os.path.join(ROOT, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

app = Flask(__name__)


FIELD_DEFS = [
    # (name, label, type, default, extra)
    # -- Personal --
    ("Age",                    "Age",                         "number", 30,  {"min": 18, "max": 65}),
    ("Gender",                 "Gender",                      "select", "Male", {"options": ["Male", "Female"]}),
    ("MaritalStatus",          "Marital Status",              "select", "Married", {"options": ["Married", "Single", "Divorced"]}),
    ("Education",              "Education (1-5)",             "number", 3,   {"min": 1, "max": 5}),
    ("EducationField",         "Education Field",             "select", "Life Sciences",
                               {"options": ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                                            "Human Resources", "Other"]}),
    ("DistanceFromHome",       "Distance From Home (km)",     "number", 5,   {"min": 0, "max": 30}),
    # -- Work --
    ("BusinessTravel",         "Business Travel",             "select", "Travel_Rarely",
                               {"options": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]}),
    ("Department",             "Department",                  "select", "Research & Development",
                               {"options": ["Research & Development", "Sales", "Human Resources"]}),
    ("JobRole",                "Job Role",                    "select", "Research Scientist",
                               {"options": ["Sales Executive", "Research Scientist",
                                            "Laboratory Technician", "Manufacturing Director",
                                            "Healthcare Representative", "Manager",
                                            "Sales Representative", "Research Director",
                                            "Human Resources"]}),
    ("JobLevel",               "Job Level (1-5)",             "number", 2,   {"min": 1, "max": 5}),
    ("OverTime",               "OverTime",                    "select", "No", {"options": ["No", "Yes"]}),
    # -- Performance --
    ("JobInvolvement",         "Job Involvement (1-4)",       "number", 3,   {"min": 1, "max": 4}),
    ("JobSatisfaction",        "Job Satisfaction (1-4)",      "number", 3,   {"min": 1, "max": 4}),
    ("EnvironmentSatisfaction","Environment Satisfaction (1-4)","number", 3,  {"min": 1, "max": 4}),
    ("WorkLifeBalance",        "Work Life Balance (1-4)",     "number", 3,   {"min": 1, "max": 4}),
    ("PerformanceRating",      "Performance Rating (1-4)",    "number", 3,   {"min": 1, "max": 4}),
    ("RelationshipSatisfaction","Relationship Satisfaction (1-4)","number",3, {"min": 1, "max": 4}),
    ("TrainingTimesLastYear",  "Training Times Last Year",    "number", 2,   {"min": 0, "max": 6}),
    # -- Compensation --
    ("MonthlyIncome",          "Monthly Income",              "number", 5000, {"min": 1000, "max": 25000}),
    ("MonthlyRate",            "Monthly Rate",                "number", 20000, {"min": 2000, "max": 30000}),
    ("HourlyRate",             "Hourly Rate",                 "number", 65,  {"min": 15, "max": 100}),
    ("DailyRate",              "Daily Rate",                  "number", 800, {"min": 100, "max": 1500}),
    ("PercentSalaryHike",      "Percent Salary Hike",         "number", 15,  {"min": 0, "max": 25}),
    ("StockOptionLevel",       "Stock Option Level (0-3)",    "number", 0,   {"min": 0, "max": 3}),
    # -- Tenure --
    ("TotalWorkingYears",      "Total Working Years",         "number", 8,   {"min": 0, "max": 40}),
    ("NumCompaniesWorked",     "Num Companies Worked",        "number", 1,   {"min": 0, "max": 10}),
    ("YearsAtCompany",         "Years At Company",            "number", 5,   {"min": 0, "max": 40}),
    ("YearsInCurrentRole",     "Years In Current Role",       "number", 3,   {"min": 0, "max": 20}),
    ("YearsSinceLastPromotion","Years Since Last Promotion",  "number", 1,   {"min": 0, "max": 15}),
    ("YearsWithCurrManager",   "Years With Current Manager",  "number", 3,   {"min": 0, "max": 20}),
]

ALL_FIELD_NAMES = [f[0] for f in FIELD_DEFS]

FIELD_GROUPS = [
    ("Personal",     ["Age", "Gender", "MaritalStatus", "Education", "EducationField", "DistanceFromHome"]),
    ("Work",         ["BusinessTravel", "Department", "JobRole", "JobLevel", "OverTime"]),
    ("Performance",  ["JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction",
                      "WorkLifeBalance", "PerformanceRating", "RelationshipSatisfaction",
                      "TrainingTimesLastYear"]),
    ("Compensation", ["MonthlyIncome", "MonthlyRate", "HourlyRate", "DailyRate",
                      "PercentSalaryHike", "StockOptionLevel"]),
    ("Tenure",       ["TotalWorkingYears", "NumCompaniesWorked", "YearsAtCompany",
                      "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]),
]

FIELD_DEFS_DICT = {f[0]: f for f in FIELD_DEFS}


def build_row(form):
    """Turn POST data into a one-row DataFrame with ALL 30 required columns."""
    data = {}
    for name, label, ftype, default, extra in FIELD_DEFS:
        val = request.form.get(name, "")
        if ftype == "number":
            try:
                data[name] = int(float(val)) if val else default
            except (ValueError, TypeError):
                data[name] = default
        else:
            data[name] = val if val else default
    return pd.DataFrame([data], columns=ALL_FIELD_NAMES)


def risk_label(prob):
    if prob >= 0.5:
        return "HIGH", "high"
    if prob >= 0.25:
        return "MODERATE", "moderate"
    return "LOW", "low"


def waterfall_b64(shap_obj, row_idx=0):
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_obj[row_idx], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def whatif(template_row, current_proba, label, field, op, val):
    row = template_row.copy()
    if op == "set":
        row.loc[0, field] = val
    elif op == "mul":
        row.loc[0, field] = int(row.loc[0, field] * val)
    new_proba = float(pipeline.predict_proba(row)[:, 1][0])
    return {"label": label, "new_proba": new_proba, "drop": max(current_proba - new_proba, 0.0)}


pipeline = None
explainer = None


def ensure_model():
    global pipeline, explainer
    if pipeline is not None:
        return
    pipeline = ex.load_pipeline(MODEL_PATH)
    df = pre.load_data(DATA_PATH)
    _, X_train, _, _, _, _, _ = pre.prepare_datasets(df)
    background = ex.get_background(pipeline, X_train, sample_size=50)
    explainer = ex.create_explainer(pipeline, background)


@app.route("/")
def index():
    ensure_model()
    return render_template("index.html", field_groups=FIELD_GROUPS,
                           defs=FIELD_DEFS_DICT)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    ensure_model()
    if request.method == "POST":
        form = request.form
    else:
        form = request.args

    row = build_row(form)
    proba = float(pipeline.predict_proba(row)[:, 1][0])
    label, level = risk_label(proba)

    shap_values, feature_names = ex.explain_raw_rows(pipeline, explainer, row)
    factors = ex.top_pushing_factors(shap_values, feature_names, n=3)
    plan = ex.build_action_plan(shap_values, 0, row, feature_names, top_n=3)
    waterfall = waterfall_b64(shap_values)

    sims = [
        whatif(row, proba, "Remove overtime", "OverTime", "set", "No"),
        whatif(row, proba, "Raise monthly income +20%", "MonthlyIncome", "mul", 1.2),
        whatif(row, proba, "Boost job satisfaction", "JobSatisfaction", "set", 4),
    ]

    return render_template("result.html", proba=proba, label=label, level=level,
                           factors=factors, plan=plan, waterfall=waterfall,
                           sims=sims, fields=row.iloc[0].to_dict())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
