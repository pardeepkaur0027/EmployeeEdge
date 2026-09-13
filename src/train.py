"""Model training, tuning, evaluation and cost-benefit utilities for EmployeeEdge.

The reusable counterpart of the modeling notebook: 

  - build the 3 candidate pipelines   (LR / HistGradientBoosting / XGBoost)
  - cross-validate + grid-search them
  - evaluate on a held-out test set
  - estimate the $ value of acting on the highest-risk employees

Everything returns/loads a `Pipeline` (preprocessor + model) so the served
predictions always run the exact same feature engineering as training.
"""

import os

import joblib

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from . import preprocess as pre

RANDOM_STATE = 42
MODEL_SAVE_PATH = "../outputs/models/model.joblib"


def build_preprocessor(numerical_cols, categorical_cols):
    return pre.build_preprocessor(numerical_cols, categorical_cols)


def build_pipelines(numerical_cols, categorical_cols, y_train):
    """Return the 3 candidate pipelines, each = preprocessor + one model.

    All three are imbalance-aware, each in its own natural way:
      - LogisticRegression        : class_weight='balanced'
      - HistGradientBoosting      : class_weight='balanced'
      - XGBoost                   : scale_pos_weight = negatives / positives
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    preprocessor = pre.build_preprocessor(numerical_cols, categorical_cols)

    pipeline_lr = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipeline_hgb = Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipeline_xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE)),
    ])
    return {
        "LogisticRegression": pipeline_lr,
        "HistGradientBoosting": pipeline_hgb,
        "XGBoost": pipeline_xgb,
    }


def cross_validate(pipelines, X_train, y_train, cv=5):
    """Mean cross-validated ROC-AUC of each default (untuned) pipeline."""
    cv_scores = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = round(float(scores.mean()), 4)
        print(f"{name:22s} default CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return cv_scores


def tune_models(pipelines, X_train, y_train, cv=5):
    """Grid-search each pipeline (ROC-AUC); return (best_estimators, best_cv)."""
    param_grids = {
        "LogisticRegression": {"model__C": [0.01, 0.1, 1.0, 10, 100]},
        "HistGradientBoosting": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [100, 200],
            "model__max_leaf_nodes": [15, 31],
        },
        "XGBoost": {
            "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
        },
    }

    best_estimators = {}
    best_cv = {}
    for name, pipe in pipelines.items():
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_estimators[name] = grid.best_estimator_
        best_cv[name] = round(float(grid.best_score_), 4)
        print(f"{name:22s} tuned  CV AUC: {grid.best_score_:.4f}  {grid.best_params_}")
    return best_estimators, best_cv


def evaluate_models(best_estimators, X_test, y_test):
    """Test-set metric table (accuracy/precision/recall/f1/AUC) per model."""
    rows = []
    for name, pipe in best_estimators.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred), 3),
            "recall": round(recall_score(y_test, y_pred), 3),
            "f1": round(f1_score(y_test, y_pred), 3),
            "auc": round(roc_auc_score(y_test, y_proba), 3),
        })
        print(f"\nConfusion matrix — {name}\n{confusion_matrix(y_test, y_pred)}")
    return (pd.DataFrame(rows).set_index("model"))[["accuracy", "precision",
                                                    "recall", "f1", "auc"]]


def optimize_threshold(pipe, X_val, y_val):
    """Find the probability cut-off that maximizes F1 on the validation set."""
    from sklearn.metrics import f1_score as _f1
    proba = pipe.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.80, 0.01):
        f1 = _f1(y_val, (proba >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def cost_benefit(pipe, X_test, y_test, monthly_income_by_index,
                 replacement_multiple=1.5, intervention_cost=2500,
                 n_intervene=30, proba_threshold=0.5):
    """Estimate the $ value of acting on the n highest-risk test employees.

    Conservative, transparent assumptions (all tunable):
      - replacing a leaver costs `replacement_multiple` x their annual salary
      - we spend `intervention_cost` per employee we act on
      - we only "save" employees who would ACTUALLY leave (true positives in
        the top-n riskiest people, using the tuned threshold)

    Returns a dict of headline numbers for the app/notebook.
    """
    from . import explain as ex
    proba = pipe.predict_proba(X_test)[:, 1]
    order = np.argsort(proba)[::-1][:n_intervene]
    at_risk_leavers = ((proba[order] >= proba_threshold) & (y_test.iloc[order] == 1)).sum()
    salaries = np.array([monthly_income_by_index.get(i, np.nan)
                         for i in X_test.index[order]]) * 12
    mean_replacement = np.nanmean(salaries) * replacement_multiple
    cost_saved = at_risk_leavers * mean_replacement
    cost_spent = n_intervene * intervention_cost
    net = cost_saved - cost_spent
    result = {
        "at_risk_leavers_caught": int(at_risk_leavers),
        "avg_replacement_cost": round(float(mean_replacement), 0),
        "cost_saved": round(float(cost_saved), 0),
        "cost_spent": round(float(cost_spent), 0),
        "net_saving": round(float(net), 0),
        "roi_multiple": round(float(net / cost_spent), 1) if cost_spent else 0.0,
    }
    print(f"Caught {at_risk_leavers}/{n_intervene} at-risk leavers in top-{n_intervene}")
    print(f"Est. savings ${cost_saved:,.0f} vs. cost ${cost_spent:,.0f} "
          f"-> net ${net:,.0f} ({result['roi_multiple']}x ROI)")
    return result


def save_model(pipeline, path=MODEL_SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved model -> {path}")


def load_model(path=MODEL_SAVE_PATH):
    return joblib.load(path)
