"""Model training and evaluation utilities for EmployeeEdge.

Builds the three candidate pipelines (Logistic Regression, HistGradientBoosting,
XGBoost), cross-validates them, tunes hyperparameters, evaluates on a held-out
test set and saves the winning model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

RANDOM_STATE = 42
MODEL_SAVE_PATH = "../outputs/models/model.joblib"


def build_pipelines(numerical_cols, categorical_cols, y_train):
    from src import preprocess as pre

    preprocessor = pre.build_preprocessor(numerical_cols, categorical_cols)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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


def run_baseline(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    return evaluate(y_test, dummy.predict(X_test),
                    dummy.predict_proba(X_test)[:, 1])


def cross_validate(pipelines, X_train, y_train, cv=5):
    return {
        name: cross_val_score(pipe, X_train, y_train, cv=cv,
                              scoring="roc_auc", n_jobs=-1).mean()
        for name, pipe in pipelines.items()
    }


def tune_models(pipelines, param_grids, X_train, y_train, cv=5):
    best = {}
    for name, pipe in pipelines.items():
        grid = GridSearchCV(pipe, param_grids[name], cv=cv,
                            scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        best[name] = grid.best_estimator_
        print(f"{name:20s} best CV AUC: {grid.best_score_:.4f}  {grid.best_params_}")
    return best


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
    }


def evaluate_models(best_estimators, X_test, y_test):
    labels = []
    for name, pipe in best_estimators.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, y_pred, y_proba)
        m["model"] = name
        labels.append(m)
    df = pd.DataFrame(labels).set_index("model")
    return df[["accuracy", "precision", "recall", "f1", "auc"]]


def save_model(pipeline, path=MODEL_SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved -> {path}")


def load_model(path=MODEL_SAVE_PATH):
    return joblib.load(path)