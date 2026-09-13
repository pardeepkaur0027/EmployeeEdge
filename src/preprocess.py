"""Reusable preprocessing utilities for EmployeeEdge.

Handles loading, cleaning, feature/target split, and the ``ColumnTransformer``
that turns raw HR rows into the model's feature matrix. Train, tune, explain
and the serving app all reuse these exact building blocks so there is a single
source of truth and zero leakage between training and serving.
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
RANDOM_STATE = 42


def load_data(path=DEFAULT_DATA_PATH):
    """Load the raw attrition CSV and print its shape."""
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def split_features_target(df_clean):
    """Return (X, y): features DataFrame + binary target Series."""
    X = df_clean.drop(columns=["Attrition"])
    y = df_clean["Attrition"].map({"Yes": 1, "No": 0})
    return X, y


def get_column_splits(df_clean):
    """Return (numerical_cols, categorical_cols) from the cleaned DataFrame."""
    numerical_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "Attrition"]
    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    """ColumnTransformer used by EVERY model pipeline (no leakage by design).

    - numerical cols: StandardScaler
    - categorical cols: OneHotEncoder (handle_unknown='ignore' so the app can
      receive category values it never saw during training)

    Feature names come out prefixed ``num__`` / ``cat__`` so the serving app and
    the SHAP explanations can map encoded features back to human-readable ones.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split on the (binary) target."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def prepare_datasets(df):
    """Full pipeline: drop junk cols -> split X/y -> split rows.

    Returns (df_clean, X_train, X_test, y_train, y_test,
             numerical_cols, categorical_cols).
    """
    df_clean = df.drop(columns=DROP_COLS)
    X, y = split_features_target(df_clean)
    numerical_cols, categorical_cols = get_column_splits(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return df_clean, X_train, X_test, y_train, y_test, numerical_cols, categorical_cols
