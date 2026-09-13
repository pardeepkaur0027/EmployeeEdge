"""Reusable preprocessing utilities for EmployeeEdge.

Handles loading, cleaning, splitting and building the ColumnTransformer
that converts raw HR data into the model-ready feature matrix.
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
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def split_features_target(df_clean):
    X = df_clean.drop(columns=["Attrition"])
    y = df_clean["Attrition"].map({"Yes": 1, "No": 0})
    return X, y


def get_column_splits(df_clean):
    numerical_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include="object").columns.tolist()
    categorical_cols.remove("Attrition")
    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols),
        ]
    )


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def prepare_datasets(df):
    """One-stop helper: clean -> X/y -> column splits -> stratified split."""
    df_clean = df.drop(columns=DROP_COLS)
    X, y = split_features_target(df_clean)
    numerical_cols, categorical_cols = get_column_splits(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return df_clean, X_train, X_test, y_train, y_test, numerical_cols, categorical_cols