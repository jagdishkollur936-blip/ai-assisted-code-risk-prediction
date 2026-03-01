import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def engineer_features(df):
    """
    Perform feature engineering:
    - Convert timestamps
    - Create duration features
    - Log transform skewed durations
    - Extract time-based features
    - Remove leakage & ID columns
    """

    df = df.copy()

    # -----------------------------
    # 1️⃣ Convert timestamp columns
    # -----------------------------
    timestamp_cols = [
        "jira_created_at",
        "work_start_at",
        "pr_created_at",
        "merged_at"
    ]

    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # -----------------------------
    # 2️⃣ Create duration features
    # -----------------------------
    if "merged_at" in df.columns and "pr_created_at" in df.columns:
        df["pr_duration_hours"] = (
            (df["merged_at"] - df["pr_created_at"]).dt.total_seconds() / 3600
        )

    if "merged_at" in df.columns and "work_start_at" in df.columns:
        df["work_duration_hours"] = (
            (df["merged_at"] - df["work_start_at"]).dt.total_seconds() / 3600
        )

    if "pr_created_at" in df.columns and "jira_created_at" in df.columns:
        df["jira_to_pr_hours"] = (
            (df["pr_created_at"] - df["jira_created_at"]).dt.total_seconds() / 3600
        )

    # -----------------------------
    # 3️⃣ Clip negative durations
    # -----------------------------
    duration_cols = ["pr_duration_hours", "work_duration_hours"]

    for col in duration_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # -----------------------------
    # 4️⃣ Log transform skewed durations
    # -----------------------------
    for col in duration_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # -----------------------------
    # 5️⃣ Extract time-based features
    # -----------------------------
    if "merged_at" in df.columns:
        df["merge_hour"] = df["merged_at"].dt.hour
        df["merge_dayofweek"] = df["merged_at"].dt.dayofweek

    # -----------------------------
    # 6️⃣ Drop leakage & ID columns
    # -----------------------------
    drop_cols = [
        # ID columns
        "pr_id",
        "jira_issue_key",
        "trace_id",
        "author_id",

        # Raw timestamps
        "jira_created_at",
        "work_start_at",
        "pr_created_at",
        "merged_at",

        # Post-failure leakage
        "root_cause_category",
        "failed_pipeline_step",
        "harness_pipeline_status",
        "error_log"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def build_feature_pipeline(X_train):
    """
    Create preprocessing pipeline for numerical and categorical features.
    """

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    return preprocessor