import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------
# 1️⃣ Feature Engineering Step (Structural Changes)
# ---------------------------------------------------
def engineer_features(df):
    """
    Create engineered features from raw dataset.
    This modifies structure (adds/drops columns).
    """

    df = df.copy()  # avoid modifying original dataframe

    # Timestamp columns
    timestamp_cols = [ "jira_created_at","work_start_at","pr_created_at","merged_at" ]

    # Convert to datetime
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Duration features
    if "merged_at" in df.columns and "pr_created_at" in df.columns:
        df["pr_duration_hours"] = ((df["merged_at"] - df["pr_created_at"]).dt.total_seconds() / 3600)

    if "merged_at" in df.columns and "work_start_at" in df.columns:
        df["work_duration_hours"] = ((df["merged_at"] - df["work_start_at"]).dt.total_seconds() / 3600)

    if "pr_created_at" in df.columns and "jira_created_at" in df.columns:
        df["jira_to_pr_hours"] = ((df["pr_created_at"] - df["jira_created_at"]).dt.total_seconds() / 3600)

    # Extract hour & weekday
    if "merged_at" in df.columns:
        df["merge_hour"] = df["merged_at"].dt.hour
        df["merge_dayofweek"] = df["merged_at"].dt.dayofweek

    # Drop ID & raw timestamp columns
    drop_cols = [ "pr_id","jira_issue_key","trace_id","jira_created_at","work_start_at","pr_created_at","merged_at"]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


# ---------------------------------------------------
# 2️⃣ Preprocessing Pipeline (Imputation + Encoding)
# ---------------------------------------------------
def build_feature_pipeline(X_train):
    """
    Create preprocessing pipeline for numeric and categorical columns.
    """

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_cols),
                      ("cat", categorical_pipeline, categorical_cols)])

    return preprocessor