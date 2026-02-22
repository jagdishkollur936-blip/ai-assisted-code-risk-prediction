from src.config import TARGET_COLUMN 




def validate_dataset(df):
    """
    validate dataset before training.
    Parameters:
        df(pd.Dataframe): Input dataset

    Raise:
        Exception: if validation failes
    """

    # 1️⃣ check if dataset is empty
    if df.empty:
        raise Exception("dataset is empty.")

    # 2️⃣ check if targetcolumn exists
    if TARGET_COLUMN not in df.columns:
        raise Exception(f"Target column {TARGET_COLUMN}not found in dataset.")


    # 3️⃣ Warn if duplicate rows exist
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Warning: Dataset contains {duplicate_count} duplicate rows.")

    # 4️⃣ Warn if missing values exist
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Dataset contains {missing_values} missing values.")

    return True