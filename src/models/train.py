from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train):
    """
    Train baseline Logistic Regression model.
    """

    model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear"
)

    model.fit(X_train, y_train)

    return model
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model
from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model