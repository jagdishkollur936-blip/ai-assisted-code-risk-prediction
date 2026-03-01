import shap
import pandas as pd


def generate_shap_values(model, X_sample):
    """
    Generate SHAP values for XGBoost model.
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values


def get_top_shap_features(shap_values, feature_names, top_n=10):
    """
    Get top contributing features globally.
    """

    import numpy as np

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)

    return feature_importance.head(top_n)