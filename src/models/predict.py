import numpy as np
from src.risk.risk_scoring import generate_risk_output

# -----------------------------
# Business-Optimized Threshold
# -----------------------------
BUSINESS_THRESHOLD = 0.45


def predict_risk(model, preprocessor, input_df):
    """
    Generate risk prediction using trained model and business-optimized threshold.

    Parameters:
    ----------
    model : trained ML model
    preprocessor : fitted preprocessing pipeline
    input_df : pandas DataFrame containing input features

    Returns:
    -------
    dict : {
        failure_probability,
        risk_score,
        risk_category,
        predicted_label,
        decision_threshold
    }
    """

    # 1️⃣ Transform input features
    X_processed = preprocessor.transform(input_df)

    # 2️⃣ Get failure probability (class 1)
    probability = model.predict_proba(X_processed)[:, 1][0]

    # Convert numpy float to Python float (important for JSON serialization)
    probability = float(probability)

    # 3️⃣ Apply business threshold
    predicted_label = int(probability >= BUSINESS_THRESHOLD)

    # 4️⃣ Generate risk score + category
    result = generate_risk_output(probability)

    # 5️⃣ Add decision metadata
    result["predicted_label"] = predicted_label
    result["decision_threshold"] = BUSINESS_THRESHOLD

    return result