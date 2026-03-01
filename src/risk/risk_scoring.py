import numpy as np
from src.config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD


def calculate_risk_score(probability):
    """
    Convert probability (0–1) to risk score (0–100)
    """
    return round(probability * 100, 2)


def assign_risk_category(risk_score):
    """
    Assign risk category based on thresholds
    """
    if risk_score < LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif risk_score < HIGH_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "High Risk"


def generate_risk_output(probability):
    """
    Generate full risk output dictionary
    """

    probability = float(probability)  # Convert numpy → Python float

    risk_score = float(calculate_risk_score(probability))
    risk_category = assign_risk_category(risk_score)

    return {
        "failure_probability": round(probability, 4),
        "risk_score": round(risk_score, 2),
        "risk_category": risk_category
    }