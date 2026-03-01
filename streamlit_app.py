import streamlit as st
import pandas as pd
import joblib

from src.models.predict import predict_risk

# --------------------------
# Load Model & Preprocessor
# --------------------------

model = joblib.load("models/xgb_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

st.title("🚀 AI-Assisted Code Risk Prediction")

st.write("Enter Pull Request details below to predict failure risk.")

# --------------------------
# Input Form
# --------------------------

task_complexity = st.slider("Task Complexity", 1, 10, 5)
pr_size_loc = st.number_input("PR Size (Lines of Code)", min_value=0, value=200)
rework_time_hours = st.number_input("Rework Time (Hours)", min_value=0.0, value=2.0)
coding_time_hours = st.number_input("Coding Time (Hours)", min_value=0.0, value=5.0)
review_time_hours = st.number_input("Review Time (Hours)", min_value=0.0, value=2.0)
deployment_pressure = st.slider("Deployment Pressure", 1, 10, 5)
ai_acceptance_rate = st.slider("AI Acceptance Rate", 0.0, 1.0, 0.5)
num_dependencies = st.number_input("Number of Dependencies", min_value=0, value=3)

author_seniority = st.selectbox(
    "Author Seniority",
    ["Junior", "Mid", "Senior"]
)

service_name = st.selectbox(
    "Service Name",
    ["Auth-Service", "Payment-Gateway", "Recommendation-Engine",
     "Search-Optimizer", "Inventory-DB"]
)

is_legacy_codebase = st.selectbox(
    "Legacy Codebase?",
    [0, 1]
)

# --------------------------
# Predict Button
# --------------------------

if st.button("Predict Risk"):

    input_data = pd.DataFrame([{
        "task_complexity": task_complexity,
        "pr_size_loc": pr_size_loc,
        "rework_time_hours": rework_time_hours,
        "coding_time_hours": coding_time_hours,
        "review_time_hours": review_time_hours,
        "deployment_pressure": deployment_pressure,
        "ai_acceptance_rate": ai_acceptance_rate,
        "num_dependencies": num_dependencies,
        "author_seniority": author_seniority,
        "service_name": service_name,
        "is_legacy_codebase": is_legacy_codebase
    }])

    result = predict_risk(model, preprocessor, input_data)

    st.subheader("Prediction Result")

    st.write(f"Failure Probability: **{result['failure_probability']}**")
    st.write(f"Risk Score: **{result['risk_score']}**")

    if result["risk_category"] == "High Risk":
        st.error("🚨 High Risk")
    elif result["risk_category"] == "Medium Risk":
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk")