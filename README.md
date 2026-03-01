# 🚀 AI-Assisted Code Risk Prediction System

> Built an end-to-end Machine Learning system to predict software deployment failures using XGBoost. 
> Optimized decision threshold to prioritize recall and reduce missed high-risk changes. 
> Deployed the model via FastAPI and integrated with a Streamlit dashboard.
An end-to-end Machine Learning system that predicts the probability of software change failures before deployment.

The system analyzes Pull Request (PR) metadata and engineering signals to classify changes into:

- ✅ Low Risk
- ⚠️ Medium Risk
- 🚨 High Risk

---

## 🎯 Business Objective

Software deployment failures can cause:

- Production outages
- Revenue loss
- Emergency rollbacks
- Engineering firefighting

The goal of this project is to:

> Predict high-risk changes early so they can receive additional review before deployment.

Because missing a risky change is more costly than raising a false alarm, the model is optimized to **maximize recall for failure cases**.

---

## 🧠 Problem Formulation

Binary Classification Problem:

- Target: `change_failure`
- 1 → Deployment failed
- 0 → Deployment succeeded

Due to class imbalance, special care was taken to:

- Use stratified train/test split
- Handle imbalance with `scale_pos_weight`
- Optimize decision threshold for recall

---

## 📊 Features Used (Production-Aligned)

### Numerical Features
- task_complexity
- pr_size_loc
- rework_time_hours
- coding_time_hours
- review_time_hours
- deployment_pressure
- ai_acceptance_rate
- num_dependencies

### Categorical Features
- author_seniority
- service_name
- is_legacy_codebase

---

## ⚙️ Model Approach

- Model: **XGBoost Classifier**
- Preprocessing:
  - Median imputation (numerical)
  - Most frequent imputation (categorical)
  - One-hot encoding
- Train/Test Split: Stratified (80/20)

---

## 🎯 Business Threshold Optimization

Instead of using default 0.5 probability threshold, threshold tuning was performed.

Final Decision Threshold:

This achieves higher recall for failure detection, reducing the chance of missing risky deployments.

---

## 📈 Evaluation Strategy
### 🔍 Model Selection & Optimization

- Handled class imbalance using `scale_pos_weight`
- Used stratified train/test split
- Evaluated using ROC-AUC and precision-recall trade-offs
- Tuned probability threshold from 0.5 → 0.35 to maximize recall

This reduced false negatives and aligned the model with real-world business risk.
Metrics evaluated:

- Precision
- Recall
- F1-score
- ROC-AUC
- Threshold sensitivity analysis

Recall for failure class was prioritized due to higher business cost of false negatives.

---

## 🏗 System Architecture


### Frontend — Streamlit
- Interactive dashboard
- User inputs PR details
- Displays:
  - Failure probability
  - Risk score (0–100)
  - Risk category

---

## 🔄 Prediction Flow

User  
→ Streamlit UI or API  
→ Preprocessing Pipeline  
→ XGBoost Model  
→ Risk Scoring Layer  
→ JSON Response  

---

## 📦 Project Structure
ai-assisted-code-risk-prediction/
│
├── src/ # Modular ML components
│ ├── data/
│ ├── features/
│ ├── models/
│ ├── risk/
│
├── models/ # Saved model artifacts
├── notebooks/ # Experimentation notebooks
├── streamlit_app.py # Interactive dashboard
└── README.md

---

