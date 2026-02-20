import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# DATA CONFIGURATION
# =========================

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "engineering_metrics.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")

TARGET_COLUMN = "change_failure"

# =========================
# MODEL CONFIGURATION
# =========================

MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "model_v1.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# =========================
# RISK CONFIGURATION
# =========================

LOW_RISK_THRESHOLD = 30
HIGH_RISK_THRESHOLD = 70