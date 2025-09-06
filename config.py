# Project Configuration
MODEL_DIR = "models"
DATA_DIR = "data"
RESULTS_DIR = "results"
NOTEBOOKS_DIR = "notebooks"

# Data Configuration
RAW_DATA_PATH = "data/raw/Fraud.csv"
DATA_DICTIONARY_PATH = "data/Data Dictionary.txt"

# Model Configuration
DEFAULT_MODELS = ["logistic_regression", "xgboost", "calibrated_xgboost"]
DEFAULT_THRESHOLDS = {
    "lr": 0.5,
    "xgb": 0.5,
    "calibrated": 0.5
}

# Training Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
QUICK_TRAINING_SAMPLE_SIZE = 50000

# App Configuration
PAGE_TITLE = "Fraud Detection System"
PAGE_ICON = "🛡️"
LAYOUT = "wide"

# Risk Levels
RISK_LEVELS = {
    "HIGH": {"threshold": 0.8, "color": "red", "icon": "🔴"},
    "MEDIUM": {"threshold": 0.5, "color": "orange", "icon": "🟡"},
    "LOW": {"threshold": 0.2, "color": "yellow", "icon": "🟠"},
    "MINIMAL": {"threshold": 0.0, "color": "green", "icon": "🟢"}
}
