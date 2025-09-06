# Project Structure

```
Fraud-Detection-System/
├── app.py                          # Main Streamlit application
├── config.py                       # Project configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # License file
├── .gitignore                      # Git ignore patterns
│
├── data/                           # Data directory
│   ├── raw/                        # Raw, unprocessed data
│   │   └── Fraud.csv              # Original fraud dataset
│   └── Data Dictionary.txt         # Data documentation
│
├── models/                         # Trained models
│   ├── legacy/                     # Old model files
│   │   └── fraud_lr_pipeline_*.joblib
│   ├── xgb_pipeline.joblib        # XGBoost model
│   └── xgb_pipeline_calibrated.joblib # Calibrated XGBoost
│
├── notebooks/                      # Jupyter notebooks
│   └── fraud_detection_case_study.ipynb # Original analysis
│
├── src/                           # Source code
│   └── model_training.py          # Comprehensive model training
│
├── scripts/                       # Utility scripts
│   └── quick_train.py             # Fast model training script
│
├── results/                       # Training results and metrics
│   └── model_evaluation_*.json    # Model performance results
│
├── docs/                          # Documentation
│   └── PROJECT_STRUCTURE.md       # This file
│
└── .venv311/                      # Virtual environment (not tracked)
```

## Directory Descriptions

### `/data`
- **`raw/`**: Original, unmodified datasets
- **Root**: Data documentation and processed datasets

### `/models`
- **`legacy/`**: Archived old model versions
- **Root**: Current production models

### `/notebooks`
- Jupyter notebooks for data exploration and analysis
- Research and experimental code

### `/src`
- Core source code for model training and utilities
- Reusable functions and classes

### `/scripts`
- Standalone scripts for specific tasks
- Quick training, data processing, etc.

### `/results`
- Model evaluation metrics
- Training logs and performance reports

### `/docs`
- Project documentation
- API references and guides

## File Naming Conventions

### Models
- `{algorithm}_pipeline_{timestamp}.joblib`
- Example: `xgb_pipeline_20250906_143022.joblib`

### Results
- `model_evaluation_{timestamp}.json`
- Example: `model_evaluation_20250906_143022.json`

### Scripts
- Descriptive names with underscores
- Example: `quick_train.py`, `data_preprocessing.py`

## Usage Guidelines

1. **Raw data** should never be modified directly
2. **Models** should be versioned with timestamps
3. **Results** should be saved after each training run
4. **Legacy files** should be moved to appropriate folders
5. **Documentation** should be updated with major changes
