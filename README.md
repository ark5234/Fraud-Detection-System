# Fraud Detection Web App

A comprehensive Streamlit web application for detecting fraudulent financial transactions using machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Trained models from the fraud detection notebook

### Installation

1. **Clone/Download the project files**
   ```bash
   # Navigate to your project directory
   cd C:\Users\Vikra\OneDrive\Desktop\Accredian
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure models are available**
   Make sure you have run the fraud detection notebook and have trained models in the `models/` directory:
   - `models/xgb_pipeline.joblib` (XGBoost model)
   - `models/xgb_pipeline_calibrated.joblib` (Calibrated XGBoost)
   - `models/fraud_lr_pipeline_*.joblib` (Logistic Regression)

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   The app will automatically open at `http://localhost:8501`

## 🎯 Features

### Single Transaction Analysis
- Manual input form for individual transactions
- Real-time fraud probability calculation
- Visual probability gauges
- Multiple model predictions with confidence levels

### Batch Processing
- CSV file upload for multiple transactions
- Batch fraud detection
- Results summary and downloadable CSV
- Sample data for testing

### Model Comparison
- Support for multiple trained models
- Configurable decision thresholds
- Side-by-side model performance

### Interactive Configuration
- Adjustable fraud detection thresholds
- Model selection options
- Real-time parameter updates

## 📊 How It Works

1. **Data Input**: Upload CSV files or enter transaction details manually
2. **Feature Engineering**: Automatically applies the same transformations used in training:
   - Transaction type encoding
   - Amount logging and high-value flags
   - Time-based features (hour, day, weekend)
   - Balance inconsistency detection
   - Account behavior flags

3. **Prediction**: Uses trained ML models to calculate fraud probabilities
4. **Decision**: Applies configurable thresholds to classify transactions
5. **Visualization**: Displays results with confidence indicators and explanations

## 🔧 Model Details

The app supports three types of models:

- **Logistic Regression**: Fast, interpretable baseline with balanced class weights
- **XGBoost**: High-accuracy gradient boosting with class imbalance handling
- **Calibrated XGBoost**: XGBoost with isotonic calibration for better probability estimates

## 📈 Usage Examples

### Single Transaction
1. Navigate to "Single Prediction" tab
2. Fill in transaction details:
   - Time step (1-744)
   - Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
   - Amount and balance information
3. Click "Analyze Transaction"
4. View fraud probability and classification

### Batch Processing
1. Go to "Batch Prediction" tab
2. Upload a CSV file with transaction data
3. Click "Analyze Batch"
4. Download results with predictions

### Threshold Tuning
1. Use sidebar sliders to adjust thresholds
2. Higher thresholds = fewer false alarms, might miss some fraud
3. Lower thresholds = catch more fraud, more false positives
4. Optimize based on business requirements

## 🛠️ Customization

### Adding New Models
1. Train your model using the same preprocessing pipeline
2. Save with joblib to the `models/` directory
3. Update the `load_models()` function in `app.py`

### Modifying Features
1. Update the `engineer_features()` function
2. Ensure consistency with training pipeline
3. Test with sample data

### Styling
- Modify CSS in the app header
- Add custom themes and colors
- Update layout and components

## 📁 File Structure

```
Accredian/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/                         # Trained model files
│   ├── xgb_pipeline.joblib
│   ├── xgb_pipeline_calibrated.joblib
│   └── fraud_lr_pipeline_*.joblib
└── fraud_detection_case_study.ipynb  # Training notebook
```

## 🚨 Important Notes

1. **Model Availability**: Ensure trained models exist before running the app
2. **Feature Consistency**: Input data must match training feature schema
3. **Threshold Selection**: Tune thresholds based on business cost of false positives vs false negatives
4. **Performance**: For large batch files, consider processing in chunks
5. **Security**: Don't expose sensitive transaction data in production

## 🐛 Troubleshooting

### Common Issues

**"No trained models found"**
- Run the training notebook first
- Check that model files exist in `models/` directory
- Verify file permissions

**"Error loading models"**
- Check Python/scikit-learn version compatibility
- Ensure all dependencies are installed
- Verify model file integrity

**"Feature mismatch errors"**
- Ensure input data has required columns
- Check data types and formats
- Verify feature engineering consistency

**Performance issues**
- Reduce batch size for large uploads
- Use sampling for SHAP explanations
- Consider model optimization

## 📞 Support

For issues with:
- Model training: Check the Jupyter notebook
- App functionality: Review error messages and logs
- Feature engineering: Ensure data format consistency
- Performance: Monitor resource usage

## 🔄 Updates

To update the app:
1. Retrain models with new data
2. Update feature engineering if needed
3. Test with sample transactions
4. Deploy updated models

---

**Happy fraud detecting! 🛡️**
