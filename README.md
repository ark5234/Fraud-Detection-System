# Fraud Detection System

A comprehensive machine learning-based web application for real-time detection and analysis of fraudulent financial transactions. This system provides an intuitive interface for fraud analysts, data scientists, and financial institutions to assess transaction risk using multiple trained models.

## Overview

This application leverages advanced machine learning algorithms to identify potentially fraudulent financial transactions in real-time. The system supports both individual transaction analysis and batch processing, making it suitable for various operational scenarios from real-time monitoring to historical data analysis.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for large datasets)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Trained machine learning models (generated from the included Jupyter notebook)

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ark5234/Fraud-Detection-System.git
cd Fraud-Detection-System
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Machine Learning Models
Before running the web application, execute the Jupyter notebook to train the models:
```bash
jupyter notebook fraud_detection_case_study.ipynb
```
Run all cells to generate the following model files in the `models/` directory:
- `xgb_pipeline.joblib` - XGBoost classifier
- `xgb_pipeline_calibrated.joblib` - Calibrated XGBoost classifier  
- `fraud_lr_pipeline_*.joblib` - Logistic regression classifier

### 5. Launch the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Core Features

### Single Transaction Analysis
Real-time analysis of individual transactions with manual input form supporting all transaction types and balance configurations. Provides immediate risk assessment with probability scores and confidence intervals.

### Batch Processing  
Upload CSV files containing multiple transactions for bulk analysis. Supports processing of large datasets with automated feature engineering and downloadable results including probability scores and classifications.

### Model Comparison
Simultaneous evaluation using multiple machine learning algorithms including Logistic Regression, XGBoost, and Calibrated models. Compare performance across different approaches to optimize detection accuracy.

### Interactive Configuration
Dynamic threshold adjustment via intuitive sliders allowing real-time optimization of false positive and false negative rates based on business requirements and risk tolerance.

## Technical Architecture

### Machine Learning Models

**Logistic Regression**
- Baseline interpretable model with balanced class weights
- Fast inference suitable for real-time applications  
- Coefficient-based explainability for regulatory compliance

**XGBoost Classifier**
- Gradient boosting algorithm optimized for imbalanced datasets
- Advanced feature interactions and non-linear pattern detection
- Superior accuracy on complex fraud patterns

**Calibrated Models**
- Isotonic calibration applied to base models
- Improved probability estimates for threshold optimization
- Better uncertainty quantification for decision support

### Feature Engineering Pipeline

The system automatically applies comprehensive feature engineering:

**Transaction Features**
- Transaction type encoding and normalization
- Amount logarithmic transformation
- High-value transaction indicators

**Temporal Features**  
- Hour-of-day and day-of-week extraction
- Weekend transaction flags
- Time-based risk patterns

**Account Features**
- Balance transition analysis
- Account emptying detection
- Merchant vs customer identification

**Risk Indicators**
- Accounting error detection
- Suspicious balance patterns
- Transaction sequence analysis

## Data Requirements

### Input Format
CSV files with the following required columns:
- `step`: Time step (integer)
- `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- `amount`: Transaction amount (float)
- `oldbalanceOrg`: Origin account balance before transaction
- `newbalanceOrig`: Origin account balance after transaction  
- `oldbalanceDest`: Destination account balance before transaction
- `newbalanceDest`: Destination account balance after transaction
- `isFlaggedFraud`: Business rule flag (0/1)

### Optional Fields
- `nameOrig`: Origin account identifier
- `nameDest`: Destination account identifier
- `isFraud`: Ground truth labels for validation

## Performance Metrics

The system reports comprehensive performance metrics:

**Classification Metrics**
- ROC AUC: Area under receiver operating characteristic curve
- PR AUC: Precision-recall area under curve (preferred for imbalanced data)
- F2 Score: Weighted harmonic mean emphasizing recall
- Precision, Recall, F1-Score: Standard classification metrics

**Calibration Metrics**
- Brier Score: Probability calibration quality
- Reliability Diagram: Visual calibration assessment
- Expected Calibration Error: Quantitative calibration measure

## Usage Guidelines

### Single Transaction Analysis
1. Navigate to Single Prediction tab
2. Input transaction details using the form
3. Select models for comparison
4. Adjust thresholds based on risk tolerance
5. Review probability scores and classifications

### Batch Processing
1. Prepare CSV file with required columns
2. Upload file via Batch Prediction tab
3. Configure analysis parameters
4. Execute batch analysis
5. Download comprehensive results

### Threshold Optimization
- **Conservative**: Higher thresholds reduce false positives but may miss fraud
- **Aggressive**: Lower thresholds catch more fraud but increase false alarms
- **Balanced**: Use F2-optimized thresholds for recall-precision balance
- **Cost-based**: Configure based on business cost of false positives vs false negatives

## Security Considerations

### Data Privacy
- No transaction data is stored permanently
- All processing occurs locally
- No external API calls for sensitive data

### Model Security  
- Models are validated against adversarial examples
- Feature importance monitoring for drift detection
- Regular retraining recommended

### Deployment Security
- Use HTTPS in production environments
- Implement proper authentication and authorization
- Monitor system logs for unusual access patterns

## Troubleshooting

### Common Issues

**Model Loading Errors**
- Verify all model files exist in `models/` directory
- Check Python package versions match training environment
- Ensure sufficient memory for model loading

**Performance Issues**
- Reduce batch size for large CSV files
- Use sampling for SHAP explanations
- Monitor memory usage during processing

**Feature Mismatch**
- Verify CSV column names match requirements
- Check data types and formats
- Ensure no missing required fields

### Error Resolution

**Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**Memory Issues**
```bash
# Process data in smaller chunks
# Reduce visualization complexity
# Clear browser cache
```

**Model Compatibility**
```bash
# Retrain models with current scikit-learn version
# Check model serialization format
```

## Development and Customization

### Adding New Models
1. Train model using same preprocessing pipeline
2. Save with joblib to `models/` directory
3. Update `load_models()` function in `app.py`
4. Add model-specific configuration

### Custom Features
1. Modify `engineer_features()` function
2. Ensure consistency with training pipeline
3. Update documentation and validation

### UI Customization
1. Modify CSS in application header
2. Update layout and component styling
3. Add custom visualizations

## API Reference

### Core Functions

**load_models()**
- Loads trained models from filesystem
- Returns dictionary of available models
- Handles missing models gracefully

**engineer_features(df)**
- Applies feature engineering pipeline
- Input: Raw transaction DataFrame
- Output: Feature-engineered DataFrame

**predict_fraud(models, df, thresholds)**
- Generates fraud predictions
- Input: Models, features, thresholds
- Output: Probabilities and classifications

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new functions
- Update documentation for changes

## License and Disclaimer

This software is provided for educational and research purposes. Users are responsible for validating model performance in their specific environment and ensuring compliance with applicable regulations.

**Important**: Always validate fraud detection systems thoroughly before production deployment. Consider regulatory requirements, fairness testing, and ongoing monitoring for model drift.

## Support and Documentation

For additional support:
- Review the Jupyter notebook for detailed model development
- Check GitHub issues for common problems
- Consult scikit-learn and XGBoost documentation for algorithm details
- Consider professional consultation for production deployments
