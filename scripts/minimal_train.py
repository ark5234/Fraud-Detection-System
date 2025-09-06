"""
Minimal Model Training - Robust and Fast
Only trains essential models without extensive hyperparameter search
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report

# Essential models with default parameters
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

def load_and_prepare_data(data_path="data/raw/Fraud.csv", sample_size=20000):
    """Load and prepare data quickly"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Quick sampling for speed
    if len(df) > sample_size:
        fraud_samples = df[df['isFraud'] == 1].sample(n=min(sample_size//10, len(df[df['isFraud'] == 1])), random_state=42)
        normal_samples = df[df['isFraud'] == 0].sample(n=sample_size - len(fraud_samples), random_state=42)
        df = pd.concat([fraud_samples, normal_samples]).reset_index(drop=True)
    
    # Remove string columns
    df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
    
    # Basic feature engineering
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], prefix='type')
    
    # Amount features
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
        df['is_high_value'] = (df['amount'] >= 200000).astype(int)
    
    # Balance features and errors
    for col in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    if all(col in df.columns for col in ['oldbalanceOrg', 'newbalanceOrig', 'amount']):
        df['error_orig'] = (df['newbalanceOrig'] + df['amount'] != df['oldbalanceOrg']).astype(int)
    
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y

def create_models():
    """Create models with good default parameters"""
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=10,
            eval_metric='logloss'
        )
    }

def train_and_save_models():
    """Train models with minimal complexity"""
    print("🚀 Starting Minimal Model Training")
    print("=" * 50)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create preprocessor
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[('num', RobustScaler(), numeric_features)],
        remainder='passthrough'
    )
    
    # Get models
    models = create_models()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trained_models = {}
    results = {}
    
    # Train each model
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"✅ {name} - ROC-AUC: {roc_auc:.4f}")
            
            # Save model
            model_path = Path("models") / f"{name}_pipeline_{timestamp}.joblib"
            joblib.dump(pipeline, model_path)
            print(f"💾 Saved to {model_path}")
            
            trained_models[name] = pipeline
            results[name] = {'roc_auc': roc_auc}
            
            # Create calibrated version
            try:
                print(f"Creating calibrated {name}...")
                calibrated = CalibratedClassifierCV(pipeline, method='sigmoid', cv=3)
                calibrated.fit(X_train, y_train)
                
                cal_path = Path("models") / f"calibrated_{name}_pipeline_{timestamp}.joblib"
                joblib.dump(calibrated, cal_path)
                print(f"✅ Calibrated {name} saved")
                
            except Exception as e:
                print(f"⚠️ Calibration failed for {name}: {e}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    # Save results
    results_path = Path("results") / f"model_evaluation_{timestamp}.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Results saved to {results_path}")
    print(f"🤖 {len(trained_models)} models trained successfully")
    
    return trained_models, results

if __name__ == "__main__":
    trained_models, results = train_and_save_models()
    
    print("\n🏆 Final Results:")
    for name, metrics in results.items():
        print(f"  {name}: ROC-AUC = {metrics['roc_auc']:.4f}")
