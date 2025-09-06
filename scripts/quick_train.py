import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, f1_score, precision_recall_curve,
    roc_curve, precision_score, recall_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV

class QuickFraudTrainer:
    def __init__(self, data_path="data/raw/Fraud.csv", model_dir="models", results_dir="results"):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.models_config = {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__class_weight': ['balanced']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20],
                    'model__class_weight': ['balanced']
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.1, 0.15],
                    'model__max_depth': [5, 7],
                    'model__scale_pos_weight': [10]
                }
            }
        }
        
        self.trained_models = {}
        self.model_scores = {}
    
    def load_and_preprocess_data(self, sample_size=None):
        print("Loading and preprocessing data...")
        
        df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        if sample_size and len(df) > sample_size:
            fraud_samples = df[df['isFraud'] == 1].sample(n=min(sample_size//10, len(df[df['isFraud'] == 1])), random_state=42)
            normal_samples = df[df['isFraud'] == 0].sample(n=sample_size - len(fraud_samples), random_state=42)
            df = pd.concat([fraud_samples, normal_samples]).reset_index(drop=True)
            print(f"Sampled to {df.shape[0]} rows for faster training")
        
        df = df.dropna()
        
        df_processed = self.engineer_features(df)
        
        target_col = 'isFraud'
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def engineer_features(self, df):
        df_fe = df.copy()
        
        columns_to_drop = ['nameOrig', 'nameDest']
        for col in columns_to_drop:
            if col in df_fe.columns:
                df_fe = df_fe.drop(columns=[col])
        
        if 'type' in df_fe.columns:
            df_fe = pd.get_dummies(df_fe, columns=['type'], prefix='type')
        
        if 'amount' in df_fe.columns:
            df_fe['amount_log'] = np.log1p(df_fe['amount'])
            df_fe['is_high_value'] = (df_fe['amount'] >= 200000).astype(int)
        
        balance_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in balance_cols:
            if col in df_fe.columns:
                df_fe[f'{col}_log'] = np.log1p(df_fe[col])
        
        if all(col in df_fe.columns for col in ['oldbalanceOrg', 'newbalanceOrig', 'amount']):
            df_fe['balance_diff_orig'] = df_fe['newbalanceOrig'] - df_fe['oldbalanceOrg']
            df_fe['error_orig'] = (df_fe['newbalanceOrig'] + df_fe['amount'] != df_fe['oldbalanceOrg']).astype(int)
        
        if all(col in df_fe.columns for col in ['oldbalanceDest', 'newbalanceDest', 'amount']):
            df_fe['balance_diff_dest'] = df_fe['newbalanceDest'] - df_fe['oldbalanceDest']
            df_fe['error_dest'] = (df_fe['oldbalanceDest'] + df_fe['amount'] != df_fe['newbalanceDest']).astype(int)
        
        return df_fe
    
    def create_preprocessing_pipeline(self, X):
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[('num', RobustScaler(), numeric_features)],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train_model(self, model_name, X_train, y_train, cv_folds=3):
        print(f"Training {model_name}...")
        
        config = self.models_config[model_name]
        preprocessor = self.create_preprocessing_pipeline(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline, config['params'], cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best score for {model_name}: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train):
        print("=" * 60)
        print("TRAINING ESSENTIAL MODELS")
        print("=" * 60)
        
        for model_name in self.models_config.keys():
            try:
                trained_model = self.train_model(model_name, X_train, y_train)
                self.trained_models[model_name] = trained_model
                print(f"✅ {model_name} trained successfully")
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
    
    def apply_calibration(self, X_train, y_train):
        print("Applying calibration...")
        
        calibrated_models = {}
        
        for name, model in self.trained_models.items():
            try:
                calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                calibrated.fit(X_train, y_train)
                calibrated_models[f'calibrated_{name}'] = calibrated
                print(f"✅ Calibrated {name}")
            except Exception as e:
                print(f"❌ Error calibrating {name}: {str(e)}")
        
        self.trained_models.update(calibrated_models)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nEVALUATING MODELS")
        print("=" * 60)
        
        evaluation_results = {}
        
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'average_precision': average_precision_score(y_test, y_pred_proba),
                    'f1_score': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'accuracy': (y_pred == y_test).mean()
                }
                
                evaluation_results[name] = metrics
                
                print(f"\n{name}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                    
            except Exception as e:
                print(f"❌ Error evaluating {name}: {str(e)}")
        
        self.model_scores = evaluation_results
        return evaluation_results
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.trained_models.items():
            model_path = self.model_dir / f"{name}_pipeline_{timestamp}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save evaluation results
        results_path = self.results_dir / f"model_evaluation_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.model_scores, f, indent=2)
        
        print(f"Saved evaluation results to {results_path}")
    
    def run_quick_training(self, sample_size=100000):
        """Run quick training pipeline"""
        print("🚀 Starting Quick Fraud Detection Model Training")
        print("=" * 60)
        
        # Load data with sampling for speed
        X, y = self.load_and_preprocess_data(sample_size=sample_size)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train models
        self.train_all_models(X_train, y_train)
        
        # Apply calibration
        self.apply_calibration(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Save models
        self.save_models()
        
        print(f"\n🎉 Quick training completed!")
        print(f"Total models trained: {len(self.trained_models)}")
        
        # Show top models
        if self.model_scores:
            print("\n🏆 Top Performing Models:")
            sorted_models = sorted(
                self.model_scores.items(), 
                key=lambda x: x[1]['roc_auc'], 
                reverse=True
            )
            
            for i, (name, scores) in enumerate(sorted_models[:3], 1):
                print(f"{i}. {name}: ROC-AUC = {scores['roc_auc']:.4f}")


def main():
    """Main execution"""
    trainer = QuickFraudTrainer()
    trainer.run_quick_training(sample_size=50000)  # Use 50K samples for speed


if __name__ == "__main__":
    main()
