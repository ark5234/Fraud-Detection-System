"""
Enhanced Model Training Script for Fraud Detection System

This module provides comprehensive model training capabilities including:
- Multiple machine learning algorithms
- Advanced calibration techniques
- Model ensemble methods
- Hyperparameter optimization
- Performance evaluation and comparison
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json

# Core ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, f1_score, precision_recall_curve,
    roc_curve, precision_score, recall_score
)

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# Calibration imports
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, RFE

class EnhancedFraudModelTrainer:
    """
    Enhanced fraud detection model trainer with multiple algorithms,
    calibration techniques, and ensemble methods.
    """
    
    def __init__(self, data_path="data/raw/Fraud.csv", model_dir="models", results_dir="results"):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models_config = {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=2000, random_state=42),
                'params': {
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__class_weight': ['balanced', None],
                    'model__solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__class_weight': ['balanced', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.05, 0.1, 0.15],
                    'model__max_depth': [3, 5, 7]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.05, 0.1, 0.15],
                    'model__max_depth': [3, 5, 7],
                    'model__scale_pos_weight': [1, 10, 20]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.05, 0.1, 0.15],
                    'model__max_depth': [3, 5, 7],
                    'model__class_weight': ['balanced', None]
                }
            },
            'svm': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf', 'linear'],
                    'model__class_weight': ['balanced', None]
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'model__hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'model__learning_rate_init': [0.001, 0.01],
                    'model__alpha': [0.0001, 0.001]
                }
            },
            'adaboost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.5, 1.0, 1.5]
                }
            }
        }
        
        self.trained_models = {}
        self.model_scores = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the fraud detection dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Basic preprocessing
        df = df.dropna()
        
        # Feature engineering (same as in original notebook)
        df_processed = self.engineer_features(df)
        
        # Separate features and target
        target_col = 'isFraud'
        if target_col not in df_processed.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def engineer_features(self, df):
        """Apply comprehensive feature engineering"""
        df_fe = df.copy()
        
        # Amount-based features
        if 'amount' in df_fe.columns:
            df_fe['amount_log'] = np.log1p(df_fe['amount'])
            df_fe['amount_squared'] = df_fe['amount'] ** 2
            df_fe['amount_sqrt'] = np.sqrt(df_fe['amount'])
        
        # Balance-based features
        balance_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in balance_cols:
            if col in df_fe.columns:
                df_fe[f'{col}_log'] = np.log1p(df_fe[col])
        
        # Transaction type encoding
        if 'type' in df_fe.columns:
            df_fe = pd.get_dummies(df_fe, columns=['type'], prefix='type')
        
        # Balance difference features
        if 'oldbalanceOrg' in df_fe.columns and 'newbalanceOrig' in df_fe.columns:
            df_fe['balance_diff_orig'] = df_fe['newbalanceOrig'] - df_fe['oldbalanceOrg']
            df_fe['balance_diff_dest'] = df_fe['newbalanceDest'] - df_fe['oldbalanceDest']
        
        # Error flags
        if 'amount' in df_fe.columns:
            df_fe['error_orig'] = (df_fe['newbalanceOrig'] + df_fe['amount'] != df_fe['oldbalanceOrg']).astype(int)
            df_fe['error_dest'] = (df_fe['oldbalanceDest'] + df_fe['amount'] != df_fe['newbalanceDest']).astype(int)
        
        return df_fe
    
    def create_preprocessing_pipeline(self, X):
        """Create preprocessing pipeline for features"""
        # Identify numeric columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train_single_model(self, model_name, X_train, y_train, cv_folds=5):
        """Train a single model with hyperparameter optimization"""
        print(f"\nTraining {model_name}...")
        
        config = self.models_config[model_name]
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X_train)
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])
        
        # Hyperparameter optimization
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline,
            config['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best score for {model_name}: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train):
        """Train all configured models"""
        print("=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)
        
        for model_name in self.models_config.keys():
            try:
                trained_model = self.train_single_model(model_name, X_train, y_train)
                self.trained_models[model_name] = trained_model
                print(f"✅ {model_name} trained successfully")
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
    
    def create_ensemble_models(self, X_train, y_train):
        """Create ensemble models from trained base models"""
        print("\nCreating ensemble models...")
        
        if len(self.trained_models) < 2:
            print("Need at least 2 trained models for ensemble")
            return
        
        # Select best performing models for ensemble
        base_models = list(self.trained_models.items())[:5]  # Top 5 models
        
        # Voting classifier (soft voting for probabilities)
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X_train)
        ensemble_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ensemble', voting_clf)
        ])
        
        ensemble_pipeline.fit(X_train, y_train)
        self.trained_models['ensemble_voting'] = ensemble_pipeline
        
        print("✅ Ensemble model created")
    
    def apply_calibration(self, X_train, y_train):
        """Apply probability calibration to models"""
        print("\nApplying probability calibration...")
        
        calibrated_models = {}
        
        for name, model in self.trained_models.items():
            if name.startswith('calibrated_'):
                continue
                
            try:
                # Platt scaling
                platt_calibrated = CalibratedClassifierCV(
                    model, method='sigmoid', cv=3
                )
                platt_calibrated.fit(X_train, y_train)
                calibrated_models[f'calibrated_platt_{name}'] = platt_calibrated
                
                # Isotonic regression
                isotonic_calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                isotonic_calibrated.fit(X_train, y_train)
                calibrated_models[f'calibrated_isotonic_{name}'] = isotonic_calibrated
                
                print(f"✅ Calibrated {name}")
                
            except Exception as e:
                print(f"❌ Error calibrating {name}: {str(e)}")
                continue
        
        self.trained_models.update(calibrated_models)
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        evaluation_results = {}
        
        for name, model in self.trained_models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
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
                continue
        
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
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        print("🚀 Starting Enhanced Fraud Detection Model Training")
        print("=" * 60)
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train all models
        self.train_all_models(X_train, y_train)
        
        # Create ensemble models
        self.create_ensemble_models(X_train, y_train)
        
        # Apply calibration
        self.apply_calibration(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Save models
        self.save_models()
        
        print("\n🎉 Training completed successfully!")
        print(f"Total models trained: {len(self.trained_models)}")
        
        # Print top performing models
        if self.model_scores:
            print("\n🏆 Top Performing Models (by ROC-AUC):")
            sorted_models = sorted(
                self.model_scores.items(), 
                key=lambda x: x[1]['roc_auc'], 
                reverse=True
            )
            
            for i, (name, scores) in enumerate(sorted_models[:5], 1):
                print(f"{i}. {name}: ROC-AUC = {scores['roc_auc']:.4f}")


def main():
    """Main execution function"""
    # Check for required packages
    try:
        import lightgbm
    except ImportError:
        print("Installing LightGBM...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm'])
    
    # Initialize and run trainer
    trainer = EnhancedFraudModelTrainer()
    trainer.run_full_training()


if __name__ == "__main__":
    main()
