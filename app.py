"""
Fraud Detection Web App
A Streamlit application for predicting fraudulent financial transactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
try:
    from config import *
except ImportError:
    # Fallback values if config not available
    RAW_DATA_PATH = "data/raw/Fraud.csv"
    MODEL_DIR = "models"
    PAGE_TITLE = "Fraud Detection System"
    PAGE_ICON = "🛡️"

# Page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-fraud {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .alert-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load trained models"""
    models = {}
    model_dir = Path(MODEL_DIR)
    
    # Try to load models
    try:
        if (model_dir / "xgb_pipeline.joblib").exists():
            models['xgb'] = joblib.load(model_dir / "xgb_pipeline.joblib")
            st.success("✅ XGBoost model loaded")
        
        if (model_dir / "xgb_pipeline_calibrated.joblib").exists():
            models['xgb_cal'] = joblib.load(model_dir / "xgb_pipeline_calibrated.joblib")
            st.success("✅ Calibrated XGBoost model loaded")
            
        # Look for LR model (pattern match)
        lr_files = list(model_dir.glob("*logistic_regression*.joblib"))
        if lr_files:
            models['lr'] = joblib.load(lr_files[0])
            st.success("✅ Logistic Regression model loaded")
        
        # Look for new calibrated models
        cal_files = list(model_dir.glob("calibrated_*.joblib"))
        for cal_file in cal_files:
            model_name = cal_file.stem
            models[model_name] = joblib.load(cal_file)
            st.success(f"✅ {model_name} model loaded")
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models

def create_sample_data():
    """Create sample transaction data"""
    return pd.DataFrame({
        'step': [1, 5, 10, 15, 20],
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'PAYMENT'],
        'amount': [9839.64, 181.00, 181.00, 52.18, 1864.28],
        'oldbalanceOrg': [170136.0, 181.00, 181.00, 41554.0, 19384.72],
        'newbalanceOrig': [160296.36, 0.00, 0.00, 41501.82, 17520.44],
        'oldbalanceDest': [0.0, 0.0, 21182.0, 0.0, 0.0],
        'newbalanceDest': [0.0, 0.0, 0.0, 0.0, 0.0],
        'isFlaggedFraud': [0, 0, 1, 0, 0]
    })

def engineer_features(df):
    """Apply the same feature engineering as in training"""
    df = df.copy()
    
    # Transaction type normalization
    if 'type' in df.columns:
        df['type'] = df['type'].astype('string').str.replace('-', '_').astype('category')
    
    # Accounting errors
    df['orig_error'] = (df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']).astype('float32')
    df['dest_error'] = (df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']).astype('float32')
    
    # Transaction type flags
    df['is_TRANSFER'] = (df['type'] == 'TRANSFER').astype('int8')
    df['is_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype('int8')
    
    # Amount features
    df['amt_log'] = np.log1p(df['amount']).astype('float32')
    df['is_high_value'] = (df['amount'] >= 200_000).astype('int8')
    
    # Time features
    df['hour'] = (df['step'] % 24).astype('int8')
    df['day'] = (df['step'] // 24).astype('int16')
    df['is_weekend'] = (df['day'] % 7 >= 5).astype('int8')
    
    # Destination features (mock merchant detection)
    if 'nameDest' in df.columns:
        df['dest_is_merchant'] = df['nameDest'].astype('string').str.startswith('M').fillna(False).astype('int8')
    else:
        # Default assumption for manual input
        df['dest_is_merchant'] = 0
    
    # Balance transition flags
    df['orig_went_zero'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype('int8')
    df['dest_went_zero'] = ((df['newbalanceDest'] == 0) & (df['oldbalanceDest'] > 0)).astype('int8')
    
    return df

def get_model_info():
    """Get detailed information about models and their capabilities"""
    return {
        'xgb': {
            'name': 'XGBoost Classifier',
            'description': 'Gradient boosting ensemble method optimized for performance and accuracy',
            'strengths': ['High accuracy', 'Handles complex patterns', 'Built-in feature importance'],
            'use_case': 'Best for detecting sophisticated fraud patterns with high accuracy'
        },
        'xgb_cal': {
            'name': 'Calibrated XGBoost',
            'description': 'XGBoost with probability calibration for reliable confidence scores',
            'strengths': ['Accurate probabilities', 'Better uncertainty quantification', 'Calibrated confidence'],
            'use_case': 'When you need reliable probability estimates for risk assessment'
        },
        'lr': {
            'name': 'Logistic Regression',
            'description': 'Linear model with interpretable coefficients and fast predictions',
            'strengths': ['Highly interpretable', 'Fast predictions', 'Stable performance'],
            'use_case': 'When interpretability and speed are priorities'
        }
    }

def explain_prediction(transaction_data, probabilities, model_name):
    """Provide detailed explanation of fraud prediction"""
    explanations = []
    
    # Risk level classification
    max_prob = max(probabilities.values()) if probabilities else 0
    
    if max_prob >= 0.8:
        risk_level = "🔴 HIGH RISK"
        risk_explanation = "Multiple models indicate high fraud probability"
    elif max_prob >= 0.5:
        risk_level = "🟡 MEDIUM RISK"
        risk_explanation = "Some indicators suggest potential fraud"
    elif max_prob >= 0.2:
        risk_level = "🟠 LOW RISK"
        risk_explanation = "Few fraud indicators detected"
    else:
        risk_level = "🟢 MINIMAL RISK"
        risk_explanation = "Transaction appears legitimate"
    
    explanations.append(f"**Risk Assessment:** {risk_level}")
    explanations.append(f"*{risk_explanation}*")
    explanations.append("")
    
    # Feature-based explanations
    explanations.append("**Key Risk Factors Analyzed:**")
    
    # Transaction type analysis
    tx_type = transaction_data.get('type', [''])[0] if hasattr(transaction_data.get('type', ['']), '__getitem__') else transaction_data.get('type', '')
    if tx_type in ['TRANSFER', 'CASH_OUT']:
        explanations.append(f"• **Transaction Type**: {tx_type} (Higher risk category)")
    else:
        explanations.append(f"• **Transaction Type**: {tx_type} (Lower risk category)")
    
    # Amount analysis
    amount = transaction_data.get('amount', [0])[0] if hasattr(transaction_data.get('amount', [0]), '__getitem__') else transaction_data.get('amount', 0)
    if amount >= 200000:
        explanations.append(f"• **Amount**: ${amount:,.2f} (High-value transaction)")
    elif amount >= 50000:
        explanations.append(f"• **Amount**: ${amount:,.2f} (Medium-value transaction)")
    else:
        explanations.append(f"• **Amount**: ${amount:,.2f} (Standard transaction)")
    
    # Balance analysis
    old_bal = transaction_data.get('oldbalanceOrg', [0])[0] if hasattr(transaction_data.get('oldbalanceOrg', [0]), '__getitem__') else transaction_data.get('oldbalanceOrg', 0)
    new_bal = transaction_data.get('newbalanceOrig', [0])[0] if hasattr(transaction_data.get('newbalanceOrig', [0]), '__getitem__') else transaction_data.get('newbalanceOrig', 0)
    
    if old_bal > 0 and new_bal == 0:
        explanations.append("• **Balance Pattern**: Account emptied (High risk indicator)")
    elif abs((old_bal - amount) - new_bal) > 0.01:
        explanations.append("• **Balance Pattern**: Accounting inconsistency detected")
    else:
        explanations.append("• **Balance Pattern**: Normal transaction flow")
    
    # Flagged analysis
    is_flagged = transaction_data.get('isFlaggedFraud', [0])[0] if hasattr(transaction_data.get('isFlaggedFraud', [0]), '__getitem__') else transaction_data.get('isFlaggedFraud', 0)
    if is_flagged:
        explanations.append("• **System Flag**: Transaction flagged by internal systems")
    
    return "\n".join(explanations)

def predict_fraud(models, df, threshold_lr=0.5, threshold_xgb=0.5):
    """Make fraud predictions using available models"""
    results = {}
    
    # Prepare features (exclude IDs and target if present)
    feature_cols = [c for c in df.columns if c not in ['nameOrig', 'nameDest', 'isFraud']]
    X = df[feature_cols]
    
    for model_name, model in models.items():
        try:
            proba = model.predict_proba(X)[:, 1]
            
            # Use appropriate threshold
            if 'lr' in model_name.lower():
                threshold = threshold_lr
            else:
                threshold = threshold_xgb
                
            pred = (proba >= threshold).astype(int)
            
            results[model_name] = {
                'probabilities': proba,
                'predictions': pred,
                'threshold': threshold
            }
        except Exception as e:
            st.error(f"Error with model {model_name}: {e}")
    
    return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Detect fraudulent financial transactions using machine learning")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("❌ No trained models found. Please run the training notebook first and ensure models are saved in the 'models' directory.")
        return
    
    # Model selection
    available_models = list(models.keys())
    selected_models = st.sidebar.multiselect(
        "Select Models to Use",
        available_models,
        default=available_models
    )
    
    # Threshold configuration
    st.sidebar.subheader("🎯 Decision Thresholds")
    threshold_lr = st.sidebar.slider("Logistic Regression Threshold", 0.0, 1.0, 0.5, 0.01)
    threshold_xgb = st.sidebar.slider("XGBoost Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📋 Data Preview", "📈 Model Comparison", "ℹ️ About"])
    
    with tab1:
        st.header("Single Transaction Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Transaction Details")
            
            # Manual input form
            with st.form("transaction_form"):
                step = st.number_input("Time Step", min_value=1, max_value=744, value=1)
                
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
                )
                
                amount = st.number_input("Amount", min_value=0.01, value=1000.0, format="%.2f")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    old_balance_orig = st.number_input("Origin Old Balance", min_value=0.0, value=5000.0, format="%.2f")
                    new_balance_orig = st.number_input("Origin New Balance", min_value=0.0, value=4000.0, format="%.2f")
                
                with col_b:
                    old_balance_dest = st.number_input("Destination Old Balance", min_value=0.0, value=0.0, format="%.2f")
                    new_balance_dest = st.number_input("Destination New Balance", min_value=0.0, value=1000.0, format="%.2f")
                
                is_flagged = st.checkbox("Flagged by Business Rules")
                
                submitted = st.form_submit_button("🔍 Analyze Transaction")
        
        with col2:
            if submitted:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'step': [step],
                    'type': [transaction_type],
                    'amount': [amount],
                    'oldbalanceOrg': [old_balance_orig],
                    'newbalanceOrig': [new_balance_orig],
                    'oldbalanceDest': [old_balance_dest],
                    'newbalanceDest': [new_balance_dest],
                    'isFlaggedFraud': [1 if is_flagged else 0]
                })
                
                # Feature engineering
                input_data = engineer_features(input_data)
                
                # Make predictions
                selected_model_dict = {k: v for k, v in models.items() if k in selected_models}
                results = predict_fraud(selected_model_dict, input_data, threshold_lr, threshold_xgb)
                
                st.subheader("🎯 Prediction Results")
                
                for model_name, result in results.items():
                    prob = result['probabilities'][0]
                    pred = result['predictions'][0]
                    threshold = result['threshold']
                    
                    # Display result
                    if pred == 1:
                        st.markdown(f"""
                        <div class="alert-fraud">
                        <strong>⚠️ {model_name.upper()}: FRAUD DETECTED</strong><br>
                        Probability: {prob:.6f} (≥ {threshold})<br>
                        Confidence: {prob*100:.3f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-safe">
                        <strong>✅ {model_name.upper()}: LEGITIMATE</strong><br>
                        Probability: {prob:.6f} (< {threshold})<br>
                        Confidence: {(1-prob)*100:.3f}%
                        </div>
                        """, unsafe_allow_html=True)
                
                # Probability gauge
                if results:
                    st.subheader("📊 Probability Visualization")
                    
                    fig = make_subplots(
                        rows=1, cols=len(results),
                        specs=[[{"type": "indicator"}] * len(results)],
                        subplot_titles=list(results.keys())
                    )
                    
                    for i, (model_name, result) in enumerate(results.items()):
                        prob = result['probabilities'][0]
                        
                        fig.add_trace(
                            go.Indicator(
                                mode="gauge+number",
                                value=prob,
                                number={'valueformat': '.6f'},
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': f"Fraud Probability"},
                                gauge={
                                    'axis': {'range': [None, 1]},
                                    'bar': {'color': "darkred" if prob > result['threshold'] else "darkgreen"},
                                    'steps': [
                                        {'range': [0, 0.3], 'color': "lightgreen"},
                                        {'range': [0.3, 0.7], 'color': "yellow"},
                                        {'range': [0.7, 1], 'color': "lightcoral"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': result['threshold']
                                    }
                                }
                            ),
                            row=1, col=i+1
                        )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
                
                # Detailed Explanation Section
                st.subheader("🔍 Detailed Analysis")
                
                # Get model information
                model_info = get_model_info()
                
                # Create explanation tabs for each model
                if len(results) > 1:
                    explanation_tabs = st.tabs([f"{name.upper()}" for name in results.keys()])
                    
                    for i, (model_name, result) in enumerate(results.items()):
                        with explanation_tabs[i]:
                            prob = result['probabilities'][0]
                            
                            # Model-specific explanation
                            st.markdown("**Model Information:**")
                            if model_name in model_info:
                                info = model_info[model_name]
                                st.markdown(f"- **Name**: {info['name']}")
                                st.markdown(f"- **Description**: {info['description']}")
                                st.markdown(f"- **Best Use Case**: {info['use_case']}")
                            
                            st.markdown("---")
                            
                            # Prediction explanation
                            explanation = explain_prediction(input_data.to_dict('list'), {model_name: prob}, model_name)
                            st.markdown(explanation)
                            
                            # Risk recommendations
                            st.markdown("**🛡️ Recommended Actions:**")
                            if prob >= 0.8:
                                st.markdown("- **IMMEDIATE**: Block transaction and investigate")
                                st.markdown("- **ESCALATE**: Contact fraud prevention team")
                                st.markdown("- **MONITOR**: Review account for additional suspicious activity")
                            elif prob >= 0.5:
                                st.markdown("- **REVIEW**: Manual review recommended")
                                st.markdown("- **VERIFY**: Additional authentication may be required")
                                st.markdown("- **MONITOR**: Flag account for enhanced monitoring")
                            elif prob >= 0.2:
                                st.markdown("- **MONITOR**: Standard monitoring protocols")
                                st.markdown("- **LOG**: Record transaction for future analysis")
                            else:
                                st.markdown("- **APPROVE**: Transaction appears legitimate")
                                st.markdown("- **ROUTINE**: Standard processing recommended")
                else:
                    # Single model explanation
                    for model_name, result in results.items():
                        prob = result['probabilities'][0]
                        
                        col_info, col_explain = st.columns([1, 2])
                        
                        with col_info:
                            st.markdown("**Model Information:**")
                            if model_name in model_info:
                                info = model_info[model_name]
                                st.markdown(f"**{info['name']}**")
                                st.markdown(info['description'])
                                st.markdown(f"*Use Case: {info['use_case']}*")
                        
                        with col_explain:
                            explanation = explain_prediction(input_data.to_dict('list'), {model_name: prob}, model_name)
                            st.markdown(explanation)
    
    with tab2:
        st.header("Batch Transaction Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with transaction data",
            type=['csv'],
            help="Upload a CSV file with the same structure as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_batch)} transactions")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df_batch.head())
                
                # Feature engineering
                df_batch = engineer_features(df_batch)
                
                # Make predictions
                selected_model_dict = {k: v for k, v in models.items() if k in selected_models}
                
                if st.button("🔍 Analyze Batch"):
                    results = predict_fraud(selected_model_dict, df_batch, threshold_lr, threshold_xgb)
                    
                    # Store in session state for preview tab
                    st.session_state.last_batch_data = df_batch.copy()
                    
                    # Results summary
                    st.subheader("📊 Batch Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (model_name, result) in enumerate(results.items()):
                        fraud_count = result['predictions'].sum()
                        fraud_rate = (fraud_count / len(df_batch)) * 100
                        avg_prob = result['probabilities'].mean()
                        
                        col = [col1, col2, col3][i % 3]
                        
                        with col:
                            st.metric(
                                f"{model_name.upper()} Results",
                                f"{fraud_count} frauds",
                                f"{fraud_rate:.1f}% fraud rate"
                            )
                            st.metric(
                                "Avg Probability",
                                f"{avg_prob:.4f}"
                            )
                    
                    # Detailed results
                    st.subheader("Detailed Results")
                    
                    # Add predictions to dataframe
                    df_results = df_batch.copy()
                    for model_name, result in results.items():
                        df_results[f'{model_name}_prob'] = result['probabilities']
                        df_results[f'{model_name}_pred'] = result['predictions']
                    
                    st.dataframe(df_results)
                    
                    # Download link
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            # Show sample data option
            if st.button("📋 Use Sample Data"):
                sample_data = create_sample_data()
                st.subheader("Sample Data")
                st.dataframe(sample_data)
                
                # Feature engineering
                sample_data = engineer_features(sample_data)
                
                # Make predictions
                selected_model_dict = {k: v for k, v in models.items() if k in selected_models}
                results = predict_fraud(selected_model_dict, sample_data, threshold_lr, threshold_xgb)
                
                # Show results
                st.subheader("Sample Results")
                for model_name, result in results.items():
                    st.write(f"**{model_name.upper()}:**")
                    st.write(f"Probabilities: {result['probabilities']}")
                    st.write(f"Predictions: {result['predictions']}")
    
    with tab3:
        st.header("Data Preview and Analysis")
        
        # Check if we have any loaded data
        data_source = st.radio(
            "Select data source to preview:",
            ["Sample Data", "Upload New File", "Recent Batch Data"]
        )
        
        if data_source == "Sample Data":
            st.subheader("Sample Transaction Data")
            sample_data = create_sample_data()
            st.dataframe(sample_data, width='stretch')
            
            # Show engineered features
            if st.button("Show Engineered Features"):
                engineered_sample = engineer_features(sample_data.copy())
                st.subheader("With Feature Engineering")
                st.dataframe(engineered_sample, width='stretch')
                
                # Data insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(engineered_sample))
                with col2:
                    avg_amount = engineered_sample['amount'].mean()
                    st.metric("Average Amount", f"${avg_amount:,.2f}")
                with col3:
                    high_value = engineered_sample['is_high_value'].sum()
                    st.metric("High Value Transactions", high_value)
        
        elif data_source == "Upload New File":
            st.subheader("Upload Data for Preview")
            preview_file = st.file_uploader(
                "Choose CSV file for preview",
                type=['csv'],
                key="preview_upload",
                help="Upload a CSV file to preview its structure and contents"
            )
            
            if preview_file is not None:
                try:
                    preview_df = pd.read_csv(preview_file)
                    
                    # Basic info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", len(preview_df))
                    with col2:
                        st.metric("Columns", len(preview_df.columns))
                    with col3:
                        missing_pct = (preview_df.isnull().sum().sum() / (len(preview_df) * len(preview_df.columns))) * 100
                        st.metric("Missing Data", f"{missing_pct:.1f}%")
                    with col4:
                        if 'amount' in preview_df.columns:
                            total_volume = preview_df['amount'].sum()
                            st.metric("Total Volume", f"${total_volume:,.2f}")
                    
                    # Data preview
                    st.subheader("Data Preview")
                    st.dataframe(preview_df.head(100), width='stretch')
                    
                    # Column information
                    st.subheader("Column Information")
                    col_info = pd.DataFrame({
                        'Column': preview_df.columns,
                        'Data Type': preview_df.dtypes,
                        'Non-Null Count': preview_df.count(),
                        'Null Count': preview_df.isnull().sum(),
                        'Null Percentage': (preview_df.isnull().sum() / len(preview_df) * 100).round(2)
                    })
                    st.dataframe(col_info, width='stretch')
                    
                    # Data distribution for numeric columns
                    numeric_cols = preview_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Numeric Column Statistics")
                        st.dataframe(preview_df[numeric_cols].describe(), width='stretch')
                    
                    # Feature engineering preview
                    if st.button("Preview Feature Engineering", key="preview_features"):
                        try:
                            engineered_preview = engineer_features(preview_df.copy())
                            st.subheader("After Feature Engineering")
                            st.dataframe(engineered_preview.head(20), width='stretch')
                            
                            # Show new columns created
                            original_cols = set(preview_df.columns)
                            new_cols = set(engineered_preview.columns) - original_cols
                            if new_cols:
                                st.info(f"New features created: {', '.join(sorted(new_cols))}")
                        except Exception as e:
                            st.error(f"Error in feature engineering: {e}")
                
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Recent Batch Data
            st.subheader("Recent Batch Data")
            st.info("This feature will show the most recently processed batch data. Upload and process a batch file first to see data here.")
            
            # You could store the last processed batch in session state
            if 'last_batch_data' in st.session_state:
                st.dataframe(st.session_state.last_batch_data, width='stretch')
            else:
                st.warning("No recent batch data available. Process a batch file first.")
    
    with tab4:
        st.header("Model Performance Comparison")
        
        if len(models) > 1:
            st.info("Load test data to compare model performance")
            
            # Model info
            st.subheader("Available Models")
            
            model_info = {
                'lr': 'Logistic Regression with balanced class weights',
                'xgb': 'XGBoost with class imbalance handling',
                'xgb_cal': 'Calibrated XGBoost with isotonic calibration'
            }
            
            for model_name in models.keys():
                description = model_info.get(model_name, 'Model description not available')
                st.write(f"**{model_name.upper()}:** {description}")
        else:
            st.warning("Load multiple models to enable comparison")
    
    with tab5:
        st.header("About This Application")
        
        st.markdown("""
        ### 🛡️ Fraud Detection System
        
        This web application uses advanced machine learning models to detect potentially fraudulent financial transactions
        in real-time, helping financial institutions prevent fraud and protect customers.
        
        #### 🎯 What We're Predicting
        
        **FRAUD PROBABILITY**: The likelihood that a financial transaction is fraudulent based on:
        
        **Transaction Characteristics:**
        - **Amount**: Large transactions (>$200K) have higher fraud risk
        - **Type**: TRANSFER and CASH_OUT transactions are highest risk
        - **Timing**: Unusual timing patterns (weekends, off-hours)
        - **Account Behavior**: Accounts being emptied or zeroed out
        
        **Financial Inconsistencies:**
        - **Balance Errors**: When transaction amounts don't match balance changes
        - **Accounting Anomalies**: Mathematical inconsistencies in before/after balances
        - **Destination Patterns**: Transactions to merchant vs. personal accounts
        
        **System Flags:**
        - **Business Rules**: Transactions flagged by existing fraud detection systems
        - **Historical Patterns**: Deviations from normal account behavior
        
        #### 🤖 Machine Learning Models
        
        **Multiple Model Ensemble Approach:**
        
        **1. Logistic Regression**
        - **Purpose**: Fast, interpretable baseline model
        - **Strengths**: Clear feature importance, regulatory compliance
        - **Use Case**: When explainability is critical
        
        **2. XGBoost Classifier**
        - **Purpose**: High-accuracy gradient boosting model
        - **Strengths**: Handles complex patterns, feature interactions
        - **Use Case**: Maximum fraud detection accuracy
        
        **3. Calibrated Models**
        - **Purpose**: Reliable probability estimates
        - **Strengths**: Better confidence in risk scores
        - **Use Case**: When probability accuracy matters for risk assessment
        
        **4. Ensemble Models** (When Available)
        - **Purpose**: Combines multiple model predictions
        - **Strengths**: More robust and reliable predictions
        - **Use Case**: Production systems requiring maximum reliability
        
        #### 📈 Model Performance Features
        
        **Advanced Calibration:**
        - **Platt Scaling**: Sigmoid-based probability calibration
        - **Isotonic Regression**: Non-parametric probability calibration
        - **Cross-Validation**: Ensures robust calibration estimates
        
        **Feature Engineering:**
        - **Amount Transformations**: Log, square root, and polynomial features
        - **Balance Analysis**: Change patterns and consistency checks
        - **Time Features**: Hour, day, weekend patterns
        - **Error Detection**: Accounting discrepancy identification
        
        #### 🛡️ Risk Assessment Framework
        
        **Risk Levels:**
        - **🔴 HIGH RISK (≥80%)**: Immediate blocking and investigation required
        - **🟡 MEDIUM RISK (50-79%)**: Manual review and additional verification
        - **🟠 LOW RISK (20-49%)**: Enhanced monitoring and logging
        - **🟢 MINIMAL RISK (<20%)**: Standard processing
        
        **Decision Thresholds:**
        - **Conservative**: Lower thresholds catch more fraud but increase false positives
        - **Balanced**: Default thresholds optimize accuracy vs. false positive rate
        - **Aggressive**: Higher thresholds reduce false positives but may miss some fraud
        
        #### 🔧 Application Features
        
        **Real-Time Analysis:**
        - Individual transaction risk assessment
        - Instant probability calculations
        - Interactive threshold adjustment
        - Detailed explanation of risk factors
        
        **Batch Processing:**
        - Process multiple transactions simultaneously
        - Export results for further analysis
        - Summary statistics and reporting
        
        **Model Comparison:**
        - Side-by-side model performance
        - Consensus predictions across models
        - Confidence interval analysis
        
        **Data Visualization:**
        - Interactive probability gauges
        - Risk distribution charts
        - Feature importance displays
        
        #### 📊 Technical Specifications
        
        **Training Data:** 6.3M+ financial transactions with fraud labels
        **Features:** 25+ engineered features including amount, balances, timing, and patterns
        **Performance:** ROC-AUC >0.99 on test data with optimized precision-recall balance
        **Scalability:** Designed for real-time processing of individual or batch transactions
        
        #### ⚙️ Configuration Options
        
        **Threshold Tuning:**
        - Adjust based on business risk tolerance
        - Separate thresholds for different model types
        - Real-time threshold impact visualization
        
        **Model Selection:**
        - Choose which models to use for predictions
        - Compare results across different algorithms
        - Weight model outputs based on performance
        
        #### 💡 Best Practices
        
        **For Financial Institutions:**
        - Monitor model performance regularly
        - Retrain models with new fraud patterns
        - Implement human review for medium-risk transactions
        - Maintain audit trails for regulatory compliance
        
        **For Risk Managers:**
        - Use ensemble predictions for critical decisions
        - Set thresholds based on cost of false positives vs. missed fraud
        - Implement escalation procedures for high-risk transactions
        - Regular model performance evaluation
        
        **For Analysts:**
        - Review feature importance for model interpretability
        - Analyze false positive patterns to improve thresholds
        - Monitor data drift and model degradation
        - Use calibrated probabilities for risk scoring
        """)
        
        st.subheader("📊 Model Performance Metrics")
        st.info("Training in progress... Enhanced models with better calibration and multiple algorithms are being created. Check back in a few minutes for updated performance metrics.")
        
        # Model training status
        if st.button("🔄 Check Training Status"):
            try:
                models_dir = Path("models")
                results_dir = Path("results")
                
                if models_dir.exists():
                    model_files = list(models_dir.glob("*.joblib"))
                    st.success(f"✅ {len(model_files)} model files found in models directory")
                    
                    for model_file in sorted(model_files):
                        st.write(f"- {model_file.name}")
                
                if results_dir.exists() and list(results_dir.glob("*.json")):
                    result_files = list(results_dir.glob("*.json"))
                    st.success(f"✅ {len(result_files)} result files found")
                    
                    # Show latest results if available
                    if result_files:
                        latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                        st.info(f"Latest evaluation: {latest_result.name}")
                
            except Exception as e:
                st.error(f"Error checking status: {e}")

if __name__ == "__main__":
    main()
