#!/bin/bash

# Fraud Detection App Launcher
# Quick start script for the Streamlit web application

echo "🛡️ Starting Fraud Detection Web App..."
echo "================================================"

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "❌ Models directory not found!"
    echo "Please run the fraud detection notebook first to train models."
    exit 1
fi

# Check if any model files exist
MODEL_COUNT=$(find models -name "*.joblib" | wc -l)
if [ $MODEL_COUNT -eq 0 ]; then
    echo "❌ No trained models found in models/ directory!"
    echo "Please run the fraud detection notebook first to train models."
    exit 1
fi

echo "✅ Found $MODEL_COUNT trained model(s)"

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly
fi

echo "🚀 Launching Streamlit app..."
echo "The app will open in your default browser at http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""

# Launch the app
streamlit run app.py
