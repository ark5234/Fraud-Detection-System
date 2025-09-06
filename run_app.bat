@echo off
REM Fraud Detection App Launcher for Windows
REM Quick start script for the Streamlit web application

echo 🛡️ Starting Fraud Detection Web App...
echo ================================================

REM Check if models directory exists
if not exist "models" (
    echo ❌ Models directory not found!
    echo Please run the fraud detection notebook first to train models.
    pause
    exit /b 1
)

REM Check if any model files exist
dir /b models\*.joblib >nul 2>&1
if errorlevel 1 (
    echo ❌ No trained models found in models/ directory!
    echo Please run the fraud detection notebook first to train models.
    pause
    exit /b 1
)

echo ✅ Found trained models

REM Check if Streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit not found. Installing...
    pip install streamlit plotly
)

echo 🚀 Launching Streamlit app...
echo The app will open in your default browser at http://localhost:8501
echo Press Ctrl+C to stop the app
echo.

REM Launch the app
streamlit run app.py

pause
