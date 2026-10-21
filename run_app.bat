@echo off
TITLE Electricity Usage Category Predictor
echo ============================================================
echo   ELECTRICITY USAGE CATEGORY PREDICTOR - EXPO LAUNCHER
echo ============================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) was not found.
    echo Please install dependencies first using:
    echo   python -m venv .venv
    echo   .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "electricity_usage.csv" (
    echo [1/3] Generating synthetic dataset...
    .venv\Scripts\python generate_dataset.py
)

if not exist "models\electricity_model.joblib" (
    echo [2/3] Training Decision Tree Classifier...
    .venv\Scripts\python train_model.py
)

echo [3/3] Starting Streamlit Web App on http://localhost:8501 ...
echo.
.venv\Scripts\streamlit run app.py
pause
