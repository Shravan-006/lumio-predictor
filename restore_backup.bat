@echo off
TITLE Restore Stable State - VoltPulse
echo ============================================================
echo   RESTORE STABLE APP STATE
echo ============================================================
echo.
echo Restoring files from backup_stable_state...

cd /d "%~dp0"

if not exist "backup_stable_state" (
    echo [ERROR] backup_stable_state folder was not found!
    pause
    exit /b 1
)

copy /Y "backup_stable_state\app.py" "app.py"
copy /Y "backup_stable_state\generate_dataset.py" "generate_dataset.py"
copy /Y "backup_stable_state\train_model.py" "train_model.py"
copy /Y "backup_stable_state\test_app.py" "test_app.py"
copy /Y "backup_stable_state\requirements.txt" "requirements.txt"
copy /Y "backup_stable_state\electricity_usage.csv" "electricity_usage.csv"
copy /Y "backup_stable_state\README.md" "README.md"
copy /Y "backup_stable_state\VIVA_NOTES.md" "VIVA_NOTES.md"
copy /Y "backup_stable_state\run_app.bat" "run_app.bat"

if not exist "models" mkdir "models"
copy /Y "backup_stable_state\models\*" "models\"

if not exist "reports" mkdir "reports"
copy /Y "backup_stable_state\reports\*" "reports\"

echo.
echo [SUCCESS] Current state has been restored to the stable snapshot!
echo.
pause
