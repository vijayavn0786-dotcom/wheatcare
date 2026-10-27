@echo off
title WheatCare AI - Final Year Engineering Project
color 0A

echo =====================================================================
echo                 WHEATCARE AI - DEEP LEARNING SYSTEM
echo        Intelligent Wheat Leaf Disease Diagnosis & Explainable AI
echo                  Final Year Engineering Capstone
echo =====================================================================
echo.
echo Initializing environment and launching Streamlit web interface...
echo.

set TF_ENABLE_ONEDNN_OPTS=0
set MPLCONFIGDIR=%TEMP%\matplotlib

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    echo [OK] Local Virtual Environment detected in current folder.
    "venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
) else if exist "..\venv\Scripts\python.exe" (
    echo [OK] Virtual Environment detected in parent folder.
    "..\venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
) else (
    echo [INFO] Searching for system Python environment...
    python -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
)

pause
