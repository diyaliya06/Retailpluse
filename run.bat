@echo off
title RetailPulse
cd /d "%~dp0"

rem Use the project1 virtual environment if it exists, otherwise system Python
if exist "..\.venv\Scripts\python.exe" (
    set PY="..\.venv\Scripts\python.exe"
) else (
    set PY=python
)

echo Installing requirements (first time can take a few minutes)...
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Install failed. Check your internet connection.
    pause
    exit /b 1
)

if not exist "artifacts\cleaned_transactions.csv" (
    echo.
    echo Running the training pipeline...
    %PY% -m src.train
    if errorlevel 1 (
        echo.
        echo Training failed. Make sure your data file is inside the data folder.
        pause
        exit /b 1
    )
)

echo.
echo Starting RetailPulse at http://localhost:8501
%PY% -m streamlit run src/app.py
pause
