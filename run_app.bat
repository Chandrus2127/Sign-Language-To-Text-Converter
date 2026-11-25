@echo off
title Sign Language To Text Conversion
cd /d "%~dp0"

echo =====================================================================
echo  Starting Sign Language To Text Conversion Application...
echo =====================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" Application.py
) else (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please make sure the Python 3.11 virtual environment is created.
    echo Running with system python fallback...
    python Application.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error code: %ERRORLEVEL%
    pause
)
