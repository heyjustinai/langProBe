@echo off
REM MetaPromptBench Startup Script for Windows

echo 🚀 Starting MetaPromptBench...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.6+ and try again
    pause
    exit /b 1
)

REM Get the directory of this script
set SCRIPT_DIR=%~dp0

echo 📁 Working directory: %SCRIPT_DIR%
echo 🐍 Using Python: python
echo.

REM Start the server
echo 🌐 Starting server on http://localhost:8000
echo 📊 Open http://localhost:8000/evaluation_viewer.html in your browser
echo.
echo Press Ctrl+C to stop the server
echo ----------------------------------------

cd /d "%SCRIPT_DIR%"
python server.py

pause
