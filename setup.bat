@echo off
REM Quick Start Script for Rainwater Harvesting Simulator (Windows)

echo.
echo =====================================
echo  RAINWATER HARVESTING SIMULATOR SETUP
echo =====================================
echo.

REM Check Python version
echo Checking Python installation...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo.

echo =====================================
echo  SETUP COMPLETE!
echo =====================================
echo.
echo To start the application, run:
echo    streamlit run app.py
echo.
echo The application will open in your browser at:
echo    http://localhost:8501
echo.
pause
