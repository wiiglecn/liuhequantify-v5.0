@echo off
REM ============================================================
REM  Macau Mark Six lottery analysis and prediction launcher
REM  Double-click to run, or execute in command prompt
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   Macau Mark Six Analysis and Prediction
echo   Working dir: %cd%
echo ============================================================
echo.

REM Adjust backtest periods / years / refresh here
python -X utf8 macau_predictor.py --backtest 30

echo.
echo ============================================================
echo   Finished. Press any key to close...
pause >nul
