@echo off
REM ============================================================
REM  Macau Mark Six Analysis GUI launcher (tkinter)
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

python -X utf8 macau_gui.py
