@echo off
REM Launch UFC Predictor GUI (Python source — avoids frozen EXE arrow.dll crash).
cd /d "%~dp0"
call "%~dp0START_DASHBOARD.bat" %*
