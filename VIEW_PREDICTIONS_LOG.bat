@echo off
cd /d "%~dp0"
set UFC_PREDICTOR_ROOT=%~dp0
set PYTHONPATH=C:\SportsBettingBot\src;%PYTHONPATH%
if exist "data\logs\predictions.log" (
  type "data\logs\predictions.log" | more
) else (
  echo No predictions.log yet. Showing via CLI fallback...
  python -m sports_bot.app.cli view-log
)
exit /b %ERRORLEVEL%
