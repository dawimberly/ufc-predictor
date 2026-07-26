@echo off
REM Recent-card backtest via SportsBettingBot CLI (uses this UFC-Predictor model/data).
cd /d "%~dp0"
set UFC_PREDICTOR_ROOT=%~dp0
set PYTHONPATH=C:\SportsBettingBot\src;%PYTHONPATH%
if "%~1"=="" (
  python -m sports_bot.app.cli backtest --last=5
) else (
  python -m sports_bot.app.cli backtest %*
)
exit /b %ERRORLEVEL%
