@echo off
REM UFC Predict CLI — next two cards + odds (Python cli_entry; supports --next-two).
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "UFC_CANONICAL_ROOT=%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    pause
    exit /b 1
)

echo UFC Predict CLI: next two cards + odds
echo.
python -u -m src.cli_entry --next-two --odds %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo.
    echo [ERROR] Exit code %EC%
    pause
)
exit /b %EC%
