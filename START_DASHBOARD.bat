@echo off
REM Reliable dashboard launch via Python (avoids frozen EXE arrow.dll crash).
setlocal
cd /d "C:\UFC-Predictor"
set "PYTHONUTF8=1"
set "UFC_CANONICAL_ROOT=C:\UFC-Predictor"

if not exist "src\ufc_dashboard.py" (
    echo [ERROR] Missing C:\UFC-Predictor\src\ufc_dashboard.py
    pause
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    pause
    exit /b 1
)

echo Launching UFC Predictor Dashboard from C:\UFC-Predictor ...
echo (Python source — latest code. Close this window after the GUI opens if it stays open.)
echo.

REM Prefer pythonw for no console; fall back to python if pythonw missing.
where pythonw >nul 2>&1
if not errorlevel 1 (
    start "" pythonw -u "src\ufc_dashboard.py" %*
    exit /b 0
)

python -u "src\ufc_dashboard.py" %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo.
    echo [ERROR] Dashboard exited with code %EC%
    echo See: C:\UFC-Predictor\data\logs\dashboard_crash.log
    pause
)
exit /b %EC%
