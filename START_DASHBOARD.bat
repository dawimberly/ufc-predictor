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

REM Prefer known installs so PATH order / Store aliases cannot pick a bare Python.
set "PYW="
set "PY="
if exist "C:\Python314\pythonw.exe" set "PYW=C:\Python314\pythonw.exe"
if exist "C:\Python314\python.exe" set "PY=C:\Python314\python.exe"
if not defined PYW if exist "%LocalAppData%\Python\pythoncore-3.14-64\pythonw.exe" set "PYW=%LocalAppData%\Python\pythoncore-3.14-64\pythonw.exe"
if not defined PY if exist "%LocalAppData%\Python\pythoncore-3.14-64\python.exe" set "PY=%LocalAppData%\Python\pythoncore-3.14-64\python.exe"
if not defined PYW if exist "%LocalAppData%\Programs\Python\Python311\pythonw.exe" set "PYW=%LocalAppData%\Programs\Python\Python311\pythonw.exe"
if not defined PY if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PY=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PYW (
    where pythonw >nul 2>&1
    if not errorlevel 1 for /f "delims=" %%I in ('where pythonw') do if not defined PYW set "PYW=%%I"
)
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 for /f "delims=" %%I in ('where python') do if not defined PY set "PY=%%I"
)

if not defined PYW if not defined PY (
    echo [ERROR] Python not found. Install Python 3.14/3.11 or add it to PATH.
    pause
    exit /b 1
)

echo Launching UFC Predictor Dashboard from C:\UFC-Predictor ...
echo (Python source — latest code. Close this window after the GUI opens if it stays open.)
echo.

if defined PYW (
    start "" "%PYW%" -u "src\ufc_dashboard.py" %*
    exit /b 0
)

"%PY%" -u "src\ufc_dashboard.py" %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo.
    echo [ERROR] Dashboard exited with code %EC%
    echo See: C:\UFC-Predictor\data\logs\dashboard_crash.log
    pause
)
exit /b %EC%
