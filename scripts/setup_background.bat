@echo off
REM Register Windows Task Scheduler jobs for UFC background processing.
REM   UFC Bot Nightly  — full Next Two Cards run daily at 5:00 AM (local)
REM   UFC Bot Startup  — auto run on user logon (cache if fresh; full only if stale)

setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
set "ROOT=%CD%"

echo.
echo === UFC Background Runner — Task Scheduler setup ===
echo   Project root: %ROOT%
echo.

if not exist "%ROOT%\src\background_runner.py" (
    echo [FAIL] src\background_runner.py not found.
    exit /b 1
)

if not exist "%ROOT%\models\ensemble_winner.joblib" (
    if not exist "%ROOT%\models\lgbm_winner.joblib" (
        echo [WARN] No trained model in models\ — background runs will fail until you train.
    )
)

set "HIDDEN=%ROOT%\scripts\run_background_hidden.vbs"
set "RUNNER=%ROOT%\scripts\run_background.bat"
if exist "%HIDDEN%" (
    set "NIGHTLY_CMD=wscript.exe \"%HIDDEN%\" full scheduled"
    set "STARTUP_CMD=wscript.exe \"%HIDDEN%\" auto startup"
) else (
    if not exist "%RUNNER%" (
        echo [FAIL] scripts\run_background.bat missing.
        exit /b 1
    )
    set "NIGHTLY_CMD=cmd /c \"%RUNNER%\" full scheduled"
    set "STARTUP_CMD=cmd /c \"%RUNNER%\" auto startup"
)

where schtasks >nul 2>&1
if errorlevel 1 (
    echo [FAIL] schtasks not found — requires Windows Task Scheduler.
    exit /b 1
)

echo Removing legacy / overlapping tasks (if present)...
schtasks /Delete /TN "UFC Bot Midnight" /F >nul 2>&1
schtasks /Delete /TN "UFC Bot Sunday" /F >nul 2>&1

echo Creating task: UFC Bot Nightly (daily 5:00 AM, one full run)...
schtasks /Create /TN "UFC Bot Nightly" /TR "%NIGHTLY_CMD%" /SC DAILY /ST 05:00 /RL LIMITED /F
if errorlevel 1 (
    echo [FAIL] Could not create UFC Bot Nightly task — try Run as Administrator.
    exit /b 1
)

echo Creating task: UFC Bot Startup (on user logon, auto/cache — not a second full scrape)...
schtasks /Create /TN "UFC Bot Startup" /TR "%STARTUP_CMD%" /SC ONLOGON /RL LIMITED /IT /F
if errorlevel 1 (
    echo [WARN] ONLOGON task denied — installing Startup folder shortcut instead...
    set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
    set "LINK_BAT=!STARTUP_FOLDER!\UFC-Bot-Background.bat"
    if not exist "!STARTUP_FOLDER!" mkdir "!STARTUP_FOLDER!"
    > "!LINK_BAT!" echo @echo off
    >> "!LINK_BAT!" echo call "%ROOT%\scripts\run_background.bat" auto startup ^>^> "%ROOT%\data\logs\background_task.log" 2^>^&1
    if exist "!LINK_BAT!" (
        echo [OK] Startup shortcut: !LINK_BAT!
    ) else (
        echo [FAIL] Could not create startup task or shortcut. Run this script as Administrator.
        exit /b 1
    )
)

echo.
echo === Setup complete ===
echo   Tasks registered for: %ROOT%
echo   Nightly: 5:00 AM local, mode=full (writes the snapshot morning open uses)
echo   Logon:   mode=auto (cache if fresh; does not block the GUI)
echo.
echo   Verify:
echo     schtasks /Query /TN "UFC Bot Nightly" /V /FO LIST
echo     schtasks /Query /TN "UFC Bot Startup"
echo.
echo   Re-register (same as this script):
echo     scripts\register_background_tasks.ps1
echo.
echo   Manual test:
echo     scripts\run_background.bat full manual
echo.
echo   Logs: data\logs\background_runner.log
echo.
endlocal
