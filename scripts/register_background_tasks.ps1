# Register UFC Bot Nightly (5:00 AM full) and UFC Bot Startup (logon auto).
# One full overnight analysis only — logon stays cache/auto and must not block the GUI.
# Usage (from project root or scripts\):  powershell -ExecutionPolicy Bypass -File scripts\register_background_tasks.ps1

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "src\background_runner.py"))) {
    $Root = $PSScriptRoot
    if (-not (Test-Path (Join-Path $Root "src\background_runner.py"))) {
        Write-Error "src\background_runner.py not found. Run from UFC-Predictor."
    }
}

$Hidden = Join-Path $Root "scripts\run_background_hidden.vbs"
$Runner = Join-Path $Root "scripts\run_background.bat"
if (Test-Path $Hidden) {
    $NightlyCmd = "wscript.exe `"$Hidden`" full scheduled"
    $StartupCmd = "wscript.exe `"$Hidden`" auto startup"
} elseif (Test-Path $Runner) {
    $NightlyCmd = "cmd /c `"$Runner`" full scheduled"
    $StartupCmd = "cmd /c `"$Runner`" auto startup"
} else {
    Write-Error "Missing scripts\run_background_hidden.vbs and scripts\run_background.bat"
}

Write-Host "=== UFC Background Runner — Task Scheduler setup ==="
Write-Host "  Project root: $Root"
Write-Host "  Nightly TR:   $NightlyCmd"

schtasks /Delete /TN "UFC Bot Midnight" /F 2>$null | Out-Null
schtasks /Delete /TN "UFC Bot Sunday" /F 2>$null | Out-Null

schtasks /Create /TN "UFC Bot Nightly" /TR "$NightlyCmd" /SC DAILY /ST 05:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) {
    Write-Error "Could not create UFC Bot Nightly. Try Run as Administrator."
}

schtasks /Create /TN "UFC Bot Startup" /TR "$StartupCmd" /SC ONLOGON /RL LIMITED /IT /F
if ($LASTEXITCODE -ne 0) {
    Write-Warning "ONLOGON task denied — falling back to setup_background.bat shortcut path."
    & (Join-Path $Root "scripts\setup_background.bat")
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Registered:"
Write-Host "  UFC Bot Nightly  — daily 5:00 AM local, mode=full (one snapshot for morning open)"
Write-Host "  UFC Bot Startup  — logon, mode=auto (cache if fresh; not a second full scrape)"
Write-Host ""
Write-Host "Verify:"
Write-Host '  schtasks /Query /TN "UFC Bot Nightly" /V /FO LIST'
Write-Host '  schtasks /Query /TN "UFC Bot Startup"'
