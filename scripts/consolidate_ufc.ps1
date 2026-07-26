# Keep one canonical UFC project (C:\UFC-Predictor). Trading bot stays separate.
# Usage:
#   powershell -File C:\UFC-Predictor\scripts\consolidate_ufc.ps1

param(
    [string]$CanonicalRoot = "C:\UFC-Predictor",
    [string]$MonorepoRoot = "C:\Users\Owner\PythonTrading"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $CanonicalRoot)) {
    throw "Canonical UFC root not found: $CanonicalRoot"
}

$MonoPredictor = Join-Path $MonorepoRoot "ufc-predictor"

Write-Host "=== UFC consolidation ===" -ForegroundColor Cyan
Write-Host "Canonical: $CanonicalRoot"
Write-Host "Monorepo stub: $MonoPredictor (docs only; trading-bot stays separate)"

# Lightweight stub sync: docs + launch hints only (no heavy data/models/dist)
if (Test-Path $MonoPredictor) {
    foreach ($name in @("CANONICAL.md", "README.md", ".env.example")) {
        $src = Join-Path $CanonicalRoot $name
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $MonoPredictor $name) -Force
        }
    }
    Write-Host "Updated monorepo stub docs -> $MonoPredictor" -ForegroundColor Green
}

$rootEnv = Join-Path $CanonicalRoot ".env"
$distEnv = Join-Path $CanonicalRoot "dist\.env"
if (-not (Test-Path $rootEnv) -and (Test-Path $distEnv)) {
    Copy-Item $distEnv $rootEnv
    Write-Host "Created $rootEnv from dist\.env" -ForegroundColor Yellow
}

if (Test-Path $rootEnv) {
    $raw = Get-Content $rootEnv -Raw
    if ($raw -notmatch "UFC_CANONICAL_ROOT=") {
        Add-Content $rootEnv "`nUFC_CANONICAL_ROOT=$CanonicalRoot"
        Write-Host "Added UFC_CANONICAL_ROOT to .env" -ForegroundColor Yellow
    }
    if ((Get-Content $rootEnv) -notmatch "^ENABLE_PROPS=") {
        Add-Content $rootEnv "ENABLE_PROPS=true"
    }
}

Write-Host ""
Write-Host "Done. Use only:" -ForegroundColor Green
Write-Host "  $CanonicalRoot"
Write-Host "  Dashboard EXE: $(Join-Path $CanonicalRoot 'dist\ufc-dashboard.exe')"
Write-Host "  Launcher:      $(Join-Path $CanonicalRoot 'START_DASHBOARD.bat')"
