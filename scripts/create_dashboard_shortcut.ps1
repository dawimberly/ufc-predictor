# Recreate desktop shortcuts with custom UFC Predictor icon.
# Targets pythonw/python directly so Windows shows the .ico (not a .bat icon).
param(
    [string]$Desktop = [Environment]::GetFolderPath("Desktop"),
    [string]$Root = "C:\UFC-Predictor"
)

$ErrorActionPreference = "Stop"
$shell = New-Object -ComObject WScript.Shell

$icon = Join-Path $Root "assets\ufc_predictor.ico"
if (-not (Test-Path $icon)) {
    Write-Error "Missing icon: $icon"
    exit 1
}

$pyw = "C:\Python314\pythonw.exe"
$py = "C:\Python314\python.exe"
if (-not (Test-Path $pyw)) {
    $pyw = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\pythonw.exe"
    $py = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
}
if (-not (Test-Path $pyw)) {
    $pyw = (Get-Command pythonw.exe).Source
    $py = (Get-Command python.exe).Source
}

function Save-Lnk {
    param(
        [string]$LinkPath,
        [string]$TargetPath,
        [string]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$Description,
        [string]$IconPath
    )
    $lnk = $shell.CreateShortcut($LinkPath)
    $lnk.TargetPath = $TargetPath
    $lnk.Arguments = $ArgumentList
    $lnk.WorkingDirectory = $WorkingDirectory
    $lnk.WindowStyle = 1
    $lnk.Description = $Description
    $lnk.IconLocation = "$IconPath,0"
    $lnk.Save()
    Write-Host "OK $LinkPath"
    Write-Host "   -> $TargetPath $ArgumentList"
    Write-Host "   icon $IconPath"
}

$dashArgs = "-u `"$Root\src\ufc_dashboard.py`""
$cliArgs = "-u -m src.cli_entry --next-two --odds"

Save-Lnk -LinkPath (Join-Path $Desktop "UFC Predictor.lnk") `
    -TargetPath $pyw -ArgumentList $dashArgs -WorkingDirectory $Root `
    -Description "UFC Predictor dashboard" -IconPath $icon

Save-Lnk -LinkPath (Join-Path $Desktop "UFC Dashboard.lnk") `
    -TargetPath $pyw -ArgumentList $dashArgs -WorkingDirectory $Root `
    -Description "UFC Predictor dashboard" -IconPath $icon

Save-Lnk -LinkPath (Join-Path $Desktop "UFC Predict CLI.lnk") `
    -TargetPath $py -ArgumentList $cliArgs -WorkingDirectory $Root `
    -Description "UFC Predict CLI (next two + odds)" -IconPath $icon

# Bust Explorer icon cache hint
(Get-Item $icon).LastWriteTime = Get-Date

Write-Host ""
Write-Host "Desktop shortcuts updated. If the icon looks old, press F5 on the Desktop."
