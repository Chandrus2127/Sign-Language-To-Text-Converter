# Sign Language To Text Conversion - PowerShell Launcher
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Starting Sign Language To Text Conversion Application..." -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    & $venvPython Application.py
} else {
    Write-Warning "Virtual environment not found at $venvPython"
    Write-Host "Attempting to run with system python..." -ForegroundColor Yellow
    python Application.py
}
