param(
    [string]$Host = '127.0.0.1',
    [int]$Port = 8000,
    [switch]$Reload
)

$venvActivate = Join-Path $PSScriptRoot 'venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    . $venvActivate
} else {
    Write-Host 'Virtual environment activation script not found. Ensure ./venv exists.' -ForegroundColor Yellow
}

$uvicornArgs = @('web.app:app', '--host', $Host, '--port', $Port)
if ($Reload) { $uvicornArgs += '--reload' }

python -m uvicorn @uvicornArgs
