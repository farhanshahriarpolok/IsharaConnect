Write-Host "Waiting for pip to finish..."
# Wait for the pip process to finish by checking for it (or just try running the python scripts, they will fail if not ready, but we can loop)
$process = Get-Process -Name "pip" -ErrorAction SilentlyContinue
if ($process) {
    Write-Host "Waiting for pip installation to complete..."
    Wait-Process -Name "pip"
}

Write-Host "Pip finished. Running Step 3: Training Baseline Model..."
.venv\Scripts\python.exe scripts/train.py
if ($LASTEXITCODE -ne 0) { Write-Error "Train failed"; exit $LASTEXITCODE }

Write-Host "Running Step 3: Exporting ONNX Model..."
.venv\Scripts\python.exe scripts/export_onnx.py
if ($LASTEXITCODE -ne 0) { Write-Error "Export failed"; exit $LASTEXITCODE }

Write-Host "Running Step 4: Launching IsharaConnect..."
.venv\Scripts\python.exe launch.py
