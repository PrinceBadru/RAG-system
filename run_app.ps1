# DocuMind AI PowerShell Launcher
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " DocuMind AI -- Enterprise Due Diligence & Audit Suite" -ForegroundColor Green
Write-Host " Starting Streamlit Application..." -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path ".\.venv\Scripts\python.exe") {
    & ".\.venv\Scripts\python.exe" -m streamlit run main.py
} else {
    python -m streamlit run main.py
}
