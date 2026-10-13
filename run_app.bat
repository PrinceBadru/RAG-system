@echo off
title DocuMind AI - Enterprise Due Diligence Suite
echo =====================================================================
echo  DocuMind AI -- Enterprise Due Diligence ^& Audit Suite
echo  Starting Streamlit Application...
echo =====================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run main.py
) else (
    python -m streamlit run main.py
)

pause
