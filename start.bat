@echo off
REM Liver Tumour Intelligence Platform V11 - start the API and web app on http://127.0.0.1:8000
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" start.py --open %*
) else (
  where py >nul 2>nul && (py -3 start.py --open %*) || (python start.py --open %*)
)
exit /b %errorlevel%
