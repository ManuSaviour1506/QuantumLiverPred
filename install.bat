@echo off
REM Liver Tumour Intelligence Platform V11 - Windows installer wrapper.
REM Usage:  install.bat            (full install)
REM         install.bat --check    (verify an existing installation)
REM Requires Python 3.10+ from python.org (tick "Add python.exe to PATH") - Node.js 20+ is optional.
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 install.py %*
) else (
  python install.py %*
)
exit /b %errorlevel%
