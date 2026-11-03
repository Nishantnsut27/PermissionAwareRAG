@echo off
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [!] Virtualenv not found.
    echo     Create it first:  py -3.14 -m venv .venv
    echo     Then install:     .venv\Scripts\pip install -r requirements.lock
    pause
    exit /b 1
)

if not exist ".env" (
    echo [!] .env file not found at project root.
    echo     Copy .env.example to .env and fill in your API keys.
    pause
    exit /b 1
)

"%PY%" scripts\run_local.py %*
exit /b %ERRORLEVEL%
