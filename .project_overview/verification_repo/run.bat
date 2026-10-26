@echo off
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [!] Virtualenv not found.
    echo     Create it first:  python -m venv .venv
    echo     Then install:     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist ".env" (
    echo [!] .env file not found at project root.
    echo     Copy .env.example to .env and fill in your API keys.
    pause
    exit /b 1
)

echo [1/2] Starting the answering API on http://127.0.0.1:8000 ...
start "Nexora API" cmd /k "%PY% answering\serve.py"

echo [2/2] Waiting for the API to become healthy...
set /a tries=0
:wait_api
"%PY%" -c "import requests,sys; sys.exit(0 if requests.get('http://127.0.0.1:8000/health', timeout=2).status_code==200 else 1)" >nul 2>&1
if errorlevel 1 (
    set /a tries+=1
    if %tries% geq 30 (
        echo [!] API did not become healthy in time. Check the API window for errors.
        pause
        exit /b 1
    )
    timeout /t 1 /nobreak >nul
    goto wait_api
)
echo       API is healthy.

echo.
echo Starting the Streamlit UI on http://localhost:8501 ...
echo (a browser window will open automatically; close both windows to stop)
"%PY%" -m streamlit run app.py
