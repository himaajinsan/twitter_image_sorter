@echo off
cd /d "%~dp0"
set "SYSTEM_PYTHON=python"
set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
  echo Creating local Python environment in %VENV_DIR% ...
  "%SYSTEM_PYTHON%" -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create the local environment. Install Python 3.11 or 3.12.
    pause
    exit /b 1
  )
)

echo Checking local components and cached models...
"%VENV_PYTHON%" startup.py
if errorlevel 1 (
  echo.
  echo Startup failed. Check the setup window or the network connection.
)
pause
