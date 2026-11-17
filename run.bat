@echo off
REM Group 14 EV Battery Digital-Twin Virtual Lab: one-command launcher for Windows. Double-click to start.
REM   run.bat              start the lab (the first run sets everything up and needs internet once)
REM   run.bat --port 8600  use a specific port
REM   run.bat --rebuild    rebuild data and the calibrated twin from data\raw, then start (about 7 min)
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONHOME="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

REM 1. A Python 3.10+ interpreter.
set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.10 or newer is required: https://www.python.org/downloads/
  echo During installation, tick "Add python.exe to PATH".
  pause
  exit /b 1
)

REM 2. A private environment in .venv (re-created if it is missing or broken).
".venv\Scripts\python.exe" -m pip --version >nul 2>nul
if errorlevel 1 (
  if exist ".venv" rmdir /s /q ".venv"
  echo ^>^> Creating the virtual environment ^(.venv^) ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)

REM 3. Dependencies: installed on the first run, and again only when requirements.txt changes.
set "SUM="
for /f "skip=1 delims=" %%h in ('certutil -hashfile requirements.txt SHA256') do if not defined SUM set "SUM=%%h"
set "OLD="
if exist ".venv\requirements.sum" set /p OLD=<".venv\requirements.sum"
if not "%SUM%"=="%OLD%" (
  echo ^>^> Installing dependencies ^(first run only: needs internet, takes a few minutes^) ...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
  if errorlevel 1 (
    echo Installing the dependencies failed. Check the internet connection and run again.
    pause
    exit /b 1
  )
  >".venv\requirements.sum" echo %SUM%
)

REM 4. Check the data and models, start the lab and open the browser.
".venv\Scripts\python.exe" scripts\launch.py %*
if errorlevel 1 pause
endlocal
