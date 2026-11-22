@echo off
rem Start the assistant. run_server.bat must already be running in another window.
rem Pass extra args through, e.g.  run_assistant.bat --text
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if exist ".venv\Scripts\python.exe" goto HAVE_VENV
echo [ERROR] .venv not found in this folder.
echo Run setup.bat first.
pause
exit /b 1

:HAVE_VENV
".venv\Scripts\python.exe" -m src.main %*
pause
