@echo off
rem Environment setup for the local voice assistant.
rem
rem Keep this file ASCII-only: cmd.exe parses .bat in the OEM codepage,
rem so UTF-8 Chinese bytes silently corrupt command lines.
rem
rem Usage:
rem   setup.bat            create venv + install dependencies
rem   setup.bat models     also download models (about 4.9 GB)
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo === [1/4] Locate a usable Python (3.10 - 3.12) ===
set "PYEXE="

rem uv is the most reliable source, it keeps its own interpreters
where uv >nul 2>nul
if not errorlevel 1 for /f "delims=" %%i in ('uv python find 3.12 2^>nul') do if not defined PYEXE set "PYEXE=%%i"
if not defined PYEXE for /f "delims=" %%i in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do if not defined PYEXE set "PYEXE=%%i"
if not defined PYEXE for /f "delims=" %%i in ('uv python find 3.11 2^>nul') do if not defined PYEXE set "PYEXE=%%i"
if not defined PYEXE for /f "delims=" %%i in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do if not defined PYEXE set "PYEXE=%%i"

if not defined PYEXE goto NO_PY
if not exist "%PYEXE%" goto NO_PY
echo Found: %PYEXE%

echo.
echo === [2/4] Create virtualenv ===
if exist ".venv\Scripts\python.exe" goto VENV_OK
"%PYEXE%" -m venv .venv
if errorlevel 1 goto VENV_FAIL
:VENV_OK
call ".venv\Scripts\activate.bat"
python -c "import sys; print('venv python', sys.version.split()[0])"

echo.
echo === [3/4] Install dependencies ===
python -m pip install -U pip
pip install -r requirements.txt
if not errorlevel 1 goto DEPS_OK

echo.
echo [WARN] full install failed, retrying core packages only
pip install numpy sounddevice PyYAML faster-whisper onnxruntime openwakeword openai huggingface_hub pyttsx3
if errorlevel 1 goto PIP_FAIL
echo [WARN] retrying piper-tts alone
pip install piper-tts
if errorlevel 1 echo [WARN] piper-tts failed - falls back to piper.exe or Windows SAPI

:DEPS_OK
echo.
echo === [4/4] Verify ===
python -c "import numpy, sounddevice, onnxruntime, faster_whisper, openai; print('core deps OK')"
if errorlevel 1 echo [WARN] some imports failed, read the messages above

if /i "%~1"=="models" goto MODELS
echo.
echo Done. Next steps:
echo   1^) python scripts\download_models.py
echo   2^) run_server.bat
echo   3^) run_assistant.bat
goto END

:MODELS
echo.
echo === Downloading models ===
python scripts\download_models.py
goto END

:NO_PY
echo.
echo [ERROR] No usable Python 3.10 - 3.12 found.
echo   Check what is installed:  py -0p
echo   Or install one:           uv python install 3.12
goto END

:VENV_FAIL
echo.
echo [ERROR] Failed to create .venv. Delete the .venv folder and retry.
goto END

:PIP_FAIL
echo.
echo [ERROR] pip install failed. Check network, or use a mirror:
echo   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

:END
echo.
if not "%VA_NOPAUSE%"=="1" pause
