@echo off
rem Start llama-server. Keep this window open.
rem Set T to your physical core count: 6 cores -> 6, 8 cores -> 8.
rem Leave some cores for ASR / TTS / audio.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

set T=8
set CTX=4096
set SERVER=%~dp0models\llama\llama-server.exe
set MODEL=%~dp0models\llm\Qwen2.5-7B-Instruct-Q4_K_M.gguf

if not exist "%SERVER%" goto NO_SERVER
if not exist "%MODEL%" goto NO_MODEL

echo Starting llama-server
echo   model       : %MODEL%
echo   threads     : %T%
echo   context     : %CTX%
echo   listening   : http://127.0.0.1:8080
echo.
"%SERVER%" -m "%MODEL%" -t %T% -c %CTX% --host 127.0.0.1 --port 8080 -ngl 0
pause
exit /b 0

:NO_SERVER
echo [ERROR] llama-server.exe not found:
echo   %SERVER%
echo Download it with:  python scripts\download_models.py --only llama
pause
exit /b 1

:NO_MODEL
echo [ERROR] model file not found:
echo   %MODEL%
echo Download it with:  python scripts\download_models.py --only llm
pause
exit /b 1
