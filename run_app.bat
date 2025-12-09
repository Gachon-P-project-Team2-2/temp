@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"

:: 가상환경 폴더 확인
if exist venv\Scripts\activate.bat goto START_APP

echo [오류] 가상환경 폴더(venv)를 찾을 수 없습니다.
echo 먼저 'py -3.13 -m venv venv' 명령어로 가상환경을 생성해주세요.
pause
exit /b

:START_APP
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%venv\Scripts\activate.bat"
"%VIRTUAL_ENV%\Scripts\python.exe" -m streamlit run app.py
pause