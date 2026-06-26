@echo off
setlocal
cd /d "%~dp0"

set "SCRIPT=%~dp0review_and_classify.py"
if not exist "%SCRIPT%" set "SCRIPT=%~dp0..\scripts\review_and_classify.py"

where python >nul 2>&1
if %errorlevel%==0 (
    python "%SCRIPT%"
    goto check_err
)

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 "%SCRIPT%"
    goto check_err
)

echo Python not found.
echo Install Python from https://www.python.org/downloads/
echo Then run: pip install pillow
pause
exit /b 1

:check_err
if errorlevel 1 (
    echo.
    echo Start failed. Try: pip install pillow
    pause
)
endlocal
