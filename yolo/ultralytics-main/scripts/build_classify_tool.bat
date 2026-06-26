@echo off
setlocal
cd /d "%~dp0.."

echo Building Image Classify Tool...
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo Python not found.
    pause
    exit /b 1
)

pip install pyinstaller pillow -q

if exist build\ImageClassifyTool rmdir /s /q build\ImageClassifyTool
if exist dist\ImageClassifyTool rmdir /s /q dist\ImageClassifyTool

pyinstaller --noconfirm --clean ^
    --windowed ^
    --name ImageClassifyTool ^
    --distpath dist ^
    --workpath build ^
    --specpath build ^
    scripts\review_and_classify.py

if errorlevel 1 (
    echo.
    echo Build failed.
    pause
    exit /b 1
)

set "OUT=release\ImageClassifyTool"
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%"
xcopy /e /i /y "dist\ImageClassifyTool\*" "%OUT%\" >nul

copy /y "scripts\release\启动.bat" "%OUT%\启动.bat" >nul
copy /y "release\ImageClassifyTool\使用说明.txt" "%OUT%\使用说明.txt" >nul 2>nul
copy /y "release\ImageClassifyTool\run.bat" "%OUT%\run.bat" >nul 2>nul

powershell -NoProfile -Command "Compress-Archive -Path '%OUT%' -DestinationPath 'release\ImageClassifyTool.zip' -Force"

echo.
echo Done folder : %CD%\%OUT%
echo Done zip    : %CD%\release\ImageClassifyTool.zip
echo Send the zip to others. No Python required.
echo.
pause
endlocal
