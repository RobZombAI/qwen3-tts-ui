@echo off
REM Build standalone Qwen3 TTS .exe with PyInstaller
cd /d "%~dp0"
call venv\Scripts\activate
pip install pyinstaller
pyinstaller Qwen3TTS.spec
echo.
echo Build complete: dist\Qwen3 TTS.exe
pause
