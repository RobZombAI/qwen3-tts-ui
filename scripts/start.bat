@echo off
REM Qwen3 TTS — Windows Launcher (double-click from Explorer)
cd /d "%~dp0"
call venv\Scripts\activate
python win_launcher.py
pause
