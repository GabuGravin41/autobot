@echo off
rem Start Autobot's butler daemon from this folder, logging to %USERPROFILE%\.autobot\logs\butler.log
rem Use this for the "start at logon" scheduled task described in BUTLER.md.
cd /d "%~dp0"
if not exist "%USERPROFILE%\.autobot\logs" mkdir "%USERPROFILE%\.autobot\logs"
set PYTHONUTF8=1
python -m autobot butler run >> "%USERPROFILE%\.autobot\logs\butler.log" 2>&1
