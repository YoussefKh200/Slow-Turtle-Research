@echo off
cd /d "%~dp0"
python run_weekly.py >> run_weekly.log 2>&1
