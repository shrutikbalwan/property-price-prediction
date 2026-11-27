@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m ml_projects.app --port 8501
) else (
  python -m ml_projects.app --port 8501
)
pause
