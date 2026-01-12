@echo off
REM TEFault Backend Service Launcher

REM Set default environment name (can be overridden)
if "%CONDA_ENV%"=="" set CONDA_ENV=tefault

REM Navigate to project root
cd /d "%~dp0"

REM Activate conda environment
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo Error: Failed to activate conda environment '%CONDA_ENV%'
    echo Available environments:
    conda env list
    pause
    exit /b 1
)

echo Starting TEFault Backend Service...
echo Backend: http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.

uvicorn apps.backend.main:app --reload --host 0.0.0.0 --port 8000

pause
