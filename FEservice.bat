@echo off
REM TEFault Frontend Service Launcher

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

REM Navigate to frontend directory
cd apps\frontend

echo Starting TEFault Frontend Service...
echo Frontend: http://localhost:3000
echo.

call npm run dev

pause
