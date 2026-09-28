@echo off
setlocal EnableExtensions

echo ============================================================
echo NeuroFusion AI - ds004302 VBM PILOT
echo 2 Worker Launcher TEST
echo ============================================================
echo.

set "SPM_EXE=C:\Users\User\Documents\NeuroFusionTools\spm_standalone\spm25.exe"
set "WORKER_SCRIPT=C:\Users\User\Documents\neurofusion\src\vbm_spm\run_external_pilot_vbm_worker.m"

echo SPM:
echo %SPM_EXE%
echo.
echo Pilot worker:
echo %WORKER_SCRIPT%
echo.

if not exist "%SPM_EXE%" (
    echo ERROR: SPM executable not found.
    echo %SPM_EXE%
    pause
    exit /b 1
)

if not exist "%WORKER_SCRIPT%" (
    echo ERROR: Pilot worker MATLAB file not found.
    echo %WORKER_SCRIPT%
    pause
    exit /b 1
)

echo.
echo Starting Worker 1...
start "NeuroFusion Pilot Worker 1" /b "%SPM_EXE%" batch "%WORKER_SCRIPT%"

timeout /t 2 /nobreak >nul

echo Starting Worker 2...
start "NeuroFusion Pilot Worker 2" /b "%SPM_EXE%" batch "%WORKER_SCRIPT%"

echo.
echo ============================================================
echo BOTH PILOT WORKERS LAUNCHED
echo ============================================================
echo.
echo Worker 1 = sub-01, sub-55
echo Worker 2 = sub-26
echo.
echo SPM command:
echo     spm25.exe batch worker_script.m
echo.
echo ============================================================
pause

endlocal