@echo off
setlocal EnableExtensions

title NeuroFusion AI - External OpenNeuro VBM - 2 Workers

echo ============================================================
echo NeuroFusion AI - External OpenNeuro VBM
echo 43-Subject Cohort
echo 2 Worker Parallel Processing
echo ============================================================
echo.

set "SPM_EXE=C:\Users\User\Documents\NeuroFusionTools\spm_standalone\spm25.exe"

set "WORKER=C:\Users\User\Documents\neurofusion\src\vbm_spm\run_external_openneuro_vbm_worker.m"

set "INPUT=C:\Users\User\Documents\neurofusion\processed_data\smri\openneuro_ds000115\preprocessed\normalized"

set "OUTPUT=C:\Users\User\Documents\neurofusion\processed_data\smri\openneuro_ds000115\vbm"

set "LOG=C:\Users\User\Documents\neurofusion\processed_data\smri\openneuro_ds000115\vbm_logs"

echo SPM:
echo %SPM_EXE%
echo.

echo Worker script:
echo %WORKER%
echo.

echo Input:
echo %INPUT%
echo.

echo Output:
echo %OUTPUT%
echo.

echo ============================================================
echo PRE-FLIGHT CHECKS
echo ============================================================
echo.

if not exist "%SPM_EXE%" (
    echo ERROR: SPM executable not found:
    echo %SPM_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%WORKER%" (
    echo ERROR: Worker MATLAB script not found:
    echo %WORKER%
    echo.
    pause
    exit /b 1
)

if not exist "%INPUT%" (
    echo ERROR: Input directory not found:
    echo %INPUT%
    echo.
    pause
    exit /b 1
)

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

if not exist "%LOG%" mkdir "%LOG%"

echo Worker script FOUND.
echo Input directory FOUND.
echo Output directory READY.
echo.

echo Checking normalized files...

dir /b "%INPUT%\*_normalized.nii.gz"

if errorlevel 1 (
    echo.
    echo ERROR: No normalized NIfTI files found.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo VERIFYING EXACT 43 FILES
echo ============================================================
echo.

set "COUNT=0"

for %%F in ("%INPUT%\*_normalized.nii.gz") do (
    set /a COUNT+=1
)

echo Found %COUNT% normalized files.

if not "%COUNT%"=="43" (
    echo.
    echo ERROR: Expected exactly 43 normalized files.
    echo Found %COUNT%.
    echo.
    pause
    exit /b 1
)

echo Exactly 43 normalized files found.
echo.

echo ============================================================
echo STARTING WORKER 1
echo ============================================================
echo.

set "VBM_WORKER_ID=1"

start "NeuroFusion VBM Worker 1" cmd /k ""%SPM_EXE%" batch "%WORKER%" --silent"

timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo STARTING WORKER 2
echo ============================================================
echo.

set "VBM_WORKER_ID=2"

start "NeuroFusion VBM Worker 2" cmd /k ""%SPM_EXE%" batch "%WORKER%" --silent"

echo.
echo ============================================================
echo BOTH OPENNEURO VBM WORKERS LAUNCHED
echo ============================================================
echo.

echo Worker 1 = global subjects 1,3,5,7,...
echo Worker 2 = global subjects 2,4,6,8,...

echo.
echo The worker command windows are intentionally kept OPEN.
echo.

echo LOG DIRECTORY:
echo %LOG%

echo.
echo IMPORTANT:
echo Do not start another VBM launcher while these are running.

echo.
echo Each worker will process only its assigned subjects.

echo.
echo ============================================================
echo Launcher finished.
echo ============================================================
echo.

pause

endlocal