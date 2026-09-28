@echo off
setlocal
set "PROJECT=C:\Users\User\Documents\neurofusion"
set "SPM=C:\Users\User\Documents\NeuroFusionTools\spm_standalone\spm25.exe"
set "MCR=C:\Program Files\MATLAB\MATLAB Runtime\R2024b\runtime\win64"
set "VBM_NUM_WORKERS=2"
set "PATH=%MCR%;%PATH%"
if not exist "%PROJECT%\processed_data\smri\vbm_spm" mkdir "%PROJECT%\processed_data\smri\vbm_spm"

echo ============================================================
echo NeuroFusion AI - SPM25 VBM
echo Starting 2 simultaneous workers
echo ============================================================
echo.

start "NeuroFusion VBM Worker 1" cmd /k "set VBM_WORKER_ID=1&&set VBM_NUM_WORKERS=2&&set PATH=%MCR%;%PATH%&&cd /d %PROJECT%&&""%SPM%"" script ""%PROJECT%\src\vbm_spm\run_vbm_batch_worker.m"""
start "NeuroFusion VBM Worker 2" cmd /k "set VBM_WORKER_ID=2&&set VBM_NUM_WORKERS=2&&set PATH=%MCR%;%PATH%&&cd /d %PROJECT%&&""%SPM%"" script ""%PROJECT%\src\vbm_spm\run_vbm_batch_worker.m"""

echo.
echo Two workers launched. Keep both worker windows open.
echo Each worker processes a disjoint subject subset.
echo.
pause
