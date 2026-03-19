@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

if not exist "results_vortex_all\real_vortex_all_p5" (
    mkdir "results_vortex_all\real_vortex_all_p5"
)

set "PYEXE=python"
where python >nul 2>nul
if errorlevel 1 (
    set "PYEXE=py -3"
)

echo ============================================================
echo Running real-vortex full batch...
echo Script : run_real_vorticity_all_p5.py
echo Data   : auto-detect from repo root
echo Output : .\results_vortex_all\real_vortex_all_p5
echo ============================================================

%PYEXE% -u run_real_vorticity_all_p5.py ^
  --out-dir ".\results_vortex_all\real_vortex_all_p5" ^
  --gate-percentile 5

if errorlevel 1 (
    echo.
    echo FAILED: script returned error.
    pause
    exit /b 1
)

echo.
echo DONE.
pause
exit /b 0
