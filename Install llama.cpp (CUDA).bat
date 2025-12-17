@echo off
setlocal enabledelayedexpansion

REM ==========================================
REM  Setup Logging
REM ==========================================
set "LOG_FILE=%~dp0llama_install_log.txt"
if exist "%LOG_FILE%" del "%LOG_FILE%"

call :Log "========================================"
call :Log "  LLAMA.CPP Auto-Downloader (Stable)"
call :Log "========================================"
call :Log "Started at: %TIME%"
call :Log "Log file: %LOG_FILE%"
echo.

set "BASE_DIR=%~dp0"
call :Log "[DEBUG] BASE_DIR = %BASE_DIR%"

REM ==========================================
REM  Detect GPU Type
REM ==========================================
call :Log "[STEP 1] Detecting hardware..."

set "BUILD_TYPE=cpu"
set "FILE_PATTERN=win-x64"

nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    call :Log "[INFO] No NVIDIA GPU detected, using CPU build"
) else (
    call :Log "[OK] NVIDIA GPU detected"
    
    set "MIN_COMPUTE_CAP=999"
    set "GPU_COUNT=0"
    
    for /f "skip=1 tokens=*" %%a in ('nvidia-smi --query-gpu=compute_cap --format=csv 2^>nul') do (
        set "CURRENT_GPU=!GPU_COUNT!"
        set /a GPU_COUNT+=1
        set "COMPUTE_CAP=%%a"
        for /f "tokens=*" %%b in ("!COMPUTE_CAP!") do set "COMPUTE_CAP=%%b"
        set "COMPUTE_CAP_NUM=!COMPUTE_CAP:.=!"
        
        call :Log "[INFO] GPU !CURRENT_GPU! Compute Capability: !COMPUTE_CAP!"
        
        if !COMPUTE_CAP_NUM! LSS !MIN_COMPUTE_CAP! (
            set "MIN_COMPUTE_CAP=!COMPUTE_CAP_NUM!"
        )
    )
    
    if !GPU_COUNT! GTR 0 (
        call :Log "[INFO] Minimum compute capability: !MIN_COMPUTE_CAP!"
        if !MIN_COMPUTE_CAP! GEQ 50 (
            call :Log "[OK] GPUs compatible with CUDA binaries"
            set "BUILD_TYPE=cuda"
            set "FILE_PATTERN=win-cuda"
        ) else (
            call :Log "[INFO] Older GPU detected, using CPU binaries"
            set "BUILD_TYPE=cpu"
            set "FILE_PATTERN=win-x64"
        )
    ) else (
        call :Log "[WARNING] Could not detect compute capability, using CPU binaries"
    )
)

call :Log "[DEBUG] BUILD_TYPE = !BUILD_TYPE!"
call :Log "[DEBUG] FILE_PATTERN = !FILE_PATTERN!"
echo.

REM ==========================================
REM  Check currently installed version
REM ==========================================
call :Log "[STEP 2] Checking installed version..."

set "CURRENT_VERSION="
set "CURRENT_DIR="

for /d %%d in ("%BASE_DIR%llama_binaries_*") do (
    set "CURRENT_DIR=%%d"
    for %%n in ("%%~nxd") do (
        set "FOLDER_NAME=%%~n"
        set "CURRENT_VERSION=!FOLDER_NAME:llama_binaries_=!"
    )
)

if defined CURRENT_VERSION (
    call :Log "[INFO] Currently installed: %CURRENT_VERSION%"
) else (
    call :Log "[INFO] No existing installation found."
)
echo.

REM ==========================================
REM  Download release list
REM ==========================================
call :Log "[STEP 3] Checking latest version on GitHub..."

set "JSON_FILE=%TEMP%\llama_releases.json"
if exist "%JSON_FILE%" del "%JSON_FILE%"

REM Using PowerShell to download JSON silently
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; try { Invoke-WebRequest -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -OutFile '%JSON_FILE%' -UseBasicParsing -ErrorAction Stop; Write-Host '[OK] Release list downloaded' } catch { Write-Host '[ERROR]' $_.Exception.Message; exit 1 }" >> "%LOG_FILE%" 2>&1

if %errorlevel% neq 0 (
    call :Log "[ERROR] Failed to download release list."
    pause
    goto :error_exit
)

if not exist "%JSON_FILE%" (
    call :Log "[ERROR] JSON file missing."
    pause
    goto :error_exit
)
call :Log "[OK] Release list downloaded."
echo.

REM ==========================================
REM  Parse releases (STABLE METHOD)
REM ==========================================
call :Log "[STEP 4] Searching for release with %BUILD_TYPE% binaries..."

set "PS_SCRIPT=%TEMP%\find_release.ps1"
if exist "%PS_SCRIPT%" del "%PS_SCRIPT%"

REM --- WRITING LINE-BY-LINE (PREVENTS CRASHES) ---
echo $ErrorActionPreference = 'Stop' >> "%PS_SCRIPT%"
echo try { >> "%PS_SCRIPT%"
echo     $content = Get-Content '%JSON_FILE%' -Raw >> "%PS_SCRIPT%"
echo     $releases = $content ^| ConvertFrom-Json >> "%PS_SCRIPT%"
echo     $pattern = '%FILE_PATTERN%' >> "%PS_SCRIPT%"
echo     foreach($r in $releases) { >> "%PS_SCRIPT%"
echo         $allAssets = $r.assets ^| Where-Object { >> "%PS_SCRIPT%"
echo             $_.name -match '^llama-.*-bin-.*win' -and $_.name -match $pattern -and $_.name -match '.zip$' >> "%PS_SCRIPT%"
echo         } >> "%PS_SCRIPT%"
echo         $sortedAssets = $allAssets ^| Sort-Object { >> "%PS_SCRIPT%"
echo             if ($_.name -match 'cuda-([0-9]+)') { [int]$matches[1] } else { 0 } >> "%PS_SCRIPT%"
echo         } -Descending >> "%PS_SCRIPT%"
echo         $main = $sortedAssets ^| Select-Object -First 1 >> "%PS_SCRIPT%"
echo         if($main) { >> "%PS_SCRIPT%"
echo             Write-Host 'RESULT_VERSION:' $r.tag_name >> "%PS_SCRIPT%"
echo             Write-Host 'RESULT_MAIN:' $main.browser_download_url >> "%PS_SCRIPT%"
echo             if ($main.name -match 'cuda-([0-9]+\.[0-9]+)') { >> "%PS_SCRIPT%"
echo                 $cudaVer = $matches[1] >> "%PS_SCRIPT%"
echo                 Write-Host 'RESULT_CUDA_VER:' $cudaVer >> "%PS_SCRIPT%"
echo                 $cudaPattern = 'cudart-.*cuda-' + $cudaVer >> "%PS_SCRIPT%"
echo                 $cuda = $r.assets ^| Where-Object { $_.name -match $cudaPattern } ^| Select-Object -First 1 >> "%PS_SCRIPT%"
echo                 if ($cuda) { Write-Host 'RESULT_CUDA:' $cuda.browser_download_url } else { Write-Host 'RESULT_CUDA: NONE' } >> "%PS_SCRIPT%"
echo             } else { Write-Host 'RESULT_CUDA: NONE' } >> "%PS_SCRIPT%"
echo             exit 0 >> "%PS_SCRIPT%"
echo         } >> "%PS_SCRIPT%"
echo     } >> "%PS_SCRIPT%"
echo     Write-Host '[PS] ERROR: No release found' >> "%PS_SCRIPT%"
echo     exit 1 >> "%PS_SCRIPT%"
echo } catch { Write-Host '[PS] ERROR:' $_.Exception.Message; exit 1 } >> "%PS_SCRIPT%"

call :Log "[DEBUG] Parsing JSON..."

set "LATEST_VERSION="
set "MAIN_URL="
set "CUDA_URL="
set "CUDA_VER="

for /f "tokens=1,* delims=:" %%a in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" 2^>^&1') do (
    set "KEY=%%a"
    set "VAL=%%b"
    for /f "tokens=*" %%v in ("!VAL!") do set "VAL=%%v"
    
    if "!KEY!"=="RESULT_VERSION" set "LATEST_VERSION=!VAL!"
    if "!KEY!"=="RESULT_MAIN" set "MAIN_URL=!VAL!"
    if "!KEY!"=="RESULT_CUDA" set "CUDA_URL=!VAL!"
    if "!KEY!"=="RESULT_CUDA_VER" set "CUDA_VER=!VAL!"
    if "!KEY!"=="[PS] ERROR" (
        call :Log "[ERROR] PowerShell script failed: !VAL!"
        pause
        goto :error_exit
    )
)

if exist "%PS_SCRIPT%" del "%PS_SCRIPT%" 2>nul
if exist "%JSON_FILE%" del "%JSON_FILE%" 2>nul

if "%CUDA_URL%"=="NONE" set "CUDA_URL="

echo.
if not defined MAIN_URL (
    call :Log "[ERROR] Could not find any release with %BUILD_TYPE% binaries."
    pause
    goto :error_exit
)

call :Log "[OK] Latest available: %LATEST_VERSION%"
if defined CUDA_VER call :Log "[INFO] CUDA Version: %CUDA_VER%"
echo.

REM ==========================================
REM  Compare versions
REM ==========================================
call :Log "[STEP 5] Comparing versions..."

if "%CURRENT_VERSION%"=="%LATEST_VERSION%" (
    echo.
    call :Log "========================================"
    call :Log "  ALREADY UP TO DATE!"
    call :Log "========================================"
    call :Log "  Installed: %CURRENT_VERSION%"
    call :Log "  Location:  %CURRENT_DIR%"
    call :Log "========================================"
    pause
    exit /b 0
)

call :Log "[INFO] Installing: %LATEST_VERSION%"
echo.

REM ==========================================
REM  Setup Directory
REM ==========================================
set "BUILD_DIR=%BASE_DIR%llama_binaries_%LATEST_VERSION%"
call :Log "[STEP 6] Preparing directory: %BUILD_DIR%"

if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"

if not exist "%BUILD_DIR%" (
    call :Log "[ERROR] Failed to create directory."
    pause
    goto :error_exit
)
echo.

REM ==========================================
REM  Download Files
REM ==========================================
call :Log "[STEP 7] Downloading main binaries..."
call :Log "[INFO] URL: %MAIN_URL%"
echo.

REM Using CURL with --ssl-no-revoke to fix error 0x80092012
curl -4 -L --ssl-no-revoke --retry 5 --retry-delay 2 -o "%BUILD_DIR%\main.zip" "%MAIN_URL%"

if %errorlevel% neq 0 (
    echo.
    call :Log "[WARNING] Curl failed. Trying fallback to PowerShell..."
    powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%MAIN_URL%' -OutFile '%BUILD_DIR%\main.zip' -UseBasicParsing"
)

if not exist "%BUILD_DIR%\main.zip" (
    call :Log "[ERROR] Download failed (File not found)."
    pause
    goto :error_exit
)

REM --- SAFETY CHECK ---
set "MAIN_SIZE=0"
for %%A in ("%BUILD_DIR%\main.zip") do set "MAIN_SIZE=%%~zA"

call :Log "[DEBUG] Main Zip Size: !MAIN_SIZE! bytes"

if !MAIN_SIZE! LSS 10000000 (
    call :Log "[ERROR] File too small (!MAIN_SIZE! bytes). Download likely failed."
    pause
    goto :error_exit
)
call :Log "[OK] Main binaries downloaded."
echo.

if defined CUDA_URL (
    call :Log "[STEP 8] Downloading CUDA runtime..."
    call :Log "[INFO] URL: !CUDA_URL!"
    echo.
    
    curl -4 -L --ssl-no-revoke --retry 5 --retry-delay 2 -o "%BUILD_DIR%\cudart.zip" "!CUDA_URL!"
    
    if !errorlevel! neq 0 (
        echo.
        call :Log "[WARNING] Curl failed. Trying fallback to PowerShell..."
        powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '!CUDA_URL!' -OutFile '%BUILD_DIR%\cudart.zip' -UseBasicParsing"
    )
    echo.
)

REM ==========================================
REM  Extract and Organize
REM ==========================================
call :Log "[STEP 9] Extracting and Organizing..."

cd /d "%BUILD_DIR%"

call :Log "[DEBUG] Extracting main.zip..."
powershell -NoProfile -Command "try { Expand-Archive -Path 'main.zip' -DestinationPath '.' -Force -ErrorAction Stop } catch { exit 1 }"
if %errorlevel% neq 0 (
    call :Log "[ERROR] Failed to extract main.zip"
    pause
    goto :error_exit
)
del main.zip

if exist cudart.zip (
    call :Log "[DEBUG] Extracting cudart.zip..."
    powershell -NoProfile -Command "try { Expand-Archive -Path 'cudart.zip' -DestinationPath '.' -Force -ErrorAction Stop } catch { exit 1 }"
    del cudart.zip
)

REM Move files from nested folders
for /d %%d in ("llama-*") do (
    call :Log "[DEBUG] Moving files from %%d..."
    xcopy "%%d\*" "." /E /Y /Q >nul
    rmdir /s /q "%%d"
)
for /d %%d in ("cudart-*") do (
    call :Log "[DEBUG] Moving files from %%d..."
    xcopy "%%d\*" "." /E /Y /Q >nul
    rmdir /s /q "%%d"
)

REM ==========================================
REM  Cleanup Old Version
REM ==========================================
if defined CURRENT_DIR (
    if exist "%CURRENT_DIR%" (
        call :Log "[INFO] Removing old version folder..."
        rmdir /s /q "%CURRENT_DIR%"
    )
)

REM ==========================================
REM  Finish
REM ==========================================
echo.
call :Log "========================================"
call :Log "  SUCCESS! Installed: %LATEST_VERSION%"
call :Log "========================================"
call :Log "  Location: %BUILD_DIR%"
echo.
pause
exit /b 0

:error_exit
echo.
call :Log "[FATAL ERROR] The script encountered an error."
if defined BUILD_DIR (
    if exist "%BUILD_DIR%" (
         call :Log "[DEBUG] Cleaning up failed dir: %BUILD_DIR%"
         rmdir /s /q "%BUILD_DIR%" 2>nul
    )
)
echo.
echo Check the log file for details: %LOG_FILE%
pause
exit /b 1

REM ==========================================
REM  Logging Helper
REM ==========================================
:Log
echo %~1
echo %~1 >> "%LOG_FILE%"
exit /b
