@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   LLAMA.CPP Auto-Downloader
echo ========================================
echo.

set "BASE_DIR=%~dp0"

echo [DEBUG] BASE_DIR = %BASE_DIR%
echo.

REM ==========================================
REM  Detect GPU Type
REM ==========================================
echo [STEP 1] Detecting hardware...

set "BUILD_TYPE=cuda-12"
set "FILE_PATTERN=win-cuda-12"

nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] No NVIDIA GPU detected, using CPU build
    set "BUILD_TYPE=cpu"
    set "FILE_PATTERN=win-x64"
) else (
    echo [OK] NVIDIA GPU detected
)

echo [DEBUG] BUILD_TYPE = %BUILD_TYPE%
echo [DEBUG] FILE_PATTERN = %FILE_PATTERN%
echo.

REM ==========================================
REM  Check currently installed version
REM ==========================================
echo [STEP 2] Checking installed version...

set "CURRENT_VERSION="
set "CURRENT_DIR="

REM Find existing llama_binaries_* folder
for /d %%d in ("%BASE_DIR%llama_binaries_*") do (
    set "CURRENT_DIR=%%d"
    for %%n in ("%%~nxd") do (
        set "FOLDER_NAME=%%~n"
        set "CURRENT_VERSION=!FOLDER_NAME:llama_binaries_=!"
    )
)

if defined CURRENT_VERSION (
    echo [INFO] Currently installed: %CURRENT_VERSION%
    echo [DEBUG] Current folder: %CURRENT_DIR%
) else (
    echo [INFO] No existing installation found
)
echo.

REM ==========================================
REM  Download release list
REM ==========================================
echo [STEP 3] Checking latest version on GitHub...

set "JSON_FILE=%TEMP%\llama_releases.json"

if exist "%JSON_FILE%" del "%JSON_FILE%"

powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -OutFile '%JSON_FILE%' -UseBasicParsing -ErrorAction Stop; Write-Host '[OK] Download successful' } catch { Write-Host '[ERROR]' $_.Exception.Message }"

if not exist "%JSON_FILE%" (
    echo [ERROR] Failed to download release list
    goto :error_exit
)

echo [OK] Release list downloaded
echo.

REM ==========================================
REM  Parse releases to find one with binaries
REM ==========================================
echo [STEP 4] Searching for release with %BUILD_TYPE% binaries...

set "PS_SCRIPT=%TEMP%\find_release.ps1"

(
echo $ErrorActionPreference = 'Stop'
echo try {
echo     $content = Get-Content '%JSON_FILE%' -Raw
echo     $releases = $content ^| ConvertFrom-Json
echo.
echo     $pattern = '%FILE_PATTERN%'
echo.
echo     foreach($r in $releases^) {
echo         $main = $r.assets ^| Where-Object { 
echo             $_.name -match '^llama-.*-bin-.*win' -and 
echo             $_.name -match $pattern -and 
echo             $_.name -match '\.zip$'
echo         } ^| Select-Object -First 1
echo.
echo         if($main^) {
echo             Write-Host 'RESULT_VERSION:' $r.tag_name
echo             Write-Host 'RESULT_MAIN:' $main.browser_download_url
echo.
echo             $cuda = $r.assets ^| Where-Object { $_.name -match '^cudart-' -and $_.name -match $pattern } ^| Select-Object -First 1
echo             if($cuda^) {
echo                 Write-Host 'RESULT_CUDA:' $cuda.browser_download_url
echo             } else {
echo                 Write-Host 'RESULT_CUDA: NONE'
echo             }
echo             exit 0
echo         }
echo     }
echo     Write-Host '[PS] ERROR: No release found with matching binaries'
echo     exit 1
echo } catch {
echo     Write-Host '[PS] ERROR:' $_.Exception.Message
echo     exit 1
echo }
) > "%PS_SCRIPT%"

set "LATEST_VERSION="
set "MAIN_URL="
set "CUDA_URL="

for /f "tokens=1,* delims=:" %%a in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" 2^>^&1') do (
    set "KEY=%%a"
    set "VAL=%%b"
    
    if "!KEY!"=="RESULT_VERSION" set "LATEST_VERSION=!VAL:~1!"
    if "!KEY!"=="RESULT_MAIN" set "MAIN_URL=!VAL:~1!"
    if "!KEY!"=="RESULT_CUDA" set "CUDA_URL=!VAL:~1!"
)

del "%PS_SCRIPT%" 2>nul
del "%JSON_FILE%" 2>nul

if "%CUDA_URL%"=="NONE" set "CUDA_URL="
if "%CUDA_URL%"==" NONE" set "CUDA_URL="

echo [DEBUG] Latest version: %LATEST_VERSION%
echo [DEBUG] Main URL: %MAIN_URL%

if not defined MAIN_URL (
    echo [ERROR] Could not find any release with %BUILD_TYPE% binaries
    goto :error_exit
)

echo [OK] Latest available: %LATEST_VERSION%
echo.

REM ==========================================
REM  Compare versions - skip if already current
REM ==========================================
echo [STEP 5] Comparing versions...

if "%CURRENT_VERSION%"=="%LATEST_VERSION%" (
    echo.
    echo ========================================
    echo   ALREADY UP TO DATE!
    echo ========================================
    echo.
    echo   Installed version: %CURRENT_VERSION%
    echo   Latest version:    %LATEST_VERSION%
    echo.
    echo   Location: %CURRENT_DIR%
    echo.
    echo   No update needed.
    echo.
    echo ========================================
    echo.
    pause
    exit /b 0
)

if defined CURRENT_VERSION (
    echo [INFO] Update available: %CURRENT_VERSION% -^> %LATEST_VERSION%
) else (
    echo [INFO] Will install: %LATEST_VERSION%
)
echo.

REM ==========================================
REM  Set up new directory with version
REM ==========================================
set "BUILD_DIR=%BASE_DIR%llama_binaries_%LATEST_VERSION%"

echo [DEBUG] New folder will be: %BUILD_DIR%
echo.

REM ==========================================
REM  Prepare directory
REM ==========================================
echo [STEP 6] Preparing directory...

if exist "%BUILD_DIR%" (
    echo [DEBUG] Removing incomplete installation...
    rmdir /s /q "%BUILD_DIR%"
)

mkdir "%BUILD_DIR%"
if not exist "%BUILD_DIR%" (
    echo [ERROR] Failed to create directory: %BUILD_DIR%
    goto :error_exit
)

echo [OK] Directory ready: %BUILD_DIR%
echo.

REM ==========================================
REM  Download main binaries
REM ==========================================
echo [STEP 7] Downloading main binaries...
echo [INFO] URL: %MAIN_URL%
echo [INFO] This may take a few minutes...

powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%MAIN_URL%' -OutFile '%BUILD_DIR%\main.zip' -UseBasicParsing -ErrorAction Stop; Write-Host '[OK] Download complete' } catch { Write-Host '[ERROR]' $_.Exception.Message }"

if not exist "%BUILD_DIR%\main.zip" (
    echo [ERROR] Failed to download main binaries
    goto :error_exit
)

for %%A in ("%BUILD_DIR%\main.zip") do set "MAIN_SIZE=%%~zA"

if %MAIN_SIZE% LSS 10000000 (
    echo [ERROR] File too small ^(%MAIN_SIZE% bytes^). Download may have failed.
    goto :error_exit
)

echo [OK] Main binaries downloaded: %MAIN_SIZE% bytes
echo.

REM ==========================================
REM  Download CUDA runtime (optional)
REM ==========================================
if defined CUDA_URL (
    echo [STEP 8] Downloading CUDA runtime...
    echo [INFO] URL: %CUDA_URL%
    echo [INFO] This is a large file ~370MB, please wait...
    
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%CUDA_URL%' -OutFile '%BUILD_DIR%\cudart.zip' -UseBasicParsing -ErrorAction Stop; Write-Host '[OK] Download complete' } catch { Write-Host '[ERROR]' $_.Exception.Message }"
    
    if exist "%BUILD_DIR%\cudart.zip" (
        for %%A in ("%BUILD_DIR%\cudart.zip") do echo [OK] CUDA runtime downloaded: %%~zA bytes
    ) else (
        echo [WARNING] CUDA runtime download failed
        echo [WARNING] GPU acceleration may not work
    )
    echo.
) else (
    echo [STEP 8] Skipping CUDA runtime ^(not available^)
    echo.
)

REM ==========================================
REM  Extract files
REM ==========================================
echo [STEP 9] Extracting files...

cd /d "%BUILD_DIR%"

echo [DEBUG] Extracting main.zip...
powershell -NoProfile -Command "try { Expand-Archive -Path 'main.zip' -DestinationPath '.' -Force -ErrorAction Stop; Write-Host '[OK] Extracted main.zip' } catch { Write-Host '[ERROR]' $_.Exception.Message }"

if exist main.zip del main.zip

if exist cudart.zip (
    echo [DEBUG] Extracting cudart.zip...
    powershell -NoProfile -Command "try { Expand-Archive -Path 'cudart.zip' -DestinationPath '.' -Force -ErrorAction Stop; Write-Host '[OK] Extracted cudart.zip' } catch { Write-Host '[ERROR]' $_.Exception.Message }"
    del cudart.zip
)

echo.

REM ==========================================
REM  Move files from subfolders
REM ==========================================
echo [STEP 10] Organizing files...

for /d %%d in ("llama-*") do (
    echo [DEBUG] Moving files from %%d to root...
    xcopy "%%d\*" "." /E /Y /Q >nul
    rmdir /s /q "%%d"
)

for /d %%d in ("cudart-*") do (
    echo [DEBUG] Moving CUDA files from %%d to root...
    xcopy "%%d\*" "." /E /Y /Q >nul
    rmdir /s /q "%%d"
)

echo.

REM ==========================================
REM  Verify installation
REM ==========================================
echo [STEP 11] Verifying installation...

set "EXE_COUNT=0"
for %%f in (*.exe) do set /a EXE_COUNT+=1

echo [DEBUG] Found %EXE_COUNT% executables

if %EXE_COUNT%==0 (
    echo [ERROR] No executables found!
    echo.
    echo Directory contents:
    dir
    goto :error_exit
)

REM ==========================================
REM  Remove old version (only after successful install)
REM ==========================================
if defined CURRENT_DIR (
    if exist "%CURRENT_DIR%" (
        echo.
        echo [STEP 12] Removing old version: %CURRENT_VERSION%
        rmdir /s /q "%CURRENT_DIR%"
        if exist "%CURRENT_DIR%" (
            echo [WARNING] Could not fully remove old folder
        ) else (
            echo [OK] Old version removed
        )
    )
)

REM ==========================================
REM  Success!
REM ==========================================
echo.
echo ========================================
echo   SUCCESS! 
echo ========================================
echo.
if defined CURRENT_VERSION (
    echo   Updated: %CURRENT_VERSION% -^> %LATEST_VERSION%
) else (
    echo   Installed: %LATEST_VERSION%
)
echo.
echo   Location: %BUILD_DIR%
echo   Executables: %EXE_COUNT%
echo.
echo Key files:
if exist "llama-cli.exe" echo   [OK] llama-cli.exe
if exist "llama-server.exe" echo   [OK] llama-server.exe  
if exist "llama-quantize.exe" echo   [OK] llama-quantize.exe
if exist "llama-bench.exe" echo   [OK] llama-bench.exe
echo.
echo ========================================
echo.
pause
exit /b 0

REM ==========================================
REM  Error handler
REM ==========================================
:error_exit
echo.
echo ========================================
echo   FAILED - See errors above
echo ========================================
echo.

REM Clean up failed installation attempt
if defined BUILD_DIR (
    if exist "%BUILD_DIR%" (
        echo [DEBUG] Cleaning up failed installation...
        rmdir /s /q "%BUILD_DIR%" 2>nul
    )
)

echo Press any key to exit...
pause >nul
exit /b 1
