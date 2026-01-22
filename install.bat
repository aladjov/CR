@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Customer Retention Framework Setup
echo ========================================
echo.

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Define virtual environment name
set "VENV_NAME=cr"
set "VENV_PATH=%SCRIPT_DIR%%VENV_NAME%"

REM Check if uv is installed
echo [1/6] Checking for uv...
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: uv is not installed.
    echo Please install uv first:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo   or visit: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
echo [+] uv is installed
echo.

REM Check if virtual environment exists
echo [2/6] Checking virtual environment...
if exist "%VENV_PATH%" (
    echo Virtual environment '%VENV_NAME%' already exists at: %VENV_PATH%
    set /p "RECREATE=Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q "%VENV_PATH%"
        echo [+] Old virtual environment removed
    ) else (
        echo Using existing virtual environment
    )
)

if not exist "%VENV_PATH%" (
    echo Creating virtual environment '%VENV_NAME%'...
    uv venv "%VENV_NAME%"
    echo [+] Virtual environment created: %VENV_PATH%
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
echo [+] Virtual environment activated
echo.

REM Install dependencies and package
echo [4/5] Installing package with all dependencies...
uv pip install -e ".[dev]"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install package and dependencies
    exit /b 1
)
echo [+] Package and dependencies installed
echo.

REM Verify installation
echo [5/5] Verifying installation...
python -m pytest --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: pytest not found after installation
    exit /b 1
)
for /f "tokens=*" %%i in ('python -m pytest --version ^| findstr /r "^pytest"') do set PYTEST_VERSION=%%i
echo [+] %PYTEST_VERSION%

python -c "import customer_retention" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Package import failed
    exit /b 1
)
echo [+] Package imports successfully
echo.

REM Print success message and next steps
echo ========================================
echo [+] Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Activate the virtual environment:
echo    %VENV_PATH%\Scripts\activate.bat
echo.
echo 2. Run all tests:
echo    python -m pytest tests/
echo.
echo 3. Run tests with coverage:
echo    python -m pytest tests/ --cov=src/customer_retention --cov-report=term-missing
echo.
echo 4. Run specific test file:
echo    python -m pytest tests/profiling/test_quality_checks.py -v
echo.
echo 5. Deactivate virtual environment when done:
echo    deactivate
echo.
echo Environment details:
echo   Virtual env: %VENV_PATH%
python --version
echo   Pytest: %PYTEST_VERSION%
echo.

pause
