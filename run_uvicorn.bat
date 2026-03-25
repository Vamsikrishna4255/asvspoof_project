@echo off
SET SCRIPT_DIR=%~dp0
IF EXIST "%SCRIPT_DIR%venv\Scripts\activate.bat" (
    call "%SCRIPT_DIR%venv\Scripts\activate.bat"
) ELSE (
    echo Virtual environment not found at "%SCRIPT_DIR%venv\Scripts\activate.bat"
)
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000 %*
