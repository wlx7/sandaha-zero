@echo off
setlocal
cd /d "%~dp0"
title SanDaHa-Zero
where uv >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  ) else (
    echo [first run] installing the uv package manager ...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  )
)
echo Preparing the Python environment (the first run downloads dependencies) ...
uv run --python 3.12 sandaha %*
if errorlevel 1 pause
