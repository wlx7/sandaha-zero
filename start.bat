@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SanDaHa-Zero 三打哈 AI
where uv >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  ) else (
    echo [首次运行] 正在安装 uv 包管理器 ...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  )
)
echo 正在准备运行环境（首次运行需要下载依赖，请稍候）...
uv run --python 3.12 sandaha %*
if errorlevel 1 pause
