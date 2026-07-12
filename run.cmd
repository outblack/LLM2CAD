@echo off
setlocal

cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
  echo cadgen02 requires 'uv' so the verified lockfile is used. 1>&2
  exit /b 127
)

uv run --locked --quiet python -m cadgen.cli %*
exit /b %ERRORLEVEL%
