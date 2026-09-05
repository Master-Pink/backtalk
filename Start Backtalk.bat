@echo off
rem Start Backtalk.bat - launches the voice line, per backtalk.md's own
rem Windows launcher spec (Phase 5.75). Windows .bat files inherit the
rem user's PATH from whatever spawned them, but a shortcut double-clicked
rem from Explorer can inherit a STALE environment (Explorer's own process
rem often predates a later `setx`, e.g. ELEVENLABS_API_KEY or a winget
rem PATH update) - so pull both fresh from the registry rather than trust
rem inheritance. Ends with a pause on error so a crash stays readable
rem instead of the window just vanishing.
cd /d "%~dp0"
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('ELEVENLABS_API_KEY','User')"`) do set "ELEVENLABS_API_KEY=%%A"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%P"
uv sync -q --inexact
uv run python -m backtalk.main
if errorlevel 1 pause
