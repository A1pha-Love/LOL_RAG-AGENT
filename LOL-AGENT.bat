@echo off
rem ===========================================================================
rem LOL-AGENT launcher
rem
rem Starts the local backend (which also serves the already-built frontend) and
rem then opens the browser once the server is actually ready.
rem
rem Why this file is ASCII-only:
rem   cmd.exe parses a .bat by the system ANSI code page (GBK on a Chinese
rem   Windows), byte by byte. A UTF-8 .bat with Chinese text gets split mid
rem   character and the fragments are executed as commands ("'OOT' is not
rem   recognized ..."). `chcp 65001` only changes the *output* code page at
rem   run time and cannot fix parsing. So all text here stays ASCII; the Chinese
rem   explanation lives in LOL-AGENT.vbs, which Windows Script Host reads with
rem   proper encoding awareness.
rem
rem Design notes:
rem   * One port, one process: the backend mounts frontend/dist, so there is no
rem     need to run Vite separately.
rem   * Poll /api/health before opening the browser. Opening it earlier shows
rem     "cannot reach this page" and looks like a failed start.
rem   * The wait loop runs inside one PowerShell call so the .bat only has to
rem     test a single exit code; nesting that logic in .bat is fragile.
rem   * The backend runs in its own window, so its logs are visible and closing
rem     that window stops the service.
rem ===========================================================================

setlocal
chcp 65001 >nul 2>&1

rem %~dp0 is this script's directory (with a trailing backslash), so the project
rem is found no matter where the script is launched from.
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%"

echo ============================================================
echo   LOL-AGENT - LoL ranked-data RAG question answering agent
echo ============================================================
echo   Project : %PROJECT_ROOT%
echo   Web UI  : %URL%
echo   Stop    : close the "LOL-AGENT backend" window
echo ============================================================
echo.

rem --- Error paths ---------------------------------------------------------
rem When launched through LOL-AGENT.lnk the console is hidden, so messages and
rem `pause` would be invisible: pause would then wait for a key that can never be
rem pressed and leave a stray process behind. In that mode the wrapper shows a
rem message box instead, and this script must not wait. LOL-AGENT.vbs sets the
rem variable below.
if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment interpreter not found:
    echo         %PYTHON%
    echo.
    echo         Run these first:
    echo           python -m venv .venv
    echo           .venv\Scripts\python.exe -m pip install -r requirements-lock.txt
    echo.
    if not defined LOL_AGENT_HIDDEN pause
    exit /b 1
)

rem --- Check 2: API key -----------------------------------------------------
if not exist "%PROJECT_ROOT%\.env" (
    echo [ERROR] .env not found ^(it holds DASHSCOPE_API_KEY^).
    echo         Run: copy .env.example .env   then fill in the API key.
    echo.
    if not defined LOL_AGENT_HIDDEN pause
    exit /b 1
)

rem --- Check 3: built frontend ----------------------------------------------
rem Without dist the backend still starts but serves no UI, so the browser would
rem show a 404. Warn instead of failing: the API is still usable.
if not exist "%PROJECT_ROOT%\frontend\dist\index.html" (
    echo [WARN] frontend\dist\index.html not found.
    echo        The backend will start but %URL% will not show the UI.
    echo        Build it with:  cd frontend ^&^& npm install ^&^& npm run build
    echo.
)

rem --- Start the backend in its own window ----------------------------------
echo Starting the backend...
start "LOL-AGENT backend" /D "%PROJECT_ROOT%" "%PYTHON%" -m uvicorn apps.api:app --host 127.0.0.1 --port %PORT% --log-level warning

rem --- Wait until it answers -------------------------------------------------
rem Poll once a second for up to 60 seconds: first start loads the knowledge
rem index, so a fixed sleep would be either too short or needlessly slow.
echo Waiting for the service to become ready (up to 60s)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$url='%URL%/api/health'; for ($i=0; $i -lt 60; $i++) { try { $r = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } } catch { }; Start-Sleep -Seconds 1 }; exit 1"

if errorlevel 1 goto :not_ready

echo Ready. Opening the browser...
start "" "%URL%"
echo.
echo Done. The backend runs in the "LOL-AGENT backend" window.
goto :done

:not_ready
echo.
echo [WARN] Timed out waiting for the service.
echo        Check the "LOL-AGENT backend" window for errors, or open %URL% manually.

:done
echo.
echo This window can be closed.
if not defined LOL_AGENT_HIDDEN pause
endlocal
