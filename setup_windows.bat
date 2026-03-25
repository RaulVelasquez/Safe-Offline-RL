@echo off
REM ============================================================
REM  setup_windows.bat
REM  Configura el proyecto ATSC en Windows:
REM    1. Verifica SUMO y SUMO_HOME
REM    2. Descarga LemgoRL (los archivos de red)
REM    3. Instala dependencias Python
REM    4. Instala el paquete en modo editable
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ATSC Project -- Setup para Windows
echo ============================================================
echo.

REM ── 1. Verificar Python ─────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.9+ desde python.org
    pause & exit /b 1
)
echo [OK] Python encontrado

REM ── 2. Verificar SUMO_HOME ──────────────────────────────────
if "%SUMO_HOME%"=="" (
    REM Intentar detectar instalacion tipica de SUMO en Windows
    if exist "C:\Program Files (x86)\Eclipse\Sumo" (
        set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"
        echo [AUTO] SUMO_HOME detectado: !SUMO_HOME!
    ) else if exist "C:\Program Files\Eclipse\Sumo" (
        set "SUMO_HOME=C:\Program Files\Eclipse\Sumo"
        echo [AUTO] SUMO_HOME detectado: !SUMO_HOME!
    ) else (
        echo [ERROR] SUMO_HOME no esta configurado y no se detecto SUMO.
        echo         Instala SUMO desde https://sumo.dlr.de/docs/Downloads.php
        echo         Luego ejecuta: set SUMO_HOME=C:\Program Files ^(x86^)\Eclipse\Sumo
        pause & exit /b 1
    )
) else (
    echo [OK] SUMO_HOME = %SUMO_HOME%
)

REM ── 3. Verificar que sumo.exe existe ────────────────────────
if not exist "%SUMO_HOME%\bin\sumo.exe" (
    echo [ERROR] No se encontro sumo.exe en %SUMO_HOME%\bin\
    echo         Verifica la instalacion de SUMO.
    pause & exit /b 1
)
echo [OK] sumo.exe encontrado en %SUMO_HOME%\bin\

REM ── 4. Agregar SUMO tools al PYTHONPATH ─────────────────────
set "PYTHONPATH=%SUMO_HOME%\tools;%PYTHONPATH%"
echo [OK] SUMO tools agregado al PYTHONPATH

REM ── 5. Crear directorios necesarios ─────────────────────────
if not exist "data\lemgo" mkdir "data\lemgo"
if not exist "data\offline_dataset" mkdir "data\offline_dataset"
if not exist "checkpoints\offline" mkdir "checkpoints\offline"
if not exist "checkpoints\online" mkdir "checkpoints\online"
if not exist "results\logs" mkdir "results\logs"
echo [OK] Directorios de trabajo creados

REM ── 6. Descargar LemgoRL ────────────────────────────────────
echo.
echo [INFO] Descargando archivos de red LemgoRL...

REM Verificar si git esta disponible
git --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Git no encontrado. Descargando LemgoRL manualmente...
    REM Intentar con curl (disponible en Windows 10+)
    curl --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Ni Git ni curl disponibles.
        echo         Descarga LemgoRL manualmente desde:
        echo         https://github.com/IntelligentTrafficSystems/LemgoRL
        echo         y coloca los archivos en: data\lemgo\
        goto :skip_lemgo
    )
    REM Descargar ZIP de LemgoRL via curl
    curl -L -o "data\lemgo_temp.zip" "https://github.com/IntelligentTrafficSystems/LemgoRL/archive/refs/heads/main.zip"
    if errorlevel 1 (
        echo [ERROR] No se pudo descargar LemgoRL.
        goto :skip_lemgo
    )
    REM Extraer con PowerShell
    powershell -Command "Expand-Archive -Path 'data\lemgo_temp.zip' -DestinationPath 'data\lemgo_extracted' -Force"
    REM Mover archivos al directorio correcto
    xcopy /s /y "data\lemgo_extracted\LemgoRL-main\*" "data\lemgo\" >nul 2>&1
    rmdir /s /q "data\lemgo_extracted"
    del "data\lemgo_temp.zip"
    echo [OK] LemgoRL descargado via curl+PowerShell
    goto :check_lemgo
)

REM Clonar con git si existe
if exist "data\lemgo\.git" (
    echo [OK] LemgoRL ya existe, actualizando...
    cd data\lemgo
    git pull --quiet
    cd ..\..
) else (
    git clone --depth=1 https://github.com/IntelligentTrafficSystems/LemgoRL.git data\lemgo_tmp 2>&1
    if errorlevel 1 (
        echo [ERROR] No se pudo clonar LemgoRL. Verifica tu conexion a internet.
        goto :skip_lemgo
    )
    xcopy /s /y "data\lemgo_tmp\*" "data\lemgo\" >nul 2>&1
    rmdir /s /q "data\lemgo_tmp"
)
echo [OK] LemgoRL listo en data\lemgo\

:check_lemgo
REM Verificar que los archivos clave existen
if not exist "data\lemgo\*.net.xml" (
    if not exist "data\lemgo\*\*.net.xml" (
        echo [WARN] No se encontro archivo .net.xml en data\lemgo\
        echo        Es posible que LemgoRL tenga una estructura diferente.
        echo        Busca el .net.xml manualmente y ajusta config.yaml
    )
)

:skip_lemgo

REM ── 7. Detectar y configurar archivos de red ─────────────────
echo.
echo [INFO] Buscando archivos de red SUMO en data\lemgo\...
for /r "data\lemgo" %%f in (*.net.xml) do (
    set "NET_FILE=%%f"
    echo [ENCONTRADO] net_file: %%f
    goto :found_net
)
echo [WARN] No se encontro .net.xml automaticamente.
goto :update_config

:found_net
for /r "data\lemgo" %%f in (*.rou.xml) do (
    set "ROUTE_FILE=%%f"
    echo [ENCONTRADO] route_file: %%f
    goto :update_config
)

:update_config
REM ── 8. Actualizar config.yaml con rutas Windows ─────────────
python -c "
import os, re

config_path = 'configs/config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Convertir backslashes a forward slashes para YAML
net_file = os.environ.get('NET_FILE', 'data/lemgo/lemgo.net.xml').replace('\\\\', '/')
route_file = os.environ.get('ROUTE_FILE', 'data/lemgo/lemgo.rou.xml').replace('\\\\', '/')

content = re.sub(r'net_file:.*', f'net_file: \"{net_file}\"', content)
content = re.sub(r'route_file:.*', f'route_file: \"{route_file}\"', content)
content = re.sub(r'sumo_binary:.*', 'sumo_binary: \"sumo\"', content)

with open(config_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('[OK] config.yaml actualizado con rutas detectadas')
" 2>&1

REM ── 9. Instalar dependencias Python ─────────────────────────
echo.
echo [INFO] Instalando dependencias Python...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    echo         Intenta: pip install -r requirements.txt
    pause & exit /b 1
)
echo [OK] Dependencias instaladas

REM ── 10. Instalar paquete en modo editable ───────────────────
pip install -e . --quiet
if errorlevel 1 (
    echo [WARN] No se pudo instalar en modo editable. Intentando alternativa...
    pip install -e . 2>&1
)
echo [OK] Paquete instalado en modo editable

REM ── 11. Verificar TraCI ─────────────────────────────────────
python -c "import traci; print('[OK] TraCI disponible:', traci.__version__)" 2>&1
if errorlevel 1 (
    echo [WARN] TraCI no disponible desde Python.
    echo        Asegurate de que SUMO_HOME este configurado permanentemente.
    echo        Agrega a tus variables de entorno del sistema:
    echo          SUMO_HOME = %SUMO_HOME%
    echo          PYTHONPATH = %SUMO_HOME%\tools
)

REM ── 12. Test rapido sin SUMO ────────────────────────────────
echo.
echo [INFO] Ejecutando tests de integracion (no requieren SUMO)...
python -m pytest tests/test_integration.py -v --tb=short -q 2>&1
echo.

echo ============================================================
echo   Setup completado.
echo.
echo   Proximos pasos:
echo     1. Verifica que config.yaml tiene las rutas correctas
echo     2. Ejecuta: python main.py --phase 1
echo     3. O sin SUMO: python simulation\run_simulation.py
echo ============================================================
echo.
pause
