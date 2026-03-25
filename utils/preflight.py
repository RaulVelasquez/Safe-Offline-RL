# utils/preflight.py
"""
Verificador de pre-vuelo (preflight check).
Valida que el entorno esté correctamente configurado ANTES de correr
cualquier fase. Detecta y reporta todos los problemas de una vez,
con instrucciones específicas para Windows y Linux/macOS.
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Colores para la terminal (funciona en Windows 10+ y Linux/macOS)
# ─────────────────────────────────────────────────────────────────────────────

IS_WINDOWS = platform.system() == "Windows"

def _c(text: str, code: str) -> str:
    """Aplica color ANSI solo si la terminal lo soporta."""
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            return text
    return f"\033[{code}m{text}\033[0m"

OK   = lambda t: _c(f"  [OK]   {t}", "32")
WARN = lambda t: _c(f"  [WARN] {t}", "33")
ERR  = lambda t: _c(f"  [ERR]  {t}", "31")
INFO = lambda t: _c(f"  [-->]  {t}", "36")


# ─────────────────────────────────────────────────────────────────────────────
# Verificaciones individuales
# ─────────────────────────────────────────────────────────────────────────────

def check_python_version() -> Tuple[bool, str]:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        return False, f"Python {major}.{minor} detectado. Se requiere Python 3.9+."
    return True, f"Python {major}.{minor} OK"


def check_sumo_home() -> Tuple[bool, str]:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        # Intentar detectar en rutas comunes de Windows
        if IS_WINDOWS:
            candidates = [
                r"C:\Program Files (x86)\Eclipse\Sumo",
                r"C:\Program Files\Eclipse\Sumo",
                r"C:\Sumo",
            ]
            for c in candidates:
                if Path(c).exists():
                    os.environ["SUMO_HOME"] = c
                    # Agregar tools al path de Python
                    tools = str(Path(c) / "tools")
                    if tools not in sys.path:
                        sys.path.insert(0, tools)
                    return True, f"SUMO_HOME detectado automáticamente: {c}"
        return False, (
            "SUMO_HOME no está configurado.\n"
            + ("        Abre una CMD como administrador y ejecuta:\n"
               "        setx SUMO_HOME \"C:\\Program Files (x86)\\Eclipse\\Sumo\"\n"
               "        Luego cierra y vuelve a abrir la terminal."
               if IS_WINDOWS else
               "        Ejecuta: export SUMO_HOME=/usr/share/sumo\n"
               "        Agrega esa línea a tu ~/.bashrc para que persista.")
        )

    tools = str(Path(sumo_home) / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return True, f"SUMO_HOME = {sumo_home}"


def check_sumo_binary() -> Tuple[bool, str]:
    sumo_home = os.environ.get("SUMO_HOME", "")
    binary = "sumo.exe" if IS_WINDOWS else "sumo"

    # Buscar en SUMO_HOME/bin primero
    sumo_bin = Path(sumo_home) / "bin" / binary
    if sumo_bin.exists():
        return True, f"sumo encontrado en {sumo_bin}"

    # Buscar en PATH
    try:
        result = subprocess.run(
            ["sumo", "--version"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return True, "sumo encontrado en PATH del sistema"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, (
        f"sumo{'  .exe' if IS_WINDOWS else ''} no encontrado.\n"
        + ("        Descarga SUMO desde https://sumo.dlr.de/docs/Downloads.php\n"
           "        Instala la versión Windows (msi) e intenta de nuevo."
           if IS_WINDOWS else
           "        sudo apt-get install sumo sumo-tools sumo-gui")
    )


def check_traci() -> Tuple[bool, str]:
    try:
        import traci  # noqa: F401
        return True, "TraCI disponible"
    except ImportError:
        sumo_home = os.environ.get("SUMO_HOME", "")
        tools = str(Path(sumo_home) / "tools") if sumo_home else ""
        return False, (
            "TraCI no disponible desde Python.\n"
            + (f"        Agrega a PYTHONPATH: {tools}\n"
               "        En CMD: set PYTHONPATH=%SUMO_HOME%\\tools;%PYTHONPATH%"
               if IS_WINDOWS else
               f"        export PYTHONPATH={tools}:$PYTHONPATH")
        )


def check_torch() -> Tuple[bool, str]:
    try:
        import torch
        device = "CUDA disponible" if torch.cuda.is_available() else "solo CPU"
        return True, f"PyTorch {torch.__version__} ({device})"
    except ImportError:
        return False, (
            "PyTorch no instalado.\n"
            "        Ejecuta: pip install torch\n"
            "        O visita: https://pytorch.org/get-started/locally/"
        )


def check_net_file(net_file: str) -> Tuple[bool, str]:
    """Verifica que el archivo de red SUMO existe."""
    p = Path(net_file)
    if p.exists():
        return True, f"net_file encontrado: {p}"

    # Intentar buscar cualquier .net.xml en data/lemgo/
    search_dirs = [Path("data/lemgo"), Path("data")]
    for d in search_dirs:
        if d.exists():
            found = list(d.rglob("*.net.xml"))
            if found:
                return False, (
                    f"'{net_file}' no existe, pero se encontró:\n"
                    f"        {found[0]}\n"
                    f"        Actualiza configs/config.yaml:\n"
                    f"          net_file: \"{found[0]}\""
                )

    return False, (
        f"'{net_file}' no existe y no hay archivos .net.xml en data/lemgo/\n"
        "        LemgoRL no está descargado. Ejecuta:\n"
        + ("        setup_windows.bat"
           if IS_WINDOWS else
           "        bash scripts/setup_lemgorl.sh")
    )


def check_route_file(route_file: str) -> Tuple[bool, str]:
    """Verifica que el archivo de rutas SUMO existe."""
    p = Path(route_file)
    if p.exists():
        return True, f"route_file encontrado: {p}"

    search_dirs = [Path("data/lemgo"), Path("data")]
    for d in search_dirs:
        if d.exists():
            found = list(d.rglob("*.rou.xml"))
            if found:
                return False, (
                    f"'{route_file}' no existe, pero se encontró:\n"
                    f"        {found[0]}\n"
                    f"        Actualiza configs/config.yaml:\n"
                    f"          route_file: \"{found[0]}\""
                )

    return False, (
        f"'{route_file}' no existe.\n"
        "        Descarga LemgoRL o ajusta route_file en configs/config.yaml."
    )


def check_package_installed() -> Tuple[bool, str]:
    """Verifica que el paquete está instalado en modo editable."""
    try:
        import phase1  # noqa: F401
        return True, "Paquete ATSC instalado correctamente (pip install -e .)"
    except ImportError:
        return False, (
            "El paquete no está instalado.\n"
            "        Ejecuta desde la raíz del proyecto:\n"
            "        pip install -e ."
        )


def check_output_dirs() -> Tuple[bool, str]:
    """Crea los directorios de salida si no existen."""
    dirs = [
        "data/offline_dataset",
        "checkpoints/offline",
        "checkpoints/online",
        "results/logs",
    ]
    created = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(d)
    if created:
        return True, f"Directorios creados: {', '.join(created)}"
    return True, "Directorios de trabajo ya existen"


# ─────────────────────────────────────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────────────────────────────────────

def run_preflight(
    net_file: str = "data/lemgo/lemgo.net.xml",
    route_file: str = "data/lemgo/lemgo.rou.xml",
    require_sumo: bool = True,
    require_torch: bool = True,
    abort_on_error: bool = True,
) -> bool:
    """
    Ejecuta todas las verificaciones de pre-vuelo.

    Parámetros
    ----------
    net_file, route_file : str
        Rutas leídas de config.yaml.
    require_sumo : bool
        Si False, omite las verificaciones de SUMO (útil para solo simulación).
    require_torch : bool
        Si False, omite la verificación de PyTorch.
    abort_on_error : bool
        Si True, llama a sys.exit(1) cuando hay errores críticos.

    Retorna
    -------
    bool : True si todo está OK, False si hay errores.
    """
    print("\n" + "=" * 60)
    print("  Pre-flight check — Verificando entorno")
    print(f"  Sistema: {platform.system()} {platform.release()}")
    print("=" * 60)

    checks: List[Tuple[str, bool, Tuple[bool, str]]] = []

    # Siempre verificar
    checks.append(("Python", True,  check_python_version()))
    checks.append(("Paquete", True, check_package_installed()))
    checks.append(("Dirs",   False, check_output_dirs()))

    if require_torch:
        checks.append(("PyTorch", True, check_torch()))

    if require_sumo:
        checks.append(("SUMO_HOME", True,  check_sumo_home()))
        checks.append(("sumo bin",  True,  check_sumo_binary()))
        checks.append(("TraCI",     True,  check_traci()))
        checks.append(("net_file",  True,  check_net_file(net_file)))
        checks.append(("route_file",True,  check_route_file(route_file)))

    errors   = []
    warnings = []

    for name, is_critical, (ok, msg) in checks:
        if ok:
            print(OK(msg))
        elif is_critical:
            print(ERR(msg))
            errors.append((name, msg))
        else:
            print(WARN(msg))
            warnings.append((name, msg))

    print("=" * 60)

    if errors:
        print(_c(f"\n  [X] {len(errors)} error(es) critico(s) encontrado(s):", "31"))
        for name, msg in errors:
            first_line = msg.split('\n')[0]
            print(_c(f"    • [{name}] {first_line}", "31"))
        print()
        print(INFO("Consulta la documentación o ejecuta:"))
        if IS_WINDOWS:
            print(INFO("  setup_windows.bat"))
        else:
            print(INFO("  bash scripts/setup_lemgorl.sh"))
        print()
        if abort_on_error:
            sys.exit(1)
        return False

    if warnings:
        print(_c(f"\n  [!] {len(warnings)} advertencia(s). El sistema puede funcionar.", "33"))

    print(_c("\n  [OK] Entorno verificado. Listo para ejecutar.\n", "32"))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Uso independiente: python utils/preflight.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verificador de entorno ATSC")
    parser.add_argument("--net_file",   default="data/lemgo/lemgo.net.xml")
    parser.add_argument("--route_file", default="data/lemgo/lemgo.rou.xml")
    parser.add_argument("--no_sumo",    action="store_true",
                        help="Omitir verificaciones de SUMO")
    parser.add_argument("--no_abort",   action="store_true",
                        help="No salir si hay errores")
    args = parser.parse_args()

    run_preflight(
        net_file=args.net_file,
        route_file=args.route_file,
        require_sumo=not args.no_sumo,
        abort_on_error=not args.no_abort,
    )
