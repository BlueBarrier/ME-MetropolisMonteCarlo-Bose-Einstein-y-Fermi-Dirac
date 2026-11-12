"""Menú principal del proyecto: 

    python scripts/run.py

Cada opción corre uno de los scripts de esta carpeta. El modo rápido sirve para comprobar que todo funciona corriendo una fracción de los datos.
"""

from __future__ import annotations

import subprocess
import sys
import time

from _common import ROOT

PY = sys.executable
ACTIONS = [
    ("1", "Simulación individual (bosones o fermiones) con resumen de errores", [PY, "scripts/interactive_cli.py"], False),
    ("2", "Comparar datos teóricos propios con la simulación (número de estados máximo a elección)", [PY, "scripts/compare_theory.py"], False),
    ("3", "Validación contra resultados exactos (N = 1, N = 3, N = 10)", [PY, "scripts/run_validation.py"], True),
    ("4", "Validación de equilibración y autocorrelación", [PY, "scripts/run_equilibration.py"], True),
    ("5", "Barridos: corte q_max, temperatura y N, ocupaciones con N = 45", [PY, "scripts/run_sweeps.py"], True),
    ("6", "Generar las figuras (fig/) y visualización", [PY, "scripts/plot_results.py"], True),
    ("7", "Ver las figuras ya guardadas", [PY, "scripts/view_figures.py"], True),
    ("8", "Animación de la relajación al equilibrio (GIF, N = 56)", [PY, "scripts/make_animation.py"], True),
    ("9", "Pruebas automatizadas (pytest)", [PY, "-m", "pytest", "-q"], False),
]
FULL_ORDER = ["3", "4", "5", "8"]  


def run_command(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd[1:])}\n", flush=True)
    t0 = time.perf_counter()
    ok = subprocess.run(cmd, cwd=ROOT).returncode == 0
    print(f"\n<<< {'terminado' if ok else 'FALLÓ'} en {time.perf_counter() - t0:.0f} s", flush=True)
    return ok


def with_fast(key: str, fast: bool):
    _, _, cmd, accepts_fast = next(a for a in ACTIONS if a[0] == key)
    return cmd + (["--fast"] if fast and accepts_fast else [])


def run_everything(fast: bool):
    for key in ["9"] + FULL_ORDER:
        if not run_command(with_fast(key, fast)):
            print("Se detuvo la secuencia porque un paso falló.")
            return
    run_command(with_fast("6", fast))


def menu() -> None:
    fast = False
    while True:
        print("\n" + "=" * 72)
        print(" Estadística cuántica por Monte Carlo: bosones y fermiones en una trampa armónica 3D")
        print("=" * 72)
        for key, desc, _, _ in ACTIONS:
            print(f"  {key}. {desc}")
        print("  10. Reproducir todo (pruebas, validación, equilibración, barridos, animación y figuras)")
        print(f"  r. Modo rápido: {'ACTIVADO (5 % de los datos)' if fast else 'desactivado'}")
        print("  0. Salir")
        choice = input("\nOpción: ").strip().lower()
        if choice == "0":
            print("Hasta luego.")
            return
        if choice == "r":
            fast = not fast
            continue
        if choice == "10":
            run_everything(fast)
            continue
        if choice in {a[0] for a in ACTIONS}:
            run_command(with_fast(choice, fast))
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    try:
        menu()
    except (KeyboardInterrupt, EOFError):
        print("\nSalida.")
