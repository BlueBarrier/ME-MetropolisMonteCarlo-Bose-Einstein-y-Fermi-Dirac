"""Función integradora: reproduce todo el proyecto con un solo comando.

Orden: pruebas → validación → equilibración → barridos → comparación experimental → figuras y tablas → animación.

    python scripts/run_all.py              # corrida completa (~30–40 min, regenera data/raw y figures/)
    python scripts/run_all.py --fast       # verificación rápida (~4 min, en data/raw_fast y figures/fast)
    python scripts/run_all.py --plots-only # sólo figuras y tablas desde los datos crudos ya guardados

Es el equivalente multiplataforma de ``make all`` (en Windows ``make`` no suele estar instalado).
"""

from __future__ import annotations

import subprocess
import sys
import time

from _common import ROOT, base_parser

STEPS = [
    ("Pruebas automatizadas", [sys.executable, "-m", "pytest", "-q"]),
    ("Validación (A2)", [sys.executable, "scripts/run_validation.py"]),
    ("Equilibración y autocorrelación (A3)", [sys.executable, "scripts/run_equilibration.py"]),
    ("Barridos en q_max, T y N", [sys.executable, "scripts/run_sweeps.py"]),
    ("Comparación con experimentos", [sys.executable, "scripts/compare_experiments.py"]),
    ("Figuras y tablas", [sys.executable, "scripts/plot_results.py"]),
    ("Animación", [sys.executable, "scripts/make_animation.py"]),
]


def main() -> None:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--plots-only", action="store_true", help="Sólo regenerar figuras/tablas/animación desde data/raw.")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    steps = STEPS
    if args.plots_only:
        steps = [STEPS[4], STEPS[5], ("Animación", STEPS[6][1] + ["--from-raw"])]
    elif args.skip_tests:
        steps = STEPS[1:]
    t0 = time.perf_counter()
    for name, cmd in steps:
        if args.fast and "pytest" not in cmd:
            cmd = cmd + ["--fast"]
        print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
        t = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"Falló el paso '{name}' (código {result.returncode}).")
        print(f"--- {name} terminado en {time.perf_counter() - t:.0f} s", flush=True)
    print(f"\nTodo listo en {(time.perf_counter() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
