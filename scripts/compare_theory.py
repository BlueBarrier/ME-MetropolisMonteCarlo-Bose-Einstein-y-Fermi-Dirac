"""Compara datos teóricos del usuario con una simulación al q_max que elija el usuario.

Modo interactivo (pregunta todo):
    python scripts/compare_theory.py

Modo directo:
    python scripts/compare_theory.py --statistics fermions --N 20 --q-max 30 --observable energy \
        --theory data/teoria/ejemplo_fermiones_N20_energia.csv
    python scripts/compare_theory.py --statistics bosons --N 20 --q-max 15 --observable occupation --T 1 \
        --theory data/teoria/ejemplo_bosones_N20_ocupacion_T1.csv

Observables: energy (x = T, y = E/N), condensate (x = T, y = ⟨n_0⟩/N),
             occupation (x = ε, y = ⟨n(ε)⟩ por estado, a una T fija).
Los datos teóricos se dan como archivo CSV de dos columnas o escritos a mano ("x y; x y; ...").

Salidas: fig/comparaciones/<nombre>.png y data/raw/comparaciones/<nombre>.csv; al final se abre la figura.
"""

from __future__ import annotations

import argparse
import secrets

import _common
import matplotlib
import numpy as np

matplotlib.use("Agg")

from interactive_cli import STAT_ALIASES, _ask, _choice
from src import plotting as pl
from src.datastore import FIG_DIR, RAW_DIR, write_table_csv
from src.theory_compare import OBSERVABLES, compare_with_theory, parse_pairs, read_theory_file, result_tag
from src.viewer import show_gallery


def interactive_args() -> argparse.Namespace:
    print("=== Comparar datos teóricos con la simulación ===")
    stat = _ask("Estadística: (b)osones o (f)ermiones", "f", _choice({"b": "bosons", "f": "fermions"}))
    N = _ask("Número de partículas N", 20, int)
    q_max = _ask("Corte energético q_max (niveles ε = 0 … q_max)", 30, int)
    obs = _ask("Observable: (e)nergía E/N vs T, (c)ondensado n0/N vs T, (o)cupación n(ε) a T fija", "e",
               _choice({"e": "energy", "c": "condensate", "o": "occupation"}))
    T = _ask("Temperatura T para la ocupación", 1.0, float) if obs == "occupation" else None
    source = _ask("Datos teóricos: ruta a un CSV, o (m)anual", "m", str)
    if source.lower() in ("m", "manual"):
        x_name = "ε" if obs == "occupation" else "T"
        print(f"Escriba pares '{x_name} valor' separados por ';' (p. ej. 0.5 1.81; 1.0 2.60):")
        text = input("> ")
        theory = None
    else:
        text, theory = None, source
    steps = _ask("Pasos por simulación", 20_000_000, lambda s: int(float(s)))
    seed = _ask("Semilla (vacío = aleatoria)", "", str)
    return argparse.Namespace(statistics=stat, N=N, q_max=q_max, observable=obs, T=T, theory=theory, data=text,
                              steps=steps, seed=int(seed) if seed else None, proposal="particle", no_show=False)


def run(args: argparse.Namespace) -> dict:
    if args.theory:
        tx, ty = read_theory_file(args.theory)
    elif args.data:
        tx, ty = parse_pairs(args.data)
    else:
        raise SystemExit("Indique --theory ARCHIVO o --data 'x y; x y; ...'")
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    print(f"\nSimulando {args.statistics}, N = {args.N}, q_max = {args.q_max}, observable = {args.observable}, "
          f"{len(tx)} puntos, {args.steps:.1e} pasos por simulación, semilla = {seed}")
    res = compare_with_theory(args.statistics, args.N, args.q_max, args.observable, tx, ty, T=args.T,
                              n_steps=args.steps, seed=seed, proposal=args.proposal)
    rows = [{"x": x, "teoria": t, "simulacion": s, "sigma": e, "z": z, "error_%": p}
            for x, t, s, e, z, p in zip(res["sim_x"], res["theory_y"], res["sim_y"], res["sim_err"],
                                        res["z"], res["error_pct"])]
    _common.print_table(rows, list(rows[0].keys()), title="Teoría frente a simulación (z = (sim − teo)/σ)")
    for note in res["truncation_notes"]:
        print("  " + note)
    zabs = np.abs(res["z"][np.isfinite(res["z"])])
    if zabs.size:
        print(f"\n|z| medio = {zabs.mean():.2f}; {np.mean(zabs < 2):.0%} de los puntos con |z| < 2. Semillas: {res['seeds']}")

    tag = result_tag(res)
    csv_path = write_table_csv(RAW_DIR / "comparaciones" / f"{tag}.csv",
                               [dict(r, estadistica=args.statistics, N=args.N, q_max=args.q_max,
                                     observable=args.observable, semilla=seed) for r in rows])
    pl.apply_style()
    png = pl.fig_user_comparison(res, FIG_DIR / "comparaciones" / f"{tag}.png")
    print(f"Tabla: {csv_path}\nFigura: {png}")
    if not args.no_show:
        show_gallery([png], title="Teoría vs simulación")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Compara datos teóricos del usuario con la simulación.")
    p.add_argument("--statistics", type=lambda s: STAT_ALIASES[s.lower()])
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--q-max", type=int, default=30)
    p.add_argument("--observable", choices=tuple(OBSERVABLES), default="energy")
    p.add_argument("--T", type=float, default=None, help="Temperatura fija (sólo para 'occupation')")
    p.add_argument("--theory", default=None, help="CSV de dos columnas (x, y)")
    p.add_argument("--data", default=None, help="Pares escritos a mano: 'x y; x y; ...'")
    p.add_argument("--steps", type=lambda s: int(float(s)), default=20_000_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--proposal", choices=("particle", "state"), default="particle")
    p.add_argument("--no-show", action="store_true", help="No abrir la ventana con la figura")
    args = p.parse_args()
    if args.statistics is None:
        args = interactive_args()
    run(args)


if __name__ == "__main__":
    main()
