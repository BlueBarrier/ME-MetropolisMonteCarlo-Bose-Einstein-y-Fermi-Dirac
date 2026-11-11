"""Interfaz de línea de comandos para correr una simulación individual.

Modo interactivo (menú con preguntas):
    python scripts/interactive_cli.py

Modo directo (reproducible, sin preguntas):
    python scripts/interactive_cli.py --statistics fermions --N 20 --T 1.5 --seed 42
    python scripts/interactive_cli.py --statistics bosons --N 35 --T-kelvin 50e-9 --omega 2*pi*100

Muestra la semilla usada, E/N y f_0 con su incertidumbre (corregida por autocorrelación),
la discrepancia en σ y el error porcentual frente a la solución canónica exacta, y la
comparación con el ensemble gran canónico (Bose–Einstein o Fermi–Dirac).
Opcionalmente guarda los datos crudos y una figura de ⟨n(ε)⟩.
"""

from __future__ import annotations

import argparse
import ast
import math
import operator
import secrets

import _common
import numpy as np

from src import analytics as an
from src.datastore import FIG_DIR, RAW_DIR, save_raw
from src.metropolis import run_mcmc_chain
from src.model import (
    energy_to_joules,
    is_closed_shell,
    kelvin_temperature,
    reduced_temperature,
)
from src.observables import compute_observables, relative_error_percent
from src.stats import discrepancy_in_sigmas

STAT_ALIASES = {"b": "bosons", "bosones": "bosons", "bosons": "bosons",
                "f": "fermions", "fermiones": "fermions", "fermions": "fermions"}


def _ask(prompt: str, default, cast=str):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return cast(raw)
        except (ValueError, KeyError):
            print("  Valor no válido, intente de nuevo.")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_number(text: str) -> float:
    """Evalúa de forma segura expresiones aritméticas simples como 2*pi*100 o 1e-7."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in ("pi", "e"):
            return {"pi": math.pi, "e": math.e}[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"Expresión no permitida: {text!r}")
    return float(ev(ast.parse(text, mode="eval").body))


def _choice(mapping: dict):
    """Convierte la primera letra de la respuesta usando ``mapping`` (KeyError si no es válida)."""
    return lambda s: mapping[s.strip().lower()[0]]


def interactive_params() -> argparse.Namespace:
    print("=== Simulación MCMC: gas ideal en trampa armónica 3D (ensemble canónico) ===")
    stat = _ask("Estadística: (b)osones o (f)ermiones", "b", _choice({"b": "bosons", "f": "fermions"}))
    N = _ask("Número de partículas N", 20, int)
    units = _ask("Temperatura en unidades (r)educidas ħω/k_B o (s)istema internacional", "r",
                 _choice({"r": "r", "s": "s"}))
    T = T_kelvin = omega = None
    if units == "s":
        omega = _ask("Frecuencia de la trampa ω [rad/s] (se aceptan expresiones, p. ej. 2*pi*100)", "2*pi*100",
                     _eval_number)
        T_kelvin = _ask("Temperatura T [K]", 5e-9, _eval_number)
    else:
        T = _ask("Temperatura reducida T = k_B T/(ħω)", 1.5, float)
    q_max = _ask("Corte energético q_max (0 = automático)", 0, int)
    proposal = _ask("Núcleo: (p)artícula / Metropolis–Hastings o (s)imétrico / estados", "p",
                    _choice({"p": "particle", "s": "state"}))
    steps = _ask("Pasos de producción", 20_000_000, lambda s: int(float(s)))
    seed = _ask("Semilla (vacío = aleatoria)", "", str)
    init = _ask("Estado inicial: (g)round o (e)xcited", "g", _choice({"g": "ground", "e": "excited"}))
    save = _ask("¿Guardar datos crudos y figura? (s/n)", "s", lambda s: s.lower().startswith("s"))
    return argparse.Namespace(statistics=stat, N=N, T=T, T_kelvin=T_kelvin, omega=omega, q_max=q_max,
                              proposal=proposal, steps=steps, seed=int(seed) if seed else None, init=init,
                              save=save, plot=save)


def run(args: argparse.Namespace) -> dict:
    if args.T is None:
        if args.T_kelvin is None or args.omega is None:
            raise SystemExit("Indique --T (reducida) o bien --T-kelvin y --omega.")
        T = reduced_temperature(args.T_kelvin, args.omega)
    else:
        T = args.T
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    stat, N = args.statistics, args.N
    q = args.q_max if args.q_max and args.q_max > 0 else an.choose_q_max(N, T, stat)
    steps = max(int(args.steps) // 20_000 * 20_000, 20_000)

    print(f"\nCorriendo: {stat}, N = {N}, T = {T:.4g} ħω/k_B, q_max = {q}, núcleo = {args.proposal}, "
          f"pasos = {steps:.2e}, semilla = {seed}")
    res = run_mcmc_chain(N, 1.0 / T, q, stat, steps, steps // 10, steps // 20_000, seed, args.init, args.proposal)
    obs = compute_observables(res["E_trace"], res["level_trace"], N)
    ex = an.exact_canonical(N, q, 1.0 / T, stat)
    gc = an.grand_canonical_observables(N, q, 1.0 / T, stat)
    m = res["metadata"]

    rows = []
    for name, key, err_key in (("E/N [ħω]", "energy_per_particle", "energy_per_particle_err"),
                               ("f_0 = <n_0>/N", "condensate_fraction", "condensate_fraction_err")):
        sim, err = obs[key], obs[err_key]
        rows.append({
            "observable": name, "simulación": sim, "σ": err,
            "canónico exacto": ex[key], "z (σ)": discrepancy_in_sigmas(sim, err, ex[key]),
            "error %": relative_error_percent(sim, ex[key]),
            "gran canónico": gc[key], "desv. % vs GC": relative_error_percent(sim, gc[key]),
        })
    _common.print_table(rows, list(rows[0].keys()), title="Resultados")
    ref = (f"T_c = {float(an.bec_critical_temperature(N)):.3f} (N→∞),  T/T_c = {T / float(an.bec_critical_temperature(N)):.3f}"
           if stat == "bosons" else
           f"ε_F = {an.fermi_energy(N)} ħω ({'capa cerrada' if is_closed_shell(N) else 'capa abierta'}),  T/T_F = {T / an.fermi_energy(N):.3f}")
    print(f"\nEscala de temperatura: {ref}")
    print(f"Potencial químico gran canónico: μ = {gc['mu']:.5g} ħω")
    print(f"τ_int(E) = {obs['tau_int_energy']:.2f} mediciones,  τ_int(n_0) = {obs['tau_int_n0']:.2f};"
          f"  error de E/N por bloques = {obs['energy_per_particle_err_blocking']:.3g}")
    print(f"Aceptación = {m['acceptance_rate']:.3%},  tiempo = {m['elapsed_seconds']:.1f} s,  semilla = {seed}")
    tau_max = max(obs["tau_int_energy"], obs["tau_int_n0"])
    if tau_max > obs["n_samples"] / 50:
        print(f"ADVERTENCIA: τ_int = {tau_max:.0f} es comparable a la corrida ({obs['n_samples']} mediciones);"
              " los errores no son confiables. Aumente --steps (≥ 50·τ_int·intervalo).")
    if args.omega:
        print(f"En SI (ω = {args.omega:.4g} rad/s): T = {kelvin_temperature(T, args.omega):.4g} K,"
              f"  E/N = {float(energy_to_joules(obs['energy_per_particle'], args.omega)):.4g} J")

    if args.save:
        tag = f"cli_{stat}_N{N}_T{T:.3g}_seed{seed}"
        path = save_raw(RAW_DIR / "cli" / f"{tag}.npz",
                        {"E_trace": res["E_trace"], "level_trace": res["level_trace"],
                         "occupation": obs["occupation"], "occupation_err": obs["occupation_err"],
                         "occupation_exact": ex["occupation"], "occupation_gc": gc["occupation"]},
                        {"run": m, "results": rows})
        print(f"Datos crudos: {path}")
        if args.plot:
            _plot(obs, ex, gc, stat, N, T, seed, FIG_DIR / "cli" / f"{tag}.png")
    return {"observables": obs, "exact": ex, "grand_canonical": gc, "metadata": m}


def _plot(obs, ex, gc, stat, N, T, seed, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import plotting as pl

    pl.apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    eps = obs["levels"]
    keep = ex["occupation"] > 1e-5
    ax.plot(eps[keep], ex["occupation"][keep], color=pl.COLOR[stat], label="canónico exacto")
    ax.plot(eps[keep], gc["occupation"][keep], color=pl.INK2, ls="--", label="gran canónico (" + ("BE" if stat == "bosons" else "FD") + ")")
    vis = keep & (obs["occupation"] > 0)
    ax.errorbar(eps[vis], obs["occupation"][vis], yerr=obs["occupation_err"][vis], fmt="o", color=pl.COLOR[stat],
                ms=4.5, capsize=2, label="simulación")
    if stat == "bosons":
        ax.set_yscale("log")
    ax.set(xlabel=r"$\varepsilon$  [$\hbar\omega$]", ylabel=r"$\langle n(\varepsilon)\rangle$ por estado",
           title=f"{pl.NAME[stat]}, N = {N}, T = {T:.3g}, semilla {seed}")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulación individual (bosones o fermiones) con resumen de errores.")
    parser.add_argument("--statistics", type=lambda s: STAT_ALIASES[s.lower()], help="bosons/fermions")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--T", type=float, default=None, help="Temperatura reducida k_B T/(ħω)")
    parser.add_argument("--T-kelvin", type=_eval_number, default=None, help="Temperatura en K (requiere --omega)")
    parser.add_argument("--omega", type=_eval_number, default=None, help="Frecuencia de la trampa en rad/s")
    parser.add_argument("--q-max", type=int, default=0, help="Corte energético (0 = automático)")
    parser.add_argument("--proposal", choices=("particle", "state"), default="particle")
    parser.add_argument("--steps", type=lambda s: int(float(s)), default=20_000_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init", choices=("ground", "excited"), default="ground")
    parser.add_argument("--save", action="store_true", help="Guardar datos crudos en data/raw/cli/")
    parser.add_argument("--plot", action="store_true", help="Guardar figura en figures/cli/ (implica --save)")
    args = parser.parse_args()
    if args.statistics is None:
        args = interactive_params()
    if args.plot:
        args.save = True
    run(args)


if __name__ == "__main__":
    main()
