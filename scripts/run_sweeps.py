"""Estudio principal: barridos en q_max, T y N, y ocupaciones por nivel.

Parte A, convergencia en el corte q_max a N y T fijos: separa el error de
  truncamiento (exacto(q_max) − exacto(q→∞)) del error estadístico
  (simulación − exacto(q_max)).
Parte B, barrido en T para N ∈ {10, 20, 35, 56} y las dos estadísticas, con el
  q_max elegido para que el truncamiento sea < 10⁻⁴ T. Cada punto se compara con la
  solución canónica exacta (E/N, f_0 y el calor específico C/N, este último medido por
  fluctuaciones de la energía). En fermiones se ajusta además Fermi–Dirac a ⟨n(ε)⟩
  simulado para obtener el potencial químico μ(T).
Parte C, ocupación media por estado ⟨n(ε)⟩ con N = 45 a T = 0.5, 2 y 4 para las dos
  estadísticas, junto con la solución canónica exacta.

Salidas: data/raw/qmax_convergence.npz, data/raw/sweep/<estadística>_N<N>_T<T>.npz,
         data/raw/sweep_summary.csv, data/raw/occupations.npz
Uso:     python scripts/run_sweeps.py [--fast] [--resume] [--only {qmax,temperature,occupations}]
"""

from __future__ import annotations

import time

import _common
import numpy as np

from src import analytics as an
from src.config import (
    BURN_FRACTION,
    N_SAMPLES,
    N_VALUES,
    Q_TOL,
    T_GRID,
    make_seed,
    output_dirs,
    reference_temperature,
    steps_for,
)
from src.datastore import load_raw, save_raw, write_table_csv
from src.metropolis import run_mcmc_chain
from src.observables import compute_observables, fit_chemical_potential, heat_capacity
from src.stats import discrepancy_in_sigmas

STAT_CODE = {"bosons": 0, "fermions": 1}
QMAX_CASES = {"bosons": (20, 2.5), "fermions": (20, 1.5)}  # T ≈ T_c y T = T_F / 2
QMAX_VALUES = (4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)


def run_point(N: int, T: float, q_max: int, stat: str, n_steps: int, seed: int) -> tuple[dict, dict]:
    res = run_mcmc_chain(N, 1.0 / T, q_max, stat, n_steps, int(BURN_FRACTION * n_steps),
                         n_steps // N_SAMPLES, seed, "ground")
    return res, compute_observables(res["E_trace"], res["level_trace"], N)


# ---------------------------------------------------------------------------
def qmax_convergence(raw_dir, fast: bool) -> None:
    print("Parte A: convergencia en q_max")
    arrays, meta = {}, {}
    for stat, (N, T) in QMAX_CASES.items():
        beta = 1.0 / T
        q_ref = an.reference_cutoff(N, T, stat)
        E_ref = an.exact_canonical(N, q_ref, beta, stat)["energy_per_particle"]
        qs = [q for q in QMAX_VALUES if stat == "bosons" or an.number_of_states(q) > N]
        E, dE, E_ex = [], [], []
        for k, q in enumerate(qs):
            steps = steps_for(N, T, stat, fast)
            res, obs = run_point(N, T, q, stat, steps, make_seed(20, STAT_CODE[stat], k))
            ex = an.exact_canonical(N, q, beta, stat)["energy_per_particle"]
            E.append(obs["energy_per_particle"])
            dE.append(obs["energy_per_particle_err"])
            E_ex.append(ex)
            print(f"  {stat:8s} q_max={q:3d}  sim={E[-1]:.4f} ± {dE[-1]:.4f}  exacto(q)={ex:.4f}"
                  f"  truncamiento={ex - E_ref:+.2e}  z={discrepancy_in_sigmas(E[-1], dE[-1], ex):.2f}")
        q_dense = np.arange(2 if stat == "bosons" else min(qs), 61)
        E_dense = [an.exact_canonical(N, int(q), beta, stat)["energy_per_particle"] for q in q_dense]
        arrays.update({
            f"{stat}_q": np.array(qs), f"{stat}_E": np.array(E), f"{stat}_err": np.array(dE),
            f"{stat}_E_exact": np.array(E_ex), f"{stat}_E_ref": np.array(E_ref),
            f"{stat}_q_dense": q_dense, f"{stat}_E_dense": np.array(E_dense),
            f"{stat}_q_chosen": np.array(an.choose_q_max(N, T, stat, Q_TOL)),
        })
        meta[stat] = {"N": N, "T": T, "q_ref": q_ref}
    save_raw(raw_dir / "qmax_convergence.npz", arrays, meta)


# ---------------------------------------------------------------------------
def temperature_sweep(raw_dir, fast: bool, resume: bool) -> None:
    print("Parte B: barrido en T y N")
    sweep_dir = raw_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    for stat in ("bosons", "fermions"):
        for iN, N in enumerate(N_VALUES):
            T_ref = reference_temperature(N, stat)
            for iT, T in enumerate(T_GRID):
                path = sweep_dir / f"{stat}_N{N}_T{T:.2f}.npz"
                if resume and path.exists():
                    _, meta = load_raw(path)
                    rows.append(meta["summary"])
                    continue
                beta = 1.0 / T
                q = an.choose_q_max(N, T, stat, Q_TOL)
                q_ref = an.reference_cutoff(N, T, stat)
                steps = steps_for(N, T, stat, fast)
                seed = make_seed(30, STAT_CODE[stat], N, iT)
                res, obs = run_point(N, T, q, stat, steps, seed)
                ex = an.exact_canonical(N, q, beta, stat)
                ex_inf = an.exact_canonical(N, q_ref, beta, stat)
                gc = an.grand_canonical_observables(N, q, beta, stat)
                m = res["metadata"]
                C, C_err = heat_capacity(res["E_trace"], T, N)
                C_ex = an.heat_capacity_canonical(N, q, T, stat)
                if stat == "fermions":
                    mu_fit, mu_fit_err = fit_chemical_potential(obs["levels"], obs["occupation"],
                                                                obs["occupation_err"], T)
                else:
                    mu_fit, mu_fit_err = float("nan"), float("nan")
                summary = {
                    "estadistica": stat, "N": N, "T": T, "T_ref": T_ref, "T_escalada": T / T_ref,
                    "q_max": q, "M": m["M"], "semilla": seed, "pasos": steps,
                    "intervalo": m["sample_interval"], "aceptacion": m["acceptance_rate"],
                    "E_N": obs["energy_per_particle"], "E_N_err": obs["energy_per_particle_err"],
                    "tau_E": obs["tau_int_energy"],
                    "f0": obs["condensate_fraction"], "f0_err": obs["condensate_fraction_err"],
                    "tau_n0": obs["tau_int_n0"],
                    "E_N_exacto": ex["energy_per_particle"], "f0_exacto": ex["condensate_fraction"],
                    "E_N_exacto_qinf": ex_inf["energy_per_particle"],
                    "f0_exacto_qinf": ex_inf["condensate_fraction"],
                    "E_N_gc": gc["energy_per_particle"], "f0_gc": gc["condensate_fraction"], "mu_gc": gc["mu"],
                    "z_E": discrepancy_in_sigmas(obs["energy_per_particle"], obs["energy_per_particle_err"],
                                                 ex["energy_per_particle"]),
                    "z_f0": discrepancy_in_sigmas(obs["condensate_fraction"], obs["condensate_fraction_err"],
                                                  ex["condensate_fraction"]),
                    "z_E_gc": discrepancy_in_sigmas(obs["energy_per_particle"], obs["energy_per_particle_err"],
                                                    gc["energy_per_particle"]),
                    "z_f0_gc": discrepancy_in_sigmas(obs["condensate_fraction"], obs["condensate_fraction_err"],
                                                     gc["condensate_fraction"]),
                    "C_N": C, "C_N_err": C_err, "C_N_exacto": C_ex,
                    "z_C": discrepancy_in_sigmas(C, C_err, C_ex),
                    "mu_ajuste": mu_fit, "mu_ajuste_err": mu_fit_err,
                    "segundos": m["elapsed_seconds"],
                }
                arrays = {
                    "E_trace": res["E_trace"].astype(np.int32),
                    "n0_trace": res["level_trace"][:, 0].astype(np.int32),
                    "levels": obs["levels"],
                    "occupation": obs["occupation"],
                    "occupation_err": obs["occupation_err"],
                    "occupation_exact": ex["occupation"],
                    "occupation_gc": gc["occupation"],
                }
                save_raw(path, arrays, {"run": m, "summary": summary})
                rows.append(summary)
                print(f"  {stat:8s} N={N:2d} T={T:4.2f} (T/Tref={T / T_ref:4.2f}) q={q:3d}"
                      f"  E/N={summary['E_N']:.4f}±{summary['E_N_err']:.4f} z={summary['z_E']:.2f}"
                      f"  f0={summary['f0']:.4f}±{summary['f0_err']:.4f} z={summary['z_f0']:.2f}"
                  f"  C/N={C:.3f}±{C_err:.3f} (exacto {C_ex:.3f})"
                      f"  ({m['elapsed_seconds']:.1f}s, total {time.perf_counter() - t_start:.0f}s)", flush=True)
    write_table_csv(raw_dir / "sweep_summary.csv", rows)
    zE = np.array([r["z_E"] for r in rows])
    zf = np.array([r["z_f0"] for r in rows])
    print(f"\n{len(rows)} corridas. z(E/N): medio {zE.mean():.2f}, máx {zE.max():.2f}, "
          f"z<2 en {np.mean(zE < 2):.0%};  z(f0): medio {zf.mean():.2f}, máx {zf.max():.2f}, z<2 en {np.mean(zf < 2):.0%}")


# ---------------------------------------------------------------------------
OCC_N = 45
OCC_TEMPS = (0.5, 2.0, 4.0)


def occupations(raw_dir, fast: bool) -> None:
    """Parte C: ⟨n(ε)⟩ por estado con N = 45 (simulación, canónico exacto y gran canónico)."""
    print(f"Parte C: ocupaciones con N = {OCC_N}")
    arrays = {"N": OCC_N, "temperatures": OCC_TEMPS}
    meta = {"N": OCC_N, "runs": {}}
    for stat in ("bosons", "fermions"):
        arrays[f"{stat}_T_ref"] = reference_temperature(OCC_N, stat)
        for k, T in enumerate(OCC_TEMPS):
            q = an.choose_q_max(OCC_N, T, stat, Q_TOL)
            seed = make_seed(60, STAT_CODE[stat], k)
            res, obs = run_point(OCC_N, T, q, stat, steps_for(OCC_N, T, stat, fast), seed)
            ex = an.exact_canonical(OCC_N, q, 1.0 / T, stat)
            gc = an.grand_canonical_observables(OCC_N, q, 1.0 / T, stat)
            key = f"{stat}_T{T:.2f}"
            arrays.update({f"{key}_levels": obs["levels"], f"{key}_sim": obs["occupation"],
                           f"{key}_sim_err": obs["occupation_err"], f"{key}_exact": ex["occupation"],
                           f"{key}_gc": gc["occupation"]})
            meta["runs"][key] = {"q_max": q, "seed": seed, "steps": res["metadata"]["n_steps"],
                                 "E_N": obs["energy_per_particle"], "E_N_exacto": ex["energy_per_particle"]}
            print(f"  {stat:8s} T={T:4.2f} q={q:3d}  E/N={obs['energy_per_particle']:.4f}"
                  f"±{obs['energy_per_particle_err']:.4f} (exacto {ex['energy_per_particle']:.4f}), semilla {seed}")
    save_raw(raw_dir / "occupations.npz", arrays, meta)


def main(fast: bool = False, resume: bool = False, only: str | None = None) -> None:
    raw_dir, _ = output_dirs(fast)
    if only in (None, "qmax"):
        qmax_convergence(raw_dir, fast)
    if only in (None, "temperature"):
        temperature_sweep(raw_dir, fast, resume)
    if only in (None, "occupations"):
        occupations(raw_dir, fast)


if __name__ == "__main__":
    parser = _common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--resume", action="store_true", help="No repetir corridas cuyo archivo ya existe.")
    parser.add_argument("--only", choices=("qmax", "temperature", "occupations"), default=None)
    args = parser.parse_args()
    main(fast=args.fast, resume=args.resume, only=args.only)
