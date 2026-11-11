"""Estudio principal: barridos en q_max, T y N (rúbrica A3 de tamaño finito y requisito 4.1.6).

Parte A, convergencia en el corte q_max a N y T fijos: separa el error de
  truncamiento (exacto(q_max) − exacto(q→∞)) del error estadístico
  (simulación − exacto(q_max)).
Parte B, barrido en T para N ∈ {10, 20, 35, 56} y las dos estadísticas, con el
  q_max elegido para que el truncamiento sea < 10⁻⁴ T. Cada punto se compara con
  (i) la solución canónica exacta, que es la validación, y
  (ii) el ensemble gran canónico (Bose–Einstein o Fermi–Dirac con μ fijado por N),
  que es la comparación entre ensembles.
Parte C, escalamiento con N de la diferencia canónico − gran canónico en E/N a
  T/T_ref fija (exponente de tamaño finito).

Salidas: data/raw/qmax_convergence.npz, data/raw/sweep/<estadística>_N<N>_T<T>.npz,
         data/raw/sweep_summary.csv, data/raw/ensemble_scaling.npz
Uso:     python scripts/run_sweeps.py [--fast] [--resume] [--only {qmax,temperature,scaling}]
"""

from __future__ import annotations

import time

import _common  # noqa: F401
import numpy as np

from src import analytics as an
from src.config import (
    BURN_FRACTION,
    N_SAMPLES,
    N_VALUES,
    PRODUCTION_PROPOSAL,
    Q_TOL,
    T_GRID,
    make_seed,
    output_dirs,
    reference_temperature,
    steps_for,
)
from src.datastore import load_raw, save_raw, write_table_csv
from src.metropolis import run_mcmc_chain
from src.observables import compute_observables
from src.stats import discrepancy_in_sigmas

STAT_CODE = {"bosons": 0, "fermions": 1}
QMAX_CASES = {"bosons": (20, 2.5), "fermions": (20, 1.5)}  # T ≈ T_c y T = T_F / 2
QMAX_VALUES = (4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)


def run_point(N: int, T: float, q_max: int, stat: str, n_steps: int, seed: int) -> tuple[dict, dict]:
    res = run_mcmc_chain(N, 1.0 / T, q_max, stat, n_steps, int(BURN_FRACTION * n_steps),
                         n_steps // N_SAMPLES, seed, "ground", PRODUCTION_PROPOSAL)
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
                summary = {
                    "estadistica": stat, "N": N, "T": T, "T_ref": T_ref, "T_escalada": T / T_ref,
                    "q_max": q, "M": m["M"], "semilla": seed, "pasos": steps,
                    "intervalo": m["sample_interval"], "aceptacion": m["acceptance_rate"],
                    "E_N": obs["energy_per_particle"], "E_N_err": obs["energy_per_particle_err"],
                    "E_N_err_bloques": obs["energy_per_particle_err_blocking"],
                    "tau_E": obs["tau_int_energy"],
                    "f0": obs["condensate_fraction"], "f0_err": obs["condensate_fraction_err"],
                    "f0_err_bloques": obs["condensate_fraction_err_blocking"], "tau_n0": obs["tau_int_n0"],
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
                      f"  ({m['elapsed_seconds']:.1f}s, total {time.perf_counter() - t_start:.0f}s)", flush=True)
    write_table_csv(raw_dir / "sweep_summary.csv", rows)
    zE = np.array([r["z_E"] for r in rows])
    zf = np.array([r["z_f0"] for r in rows])
    print(f"\n{len(rows)} corridas. z(E/N): medio {zE.mean():.2f}, máx {zE.max():.2f}, "
          f"z<2 en {np.mean(zE < 2):.0%};  z(f0): medio {zf.mean():.2f}, máx {zf.max():.2f}, z<2 en {np.mean(zf < 2):.0%}")


# ---------------------------------------------------------------------------
def _power_law_fit(N, y, dy):
    """Ajuste ponderado log|y| = a + b log N. Devuelve (b, σ_b)."""
    x, ly = np.log(N), np.log(np.abs(y))
    w = np.abs(y) / dy  # 1/σ(log|y|)
    coef, cov = np.polyfit(x, ly, 1, w=w, cov="unscaled")
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def ensemble_scaling(raw_dir, fast: bool) -> None:
    """Parte C: (E_can − E_gc)/E_gc a T/T_ref fija y varios N, para estimar el exponente."""
    print("Parte C: escalamiento de la diferencia canónico − gran canónico")
    arrays, meta = {}, {}
    for stat in ("bosons", "fermions"):
        t = SCALING_T[stat]
        d, dd, d_ex, Ts, qs, seeds = [], [], [], [], [], []
        for k, N in enumerate(SCALING_N):
            T = t * reference_temperature(N, stat)
            q = an.choose_q_max(N, T, stat, Q_TOL)
            seed = make_seed(40, STAT_CODE[stat], N)
            res, obs = run_point(N, T, q, stat, SCALING_STEPS_FACTOR[stat] * steps_for(N, T, stat, fast), seed)
            ex = an.exact_canonical(N, q, 1.0 / T, stat)["energy_per_particle"]
            gc = an.grand_canonical_observables(N, q, 1.0 / T, stat)["energy_per_particle"]
            d.append((obs["energy_per_particle"] - gc) / gc)
            dd.append(obs["energy_per_particle_err"] / gc)
            d_ex.append((ex - gc) / gc)
            Ts.append(T); qs.append(q); seeds.append(seed)
            print(f"  {stat:8s} N={N:3d} T={T:.3f} q={q:3d}  sim: dE/E={d[-1]:+.5f} ± {dd[-1]:.5f}"
                  f"  exacto: {d_ex[-1]:+.5f}  z={discrepancy_in_sigmas(obs['energy_per_particle'], obs['energy_per_particle_err'], ex):.2f}")
        b, db = _power_law_fit(np.array(SCALING_N), np.array(d), np.array(dd))
        # Comparación justa: el mismo ajuste (mismos N y mismos pesos) aplicado a los valores exactos.
        # Con N pequeño la pendiente local todavía cambia (régimen preasintótico).
        b_same, _ = _power_law_fit(np.array(SCALING_N), np.array(d_ex), np.array(dd))
        N_dense = np.array(SCALING_DENSE[stat])
        delta_dense = []
        for N in N_dense:
            T = t * reference_temperature(int(N), stat)
            q = an.reference_cutoff(int(N), T, stat)
            ex = an.exact_canonical(int(N), q, 1.0 / T, stat)["energy_per_particle"]
            gc = an.grand_canonical_observables(int(N), q, 1.0 / T, stat)["energy_per_particle"]
            delta_dense.append((ex - gc) / gc)
        delta_dense = np.array(delta_dense)
        b_ex, _ = _power_law_fit(N_dense, delta_dense, 1e-3 * np.abs(delta_dense))
        print(f"  {stat}: exponente simulación b = {b:.3f} ± {db:.3f} (z = {abs(b - b_same) / db:.2f});"
              f"  exacto con el mismo ajuste b = {b_same:.3f};  exacto N={N_dense[0]}–{N_dense[-1]} b = {b_ex:.3f}")
        arrays.update({
            f"{stat}_N": np.array(SCALING_N), f"{stat}_T": np.array(Ts), f"{stat}_q": np.array(qs),
            f"{stat}_delta": np.array(d), f"{stat}_delta_err": np.array(dd), f"{stat}_delta_exact": np.array(d_ex),
            f"{stat}_slope": np.array(b), f"{stat}_slope_err": np.array(db), f"{stat}_slope_exact": np.array(b_ex),
            f"{stat}_slope_exact_same": np.array(b_same),
            f"{stat}_N_dense": N_dense, f"{stat}_delta_dense": delta_dense, f"{stat}_t": np.array(t),
        })
        meta[stat] = {"t": t, "seeds": seeds}
    save_raw(raw_dir / "ensemble_scaling.npz", arrays, meta)


SCALING_N = (10, 20, 35, 56, 84)
SCALING_T = {"bosons": 0.5, "fermions": 0.25}  # T/T_c y T/T_F
# Señal pequeña (~1/N): corridas más largas que en el barrido para resolverla.
SCALING_STEPS_FACTOR = {"bosons": 4, "fermions": 2}
SCALING_DENSE = {"bosons": (10, 14, 20, 28, 35, 45, 56, 70, 84, 100, 120),
                 "fermions": (10, 20, 35, 56, 84, 120, 165, 220)}


def main(fast: bool = False, resume: bool = False, only: str | None = None) -> None:
    raw_dir, _ = output_dirs(fast)
    if only in (None, "qmax"):
        qmax_convergence(raw_dir, fast)
    if only in (None, "temperature"):
        temperature_sweep(raw_dir, fast, resume)
    if only in (None, "scaling"):
        ensemble_scaling(raw_dir, fast)


if __name__ == "__main__":
    parser = _common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--resume", action="store_true", help="No repetir corridas cuyo archivo ya existe.")
    parser.add_argument("--only", choices=("qmax", "temperature", "scaling"), default=None)
    args = parser.parse_args()
    main(fast=args.fast, resume=args.resume, only=args.only)
