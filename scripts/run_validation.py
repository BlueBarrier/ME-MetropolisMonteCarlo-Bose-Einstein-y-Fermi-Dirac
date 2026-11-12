"""Validación cuantitativa contra resultados exactos (algoritmo de Metropolis–Hastings).

Tres casos, cada uno comparado en unidades de σ (z = |sim − exacto|/σ):
  V1. N = 1 contra ⟨E⟩ = 3/(e^β − 1) (trampa infinita). Con una sola partícula bosones
      y fermiones coinciden.
  V2. N = 3 (bosones y fermiones, q_max = 6) contra la enumeración de todas las configuraciones.
  V3. N = 10 (q_max = 12) a T = 0.75, 1.5 y 3 contra la solución canónica exacta.

(La coincidencia entre la función generatriz y la enumeración se comprueba en tests/test_core.py.)

Salidas: data/raw/validation.npz, data/raw/validation_table.csv
Uso:     python scripts/run_validation.py [--fast]
"""

from __future__ import annotations

import _common
import numpy as np

from src import analytics as an
from src.config import BURN_FRACTION, make_seed, output_dirs
from src.datastore import save_raw, write_table_csv
from src.metropolis import run_mcmc_chain
from src.model import build_state_catalog
from src.observables import compute_observables, relative_error_percent
from src.stats import discrepancy_in_sigmas

N_SAMPLES = 20_000


def simulate(N, T, q_max, statistics, n_steps, seed):
    """Corre una cadena (descartando n_burn = BURN_FRACTION de los pasos) y devuelve sus observables."""
    n_steps -= n_steps % N_SAMPLES
    res = run_mcmc_chain(N, 1.0 / T, q_max, statistics, n_steps, int(BURN_FRACTION * n_steps),
                         n_steps // N_SAMPLES, seed, "ground")
    return res["metadata"], compute_observables(res["E_trace"], res["level_trace"], N)


def row(case, meta, observable, sim, err, exact):
    """Una fila de la tabla de validación."""
    return {
        "caso": case, "estadistica": meta["statistics"], "N": meta["N"], "T": meta["T"],
        "q_max": meta["q_max"], "semilla": meta["seed"], "observable": observable,
        "simulacion": sim, "sigma": err, "exacto": exact,
        "z": discrepancy_in_sigmas(sim, err, exact), "error_%": relative_error_percent(sim, exact),
    }


def main(fast: bool = False) -> None:
    raw_dir, _ = output_dirs(fast)
    scale = 0.05 if fast else 1.0
    rows, arrays = [], {}

    # V1: una partícula contra la fórmula analítica
    print("V1: N = 1 contra <E> = 3/(e^beta - 1)")
    temps = (0.5, 1.0, 2.0, 3.0, 5.0)
    E, dE = [], []
    for k, T in enumerate(temps):
        q = an.choose_q_max(1, T, "bosons")
        meta, obs = simulate(1, T, q, "bosons", int(20_000_000 * scale), make_seed(1, k))
        exact = float(an.exact_single_particle_energy(1.0 / T))
        rows.append(row("V1", meta, "E/N", obs["energy_per_particle"], obs["energy_per_particle_err"], exact))
        E.append(obs["energy_per_particle"])
        dE.append(obs["energy_per_particle_err"])
        print(f"  T={T:3.1f}  E = {E[-1]:.4f} ± {dE[-1]:.4f}  exacto {exact:.4f}  z = {rows[-1]['z']:.2f}")
    arrays["n1_T"], arrays["n1_E"], arrays["n1_err"] = map(np.array, (temps, E, dE))
    arrays["n1_curve_T"] = np.linspace(0.2, 5.2, 300)
    arrays["n1_curve_E"] = an.exact_single_particle_energy(1.0 / arrays["n1_curve_T"])

    # V2: tres partículas contra la enumeración directa
    print("V2: N = 3, q_max = 6, contra la enumeración de todas las configuraciones")
    q3 = 6
    energies, _ = build_state_catalog(q3)
    for s_code, stat in enumerate(("bosons", "fermions")):
        for k, T in enumerate((0.5, 1.0, 2.0)):
            meta, obs = simulate(3, T, q3, stat, int(10_000_000 * scale), make_seed(2, s_code, k))
            ex = an.exact_canonical_enumeration(3, energies, 1.0 / T, stat)
            occ_exact = np.array([ex["occupation_per_state"][energies == l].mean() for l in range(q3 + 1)])
            rows.append(row("V2", meta, "E/N", obs["energy_per_particle"], obs["energy_per_particle_err"], ex["mean_energy"] / 3))
            rows.append(row("V2", meta, "f0", obs["condensate_fraction"], obs["condensate_fraction_err"], occ_exact[0] / 3))
            print(f"  {stat:8s} T={T:3.1f}  E/N = {obs['energy_per_particle']:.4f} ± {obs['energy_per_particle_err']:.4f}"
                  f"  exacto {ex['mean_energy'] / 3:.4f}  z = {rows[-2]['z']:.2f}")
            if T == 1.0:  # ocupaciones por nivel para la figura
                arrays[f"n3_{stat}_occ"] = obs["occupation"]
                arrays[f"n3_{stat}_occ_err"] = obs["occupation_err"]
                arrays[f"n3_{stat}_occ_exact"] = occ_exact
    arrays["n3_levels"] = np.arange(q3 + 1)

    # V3: diez partículas contra la solución canónica exacta (función generatriz)
    print("V3: N = 10, q_max = 12, contra la solución canónica exacta")
    for s_code, stat in enumerate(("bosons", "fermions")):
        for k, T in enumerate((0.75, 1.5, 3.0)):
            ex = an.exact_canonical(10, 12, 1.0 / T, stat)
            meta, obs = simulate(10, T, 12, stat, int(40_000_000 * scale), make_seed(3, s_code, k))
            rows.append(row("V3", meta, "E/N", obs["energy_per_particle"], obs["energy_per_particle_err"], ex["energy_per_particle"]))
            rows.append(row("V3", meta, "f0", obs["condensate_fraction"], obs["condensate_fraction_err"], ex["condensate_fraction"]))
            print(f"  {stat:8s} T={T:4.2f}  E/N = {obs['energy_per_particle']:.4f} ± {obs['energy_per_particle_err']:.4f}"
                  f"  exacto {ex['energy_per_particle']:.4f}  z = {rows[-2]['z']:.2f}")

    save_raw(raw_dir / "validation.npz", arrays, {"description": "Validación contra resultados exactos",
                                                    "algorithm": "metropolis-hastings"})
    write_table_csv(raw_dir / "validation_table.csv", rows)
    zs = np.array([r["z"] for r in rows])
    print(f"\n{len(zs)} comparaciones: z medio = {zs.mean():.2f}, z máximo = {zs.max():.2f}, "
          f"z < 2 en {np.mean(zs < 2):.0%}.  Datos en {raw_dir}")


if __name__ == "__main__":
    args = _common.base_parser(__doc__.splitlines()[0]).parse_args()
    main(fast=args.fast)
