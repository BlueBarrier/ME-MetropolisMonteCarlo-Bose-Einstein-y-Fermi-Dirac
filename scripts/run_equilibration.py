"""Equilibración y autocorrelación (rúbrica A3).

Para cada estadística se corren dos cadenas con los mismos parámetros y distinta
condición inicial:
  - "ground"  : estado base (fría, E por debajo del equilibrio),
  - "excited" : configuración aleatoria a T = ∞ (caliente, E muy por encima).
Las dos deben converger al mismo valor estacionario (la solución canónica exacta).
El primer cruce de sus promedios móviles acota el tiempo de equilibración y se
compara con la termalización n_burn que usan los barridos (10 % de la corrida).
Después se calcula C(t), τ_int y el análisis de bloques sobre la producción.

Salidas: data/raw/equilibration.npz
Uso:     python scripts/run_equilibration.py [--fast]
"""

from __future__ import annotations

import _common  # noqa: F401
import numpy as np

from src import analytics as an
from src.config import BURN_FRACTION, N_SAMPLES, PRODUCTION_PROPOSAL, make_seed, output_dirs, steps_for
from src.datastore import save_raw
from src.metropolis import run_mcmc_chain
from src.stats import (
    autocorrelation_function,
    blocking_analysis,
    discrepancy_in_sigmas,
    estimate_burn_in,
    statistical_uncertainty,
)

CASES = {"bosons": (35, 2.0), "fermions": (35, 1.0)}  # (N, T): T/T_c ≈ 0.65 y T/T_F = 0.25
BURN_RECORD_POINTS = 5000


def main(fast: bool = False) -> None:
    raw_dir, _ = output_dirs(fast)
    arrays, meta = {}, {"cases": {}}
    for s_code, (stat, (N, T)) in enumerate(CASES.items()):
        q = an.choose_q_max(N, T, stat)
        n_steps = steps_for(N, T, stat, fast)
        n_burn = int(BURN_FRACTION * n_steps)
        burn_interval = max(1, n_burn // BURN_RECORD_POINTS)
        interval = n_steps // N_SAMPLES
        exact = an.exact_canonical(N, q, 1.0 / T, stat)["energy_per_particle"]
        print(f"{stat}: N={N}, T={T}, q_max={q}, pasos={n_steps:.2e}, n_burn={n_burn:.2e}, E/N exacto={exact:.4f}")
        chains = {}
        for m_code, mode in enumerate(("ground", "excited")):
            seed = make_seed(10, s_code, m_code)
            res = run_mcmc_chain(N, 1.0 / T, q, stat, n_steps, n_burn, interval, seed, mode,
                                 PRODUCTION_PROPOSAL, burn_in_interval=burn_interval)
            chains[mode] = res
            burn = res["burn_E_trace"] / N
            prod = res["E_trace"] / N
            mean, tau, err = statistical_uncertainty(prod)
            arrays[f"{stat}_{mode}_burn_E"] = burn.astype(np.float32)
            arrays[f"{stat}_{mode}_prod_E"] = prod.astype(np.float32)
            arrays[f"{stat}_{mode}_E0"] = np.array(res["metadata"]["E_initial"] / N)
            meta["cases"][f"{stat}_{mode}"] = res["metadata"]
            print(f"  {mode:8s} E0/N={res['metadata']['E_initial'] / N:7.3f}  <E>/N={mean:.4f} ± {err:.4f}"
                  f"  z vs exacto={discrepancy_in_sigmas(mean, err, exact):.2f}  semilla={seed}")

        # Cruce de las dos cadenas durante la termalización.
        cross = estimate_burn_in(arrays[f"{stat}_ground_burn_E"], arrays[f"{stat}_excited_burn_E"], window=25)
        cross_steps = None if cross is None else int((cross + 1) * burn_interval)
        # Compatibilidad entre las dos cadenas ya en producción.
        mg, _, eg = statistical_uncertainty(arrays[f"{stat}_ground_prod_E"])
        me, _, ee = statistical_uncertainty(arrays[f"{stat}_excited_prod_E"])
        z_chains = discrepancy_in_sigmas(mg - me, float(np.hypot(eg, ee)), 0.0)
        print(f"  cruce de cadenas a {cross_steps} pasos (n_burn = {n_burn});"
              f" diferencia entre cadenas en producción: z = {z_chains:.2f}")

        # Autocorrelación y bloques sobre la producción de la cadena fría.
        prod = arrays[f"{stat}_ground_prod_E"].astype(float)
        acf = autocorrelation_function(prod, max_lag=min(prod.size // 4, 2000))
        mean, tau, err = statistical_uncertainty(prod)
        _, tau_fz, err_fz = statistical_uncertainty(prod, method="first_zero")
        blk = blocking_analysis(prod)
        naive = float(prod.std(ddof=1) / np.sqrt(prod.size))
        print(f"  tau_int = {tau:.2f} muestras (Sokal), {tau_fz:.2f} (primer cero);"
              f" error ingenuo {naive:.2e}, con tau_int {err:.2e}, bloques {blk['plateau_error']:.2e}")

        arrays.update({
            f"{stat}_burn_steps": (np.arange(1, arrays[f"{stat}_ground_burn_E"].size + 1) * burn_interval),
            f"{stat}_prod_steps": n_burn + np.arange(1, prod.size + 1) * interval,
            f"{stat}_exact": np.array(exact),
            f"{stat}_n_burn": np.array(n_burn),
            f"{stat}_cross_steps": np.array(-1 if cross_steps is None else cross_steps),
            f"{stat}_z_chains": np.array(z_chains),
            f"{stat}_acf": acf,
            f"{stat}_tau": np.array(tau),
            f"{stat}_tau_first_zero": np.array(tau_fz),
            f"{stat}_err_tau": np.array(err),
            f"{stat}_err_naive": np.array(naive),
            f"{stat}_block_sizes": blk["block_sizes"],
            f"{stat}_block_errors": blk["errors"],
            f"{stat}_block_error_err": blk["error_of_errors"],
            f"{stat}_sample_interval": np.array(interval),
        })
        meta[f"{stat}_params"] = {"N": N, "T": T, "q_max": q}

    save_raw(raw_dir / "equilibration.npz", arrays, meta)
    print(f"Datos guardados en {raw_dir / 'equilibration.npz'}")


if __name__ == "__main__":
    args = _common.base_parser(__doc__.splitlines()[0]).parse_args()
    main(fast=args.fast)
