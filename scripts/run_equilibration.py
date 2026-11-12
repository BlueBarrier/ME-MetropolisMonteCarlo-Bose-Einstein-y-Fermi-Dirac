"""Equilibración y autocorrelación.

Para cada estadística se corren dos cadenas con los mismos parámetros y distinta
condición inicial:
  - "ground"  : estado base (fría, E por debajo del equilibrio),
  - "excited" : configuración aleatoria a T = ∞ (caliente, E muy por encima).
Las dos deben converger al mismo valor estacionario (la solución canónica exacta).
El primer cruce de sus promedios móviles acota el tiempo de equilibración y se
compara con la termalización n_burn que usan los barridos (10 % de la corrida).
Después se calcula C(t) y τ_int sobre la producción.

Al final se verifica que la misma n_burn que usan los barridos (BURN_FRACTION de los pasos)
alcance en TODOS los puntos (estadística, N, T) del barrido: las cadenas fría y caliente deben
cruzarse antes de n_burn.

Salidas: data/raw/equilibration.npz, data/raw/burn_check.csv
Uso:     python scripts/run_equilibration.py [--fast] [--only-check]
"""

from __future__ import annotations

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
    steps_for,
)
from src.datastore import save_raw, write_table_csv
from src.metropolis import run_mcmc_chain
from src.stats import (
    autocorrelation_function,
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
            res = run_mcmc_chain(N, 1.0 / T, q, stat, n_steps, n_burn, interval, seed, mode, burn_in_interval=burn_interval)
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

        # Autocorrelación sobre la producción de la cadena fría.
        prod = arrays[f"{stat}_ground_prod_E"].astype(float)
        acf = autocorrelation_function(prod, max_lag=min(prod.size // 4, 2000))
        mean, tau, err = statistical_uncertainty(prod)
        _, tau_fz, err_fz = statistical_uncertainty(prod, method="first_zero")
        naive = float(prod.std(ddof=1) / np.sqrt(prod.size))
        print(f"  tau_int = {tau:.2f} muestras (Sokal), {tau_fz:.2f} (primer cero);"
              f" error ingenuo {naive:.2e}, con tau_int {err:.2e}")

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
            f"{stat}_sample_interval": np.array(interval),
        })
        meta[f"{stat}_params"] = {"N": N, "T": T, "q_max": q}

    save_raw(raw_dir / "equilibration.npz", arrays, meta)
    print(f"Datos guardados en {raw_dir / 'equilibration.npz'}")
    verify_burn_in(raw_dir, fast)


def verify_burn_in(raw_dir, fast: bool = False) -> list[dict]:
    """Comprueba, para CADA punto del barrido (estadística, N, T), que la termalización usada
    para la estadística (n_burn = BURN_FRACTION · pasos, la misma de run_sweeps.py) basta.

    Se corren dos cadenas sólo durante n_burn pasos, una desde el estado base y otra desde
    T = ∞, con el mismo q_max y número de pasos que el barrido. El punto pasa si sus
    promedios móviles se cruzan antes de n_burn. Salida: data/raw/burn_check.csv.
    """
    print("\nVerificación de n_burn en todos los puntos del barrido")
    rows = []
    for s_code, stat in enumerate(("bosons", "fermions")):
        for N in N_VALUES:
            for iT, T in enumerate(T_GRID):
                q = an.choose_q_max(N, T, stat, Q_TOL)
                n_steps = steps_for(N, T, stat, fast)
                n_burn = int(BURN_FRACTION * n_steps)
                interval = max(1, n_burn // BURN_RECORD_POINTS)
                traces = []
                for m_code, mode in enumerate(("ground", "excited")):
                    res = run_mcmc_chain(N, 1.0 / T, q, stat, N_SAMPLES, n_burn, N_SAMPLES,
                                         make_seed(70, s_code, N, iT, m_code), mode,
                                         burn_in_interval=interval)
                    traces.append(res["burn_E_trace"] / N)
                cross = estimate_burn_in(traces[0], traces[1], window=25)
                cross_steps = None if cross is None else int((cross + 1) * interval)
                ok = cross_steps is not None and cross_steps < n_burn
                rows.append({"estadistica": stat, "N": N, "T": T, "q_max": q, "n_burn": n_burn,
                             "cruce_pasos": -1 if cross_steps is None else cross_steps,
                             "cruce/n_burn": float("nan") if cross_steps is None else cross_steps / n_burn,
                             "suficiente": ok})
                flag = "ok" if ok else "INSUFICIENTE"
                print(f"  {stat:8s} N={N:2d} T={T:4.2f}  n_burn={n_burn:.1e}  cruce={rows[-1]['cruce_pasos']:.1e}"
                      f"  ({rows[-1]['cruce/n_burn']:.3f} de n_burn)  {flag}", flush=True)
    write_table_csv(raw_dir / "burn_check.csv", rows)
    bad = [r for r in rows if not r["suficiente"]]
    worst = max(r["cruce/n_burn"] for r in rows if r["suficiente"]) if len(bad) < len(rows) else float("nan")
    print(f"{len(rows) - len(bad)}/{len(rows)} puntos con n_burn suficiente; peor caso: cruce a {worst:.1%} de n_burn."
          + (f"  Revisar: {[(r['estadistica'], r['N'], r['T']) for r in bad]}" if bad else ""))
    return rows


if __name__ == "__main__":
    parser = _common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--only-check", action="store_true", help="Sólo verificar n_burn en todo el barrido")
    args = parser.parse_args()
    if args.only_check:
        verify_burn_in(output_dirs(args.fast)[0], args.fast)
    else:
        main(fast=args.fast)
