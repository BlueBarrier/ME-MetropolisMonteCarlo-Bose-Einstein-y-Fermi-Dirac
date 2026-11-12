"""Validación cuantitativa contra resultados exactos.

Casos:
  V1. N = 1 (las dos estadísticas coinciden) contra ⟨E⟩ = 3/(e^β − 1) en la trampa infinita,
      con los dos núcleos. El corte se elige para que el truncamiento sea despreciable.
  V2. N = 3 (bosones y fermiones, q_max = 6) contra enumeración directa de todas las
      configuraciones. Se usa el núcleo simétrico aprobado.
  V3. N = 10 (q_max = 12): núcleo simétrico vs núcleo de Hastings vs solución canónica exacta.
  V4. Función generatriz vs enumeración directa (N ≤ 4): acuerdo a precisión de máquina.

Salidas: data/raw/validation.npz, data/raw/validation_table.csv,
         data/raw/validation_exact_methods.csv
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


def _run(N, T, q_max, statistics, proposal, n_steps, seed):
    n_steps -= n_steps % N_SAMPLES
    res = run_mcmc_chain(
        N, 1.0 / T, q_max, statistics, n_steps, int(BURN_FRACTION * n_steps), n_steps // N_SAMPLES, seed, "ground", proposal
    )
    obs = compute_observables(res["E_trace"], res["level_trace"], N)
    return res, obs


def _row(case, res, obs, observable, sim, err, exact, tau):
    m = res["metadata"]
    return {
        "caso": case,
        "estadistica": m["statistics"],
        "nucleo": m["proposal"],
        "N": m["N"],
        "T": m["T"],
        "q_max": m["q_max"],
        "pasos": m["n_steps"],
        "semilla": m["seed"],
        "observable": observable,
        "simulacion": sim,
        "sigma": err,
        "exacto": exact,
        "z": discrepancy_in_sigmas(sim, err, exact),
        "error_%": relative_error_percent(sim, exact),
        "tau_int": tau,
        "aceptacion": m["acceptance_rate"],
    }


def main(fast: bool = False) -> None:
    raw_dir, _ = output_dirs(fast)
    scale = 0.05 if fast else 1.0
    rows, arrays = [], {}

    # ---------------- V1: N = 1 contra la fórmula analítica -----------------
    print("V1: N = 1 contra <E> = 3/(e^beta - 1)")
    for proposal, temps, steps in (
        ("state", (0.5, 0.75, 1.0, 1.5), 200_000_000),
        ("particle", (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0), 20_000_000),
    ):
        E, dE = [], []
        for k, T in enumerate(temps):
            q = an.choose_q_max(1, T, "bosons")
            res, obs = _run(1, T, q, "bosons", proposal, int(steps * scale), make_seed(1, k, 0 if proposal == "state" else 1))
            exact = float(an.exact_single_particle_energy(1.0 / T))
            rows.append(_row("V1 N=1 analitico", res, obs, "E/N", obs["energy_per_particle"],
                             obs["energy_per_particle_err"], exact,
                             obs["tau_int_energy"]))
            E.append(obs["energy_per_particle"])
            dE.append(obs["energy_per_particle_err"])
            print(f"  {proposal:8s} T={T:4.2f} q_max={q:3d}  E={E[-1]:.4f} ± {dE[-1]:.4f}  exacto={exact:.4f}"
                  f"  z={rows[-1]['z']:.2f}")
        arrays[f"n1_{proposal}_T"] = np.array(temps)
        arrays[f"n1_{proposal}_E"] = np.array(E)
        arrays[f"n1_{proposal}_err"] = np.array(dE)
    T_curve = np.linspace(0.2, 5.2, 300)
    arrays["n1_curve_T"] = T_curve
    arrays["n1_curve_E"] = an.exact_single_particle_energy(1.0 / T_curve)

    # ---------------- V2: N = 3 contra enumeración directa -------------------
    print("V2: N = 3 (q_max = 6) contra enumeración directa, núcleo simétrico")
    q3 = 6
    energies, _ = build_state_catalog(q3)
    for s_code, stat in enumerate(("bosons", "fermions")):
        for k, T in enumerate((0.5, 1.0, 2.0)):
            res, obs = _run(3, T, q3, stat, "state", int(40_000_000 * scale), make_seed(2, s_code, k))
            ex = an.exact_canonical_enumeration(3, energies, 1.0 / T, stat)
            occ_exact = np.array([ex["occupation_per_state"][energies == l].mean() for l in range(q3 + 1)])
            rows.append(_row("V2 N=3 enumeracion", res, obs, "E/N", obs["energy_per_particle"],
                             obs["energy_per_particle_err"],
                             ex["mean_energy"] / 3, obs["tau_int_energy"]))
            rows.append(_row("V2 N=3 enumeracion", res, obs, "<n_0>/N", obs["condensate_fraction"],
                             obs["condensate_fraction_err"],
                             occ_exact[0] / 3, obs["tau_int_n0"]))
            # z por nivel sólo donde la población exacta g<n> es apreciable (≥ 0.01 partículas):
            # en niveles casi vacíos hay muy pocos eventos y τ_int no se puede estimar bien.
            populated = obs["degeneracy"] * occ_exact >= 0.01
            z_levels = np.abs(obs["occupation"] - occ_exact)[populated] / obs["occupation_err"][populated]
            for l in np.nonzero(populated)[0]:
                rows.append(_row("V2 N=3 enumeracion", res, obs, f"<n(eps={l})>", obs["occupation"][l],
                                 obs["occupation_err"][l], occ_exact[l], float("nan")))
            print(f"  {stat:8s} T={T:3.1f}  E/N={obs['energy_per_particle']:.4f} ± {obs['energy_per_particle_err']:.4f}"
                  f"  exacto={ex['mean_energy'] / 3:.4f}  z_E={rows[-2 - populated.sum()]['z']:.2f}"
                  f"  max z(n(eps))={z_levels.max():.2f} en {populated.sum()} niveles")
            if T == 1.0:
                arrays[f"n3_{stat}_occ"] = obs["occupation"]
                arrays[f"n3_{stat}_occ_err"] = obs["occupation_err"]
                arrays[f"n3_{stat}_occ_exact"] = occ_exact
    arrays["n3_levels"] = np.arange(q3 + 1)

    # ---------------- V3: los dos núcleos contra la solución canónica exacta --
    print("V3: N = 10, q_max = 12, T = 1.5 — simétrico vs Hastings vs exacto")
    for s_code, stat in enumerate(("bosons", "fermions")):
        ex = an.exact_canonical(10, 12, 1.0 / 1.5, stat)
        for proposal, steps in (("state", 200_000_000), ("particle", 40_000_000)):
            res, obs = _run(10, 1.5, 12, stat, proposal, int(steps * scale), make_seed(3, s_code, 0 if proposal == "state" else 1))
            rows.append(_row("V3 nucleos N=10", res, obs, "E/N", obs["energy_per_particle"],
                             obs["energy_per_particle_err"],
                             ex["energy_per_particle"], obs["tau_int_energy"]))
            rows.append(_row("V3 nucleos N=10", res, obs, "<n_0>/N", obs["condensate_fraction"],
                             obs["condensate_fraction_err"],
                             ex["condensate_fraction"], obs["tau_int_n0"]))
            print(f"  {stat:8s} {proposal:8s} E/N={obs['energy_per_particle']:.4f} ± {obs['energy_per_particle_err']:.4f}"
                  f"  exacto={ex['energy_per_particle']:.4f}  z={rows[-2]['z']:.2f}"
                  f"  tau_int={obs['tau_int_energy']:.1f} muestras  aceptación={res['metadata']['acceptance_rate']:.3g}")

    # ---------------- V4: métodos exactos entre sí ----------------------------
    exact_rows = []
    for stat in ("bosons", "fermions"):
        for N in (1, 2, 3, 4):
            q = 4 if N == 4 else 6
            e, _ = build_state_catalog(q)
            for beta in (0.25, 1.0, 4.0):
                bf = an.exact_canonical_enumeration(N, e, beta, stat)
                gf = an.exact_canonical(N, q, beta, stat)
                occ_bf = np.array([bf["occupation_per_state"][e == l].mean() for l in range(q + 1)])
                exact_rows.append({
                    "estadistica": stat, "N": N, "q_max": q, "beta": beta,
                    "configuraciones": bf["n_configurations"],
                    "E_enumeracion": bf["mean_energy"], "E_generatriz": gf["mean_energy"],
                    "max_dif_E": abs(bf["mean_energy"] - gf["mean_energy"]),
                    "max_dif_ocupacion": float(np.abs(occ_bf - gf["occupation"]).max()),
                })
    worst = max(max(r["max_dif_E"], r["max_dif_ocupacion"]) for r in exact_rows)
    print(f"V4: función generatriz vs enumeración, máxima diferencia = {worst:.2e}")

    save_raw(raw_dir / "validation.npz", arrays, {"description": "Validación A2", "n_samples": N_SAMPLES})
    write_table_csv(raw_dir / "validation_table.csv", rows)
    write_table_csv(raw_dir / "validation_exact_methods.csv", exact_rows)
    zs = np.array([r["z"] for r in rows])
    print(f"\n{len(zs)} comparaciones: z medio = {zs.mean():.2f}, z máximo = {zs.max():.2f}, "
          f"fracción con z < 2: {np.mean(zs < 2):.0%}")
    print(f"Datos guardados en {raw_dir}")


if __name__ == "__main__":
    args = _common.base_parser(__doc__.splitlines()[0]).parse_args()
    main(fast=args.fast)
