"""Comparación con datos experimentales publicados (equilibrio, variables escaladas).

Experimentos (datos digitalizados en data/external/, ver su README):
  - Fermiones: DeMarco y Jin, Science 285, 1703 (1999), ⁴⁰K, N ≈ 7×10⁵. Exceso de energía
    δU/U_cl = (U − 3NkT)/(3NkT) frente a T/T_F, con k_B T_F = ħω̄ (6N)^{1/3}.
  - Bosones: Ensher et al., PRL 77, 4984 (1996), ⁸⁷Rb (figura reproducida en Cornell, Ensher
    y Wieman, cond-mat/9903109). Energía de liberación (cinética + interacción) por partícula
    E_rel/(N k_B T_0) frente a T/T_0, con k_B T_0 = ħω̄ (N/ζ(3))^{1/3}.

Las simulaciones (N ≤ 56) se llevan a las mismas variables:
  - se suma la energía de punto cero 3ħω/2 por partícula, porque la energía medida (y la
    teoría semiclásica) la incluye, aunque en el resto del proyecto se resta;
  - fermiones: δU/U_cl = (E/N + 3/2 − 3T)/(3T) y T/T_F con T_F = (6N)^{1/3}
    (la convención experimental, no el ε_F de capas cerradas);
  - bosones: E_rel = (E/N + 3/2)/2 (virial: cinética = total/2) y T/T_0 con T_0 = (N/ζ(3))^{1/3}.

Lo único que se compara son propiedades de equilibrio: los pasos de Monte Carlo no son tiempo físico.

Salidas: data/raw/experimental_comparison.npz, data/raw/experimental_comparison.csv
Uso:     python scripts/compare_experiments.py [--fast]
"""

from __future__ import annotations

from pathlib import Path

import _common  # noqa: F401
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from plot_results import exact_curves
from src import analytics as an
from src.config import N_VALUES, output_dirs
from src.datastore import ROOT, read_table_csv, save_raw, write_table_csv

EXT = ROOT / "data" / "external"
FERMI_CSV = EXT / "demarco_jin_1999_fig4.csv"
BOSE_CSV = EXT / "ensher_1996_energy_via_cornell_review_fig5.csv"
T_TRANSITION_ENSHER = 0.94  # T_c/T_0 medido (Ensher et al. 1996): 0.94(5)


def read_external(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = {}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if "sigma_y" in line:
                meta["sigma_y"] = float(line.rsplit("=", 1)[1])
            continue
        try:
            rows.append([float(v) for v in line.split(",")])
        except ValueError:
            continue  # encabezado
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], meta


def to_fermi_variables(T, E_per_N, err, N):
    TF = (6.0 * N) ** (1.0 / 3.0)
    return T / TF, (E_per_N + 1.5 - 3.0 * T) / (3.0 * T), err / (3.0 * T)


def to_bose_variables(T, E_per_N, err, N):
    T0 = float(an.bec_critical_temperature(N))
    return T / T0, 0.5 * (E_per_N + 1.5) / T0, 0.5 * err / T0


def main(fast: bool = False) -> None:
    raw_dir, _ = output_dirs(fast)
    rows = read_table_csv(raw_dir / "sweep_summary.csv")
    curves = exact_curves(raw_dir, N_VALUES)
    arrays, table = {}, []

    # ------------------------------------------------------------------ fermiones
    x, y, meta = read_external(FERMI_CSV)
    sigma = meta["sigma_y"]
    th = an.fermi_excess_energy_thermodynamic(x)
    chi2_1 = float(np.sum(((y - th) / sigma) ** 2))
    n = x.size

    def chi2_of(s):
        return float(np.sum(((y - an.fermi_excess_energy_thermodynamic(s * x)) / sigma) ** 2))

    fit = minimize_scalar(chi2_of, bounds=(0.5, 1.5), method="bounded", options={"xatol": 1e-5})
    s_best, chi2_s = float(fit.x), float(fit.fun)
    s_grid = np.linspace(s_best - 0.1, s_best + 0.1, 401)
    c_grid = np.array([chi2_of(s) for s in s_grid])
    inside = s_grid[c_grid <= chi2_s + 1.0]
    s_err = float(0.5 * (inside.max() - inside.min()))
    table += [
        {"experimento": "DeMarco y Jin 1999 (40K)", "comparacion": "gas de Fermi ideal, N→∞", "puntos": n,
         "chi2": chi2_1, "gl": n, "chi2_red": chi2_1 / n, "p": float(chi2.sf(chi2_1, n)),
         "nota": f"sigma_y = {sigma:.4f} (barra típica de la figura)"},
        {"experimento": "DeMarco y Jin 1999 (40K)", "comparacion": "ideal con T/T_F reescalado por s", "puntos": n,
         "chi2": chi2_s, "gl": n - 1, "chi2_red": chi2_s / (n - 1), "p": float(chi2.sf(chi2_s, n - 1)),
         "nota": f"s = {s_best:.3f} ± {s_err:.3f} (sistemático de T/T_F declarado: 20 %)"},
    ]
    print(f"Fermiones: chi2/gl = {chi2_1:.1f}/{n} con la escala nominal; "
          f"escala ajustada s = {s_best:.3f} ± {s_err:.3f}, chi2/gl = {chi2_s:.1f}/{n - 1}")
    t_dense = np.linspace(0.12, 3.3, 260)
    arrays.update({"fermi_exp_x": x, "fermi_exp_y": y, "fermi_sigma": np.array(sigma),
                   "fermi_th_x": t_dense, "fermi_th_y": an.fermi_excess_energy_thermodynamic(t_dense),
                   "fermi_scale": np.array(s_best), "fermi_scale_err": np.array(s_err),
                   "fermi_resid": (y - th) / sigma,
                   "fermi_resid_scaled": (y - an.fermi_excess_energy_thermodynamic(s_best * x)) / sigma})
    for N in N_VALUES:
        sel = sorted((r for r in rows if r["estadistica"] == "fermions" and r["N"] == N), key=lambda r: r["T"])
        T = np.array([r["T"] for r in sel]); E = np.array([r["E_N"] for r in sel]); dE = np.array([r["E_N_err"] for r in sel])
        mx, my, me = to_fermi_variables(T, E, dE, N)
        Tc = curves[f"fermions_N{N}_T"]
        cx, cy, _ = to_fermi_variables(Tc, curves[f"fermions_N{N}_E"], 0 * Tc, N)
        arrays.update({f"fermi_N{N}_x": mx, f"fermi_N{N}_y": my, f"fermi_N{N}_err": me,
                       f"fermi_N{N}_curve_x": cx, f"fermi_N{N}_curve_y": cy})
        band = (mx >= 0.3) & (mx <= 1.3)
        dev = my[band] - an.fermi_excess_energy_thermodynamic(mx[band])
        table.append({"experimento": f"simulación FD N={N}", "comparacion": "gas de Fermi ideal, N→∞",
                      "puntos": int(band.sum()), "desviacion_media": float(dev.mean()),
                      "desviacion_max": float(np.abs(dev).max()),
                      "nota": "0.3 ≤ T/T_F ≤ 1.3, T_F = (6N)^(1/3), con punto cero"})
        print(f"  simulación N={N}: desviación media respecto al límite termodinámico {dev.mean():+.4f}")

    # ------------------------------------------------------------------ bosones
    x, y, _ = read_external(BOSE_CSV)
    th = an.bose_release_energy_thermodynamic(x)
    # Dispersión empírica: residuos respecto a ajustes cuadráticos por separado arriba y abajo de 0.94 T_0,
    # como en la figura original.
    resid_fit = np.empty_like(y)
    for mask in (x < T_TRANSITION_ENSHER, x >= T_TRANSITION_ENSHER):
        p = np.polyfit(x[mask], y[mask], 2)
        resid_fit[mask] = y[mask] - np.polyval(p, x[mask])
    sigma_emp = float(np.sqrt(np.sum(resid_fit**2) / (x.size - 6)))
    above, below = x > 1.0, x < 0.9
    d_above = y[above] - th[above]
    d_below = y[below] - th[below]
    d_class = y[above] - 1.5 * x[above]
    table += [
        {"experimento": "Ensher et al. 1996 (87Rb)", "comparacion": "gas de Bose ideal, N→∞, T > T_0",
         "puntos": int(above.sum()), "desviacion_media": float(d_above.mean()),
         "error_media": sigma_emp / np.sqrt(above.sum()),
         "nota": f"dispersión empírica sigma = {sigma_emp:.3f} k_B T_0"},
        {"experimento": "Ensher et al. 1996 (87Rb)", "comparacion": "gas clásico 1.5 T/T_0, T > T_0",
         "puntos": int(above.sum()), "desviacion_media": float(d_class.mean()),
         "error_media": sigma_emp / np.sqrt(above.sum()), "nota": ""},
        {"experimento": "Ensher et al. 1996 (87Rb)", "comparacion": "gas de Bose ideal, N→∞, T < 0.9 T_0",
         "puntos": int(below.sum()), "desviacion_media": float(d_below.mean()),
         "error_media": sigma_emp / np.sqrt(below.sum()),
         "nota": "exceso atribuible a la energía de interacción del condensado"},
    ]
    print(f"Bosones: dispersión empírica {sigma_emp:.3f}; T > T_0: dato − ideal = {d_above.mean():+.3f} ± "
          f"{sigma_emp / np.sqrt(above.sum()):.3f}, dato − clásico = {d_class.mean():+.3f};"
          f"  T < 0.9 T_0: dato − ideal = {d_below.mean():+.3f}")
    t_dense = np.linspace(0.3, 1.8, 200)
    arrays.update({"bose_exp_x": x, "bose_exp_y": y, "bose_sigma_emp": np.array(sigma_emp),
                   "bose_th_x": t_dense, "bose_th_y": an.bose_release_energy_thermodynamic(t_dense),
                   "bose_resid": y - th, "bose_T_transition": np.array(T_TRANSITION_ENSHER)})
    for N in N_VALUES:
        sel = sorted((r for r in rows if r["estadistica"] == "bosons" and r["N"] == N), key=lambda r: r["T"])
        T = np.array([r["T"] for r in sel]); E = np.array([r["E_N"] for r in sel]); dE = np.array([r["E_N_err"] for r in sel])
        mx, my, me = to_bose_variables(T, E, dE, N)
        Tc = curves[f"bosons_N{N}_T"]
        cx, cy, _ = to_bose_variables(Tc, curves[f"bosons_N{N}_E"], 0 * Tc, N)
        arrays.update({f"bose_N{N}_x": mx, f"bose_N{N}_y": my, f"bose_N{N}_err": me,
                       f"bose_N{N}_curve_x": cx, f"bose_N{N}_curve_y": cy})

    save_raw(raw_dir / "experimental_comparison.npz", arrays,
             {"sources": {"fermions": FERMI_CSV.name, "bosons": BOSE_CSV.name}, "n_values": list(N_VALUES)})
    write_table_csv(raw_dir / "experimental_comparison.csv", table)
    print(f"Resultados en {raw_dir / 'experimental_comparison.csv'}")


if __name__ == "__main__":
    args = _common.base_parser(__doc__.splitlines()[0]).parse_args()
    main(fast=args.fast)
