"""Regenera todas las figuras (PNG en fig/) y tablas a partir de data/raw, sin volver a simular,
y al terminar abre una ventana para verlas una por una (cierre la ventana para salir).

Figuras en fig/ (nombre interno para --only → archivo):
  fig_validation              → 01_validacion_contra_resultados_exactos.png        (run_validation.py)
  fig_equilibration           → 02_equilibracion_desde_dos_estados_iniciales.png   (run_equilibration.py)
  fig_autocorrelation         → 03_autocorrelacion_de_la_energia.png               (run_equilibration.py)
  fig_qmax_convergence        → 04_convergencia_en_el_corte_energetico_qmax.png    (run_sweeps.py, parte A)
  fig_energy_vs_T             → 05_energia_por_particula_vs_temperatura.png        (run_sweeps.py, parte B)
  fig_condensate_fraction     → 06_fraccion_de_condensado_bosones_vs_T.png         (run_sweeps.py, parte B)
  fig_occupations             → 07_ocupaciones_bosones_vs_fermiones_N45.png        (run_sweeps.py, parte C)
  fig_ensemble_comparison     → 08_energia_canonico_vs_gran_canonico_BE_FD.png     (run_sweeps.py, parte B)
Tablas en data/tablas/: tabla_validacion.tex, tabla_resultados.csv, tabla_resultados.tex

Uso: python scripts/plot_results.py [--fast] [--only NOMBRE] [--no-show]
"""

from __future__ import annotations

import _common  
import matplotlib

matplotlib.use("Agg")
import numpy as np  

from src import analytics as an  
from src import plotting as pl  
from src.config import N_VALUES, output_dirs, table_dir  
from src.datastore import load_raw, read_table_csv, save_raw, write_table_csv
from src.viewer import show_gallery  



def exact_curves(raw_dir, n_values) -> dict:
    """Curvas exactas densas en T (canónico y gran canónico). Se guardan en caché porque tardan ~1 min."""
    path = raw_dir / "exact_curves.npz"
    if path.exists():
        arrays, meta = load_raw(path)
        if meta.get("n_values") == list(n_values):
            return arrays
    print("Calculando curvas exactas (guardadas en data/raw/exact_curves.npz)...")
    out = {}
    T_dense = np.concatenate([np.linspace(0.12, 1.0, 23), np.linspace(1.05, 5.2, 42)])
    for stat in ("bosons", "fermions"):
        for N in n_values:
            E, f0, Eg, f0g = [], [], [], []
            for T in T_dense:
                q = an.reference_cutoff(N, T, stat)
                ex = an.exact_canonical(N, q, 1.0 / T, stat)
                gc = an.grand_canonical_observables(N, q, 1.0 / T, stat)
                E.append(ex["energy_per_particle"]); f0.append(ex["condensate_fraction"])
                Eg.append(gc["energy_per_particle"]); f0g.append(gc["condensate_fraction"])
            out.update({f"{stat}_N{N}_T": T_dense, f"{stat}_N{N}_E": np.array(E), f"{stat}_N{N}_f0": np.array(f0),
                        f"{stat}_N{N}_E_gc": np.array(Eg), f"{stat}_N{N}_f0_gc": np.array(f0g)})
    save_raw(path, out, {"n_values": list(n_values), "description": "curvas exactas, corte q_ref ~ 40 T"})
    return out


def _sig(x: float, err: float) -> str:
    """Valor ± error con cifras significativas coherentes (1–2 cifras en el error)."""
    if not np.isfinite(err) or err <= 0:
        return f"{x:.4g}"
    d = -int(np.floor(np.log10(err))) + (1 if err / 10 ** np.floor(np.log10(err)) < 2 else 0)
    d = max(d, 0)
    return f"{x:.{d}f} \\pm {err:.{d}f}"


def tables(val_rows, rows, fig_dir) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{llrrrlrr}", r"\hline",
             r"Caso & Núcleo & $N$ & $T$ & $q_{max}$ & Observable & Simulación & $z$ \\", r"\hline"]
    for r in val_rows:
        if not str(r["observable"]).startswith("<n(eps"):
            obs = r["observable"].replace("<n_0>/N", r"$\langle n_0\rangle/N$").replace("E/N", "$E/N$")
            lines.append(f"{r['caso'].split()[0]} & {r['nucleo']} & {r['N']} & {r['T']:g} & {r['q_max']} & {obs} & "
                         f"${_sig(r['simulacion'], r['sigma'])}$ & {r['z']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (fig_dir / "tabla_validacion.tex").write_text("\n".join(lines), encoding="utf-8")

    sel = []
    for r in rows:
        sel.append({
            "estadistica": r["estadistica"], "N": r["N"], "T": r["T"], "T/T_ref": round(r["T_escalada"], 3),
            "q_max": r["q_max"], "semilla": r["semilla"],
            "E/N": r["E_N"], "sigma_E/N": r["E_N_err"], "E/N exacto": r["E_N_exacto"], "z_E": r["z_E"],
            "f0": r["f0"], "sigma_f0": r["f0_err"], "f0 exacto": r["f0_exacto"], "z_f0": r["z_f0"],
            "E/N gran canonico": r["E_N_gc"], "f0 gran canonico": r["f0_gc"], "mu": r["mu_gc"],
            "error_%_E": 100 * abs(r["E_N"] - r["E_N_exacto"]) / r["E_N_exacto"],
        })
    write_table_csv(fig_dir / "tabla_resultados.csv", sel)
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Estadística & $N$ & $T/T_{ref}$ & $E/N$ (sim.) & $E/N$ (exacto) & $f_0$ (sim.) \\", r"\hline"]
    for r in rows:
        if abs(r["T_escalada"] - 0.5) < 0.13 or abs(r["T_escalada"] - 1.0) < 0.13:
            lines.append(f"{'BE' if r['estadistica'] == 'bosons' else 'FD'} & {r['N']} & {r['T_escalada']:.2f} & "
                         f"${_sig(r['E_N'], r['E_N_err'])}$ & {r['E_N_exacto']:.4f} & ${_sig(r['f0'], r['f0_err'])}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (fig_dir / "tabla_resultados.tex").write_text("\n".join(lines), encoding="utf-8")


def main(fast: bool = False, only: str | None = None, show: bool = True) -> list:
    raw_dir, fig_dir = output_dirs(fast)
    pl.apply_style()
    made = []

    def want(name):
        return only is None or only == name

    if (raw_dir / "validation.npz").exists():
        val, _ = load_raw(raw_dir / "validation.npz")
        val_rows = read_table_csv(raw_dir / "validation_table.csv")
        if want("fig_validation"):
            made += pl.fig_validation(val, val_rows, fig_dir)
    else:
        val_rows = []
    if (raw_dir / "equilibration.npz").exists():
        eq, _ = load_raw(raw_dir / "equilibration.npz")
        if want("fig_equilibration"):
            made += pl.fig_equilibration(eq, fig_dir)
        if want("fig_autocorrelation"):
            made += pl.fig_autocorrelation(eq, fig_dir)
    if (raw_dir / "qmax_convergence.npz").exists() and want("fig_qmax_convergence"):
        qm, qm_meta = load_raw(raw_dir / "qmax_convergence.npz")
        made += pl.fig_qmax_convergence(qm, qm_meta, fig_dir)
    if (raw_dir / "sweep_summary.csv").exists():
        rows = read_table_csv(raw_dir / "sweep_summary.csv")
        curves = exact_curves(raw_dir, N_VALUES)
        if want("fig_energy_vs_T"):
            made += pl.fig_energy_vs_T(rows, curves, N_VALUES, fig_dir)
        if want("fig_condensate_fraction"):
            made += pl.fig_condensate_fraction(rows, curves, N_VALUES, fig_dir)
        if want("fig_ensemble_comparison"):
            made += pl.fig_ensemble_comparison(rows, curves, N_VALUES, fig_dir)
        if (raw_dir / "occupations.npz").exists() and want("fig_occupations"):
            occ, _ = load_raw(raw_dir / "occupations.npz")
            made += pl.fig_occupations(occ, fig_dir)
        if only is None:
            tdir = table_dir(fast)
            tables(val_rows, rows, tdir)
            print(f"Tablas en {tdir}")
    made = sorted(made)  # orden de lectura: 01, 02, ...
    for p in made:
        print(f"  {p.name}")
    print(f"{len(made)} figuras PNG guardadas en {fig_dir}")
    if show:
        show_gallery(made)
    return made


if __name__ == "__main__":
    parser = _common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="Nombre de una sola figura, p. ej. fig_validation")
    parser.add_argument("--no-show", action="store_true", help="No abrir la ventana de visualización")
    args = parser.parse_args()
    main(fast=args.fast, only=args.only, show=not args.no_show)
