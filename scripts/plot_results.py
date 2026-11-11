"""Regenera todas las figuras y tablas del reporte a partir de data/raw (sin volver a simular).

Figuras (PDF + PNG en figures/):
  fig_validation           ← run_validation.py
  fig_equilibration        ← run_equilibration.py
  fig_autocorrelation      ← run_equilibration.py
  fig_qmax_convergence     ← run_sweeps.py (parte A)
  fig_energy_vs_T          ← run_sweeps.py (parte B)
  fig_condensate_fraction  ← run_sweeps.py (parte B)
  fig_occupations          ← run_sweeps.py (parte B)
  fig_ensemble_comparison  ← run_sweeps.py (partes B y C)
  fig_z_scores             ← run_sweeps.py (parte B)
  fig_experimental_comparison ← compare_experiments.py (datos de data/external/)
Tablas: figures/tabla_validacion.tex, figures/tabla_resultados.csv, figures/tabla_resultados.tex

Uso: python scripts/plot_results.py [--fast] [--only NOMBRE]
"""

from __future__ import annotations

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from src import analytics as an  # noqa: E402
from src import plotting as pl  # noqa: E402
from src.config import N_VALUES, output_dirs  # noqa: E402
from src.datastore import load_raw, read_table_csv, save_raw, write_table_csv  # noqa: E402

OCC_N = 35
OCC_TEMPS = (0.5, 2.0, 4.0)


def exact_curves(raw_dir, n_values) -> dict:
    """Curvas exactas densas en T (canónico y gran canónico). Se guardan en caché porque tardan ~1 min."""
    path = raw_dir / "exact_curves.npz"
    if path.exists():
        arrays, meta = load_raw(path)
        if meta.get("n_values") == list(n_values):
            return arrays
    print("Calculando curvas exactas (se guardan en data/raw/exact_curves.npz)...")
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


def main(fast: bool = False, only: str | None = None) -> None:
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
        scaling = load_raw(raw_dir / "ensemble_scaling.npz")[0] if (raw_dir / "ensemble_scaling.npz").exists() else None
        if want("fig_energy_vs_T"):
            made += pl.fig_energy_vs_T(rows, curves, N_VALUES, fig_dir)
        if want("fig_condensate_fraction"):
            made += pl.fig_condensate_fraction(rows, curves, N_VALUES, fig_dir)
        if want("fig_ensemble_comparison"):
            made += pl.fig_ensemble_comparison(rows, curves, scaling, N_VALUES, fig_dir)
        if want("fig_z_scores"):
            made += pl.fig_z_scores(rows, fig_dir)
        if want("fig_occupations"):
            runs = {}
            for stat in ("bosons", "fermions"):
                for T in OCC_TEMPS:
                    arr, meta = load_raw(raw_dir / "sweep" / f"{stat}_N{OCC_N}_T{T:.2f}.npz")
                    runs[(stat, T)] = (arr, meta["summary"])
            made += pl.fig_occupations(runs, OCC_N, OCC_TEMPS, fig_dir)
        if (raw_dir / "experimental_comparison.npz").exists() and want("fig_experimental_comparison"):
            cmp, _ = load_raw(raw_dir / "experimental_comparison.npz")
            made += pl.fig_experimental_comparison(cmp, N_VALUES, fig_dir)
        if only is None:
            tables(val_rows, rows, fig_dir)
            made += [fig_dir / "tabla_validacion.tex", fig_dir / "tabla_resultados.csv", fig_dir / "tabla_resultados.tex"]
    for p in made:
        print(f"  {p.relative_to(fig_dir.parents[1] if fast else fig_dir.parent)}")
    print(f"{len(made)} archivos generados en {fig_dir}")
    return made


if __name__ == "__main__":
    parser = _common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="Nombre de una sola figura, p. ej. fig_validation")
    args = parser.parse_args()
    main(fast=args.fast, only=args.only)
