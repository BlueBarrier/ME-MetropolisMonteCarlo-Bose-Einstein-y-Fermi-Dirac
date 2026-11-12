"""Figuras del proyecto. Cada función recibe datos ya cargados de data/raw y guarda un PNG en fig/.

Convenciones visuales (las mismas en todas las figuras):
- Bosones en azul y fermiones en naranja; los tamaños N van de claro a oscuro
  dentro de cada tono y además llevan un marcador distinto.
- Simulación: marcadores con barras de error (1σ, corregidas por autocorrelación).
- Solución canónica exacta: línea continua del color de la serie.
- Ensemble gran canónico (BE/FD): línea discontinua. Límite termodinámico o
  referencia clásica: línea punteada.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import analytics as an

COLOR = {"bosons": "#2a78d6", "fermions": "#eb6834"}
RAMP = {
    "bosons": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    "fermions": ["#f3a17c", "#eb6834", "#b8461a", "#7a2c0c"],
}
KERNEL_COLOR = {"state": "#6250d6", "particle": "#1baf7a"}
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NAME = {"bosons": "Bosones (BE)", "fermions": "Fermiones (FD)"}
KERNEL_NAME = {"state": "Metropolis simétrico", "particle": "Metropolis–Hastings"}

# Escala global del tamaño de las figuras (con 0.62 una figura de tres paneles mide ~8 pulgadas).
FIG_SCALE = 0.62


def _fs(width: float, height: float) -> tuple[float, float]:
    return (width * FIG_SCALE, height * FIG_SCALE)


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 1.5,
        "legend.frameon": True, "legend.framealpha": 0.85, "legend.edgecolor": "none",
        "legend.facecolor": "white", "savefig.dpi": 200, "figure.dpi": 100,
        "mathtext.fontset": "dejavusans", "axes.titlelocation": "left",
    })


# Nombre interno de cada figura -> nombre del archivo PNG (numerado en el orden de lectura).
FIGURE_FILES = {
    "fig_validation": "01_validacion_contra_resultados_exactos",
    "fig_equilibration": "02_equilibracion_desde_dos_estados_iniciales",
    "fig_autocorrelation": "03_autocorrelacion_de_la_energia",
    "fig_qmax_convergence": "04_convergencia_en_el_corte_energetico_qmax",
    "fig_energy_vs_T": "05_energia_por_particula_vs_temperatura",
    "fig_condensate_fraction": "06_fraccion_de_condensado_bosones_vs_T",
    "fig_occupations": "07_ocupaciones_bosones_vs_fermiones_N45",
    "fig_ensemble_comparison": "08_energia_canonico_vs_gran_canonico_BE_FD",
}


def save(fig, out_dir: Path, name: str) -> list[Path]:
    """Guarda la figura como PNG en ``out_dir`` con su nombre descriptivo."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FIGURE_FILES.get(name, name)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _eb(ax, x, y, yerr, color, marker="o", label=None, filled=True, ms=4.5, zorder=3):
    return ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, mfc=color if filled else "white",
                       mec=color, ms=ms, elinewidth=1.0, capsize=2, label=label, zorder=zorder, lw=0)


def _rows(rows, stat, N=None):
    sel = [r for r in rows if r["estadistica"] == stat and (N is None or r["N"] == N)]
    return sorted(sel, key=lambda r: r["T"])


def _col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


# ---------------------------------------------------------------------------
# Validación contra resultados exactos
# ---------------------------------------------------------------------------
def fig_validation(val: dict, val_rows: list[dict], out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=_fs(13, 3.9), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    ax = axes[0]
    ax.plot(val["n1_curve_T"], val["n1_curve_E"], color=INK, lw=1.2, label=r"exacto $3/(e^{\beta}-1)$")
    ax.plot(val["n1_curve_T"], 3 * val["n1_curve_T"], color=MUTED, ls=":", lw=1.2, label=r"clásico $3k_BT$")
    for prop, mk, filled in (("state", "o", False), ("particle", "s", True)):
        _eb(ax, val[f"n1_{prop}_T"], val[f"n1_{prop}_E"], val[f"n1_{prop}_err"], KERNEL_COLOR[prop],
            marker=mk, filled=filled, label=KERNEL_NAME[prop])
    ax.set(xlabel=r"$T$  [$\hbar\omega/k_B$]", ylabel=r"$\langle E\rangle$  [$\hbar\omega$]",
           title="(a) N = 1, trampa infinita")
    ax.legend(loc="upper left")

    ax = axes[1]
    eps = val["n3_levels"]
    for k, stat in enumerate(("bosons", "fermions")):
        ax.plot(eps, val[f"n3_{stat}_occ_exact"], color=COLOR[stat], lw=1.0, alpha=0.9)
        _eb(ax, eps + (k - 0.5) * 0.12, val[f"n3_{stat}_occ"], val[f"n3_{stat}_occ_err"], COLOR[stat],
            marker=MARKERS[k], label=f"{'BE' if stat == 'bosons' else 'FD'}: simulación")
    ax.plot([], [], color=INK2, lw=1.0, label="enumeración exacta")
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 2.0)
    ax.set(xlabel=r"nivel $\varepsilon$  [$\hbar\omega$]", ylabel=r"$\langle n(\varepsilon)\rangle$ por estado",
           title=r"(b) N = 3, $q_{max}$ = 6, T = 1")
    ax.legend(loc="lower left")

    ax = axes[2]
    groups = [("V1 N=1 analitico", "N=1\nanalítico"), ("V2 N=3 enumeracion", "N=3\nenumeración"),
              ("V3 nucleos N=10", "N=10\n2 núcleos")]
    ax.axvspan(0, 2, color=GRID, alpha=0.6, lw=0, label="|z| < 2")
    rng = np.random.default_rng(0)
    for gi, (case, label) in enumerate(groups):
        for r in (r for r in val_rows if r["caso"] == case and np.isfinite(r["z"])):
            c = KERNEL_COLOR[r["nucleo"]] if case != "V2 N=3 enumeracion" else COLOR[r["estadistica"]]
            ax.plot(r["z"], gi + rng.uniform(-0.18, 0.18), "o", ms=4, color=c, mfc=c, alpha=0.85)
    ax.set_yticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_xlim(0, max(3.2, max(r["z"] for r in val_rows if np.isfinite(r["z"])) + 0.3))
    ax.set(xlabel=r"$z = |\bar A - A_{exacto}|/\sigma_{\bar A}$", title="(c) Discrepancias en σ")
    ax.grid(axis="y", visible=False)
    for prop in ("state", "particle"):
        ax.plot([], [], "o", ms=4, color=KERNEL_COLOR[prop], label=KERNEL_NAME[prop])
    for stat in ("bosons", "fermions"):
        ax.plot([], [], "o", ms=4, color=COLOR[stat], label=f"N = 3, {'BE' if stat == 'bosons' else 'FD'}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.3)
    fig.tight_layout()
    return save(fig, out_dir, "fig_validation")


# ---------------------------------------------------------------------------
# Equilibración y autocorrelación
# ---------------------------------------------------------------------------
def fig_equilibration(eq: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=_fs(11, 3.8))
    for ax, stat in zip(axes, ("bosons", "fermions")):
        ramp = RAMP[stat]
        for mode, color, label in (("ground", ramp[3], "inicio frío (estado base)"),
                                   ("excited", ramp[1], r"inicio caliente ($T=\infty$)")):
            burn_t, prod_t = eq[f"{stat}_burn_steps"], eq[f"{stat}_prod_steps"]
            stride = max(1, prod_t.size // 2000)
            t = np.concatenate([[1], burn_t, prod_t[::stride]])
            E = np.concatenate([[eq[f"{stat}_{mode}_E0"]], eq[f"{stat}_{mode}_burn_E"], eq[f"{stat}_{mode}_prod_E"][::stride]])
            ax.plot(t, E, color=color, lw=0.9, label=label)
        ax.axhline(float(eq[f"{stat}_exact"]), color=INK, ls="--", lw=1.1, label="canónico exacto")
        nb = int(eq[f"{stat}_n_burn"])
        ax.axvline(nb, color=INK2, ls=":", lw=1.2)
        ax.text(nb * 1.1, ax.get_ylim()[1] * 0.97, r"$n_{burn}$", color=INK2, va="top")
        cross = int(eq[f"{stat}_cross_steps"])
        if cross > 0:
            ax.axvline(cross, color=MUTED, ls="-.", lw=1.0, label=f"cruce de cadenas ({cross:.1e} pasos)")
        ax.set_xscale("log")
        ax.set(xlabel="pasos de Monte Carlo (propuestas)", ylabel=r"$E/N$  [$\hbar\omega$]",
               title=f"({'a' if stat == 'bosons' else 'b'}) {NAME[stat]}, z entre cadenas = {float(eq[f'{stat}_z_chains']):.2f}")
        ax.legend(loc="center right")
    fig.tight_layout()
    return save(fig, out_dir, "fig_equilibration")


def fig_autocorrelation(eq: dict, out_dir: Path):
    """C(t) de E/N en producción, con τ_int y la comparación entre el error ingenuo y el corregido."""
    fig, ax = plt.subplots(figsize=_fs(8.5, 4.2))
    for stat in ("bosons", "fermions"):
        acf, tau = eq[f"{stat}_acf"], float(eq[f"{stat}_tau"])
        ratio = float(eq[f"{stat}_err_tau"]) / float(eq[f"{stat}_err_naive"])
        lag = np.arange(acf.size)
        cut = min(acf.size, int(10 * tau) + 5)
        ax.plot(lag[:cut], acf[:cut], color=COLOR[stat],
                label=fr"{NAME[stat]}: $\tau_{{int}}$ = {tau:.1f}  (σ corregido = {ratio:.1f}× σ ingenuo)")
        ax.axvline(tau, color=COLOR[stat], ls=":", lw=1.0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set(xlabel="retardo t  [mediciones]", ylabel=r"$C(t)$ de $E/N$",
           title=r"Autocorrelación normalizada; $\sigma_{\bar A}^2 = 2\tau_{int}\sigma_A^2/N$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save(fig, out_dir, "fig_autocorrelation")


# ---------------------------------------------------------------------------
# Corte energético (truncamiento vs estadística)
# ---------------------------------------------------------------------------
def fig_qmax_convergence(qm: dict, meta: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=_fs(11, 6.2), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    for col, stat in enumerate(("bosons", "fermions")):
        N, T = meta[stat]["N"], meta[stat]["T"]
        ax = axes[0, col]
        ax.plot(qm[f"{stat}_q_dense"], qm[f"{stat}_E_dense"], color=COLOR[stat], lw=1.2,
                label=r"exacto con corte $q_{max}$")
        ax.axhline(float(qm[f"{stat}_E_ref"]), color=INK, ls="--", lw=1.0, label=r"exacto, $q_{max}\to\infty$")
        _eb(ax, qm[f"{stat}_q"], qm[f"{stat}_E"], qm[f"{stat}_err"], COLOR[stat], label="simulación")
        ax.axvline(int(qm[f"{stat}_q_chosen"]), color=MUTED, ls=":", lw=1.2, label=r"$q_{max}$ elegido")
        ax.set(ylabel=r"$E/N$  [$\hbar\omega$]", title=f"({'ab'[col]}) {NAME[stat]}, N = {N}, T = {T}")
        ax.legend(loc="lower right")
        ax = axes[1, col]
        trunc = np.abs(qm[f"{stat}_E_dense"] - float(qm[f"{stat}_E_ref"]))
        ax.semilogy(qm[f"{stat}_q_dense"], np.where(trunc > 0, trunc, np.nan), color=COLOR[stat],
                    label=r"error de truncamiento $|E_{q}-E_{\infty}|/N$")
        ax.semilogy(qm[f"{stat}_q"], qm[f"{stat}_err"], "o", color=INK2, ms=4, mfc="white",
                    label=r"error estadístico $\sigma_{E/N}$")
        ax.axhline(1e-4 * T, color=MUTED, ls=":", lw=1.0, label=r"tolerancia $10^{-4}T$")
        ax.axvline(int(qm[f"{stat}_q_chosen"]), color=MUTED, ls=":", lw=1.2)
        ax.set(xlabel=r"corte $q_{max}$", ylabel=r"error en $E/N$  [$\hbar\omega$]")
        ax.set_ylim(1e-8, None)
        if col == 0:  # las mismas entradas en los dos paneles: una sola leyenda
            ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out_dir, "fig_qmax_convergence")


# ---------------------------------------------------------------------------
# Resultados principales
# ---------------------------------------------------------------------------
def fig_energy_vs_T(rows: list[dict], curves: dict, n_values, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=_fs(13.5, 4.0))
    T_cl = np.linspace(0.15, 5.2, 200)
    for ax, stat in zip(axes[:2], ("bosons", "fermions")):
        for k, N in enumerate(n_values):
            c = RAMP[stat][k]
            ax.plot(curves[f"{stat}_N{N}_T"], curves[f"{stat}_N{N}_E"], color=c, lw=1.0)
            sel = _rows(rows, stat, N)
            _eb(ax, _col(sel, "T"), _col(sel, "E_N"), _col(sel, "E_N_err"), c, marker=MARKERS[k], label=f"N = {N}")
        ax.plot(T_cl, an.classical_energy_per_particle(1 / T_cl), color=MUTED, ls=":", lw=1.3,
                label="Maxwell–Boltzmann")
        ax.set(xlabel=r"$T$  [$\hbar\omega/k_B$]", ylabel=r"$\langle E\rangle/N$  [$\hbar\omega$]",
               title=f"({'a' if stat == 'bosons' else 'b'}) {NAME[stat]}")
        ax.legend(loc="upper left")
    ax = axes[2]
    N = n_values[2]
    for k, stat in enumerate(("bosons", "fermions")):
        ax.plot(curves[f"{stat}_N{N}_T"], curves[f"{stat}_N{N}_E"], color=COLOR[stat], lw=1.0)
        sel = _rows(rows, stat, N)
        _eb(ax, _col(sel, "T"), _col(sel, "E_N"), _col(sel, "E_N_err"), COLOR[stat], marker=MARKERS[k],
            label=NAME[stat])
    ax.plot(T_cl, an.classical_energy_per_particle(1 / T_cl), color=MUTED, ls=":", lw=1.3, label="Maxwell–Boltzmann")
    Tc = float(an.bec_critical_temperature(N))
    E0 = an.ground_state_energy_per_particle(N, "fermions")
    ax.axvline(Tc, color=COLOR["bosons"], ls="--", lw=0.9)
    ax.text(Tc + 0.05, 0.3, r"$T_c$", color=COLOR["bosons"])
    ax.axhline(E0, color=COLOR["fermions"], ls="--", lw=0.9)
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, 9)
    ax.text(0.05, E0 - 0.6, r"$E_0/N$ ($T=0$)", color=COLOR["fermions"], fontsize=7)
    ax.set(xlabel=r"$T$  [$\hbar\omega/k_B$]", ylabel=r"$\langle E\rangle/N$  [$\hbar\omega$]",
           title=f"(c) Contraste a baja T, N = {N}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, out_dir, "fig_energy_vs_T")


def fig_condensate_fraction(rows: list[dict], curves: dict, n_values, out_dir: Path):
    fig, ax = plt.subplots(figsize=_fs(10.5, 4.6))
    t = np.linspace(0, 1.6, 300)
    ax.plot(t, np.clip(1 - t**3, 0, None), color=INK, ls="--", lw=1.2, label=r"$N\to\infty$: $1-(T/T_c)^3$")
    for k, N in enumerate(n_values):
        c = RAMP["bosons"][k]
        Tc = float(an.bec_critical_temperature(N))
        ax.plot(curves[f"bosons_N{N}_T"] / Tc, curves[f"bosons_N{N}_f0"], color=c, lw=1.0)
        ax.plot(t, an.condensate_fraction_finite_size(t * Tc, N), color=c, ls=":", lw=1.1)
        sel = _rows(rows, "bosons", N)
        _eb(ax, _col(sel, "T") / Tc, _col(sel, "f0"), _col(sel, "f0_err"), c, marker=MARKERS[k], label=f"N = {N}")
    ax.plot([], [], color=INK2, lw=1.0, label="canónico exacto")
    ax.plot([], [], color=INK2, ls=":", lw=1.1, label=r"$1-t^3-3C\,t^2N^{-1/3}$ (tamaño finito)")
    ax.set(xlabel=r"$T/T_c$,  $T_c = (N/\zeta(3))^{1/3}\,\hbar\omega/k_B$", ylabel=r"$f_0 = \langle n_0\rangle/N$",
           title="Fracción de condensado (bosones)", xlim=(0, 1.6), ylim=(-0.02, 1.02))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return save(fig, out_dir, "fig_condensate_fraction")


def fig_occupations(occ: dict, out_dir: Path):
    """Ocupación media por estado: simulación (puntos pequeños, sin barras), canónico exacto y gran canónico.

    ``occ`` es el contenido de data/raw/occupations.npz (ver scripts/run_sweeps.py, parte C).
    """
    N = int(occ["N"])
    temps = [float(t) for t in occ["temperatures"]]
    fig, axes = plt.subplots(2, len(temps), figsize=_fs(4.4 * len(temps), 7.0))
    for row, stat in enumerate(("bosons", "fermions")):
        tag = "BE" if stat == "bosons" else "FD"
        T_ref = float(occ[f"{stat}_T_ref"])
        for col, T in enumerate(temps):
            ax = axes[row, col]
            key = f"{stat}_T{T:.2f}"
            eps, sim = occ[f"{key}_levels"], occ[f"{key}_sim"]
            ex, gc = occ[f"{key}_exact"], occ[f"{key}_gc"]
            keep = ex > 1e-4
            xmax = int(eps[keep].max()) if keep.any() else int(eps.max())
            m = eps <= xmax
            # Canónico (banda gruesa clara) y gran canónico (discontinua oscura): se distinguen aunque coincidan.
            ax.plot(eps[m], ex[m], color=COLOR[stat], lw=3.2, alpha=0.35, solid_capstyle="round",
                    label="canónico exacto", zorder=1)
            ax.plot(eps[m], gc[m], color=INK, ls=(0, (4, 2)), lw=0.9, label=f"gran canónico ({tag})", zorder=2)
            vis = m & (sim > 0)
            ax.plot(eps[vis], sim[vis], ls="none", marker="D", ms=2.6, mfc=RAMP[stat][3], mec=RAMP[stat][3],
                    label="simulación", zorder=3)
            if stat == "bosons":
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.03, 1.05)
                eF = an.fermi_energy(N)
                ax.axvline(eF, color=MUTED, ls=":", lw=1.0)
                ax.text(eF + 0.2, 1.0, r"$\varepsilon_F$", color=MUTED, va="top")
            ref = r"T/T_c" if stat == "bosons" else r"T/T_F"
            ax.set_title(f"({'abcdef'[row * len(temps) + col]}) {tag}, T = {T:g}  (${ref}$ = {T / T_ref:.2f})")
            ax.set_xlabel(r"$\varepsilon$  [$\hbar\omega$]")
            if col == 0:
                ax.set_ylabel(r"$\langle n(\varepsilon)\rangle$ por estado")
            if col == len(temps) - 1:
                ax.legend(loc="upper right")
    fig.suptitle(f"Ocupación media por estado, N = {N}", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    return save(fig, out_dir, "fig_occupations")


def fig_ensemble_comparison(rows: list[dict], curves: dict, n_values, out_dir: Path):
    """Diferencia relativa de energía entre los ensembles canónico y gran canónico (BE y FD)."""
    fig, axes = plt.subplots(1, 2, figsize=_fs(11.5, 4.0))
    for ax, stat, letter in ((axes[0], "bosons", "a"), (axes[1], "fermions", "b")):
        for k, N in enumerate(n_values):
            c = RAMP[stat][k]
            Tr = an.bec_critical_temperature(N) if stat == "bosons" else an.fermi_energy(N)
            Tc_, E, Eg = curves[f"{stat}_N{N}_T"], curves[f"{stat}_N{N}_E"], curves[f"{stat}_N{N}_E_gc"]
            ax.plot(Tc_ / Tr, (E - Eg) / Eg, color=c, lw=1.0)
            sel = _rows(rows, stat, N)
            Eg_s = _col(sel, "E_N_gc")
            _eb(ax, _col(sel, "T") / Tr, (_col(sel, "E_N") - Eg_s) / Eg_s, _col(sel, "E_N_err") / Eg_s, c,
                marker=MARKERS[k], label=f"N = {N}")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set(xlabel=r"$T/T_c$" if stat == "bosons" else r"$T/T_F$", ylabel=r"$(E^{can}-E^{gc})/E^{gc}$",
               title=f"({letter}) {NAME[stat]}: energía, canónico − gran canónico", xlim=(0, 1.6))
        ax.legend(loc="upper right" if stat == "bosons" else "lower right")
    fig.tight_layout()
    return save(fig, out_dir, "fig_ensemble_comparison")


# ---------------------------------------------------------------------------
# Comparación con datos teóricos ingresados por el usuario
# ---------------------------------------------------------------------------
def fig_user_comparison(result: dict, out_path: Path) -> Path:
    """Teoría del usuario (línea con puntos) frente a la simulación (marcadores con σ) y residuos z."""
    stat, obs = result["statistics"], result["observable"]
    x_t, y_t = np.asarray(result["theory_x"]), np.asarray(result["theory_y"])
    x_s, y_s, e_s = np.asarray(result["sim_x"]), np.asarray(result["sim_y"]), np.asarray(result["sim_err"])
    z = np.asarray(result["z"])
    fig, (ax, axr) = plt.subplots(2, 1, figsize=_fs(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    order = np.argsort(x_t)
    ax.plot(x_t[order], y_t[order], color=INK, lw=1.2, marker="o", ms=3.5, mfc="white", label="datos teóricos (usuario)")
    _eb(ax, x_s, y_s, e_s, COLOR[stat], marker="s", label=f"simulación ({'BE' if stat == 'bosons' else 'FD'})")
    if obs == "occupation":
        ax.set_yscale("log" if stat == "bosons" else "linear")
    ax.set(ylabel=result["y_label"],
           title=f"{NAME[stat]}, N = {result['N']}, $q_{{max}}$ = {result['q_max']}"
                 + (f", T = {result['T']:g}" if obs == "occupation" else ""))
    ax.legend(loc="best")
    axr.axhspan(-2, 2, color=GRID, alpha=0.7, lw=0, label="|z| < 2")
    axr.axhline(0, color=INK, lw=0.8)
    axr.plot(x_s, z, "o", color=COLOR[stat], ms=4)
    axr.set(xlabel=result["x_label"], ylabel=r"$(A_{sim}-A_{teo})/\sigma$")
    axr.legend(loc="upper right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
