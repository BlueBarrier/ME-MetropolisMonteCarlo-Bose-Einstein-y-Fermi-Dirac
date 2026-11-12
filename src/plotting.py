"""Figuras del proyecto. Cada función recibe datos ya cargados de data/raw y guarda un PNG en fig/.

Convenciones visuales (las mismas en todas las figuras):
- Bosones en azul y fermiones en naranja; los tamaños N van de claro a oscuro
  dentro de cada tono y además llevan un marcador distinto.
- Simulación: marcadores con barras de error (1σ, corregidas por autocorrelación).
- Solución canónica exacta: línea continua del color de la serie.
- Límite termodinámico (N → ∞): línea discontinua negra. Referencia clásica: línea punteada.
- Ocupaciones: histograma del número medio de ocupación por estado ⟨n(ε)⟩ = ⟨N(ε)⟩/g(ε), con eje lineal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from . import analytics as an

COLOR = {"bosons": "#2a78d6", "fermions": "#eb6834"}
RAMP = {
    "bosons": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    "fermions": ["#f3a17c", "#eb6834", "#b8461a", "#7a2c0c"],
}
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NAME = {"bosons": "Bosones (BE)", "fermions": "Fermiones (FD)"}

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
    "fig_validation_n1": "01a_validacion_N1_energia_vs_formula_analitica",
    "fig_validation_n3": "01b_validacion_N3_ocupaciones_vs_enumeracion",
    "fig_equilibration_bosons": "02a_equilibracion_bosones_desde_dos_estados_iniciales",
    "fig_equilibration_fermions": "02b_equilibracion_fermiones_desde_dos_estados_iniciales",
    "fig_autocorrelation": "03_autocorrelacion_de_la_energia",
    "fig_qmax_convergence": "04_convergencia_en_el_corte_energetico_qmax",
    "fig_energy_vs_T": "05_energia_por_particula_vs_temperatura",
    "fig_condensate_fraction": "06_fraccion_de_condensado_bosones_vs_T",
    "fig_occupations": "07_ocupaciones_bosones_vs_fermiones_N45",
    "fig_heat_capacity_bosons": "08a_calor_especifico_bosones_transicion_lambda",
    "fig_heat_capacity_fermions": "08b_calor_especifico_fermiones",
    "fig_chemical_potential_fermions": "08c_potencial_quimico_fermiones",
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


def _plain_axis(ax, integer=False):
    """Eje y lineal sin notación científica ni desplazamiento."""
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    if integer:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _level_histogram(ax, eps, sim_pop, exact_pop, color, dark, sim_label="simulación",
                     exact_label="canónico exacto"):
    """Histograma por nivel ε: barras = simulación, escalones negros = exacto."""
    ax.bar(eps, sim_pop, width=0.8, color=color, alpha=0.75, edgecolor=dark, lw=0.6, label=sim_label, zorder=2)
    ax.step(np.append(eps, eps[-1] + 1) - 0.5, np.append(exact_pop, exact_pop[-1]), where="post",
            color=INK, lw=1.1, label=exact_label, zorder=3)
    ax.set_ylim(0, None)
    ax.set_xlim(eps[0] - 0.6, eps[-1] + 0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _plain_axis(ax)


def _rows(rows, stat, N=None):
    sel = [r for r in rows if r["estadistica"] == stat and (N is None or r["N"] == N)]
    return sorted(sel, key=lambda r: r["T"])


def _col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


# ---------------------------------------------------------------------------
# Validación contra resultados exactos
# ---------------------------------------------------------------------------
def fig_validation(val: dict, val_rows: list[dict], out_dir: Path) -> list[Path]:
    """Validación en dos figuras separadas: N = 1 (energía) y N = 3 (ocupación media por estado)."""
    return _fig_validation_n1(val, out_dir) + _fig_validation_n3(val, out_dir)


def _fig_validation_n1(val: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(val["n1_curve_T"], val["n1_curve_E"], color=INK, lw=1.4, label=r"exacto  $3/(e^{\beta}-1)$")
    ax.plot(val["n1_curve_T"], 3 * val["n1_curve_T"] - 1.5, color=MUTED, ls=":", lw=1.3,
            label=r"límite clásico  $3k_BT - \frac{3}{2}\hbar\omega$")
    _eb(ax, val["n1_T"], val["n1_E"], val["n1_err"], "#1baf7a", marker="s", ms=6,
        label="simulación (Metropolis–Hastings)")
    ax.set(xlabel=r"$T$  [$\hbar\omega/k_B$]", ylabel=r"$\langle E\rangle$  [$\hbar\omega$]",
           title="Validación con N = 1: energía media frente a la fórmula analítica", xlim=(0, 5.3))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return save(fig, out_dir, "fig_validation_n1")


def _fig_validation_n3(val: dict, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    eps = val["n3_levels"]
    for ax, stat in zip(axes, ("bosons", "fermions")):
        _level_histogram(ax, eps, val[f"n3_{stat}_occ"], val[f"n3_{stat}_occ_exact"], COLOR[stat],
                         RAMP[stat][3], sim_label="simulación (Metropolis–Hastings)", exact_label="enumeración exacta")
        ax.set(xlabel=r"nivel $\varepsilon$  [$\hbar\omega$]", ylabel=r"número medio de ocupación  $\langle n(\varepsilon)\rangle$",
               title=f"{NAME[stat]}, N = 3, $q_{{max}}$ = 6, T = 1")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Validación con N = 3: número medio de ocupación frente a la enumeración de todas las configuraciones",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return save(fig, out_dir, "fig_validation_n3")


# ---------------------------------------------------------------------------
# Equilibración y autocorrelación
# ---------------------------------------------------------------------------
def fig_equilibration(eq: dict, out_dir: Path, params: dict | None = None) -> list[Path]:
    """Equilibración en dos figuras separadas, una por estadística (02a bosones, 02b fermiones)."""
    out = []
    for stat in ("bosons", "fermions"):
        fig, ax = plt.subplots(figsize=_fs(9.5, 5.0))
        ramp = RAMP[stat]
        for mode, color, label in (("ground", ramp[3], "inicio frío (estado base)"),
                                   ("excited", ramp[1], r"inicio caliente ($T=\infty$)")):
            burn_t, prod_t = eq[f"{stat}_burn_steps"], eq[f"{stat}_prod_steps"]
            stride = max(1, prod_t.size // 2000)
            t = np.concatenate([[1], burn_t, prod_t[::stride]])
            E = np.concatenate([[eq[f"{stat}_{mode}_E0"]], eq[f"{stat}_{mode}_burn_E"], eq[f"{stat}_{mode}_prod_E"][::stride]])
            ax.plot(t, E, color=color, lw=0.9, label=label)
        p = (params or {}).get(f"{stat}_params", {})
        eq_label = "E/N de equilibrio (canónico exacto)"
        if p:
            eq_label += f", T = {p['T']:g}, N = {p['N']}"
        ax.axhline(float(eq[f"{stat}_exact"]), color=INK, ls="--", lw=1.1, label=eq_label)
        nb = int(eq[f"{stat}_n_burn"])
        ax.axvline(nb, color=INK2, ls=":", lw=1.2)
        ax.text(nb * 1.1, ax.get_ylim()[1] * 0.97, r"$n_{burn}$", color=INK2, va="top")
        cross = int(eq[f"{stat}_cross_steps"])
        if cross > 0:
            ax.axvline(cross, color=MUTED, ls="-.", lw=1.0, label=f"cruce de cadenas ({cross:.1e} pasos)")
        ax.set_xscale("log")
        ax.set(xlabel="pasos de Monte Carlo (propuestas)", ylabel=r"$E/N$  [$\hbar\omega$]",
               title=NAME[stat])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)
        fig.tight_layout()
        out += save(fig, out_dir, f"fig_equilibration_{stat}")
    return out


def fig_autocorrelation(eq: dict, out_dir: Path):
    """C(t) de E/N en producción, con el tiempo de autocorrelación integrado τ_int de cada estadística."""
    fig, ax = plt.subplots(figsize=_fs(8.5, 4.2))
    for stat in ("bosons", "fermions"):
        acf, tau = eq[f"{stat}_acf"], float(eq[f"{stat}_tau"])
        lag = np.arange(acf.size)
        cut = min(acf.size, int(10 * tau) + 5)
        ax.plot(lag[:cut], acf[:cut], color=COLOR[stat],
                label=fr"{NAME[stat]}: $\tau_{{int}}$ = {tau:.1f} mediciones")
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
    ax.plot(t, np.clip(1 - t**3, 0, None), color=INK, ls="--", lw=1.4,
            label=r"teoría: $\langle N_0\rangle/N = 1-(T/T_c)^3$")
    for k, N in enumerate(n_values):
        c = RAMP["bosons"][k]
        Tc = float(an.bec_critical_temperature(N))
        ax.plot(curves[f"bosons_N{N}_T"] / Tc, curves[f"bosons_N{N}_f0"], color=c, lw=1.0)
        sel = _rows(rows, "bosons", N)
        _eb(ax, _col(sel, "T") / Tc, _col(sel, "f0"), _col(sel, "f0_err"), c, marker=MARKERS[k], label=f"N = {N}")
    ax.plot([], [], color=INK2, lw=1.0, label="canónico exacto")
    ax.set(xlabel=r"$T/T_c$,  $T_c = (N/\zeta(3))^{1/3}\,\hbar\omega/k_B$", ylabel=r"$f_0 = \langle n_0\rangle/N$",
           title="Fracción de condensado (bosones)", xlim=(0, 1.6), ylim=(-0.02, 1.02))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return save(fig, out_dir, "fig_condensate_fraction")


def fig_occupations(occ: dict, out_dir: Path):
    """Histograma del número medio de ocupación por estado ⟨n(ε)⟩: simulación (barras) y canónico exacto.

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
            eps, sim, ex = occ[f"{key}_levels"], occ[f"{key}_sim"], occ[f"{key}_exact"]
            g = (eps + 1) * (eps + 2) / 2
            keep = g * ex > 0.02  # niveles con al menos ~0.02 partículas en promedio
            xmax = int(eps[keep].max()) if keep.any() else int(eps.max())
            m = eps <= xmax
            _level_histogram(ax, eps[m], sim[m], ex[m], COLOR[stat], RAMP[stat][3])
            if stat == "fermions":
                ax.set_ylim(0, 1.08)
                eF = an.fermi_energy(N)
                ax.axvline(eF + 0.5, color=MUTED, ls=":", lw=1.0)
                ax.text(eF + 0.6, ax.get_ylim()[1] * 0.95, r"$\varepsilon_F$", color=MUTED, va="top")
            ref = r"T/T_c" if stat == "bosons" else r"T/T_F"
            ax.set_title(f"({'abcdef'[row * len(temps) + col]}) {tag}, T = {T:g}  (${ref}$ = {T / T_ref:.2f})")
            ax.set_xlabel(r"$\varepsilon$  [$\hbar\omega$]")
            if col == 0:
                ax.set_ylabel(r"número medio de ocupación  $\langle n(\varepsilon)\rangle$")
            if col == len(temps) - 1:
                ax.legend(loc="upper right")
    fig.suptitle(f"Número medio de ocupación por estado en cada nivel de energía, N = {N}", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    return save(fig, out_dir, "fig_occupations")


# ---------------------------------------------------------------------------
# Calor específico (transición lambda) y potencial químico
# ---------------------------------------------------------------------------
def fig_heat_capacity_bosons(rows: list[dict], curves: dict, n_values, out_dir: Path):
    """C/(N k_B) de los bosones frente a T/T_c: el pico que crece con N es la transición lambda."""
    fig, ax = plt.subplots(figsize=_fs(10.5, 5.0))
    t = np.concatenate([np.linspace(0.02, 1.0, 120), np.linspace(1.0005, 3.0, 160)])
    C_tl = an.bose_heat_capacity_thermodynamic(t)
    below = t <= 1.0
    ax.plot(t[below], C_tl[below], color=INK, ls="--", lw=1.5, label=r"teoría $N\to\infty$")
    ax.plot(t[~below], C_tl[~below], color=INK, ls="--", lw=1.5)
    ax.plot([1, 1], [C_tl[~below][0], C_tl[below][-1]], color=INK, ls="--", lw=0.8)
    for k, N in enumerate(n_values):
        c = RAMP["bosons"][k]
        Tc = float(an.bec_critical_temperature(N))
        ax.plot(curves[f"bosons_N{N}_T"] / Tc, curves[f"bosons_N{N}_C"], color=c, lw=1.1)
        sel = _rows(rows, "bosons", N)
        _eb(ax, _col(sel, "T") / Tc, _col(sel, "C_N"), _col(sel, "C_N_err"), c, marker=MARKERS[k], label=f"N = {N}")
    ax.axhline(3.0, color=MUTED, ls=":", lw=1.2, label=r"límite clásico $3k_B$")
    ax.axvline(1.0, color=AXIS, lw=0.8)
    ax.annotate(r"$\frac{12\zeta(4)}{\zeta(3)}\left(\frac{T}{T_c}\right)^3$", xy=(0.8, 12 * an.ZETA4 / an.ZETA3 * 0.512),
                xytext=(0.32, 8.0), fontsize=9, color=INK2, arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.8})
    ax.plot([], [], color=INK2, lw=1.1, label="canónico exacto (N finito)")
    ax.set(xlabel=r"$T/T_c$,  $T_c = (N/\zeta(3))^{1/3}\,\hbar\omega/k_B$", ylabel=r"$C_V/(N k_B)$",
           title="Calor específico de los bosones: transición lambda en la trampa armónica", xlim=(0, 3.0), ylim=(0, 11.5))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save(fig, out_dir, "fig_heat_capacity_bosons")


def fig_heat_capacity_fermions(rows: list[dict], curves: dict, n_values, out_dir: Path):
    """C/(N k_B) de los fermiones frente a T/E_F: lineal a T baja (Sommerfeld) y → 3 a T alta."""
    fig, ax = plt.subplots(figsize=_fs(10.5, 5.0))
    t = np.linspace(0.01, 2.6, 220)
    ax.plot(t, an.fermi_heat_capacity_thermodynamic(t), color=INK, ls="--", lw=1.5, label=r"teoría $N\to\infty$")
    ts = t[t < 0.3]
    ax.plot(ts, np.pi**2 * ts, color=INK2, ls="-.", lw=1.1, label=r"Sommerfeld: $\pi^2\,T/E_F$")
    for k, N in enumerate(n_values):
        c = RAMP["fermions"][k]
        EF = float(an.fermi_energy_semiclassical(N))
        ax.plot(curves[f"fermions_N{N}_T"] / EF, curves[f"fermions_N{N}_C"], color=c, lw=1.1)
        sel = _rows(rows, "fermions", N)
        _eb(ax, _col(sel, "T") / EF, _col(sel, "C_N"), _col(sel, "C_N_err"), c, marker=MARKERS[k], label=f"N = {N}")
    ax.axhline(3.0, color=MUTED, ls=":", lw=1.2, label=r"límite clásico $3k_B$")
    ax.plot([], [], color=INK2, lw=1.1, label="canónico exacto (N finito)")
    ax.set(xlabel=r"$T/E_F$,  $E_F = (6N)^{1/3}\,\hbar\omega$", ylabel=r"$C_V/(N k_B)$",
           title="Calor específico de los fermiones: crecimiento lineal y límite clásico", xlim=(0, 2.6), ylim=(0, 3.4))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, out_dir, "fig_heat_capacity_fermions")


def fig_chemical_potential_fermions(rows: list[dict], curves: dict, n_values, out_dir: Path):
    """μ(T) de los fermiones: parte de E_F, baja como 1 − (π²/3)(T/E_F)² y se vuelve negativo (régimen clásico).

    μ se mide desde el fondo de la trampa (se suma la energía de punto cero 3ħω/2) para comparar
    con el límite continuo; los puntos son el μ de un ajuste de Fermi–Dirac a ⟨n(ε)⟩ simulado.
    """
    fig, ax = plt.subplots(figsize=_fs(10.5, 5.0))
    t = np.linspace(0.01, 2.6, 220)
    ax.plot(t, an.fermi_chemical_potential_thermodynamic(t), color=INK, ls="--", lw=1.5, label=r"teoría $N\to\infty$")
    ts = t[t < 0.45]
    ax.plot(ts, 1 - np.pi**2 / 3 * ts**2, color=INK2, ls="-.", lw=1.1, label=r"Sommerfeld: $1-\frac{\pi^2}{3}(T/E_F)^2$")
    for k, N in enumerate(n_values):
        c = RAMP["fermions"][k]
        EF = float(an.fermi_energy_semiclassical(N))
        ax.plot(curves[f"fermions_N{N}_T"] / EF, (curves[f"fermions_N{N}_mu"] + 1.5) / EF, color=c, lw=1.1)
        sel = _rows(rows, "fermions", N)
        _eb(ax, _col(sel, "T") / EF, (_col(sel, "mu_ajuste") + 1.5) / EF, _col(sel, "mu_ajuste_err") / EF, c,
            marker=MARKERS[k], label=f"N = {N}")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot([], [], color=INK2, lw=1.1, label=r"Fermi–Dirac con $\langle N\rangle = N$ (niveles discretos)")
    ax.set(xlabel=r"$T/E_F$,  $E_F = (6N)^{1/3}\,\hbar\omega$",
           ylabel=r"$(\mu + \frac{3}{2}\hbar\omega)/E_F$",
           title="Potencial químico de los fermiones (ajuste de Fermi–Dirac a la ocupación simulada)", xlim=(0, 2.6))
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out_dir, "fig_chemical_potential_fermions")


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
        _plain_axis(ax)
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
