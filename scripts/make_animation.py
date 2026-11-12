"""Animación de la relajación al equilibrio.

Bosones y fermiones (N = 56, T = 1) parten de una configuración caliente (T = ∞).
La animación tiene dos partes:

1. Relajación: de t = 0 hasta n_burn, la MISMA termalización que se descarta antes de medir
   (BURN_FRACTION de los pasos, como en los barridos). Los fotogramas se espacian en escala
   logarítmica porque al principio todo cambia muy rápido.
2. Equilibrio (los últimos ~4 s): pasos posteriores a n_burn. Se muestra la configuración
   instantánea y, con puntos, el promedio acumulado desde n_burn, que se acerca al canónico exacto.

Salidas: data/raw/animation.npz, fig/09_animacion_relajacion_al_equilibrio_N56.gif
Uso:     python scripts/make_animation.py [--fast] [--from-raw]
"""

from __future__ import annotations

import _common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src import analytics as an
from src import plotting as pl
from src.config import BURN_FRACTION, make_seed, output_dirs, steps_for
from src.datastore import load_raw, save_raw
from src.metropolis import run_mcmc_chain
from src.model import level_degeneracy

N, T = 56, 1.0
FPS = 12
RELAX_FRAMES = 96          # ~8 s de relajación
EQUILIBRIUM_FRAMES = 48    # ~4 s finales en equilibrio
BOSON_EPS_MAX = 3          # niveles que se muestran en el panel de bosones
RECORD_POINTS = 8000       # registros por fase (se eligen los fotogramas entre ellos)


def simulate(raw_dir, fast: bool) -> None:
    arrays, meta = {}, {}
    for s_code, stat in enumerate(("bosons", "fermions")):
        q = an.choose_q_max(N, T, stat)
        n_burn = int(BURN_FRACTION * steps_for(N, T, stat, fast))
        n_eq = n_burn  # fase de equilibrio tan larga como la termalización
        burn_interval = max(1, n_burn // RECORD_POINTS)
        eq_interval = max(1, n_eq // RECORD_POINTS)
        seed = make_seed(50, s_code)
        res = run_mcmc_chain(N, 1.0 / T, q, stat, n_eq, n_burn, eq_interval, seed, "excited", burn_in_interval=burn_interval)
        burn, prod = res["burn_level_trace"], res["level_trace"]
        # Relajación: fotogramas logarítmicos hasta n_burn (el último es justo n_burn).
        i_b = np.unique(np.geomspace(1, burn.shape[0], RELAX_FRAMES).astype(int) - 1)
        # Equilibrio: fotogramas lineales después de n_burn y promedio acumulado desde n_burn.
        i_e = np.unique(np.linspace(0, prod.shape[0] - 1, EQUILIBRIUM_FRAMES).astype(int))
        running = np.cumsum(prod, axis=0) / np.arange(1, prod.shape[0] + 1)[:, None]
        arrays[f"{stat}_frames"] = np.concatenate([burn[i_b], prod[i_e]]).astype(np.int16)
        arrays[f"{stat}_running"] = np.concatenate([np.full((i_b.size, burn.shape[1]), np.nan), running[i_e]])
        arrays[f"{stat}_steps"] = np.concatenate([(i_b + 1) * burn_interval, n_burn + (i_e + 1) * eq_interval])
        arrays[f"{stat}_energy"] = np.concatenate([res["burn_E_trace"][i_b], res["E_trace"][i_e]]) / N
        arrays[f"{stat}_n_relax"] = np.array(i_b.size)
        arrays[f"{stat}_n_burn"] = np.array(n_burn)
        ex = an.exact_canonical(N, q, 1.0 / T, stat)
        arrays[f"{stat}_exact"] = ex["occupation"]
        arrays[f"{stat}_E_exact"] = np.array(ex["energy_per_particle"])
        meta[stat] = {"q_max": q, "seed": seed, "n_burn": n_burn, "n_eq": n_eq}
        print(f"{stat}: n_burn = {n_burn:.1e}; E/N al final = {res['E_trace'][-1] / N:.3f} "
              f"(promedio en equilibrio {res['E_trace'].mean() / N:.3f}, exacto {ex['energy_per_particle']:.3f})")
    save_raw(raw_dir / "animation.npz", arrays, {"N": N, "T": T, **meta})


def animate(raw_dir, fig_dir) -> None:
    data, meta = load_raw(raw_dir / "animation.npz")
    pl.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    artists = []
    n_frames = min(data["bosons_frames"].shape[0], data["fermions_frames"].shape[0])
    for ax, stat in zip(axes, ("bosons", "fermions")):
        exact = data[f"{stat}_exact"]
        eps = np.arange(exact.size)
        g = level_degeneracy(eps)
        ax.step(np.append(eps, eps[-1] + 1) - 0.5, np.append(exact, exact[-1]), where="post",
                color=pl.INK, lw=1.2, label="promedio canónico exacto")
        bars = ax.bar(eps, data[f"{stat}_frames"][0] / g, color=pl.COLOR[stat], width=0.8, alpha=0.85,
                      label="configuración instantánea")
        (avg,) = ax.plot([], [], ls="none", marker="o", ms=4.5, color=pl.INK, mfc="white",
                         label="promedio acumulado desde $n_{burn}$")
        top = max(float((data[f"{stat}_frames"] / g).max()), float(exact.max()))
        ax.set_ylim(0, 1.15 * top if stat == "bosons" else 1.08)
        # Bosones: sólo los niveles ε = 0 … BOSON_EPS_MAX, donde está casi toda la ocupación.
        ax.set_xlim(-0.6, (BOSON_EPS_MAX if stat == "bosons" else eps[-1]) + 0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl._plain_axis(ax)
        if stat == "fermions":
            ax.axvline(an.fermi_energy(N) + 0.5, color=pl.MUTED, ls=":", lw=1.0)
        ax.set(xlabel=r"$\varepsilon$  [$\hbar\omega$]", ylabel=r"número de ocupación por estado  $n(\varepsilon)$",
               title=f"{pl.NAME[stat]}, N = {N}, T = {T:g}")
        ax.legend(loc="upper right", fontsize=7)
        txt = ax.text(0.98, 0.70, "", transform=ax.transAxes, color=pl.INK2, ha="right", fontsize=8)
        phase = ax.text(0.98, 0.62, "", transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold")
        artists.append((stat, bars, avg, g, eps, txt, phase))
    fig.tight_layout()

    def update(k):
        out = []
        for stat, bars, avg, g, eps, txt, phase in artists:
            for b, h in zip(bars, data[f"{stat}_frames"][k] / g):
                b.set_height(h)
            n_burn = int(data[f"{stat}_n_burn"])
            step = int(data[f"{stat}_steps"][k])
            txt.set_text(f"paso {step:,}   E/N = {data[f'{stat}_energy'][k]:.2f}"
                         f"  (exacto {float(data[f'{stat}_E_exact']):.2f})")
            if k < int(data[f"{stat}_n_relax"]):
                phase.set_text(f"relajación: {step / n_burn:.0%} de n_burn")
                phase.set_color(pl.INK2)
                avg.set_data([], [])
            else:
                phase.set_text("EQUILIBRIO (después de n_burn)")
                phase.set_color("#0a7d3b")
                avg.set_data(eps, data[f"{stat}_running"][k] / g)
            out += list(bars) + [avg, txt, phase]
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "09_animacion_relajacion_al_equilibrio_N56.gif"
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=90)
    plt.close(fig)
    print(f"Animación guardada en {path} ({n_frames} fotogramas, {n_frames / FPS:.0f} s; "
          f"últimos {EQUILIBRIUM_FRAMES / FPS:.0f} s en equilibrio)")


def main(fast: bool = False, from_raw: bool = False) -> None:
    raw_dir, fig_dir = output_dirs(fast)
    if not (from_raw and (raw_dir / "animation.npz").exists()):
        simulate(raw_dir, fast)
    animate(raw_dir, fig_dir)


if __name__ == "__main__":
    parser = _common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--from-raw", action="store_true", help="Reusar data/raw/animation.npz sin simular.")
    args = parser.parse_args()
    main(fast=args.fast, from_raw=args.from_raw)
