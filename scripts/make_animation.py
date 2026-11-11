"""Animación de la relajación al equilibrio (para la presentación).

Bosones y fermiones (N = 35, T = 1) parten de una configuración caliente (T = ∞).
Se anima la ocupación por estado de cada nivel, que se va acercando a la solución
canónica exacta: los bosones se condensan en ε = 0 y los fermiones llenan el mar
de Fermi hasta ε_F.

Salidas: data/raw/animation.npz, figures/animation_equilibration.gif
Uso:     python scripts/make_animation.py [--fast] [--from-raw]
"""

from __future__ import annotations

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from src import analytics as an  # noqa: E402
from src import plotting as pl  # noqa: E402
from src.config import PRODUCTION_PROPOSAL, make_seed, output_dirs  # noqa: E402
from src.datastore import load_raw, save_raw  # noqa: E402
from src.metropolis import run_mcmc_chain  # noqa: E402
from src.model import level_degeneracy  # noqa: E402

N, T = 35, 1.0
N_FRAMES = 120


def simulate(raw_dir, fast: bool) -> None:
    arrays, meta = {}, {}
    total = 400_000 if fast else 4_000_000
    interval = 500
    for s_code, stat in enumerate(("bosons", "fermions")):
        q = an.choose_q_max(N, T, stat)
        seed = make_seed(50, s_code)
        res = run_mcmc_chain(N, 1.0 / T, q, stat, interval, total, interval, seed, "excited",
                             PRODUCTION_PROPOSAL, burn_in_interval=interval)
        levels = res["burn_level_trace"]
        # Fotogramas espaciados logarítmicamente: la relajación inicial es muy rápida.
        idx = np.unique(np.geomspace(1, levels.shape[0], N_FRAMES).astype(int) - 1)
        arrays[f"{stat}_frames"] = levels[idx].astype(np.int16)
        arrays[f"{stat}_steps"] = (idx + 1) * interval
        arrays[f"{stat}_energy"] = res["burn_E_trace"][idx] / N
        arrays[f"{stat}_exact"] = an.exact_canonical(N, q, 1.0 / T, stat)["occupation"]
        meta[stat] = {"q_max": q, "seed": seed}
    save_raw(raw_dir / "animation.npz", arrays, {"N": N, "T": T, **meta})


def animate(raw_dir, fig_dir) -> None:
    data, meta = load_raw(raw_dir / "animation.npz")
    pl.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    artists = []
    n_frames = min(data["bosons_frames"].shape[0], data["fermions_frames"].shape[0])
    for ax, stat in zip(axes, ("bosons", "fermions")):
        exact = data[f"{stat}_exact"]
        eps = np.arange(exact.size)
        g = level_degeneracy(eps)
        ax.plot(eps, exact, color=pl.INK, ls="--", lw=1.2, label="promedio canónico exacto")
        bars = ax.bar(eps, data[f"{stat}_frames"][0] / g, color=pl.COLOR[stat], width=0.8, label="configuración instantánea")
        if stat == "bosons":
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.5 * N)
        else:
            ax.set_ylim(0, 1.05)
            ax.axvline(an.fermi_energy(N), color=pl.MUTED, ls=":", lw=1.0)
        ax.set(xlabel=r"$\varepsilon$  [$\hbar\omega$]", ylabel=r"$n(\varepsilon)$ por estado",
               title=f"{pl.NAME[stat]}, N = {N}, T = {T:g}")
        ax.legend(loc="upper right")
        txt = ax.text(0.98, 0.78, "", transform=ax.transAxes, color=pl.INK2, ha="right")
        artists.append((stat, bars, g, txt))
    fig.tight_layout()

    def update(k):
        out = []
        for stat, bars, g, txt in artists:
            occ = data[f"{stat}_frames"][k] / g
            for b, h in zip(bars, occ):
                b.set_height(h if h > 0 else 1e-9)
            txt.set_text(f"paso {int(data[f'{stat}_steps'][k]):,}   E/N = {data[f'{stat}_energy'][k]:.2f}")
            out += list(bars) + [txt]
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "animation_equilibration.gif"
    anim.save(path, writer=PillowWriter(fps=12), dpi=90)
    plt.close(fig)
    print(f"Animación guardada en {path}")


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
