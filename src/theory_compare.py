"""Comparación de la simulación con datos teóricos ingresados por el usuario.

El usuario elige la estadística, N, el corte energético q_max (impuesto por él) y un observable,
y proporciona sus valores teóricos como pares (x, y):

- ``"energy"``      : x = T (reducida, ħω/k_B), y = ⟨E⟩/N  → una simulación por cada T.
- ``"condensate"``  : x = T,                    y = f_0 = ⟨n_0⟩/N → una simulación por cada T.
- ``"occupation"``  : x = ε (nivel, entero),    y = ⟨n(ε)⟩ por estado, a una T fija → una simulación.

Para cada punto se calcula la discrepancia z = (A_sim − A_teo)/σ_sim y el error porcentual.
Los datos se leen de un CSV de dos columnas (se ignoran el encabezado y las líneas con #)
o de texto con pares "x y" o "x,y" separados por ";" o saltos de línea.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import analytics as an
from .config import BURN_FRACTION
from .metropolis import run_mcmc_chain
from .model import number_of_states
from .observables import compute_observables

OBSERVABLES = {
    "energy": {"x": r"$T$  [$\hbar\omega/k_B$]", "y": r"$\langle E\rangle/N$  [$\hbar\omega$]", "name": "energia"},
    "condensate": {"x": r"$T$  [$\hbar\omega/k_B$]", "y": r"$f_0 = \langle n_0\rangle/N$", "name": "fraccion_fundamental"},
    "occupation": {"x": r"$\varepsilon$  [$\hbar\omega$]", "y": r"$\langle n(\varepsilon)\rangle$ por estado", "name": "ocupacion"},
}
N_SAMPLES = 20_000


def parse_pairs(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Convierte texto con pares "x y" / "x,y" (separados por ';' o líneas) en dos arreglos."""
    xs, ys = [], []
    for chunk in re.split(r"[;\n]", text):
        chunk = chunk.split("#", 1)[0].strip()
        if not chunk:
            continue
        parts = [p for p in re.split(r"[,\s\t]+", chunk) if p]
        if len(parts) < 2:
            raise ValueError(f"Par incompleto: {chunk!r}")
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue  # encabezado
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError("No se encontraron pares (x, y) numéricos.")
    return np.array(xs), np.array(ys)


def read_theory_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lee un archivo CSV/TXT de dos columnas (x, y)."""
    return parse_pairs(Path(path).read_text(encoding="utf-8"))


def truncation_estimate(N: int, q_max: int, T: float, statistics: str) -> float:
    """Error de truncamiento exacto en ⟨E⟩/N por usar q_max en vez de un corte "infinito"."""
    return an.truncation_error(N, q_max, 1.0 / T, statistics)


def compare_with_theory(
    statistics: str,
    N: int,
    q_max: int,
    observable: str,
    theory_x,
    theory_y,
    T: float | None = None,
    n_steps: int = 20_000_000,
    seed: int = 12345,
    proposal: str = "particle",
    verbose: bool = True,
) -> dict:
    """Corre la simulación con el q_max dado y la compara punto por punto con la teoría del usuario.

    La semilla de la k-ésima simulación es ``seed + k``; todas quedan en el resultado.
    """
    if observable not in OBSERVABLES:
        raise ValueError(f"observable debe ser uno de {tuple(OBSERVABLES)}")
    if statistics == "fermions" and number_of_states(q_max) <= N:
        raise ValueError(f"Con fermiones se necesita M > N: q_max = {q_max} da {number_of_states(q_max)} estados.")
    theory_x, theory_y = np.asarray(theory_x, float), np.asarray(theory_y, float)
    n_steps = max(int(n_steps) // N_SAMPLES * N_SAMPLES, N_SAMPLES)

    sim_x, sim_y, sim_err, seeds, notes = [], [], [], [], []
    if observable == "occupation":
        if T is None:
            raise ValueError("Para la ocupación se necesita una temperatura T.")
        res = run_mcmc_chain(N, 1.0 / T, q_max, statistics, n_steps, int(BURN_FRACTION * n_steps), n_steps // N_SAMPLES,
                             seed, "ground", proposal)
        obs = compute_observables(res["E_trace"], res["level_trace"], N)
        seeds.append(seed)
        for x in theory_x:
            lvl = int(round(x))
            if not 0 <= lvl <= q_max:
                raise ValueError(f"El nivel ε = {x:g} está fuera del corte q_max = {q_max}.")
            sim_x.append(lvl)
            sim_y.append(obs["occupation"][lvl])
            sim_err.append(obs["occupation_err"][lvl])
        notes.append(f"T = {T:g}: error de truncamiento en E/N ≈ {truncation_estimate(N, q_max, T, statistics):+.2e}")
    else:
        key = "energy_per_particle" if observable == "energy" else "condensate_fraction"
        for k, Tk in enumerate(theory_x):
            if Tk <= 0:
                raise ValueError("Las temperaturas deben ser positivas.")
            s = seed + k
            res = run_mcmc_chain(N, 1.0 / Tk, q_max, statistics, n_steps, int(BURN_FRACTION * n_steps), n_steps // N_SAMPLES,
                                 s, "ground", proposal)
            obs = compute_observables(res["E_trace"], res["level_trace"], N)
            sim_x.append(Tk)
            sim_y.append(obs[key])
            sim_err.append(obs[f"{key}_err"])
            seeds.append(s)
            trunc = truncation_estimate(N, q_max, Tk, statistics)
            notes.append(f"T = {Tk:g}: error de truncamiento en E/N ≈ {trunc:+.2e}")
            if verbose:
                print(f"  T = {Tk:g}: simulación {sim_y[-1]:.5g} ± {sim_err[-1]:.2g}  (teoría {theory_y[k]:.5g}, semilla {s})")

    sim_x, sim_y, sim_err = np.array(sim_x, float), np.array(sim_y), np.array(sim_err)
    safe_err = np.where(sim_err > 0, sim_err, np.nan)
    z = (sim_y - theory_y) / safe_err
    with np.errstate(divide="ignore", invalid="ignore"):
        err_pct = 100.0 * np.abs(sim_y - theory_y) / np.abs(theory_y)
    return {
        "statistics": statistics, "N": int(N), "q_max": int(q_max), "observable": observable, "T": T,
        "proposal": proposal, "n_steps": n_steps, "seeds": seeds,
        "theory_x": theory_x, "theory_y": theory_y,
        "sim_x": sim_x, "sim_y": sim_y, "sim_err": sim_err, "z": z, "error_pct": err_pct,
        "x_label": OBSERVABLES[observable]["x"], "y_label": OBSERVABLES[observable]["y"],
        "truncation_notes": notes,
    }


def result_tag(result: dict) -> str:
    """Nombre de archivo descriptivo para una comparación."""
    tag = f"comparacion_teoria_{OBSERVABLES[result['observable']]['name']}_{'bosones' if result['statistics'] == 'bosons' else 'fermiones'}"
    tag += f"_N{result['N']}_qmax{result['q_max']}"
    if result["observable"] == "occupation":
        tag += f"_T{result['T']:g}"
    return tag + f"_semilla{result['seeds'][0]}"
