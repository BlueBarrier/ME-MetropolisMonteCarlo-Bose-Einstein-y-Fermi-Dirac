"""Observables físicos a partir de las trazas de una cadena de Markov.

- Energía media por partícula ⟨E⟩/N.
- Fracción en el estado fundamental f_0 = ⟨n_0⟩/N (condensado en el caso de bosones).
- Ocupación media por estado de cada nivel: ⟨n(ε)⟩ = ⟨N(ε)⟩ / g(ε), donde N(ε) es el
  número de partículas en los g(ε) estados degenerados del nivel ε.

Cada valor va con su error estándar corregido por autocorrelación (τ_int) y con
el error por bloques como comprobación cruzada.
"""

from __future__ import annotations

import numpy as np

from .model import level_degeneracy
from .stats import blocking_analysis, discrepancy_in_sigmas, statistical_uncertainty


def compute_observables(E_trace: np.ndarray, level_trace: np.ndarray, N: int) -> dict:
    """Promedios con incertidumbre a partir de las trazas de producción.

    Parameters
    ----------
    E_trace : energía total en cada medición, forma (n_samples,).
    level_trace : partículas por nivel en cada medición, forma (n_samples, q_max + 1).
    N : número de partículas.
    """
    # ``resolution`` = salto mínimo de cada observable (ver stats.statistical_uncertainty).
    E_per_N = np.asarray(E_trace, dtype=float) / N
    e_mean, e_tau, e_err = statistical_uncertainty(E_per_N, resolution=1.0 / N)
    e_block = blocking_analysis(E_per_N)["plateau_error"]

    n0 = np.asarray(level_trace[:, 0], dtype=float)
    f0_mean, f0_tau, f0_err = statistical_uncertainty(n0 / N, resolution=1.0 / N)
    f0_block = blocking_analysis(n0 / N)["plateau_error"]

    q_max = level_trace.shape[1] - 1
    g = level_degeneracy(np.arange(q_max + 1))
    occ_mean = np.empty(q_max + 1)
    occ_err = np.empty(q_max + 1)
    for eps in range(q_max + 1):
        m, _, err = statistical_uncertainty(level_trace[:, eps] / g[eps], resolution=1.0 / g[eps])
        occ_mean[eps], occ_err[eps] = m, err

    return {
        "N": int(N),
        "n_samples": int(E_per_N.size),
        "energy_per_particle": e_mean,
        "energy_per_particle_err": e_err,
        "energy_per_particle_err_blocking": e_block,
        "tau_int_energy": e_tau,
        "condensate_fraction": f0_mean,
        "condensate_fraction_err": f0_err,
        "condensate_fraction_err_blocking": f0_block,
        "tau_int_n0": f0_tau,
        "levels": np.arange(q_max + 1),
        "degeneracy": g,
        "occupation": occ_mean,
        "occupation_err": occ_err,
    }


def relative_error_percent(sim: float, reference: float) -> float:
    """Error relativo porcentual 100·|sim − ref|/|ref| (NaN si ref = 0)."""
    return float(100.0 * abs(sim - reference) / abs(reference)) if reference != 0 else float("nan")


def compare_with_reference(obs: dict, reference: dict, label: str = "exacto") -> list[dict]:
    """Tabla de comparación (discrepancia en σ y error %) de E/N y f_0 contra una referencia.

    ``reference`` es la salida de ``analytics.exact_canonical`` o de
    ``analytics.grand_canonical_observables``.
    """
    rows = []
    for key, err_key, name in (
        ("energy_per_particle", "energy_per_particle_err", "E/N"),
        ("condensate_fraction", "condensate_fraction_err", "f_0 = <n_0>/N"),
    ):
        sim, err, ref = obs[key], obs[err_key], reference[key]
        rows.append(
            {
                "observable": name,
                "simulacion": sim,
                "sigma": err,
                label: ref,
                "z": discrepancy_in_sigmas(sim, err, ref),
                "error_%": relative_error_percent(sim, ref),
            }
        )
    return rows


def occupation_residuals(obs: dict, reference_occupation: np.ndarray, min_err: float = 0.0) -> np.ndarray:
    """Residuos normalizados (⟨n(ε)⟩_sim - ⟨n(ε)⟩_ref) / σ por nivel.

    Los niveles con σ ≤ min_err (p. ej. niveles que nunca se ocupan) se marcan con NaN.
    """
    err = obs["occupation_err"]
    res = (obs["occupation"] - reference_occupation[: err.size]) / np.where(err > min_err, err, np.nan)
    return res
