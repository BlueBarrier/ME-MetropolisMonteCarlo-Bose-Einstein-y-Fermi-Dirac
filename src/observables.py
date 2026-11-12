"""Observables físicos a partir de las trazas de una cadena de Markov.

- Energía media por partícula ⟨E⟩/N.
- Fracción en el estado fundamental f_0 = ⟨n_0⟩/N (condensado en el caso de bosones).
- Ocupación media por estado de cada nivel: ⟨n(ε)⟩ = ⟨N(ε)⟩ / g(ε), donde N(ε) es el
  número de partículas en los g(ε) estados degenerados del nivel ε.

Cada valor va con su error estándar corregido por autocorrelación (τ_int).
"""

from __future__ import annotations

import numpy as np

from .model import level_degeneracy
from .stats import discrepancy_in_sigmas, statistical_uncertainty


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

    n0 = np.asarray(level_trace[:, 0], dtype=float)
    f0_mean, f0_tau, f0_err = statistical_uncertainty(n0 / N, resolution=1.0 / N)

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
        "tau_int_energy": e_tau,
        "condensate_fraction": f0_mean,
        "condensate_fraction_err": f0_err,
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


def heat_capacity(E_trace: np.ndarray, T: float, N: int) -> tuple[float, float]:
    """C/(N k_B) por fluctuaciones de la energía en el ensemble canónico:

        C = (⟨E²⟩ − ⟨E⟩²) / (k_B T²).

    El error se obtiene tratando (E − Ē)² como una serie más (con su τ_int).
    """
    E = np.asarray(E_trace, dtype=float)
    dE2 = (E - E.mean()) ** 2
    var, _, var_err = statistical_uncertainty(dE2, resolution=1.0)
    var *= E.size / (E.size - 1)  # varianza insesgada
    return float(var / (T**2 * N)), float(var_err / (T**2 * N))


def fit_chemical_potential(levels: np.ndarray, occupation: np.ndarray, occupation_err: np.ndarray,
                           T: float, statistics: str = "fermions") -> tuple[float, float]:
    """μ efectivo: ajuste de Fermi–Dirac (o Bose–Einstein) a ⟨n(ε)⟩ simulado, por mínimos cuadrados
    ponderados con σ de cada nivel. Devuelve (μ, σ_μ)."""
    from scipy.optimize import least_squares
    from scipy.special import expit

    eps = np.asarray(levels, dtype=float)
    n = np.asarray(occupation, dtype=float)
    err = np.asarray(occupation_err, dtype=float)
    err = np.where(err > 0, err, max(float(err[err > 0].min()) if np.any(err > 0) else 1e-6, 1e-9))
    beta = 1.0 / T
    sign = 1.0 if statistics == "fermions" else -1.0

    def model(mu):
        if statistics == "fermions":
            return expit(-beta * (eps - mu))
        return 1.0 / np.expm1(np.clip(beta * (eps - mu), 1e-12, None))

    mu0 = float(eps[np.argmin(np.abs(n - 0.5))]) if statistics == "fermions" else -T
    hi = None if sign > 0 else -1e-9
    res = least_squares(lambda m: (model(m[0]) - n) / err, x0=[mu0 if hi is None else min(mu0, hi)],
                        bounds=([-np.inf], [np.inf if hi is None else hi]))
    J = res.jac
    dof = max(eps.size - 1, 1)
    chi2_red = float(np.sum(res.fun**2) / dof)
    cov = np.linalg.pinv(J.T @ J) * max(chi2_red, 1.0)
    return float(res.x[0]), float(np.sqrt(cov[0, 0]))
