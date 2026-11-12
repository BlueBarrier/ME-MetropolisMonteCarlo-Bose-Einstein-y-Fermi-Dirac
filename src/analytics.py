"""Resultados exactos y límites conocidos para validar la simulación.

1. N = 1 en la trampa infinita: ⟨E⟩ = 3 / (e^β - 1). Aquí las dos estadísticas coinciden.
2. Enumeración directa (fuerza bruta) para N ≤ 4 con corte finito.
3. Solución canónica exacta para cualquier N mediante funciones generatrices.
   La función de partición canónica de N partículas es el coeficiente de t^N en

       Ξ(t) = Π_ε f_ε(t),   f_ε(t) = (1 + x_ε t)^{g(ε)}   (fermiones)
                            f_ε(t) = (1 - x_ε t)^{-g(ε)}  (bosones),   x_ε = e^{-βε}.

   La ocupación media de un estado del nivel ε es ⟨n⟩ = x ∂ ln Z_N / ∂x, y da

       ⟨n(ε)⟩ = x_ε Z_{N-1}[g(ε) → g(ε) ∓ 1] / Z_N     (− fermiones, + bosones).

   Todos los coeficientes son positivos y se trabaja con logaritmos, así que el
   cálculo es numéricamente estable a cualquier temperatura (no hay cancelaciones).
4. Ensemble gran canónico: distribuciones de Fermi–Dirac y Bose–Einstein con μ
   fijado por ⟨N⟩ = N (esto permite comparar entre ensembles).
5. Límite termodinámico: T_c = (N/ζ(3))^{1/3}, su corrección de tamaño finito y
   la energía de Fermi.
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, gammaln, zeta
from scipy.special import gamma as gamma_fn

from .model import STATISTICS, fermi_level, ground_state_energy, level_degeneracy, number_of_states

ZETA2 = float(zeta(2))
ZETA3 = float(zeta(3))
# Coeficiente de la corrección de tamaño finito (Grossmann y Holthaus 1995):
# ΔT_c/T_c ≈ -C_FS N^{-1/3},  C_FS = ζ(2) / (2 ζ(3)^{2/3}) ≈ 0.7275.
C_FS = ZETA2 / (2.0 * ZETA3 ** (2.0 / 3.0))


# ---------------------------------------------------------------------------
# 1. Una partícula
# ---------------------------------------------------------------------------
def exact_single_particle_energy(beta):
    """⟨E⟩ exacta de N = 1 en la trampa infinita: 3/(e^β - 1)."""
    return 3.0 / np.expm1(beta)


def exact_single_particle_energy_truncated(beta: float, q_max: int) -> float:
    """⟨E⟩ de N = 1 con el corte ε ≤ q_max: Σ g ε e^{-βε} / Σ g e^{-βε}."""
    eps = np.arange(q_max + 1)
    w = level_degeneracy(eps) * np.exp(-beta * eps)
    return float(np.sum(eps * w) / np.sum(w))


def classical_energy_per_particle(beta):
    """E/N de partículas distinguibles (Maxwell–Boltzmann) en la trampa: 3/(e^β - 1).

    Es la referencia "clásica" (sin efectos de intercambio). Tiende a 3T si T ≫ 1.
    """
    return exact_single_particle_energy(beta)


# ---------------------------------------------------------------------------
# 2. Enumeración directa (fuerza bruta)
# ---------------------------------------------------------------------------
def exact_canonical_enumeration(N: int, energies: np.ndarray, beta: float, statistics: str) -> dict:
    """Recorre todas las configuraciones con Σ n_i = N (sólo para sistemas pequeños).

    Devuelve Z_N (relativa al estado base), ⟨E⟩, ⟨n_i⟩ por estado y el número de
    configuraciones. Se usa para comprobar el método de funciones generatrices.
    """
    _check(statistics)
    energies = np.asarray(energies)
    M = energies.size
    gen = combinations_with_replacement if statistics == "bosons" else combinations
    configs = np.array(list(gen(range(M), N)), dtype=np.int64)  # índices de los estados ocupados
    if configs.shape[0] > 5_000_000:
        raise ValueError("Demasiadas configuraciones para enumerar; reduzca N o q_max.")
    E = energies[configs].sum(axis=1).astype(float)
    E_min = E.min()
    w = np.exp(-beta * (E - E_min))
    Z = w.sum()
    occ = np.zeros(M)
    for col in range(N):
        np.add.at(occ, configs[:, col], w)
    return {
        "Z_relative": float(Z),
        "log_Z": float(np.log(Z) - beta * E_min),
        "mean_energy": float(np.sum(w * E) / Z),
        "occupation_per_state": occ / Z,
        "n_configurations": int(configs.shape[0]),
    }


# ---------------------------------------------------------------------------
# 3. Solución canónica exacta (funciones generatrices en espacio logarítmico)
# ---------------------------------------------------------------------------
def _logsumexp_rows(A: np.ndarray) -> np.ndarray:
    m = A.max(axis=1)
    finite = np.isfinite(m)
    out = np.full(A.shape[0], -np.inf)
    out[finite] = m[finite] + np.log(np.exp(A[finite] - m[finite, None]).sum(axis=1))
    return out


def _log_poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Producto de dos polinomios dados por log-coeficientes, truncado al grado len(a)-1."""
    K = a.size
    k = np.arange(K)[:, None]
    m = np.arange(K)[None, :]
    d = k - m
    B = np.where(d >= 0, b[np.clip(d, 0, None)], -np.inf)
    return _logsumexp_rows(a[None, :] + B)


def _log_binom(n, k):
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def _level_log_coefficients(eps: int, g: int, beta: float, N: int, statistics: str) -> np.ndarray:
    """log de los coeficientes de t^k (k = 0..N) del factor f_ε(t) de un nivel con g estados."""
    k = np.arange(N + 1, dtype=float)
    c = np.full(N + 1, -np.inf)
    if g == 0:
        c[0] = 0.0
        return c
    if statistics == "fermions":
        valid = k <= g
        c[valid] = _log_binom(float(g), k[valid]) - beta * eps * k[valid]
    else:
        c[:] = _log_binom(k + g - 1.0, k) - beta * eps * k
    return c


def canonical_log_partition_functions(N: int, q_max: int, beta: float, statistics: str) -> np.ndarray:
    """ln Z_k para k = 0, ..., N (ensemble canónico, corte ε ≤ q_max)."""
    _check(statistics)
    P = np.full(N + 1, -np.inf)
    P[0] = 0.0
    for eps in range(q_max + 1):
        g = int(level_degeneracy(eps))
        P = _log_poly_mul(P, _level_log_coefficients(eps, g, beta, N, statistics))
    return P


def exact_canonical(N: int, q_max: int, beta: float, statistics: str) -> dict:
    """Solución canónica exacta para N partículas con corte q_max.

    Returns
    -------
    dict con ``levels`` (ε = 0..q_max), ``degeneracy`` g(ε), ``occupation``
    ⟨n(ε)⟩ por estado, ``level_population`` g⟨n⟩, ``mean_energy`` ⟨E⟩,
    ``energy_per_particle`` ⟨E⟩/N, ``n0`` = ⟨n_0⟩, ``condensate_fraction`` ⟨n_0⟩/N
    y ``log_Z``.
    """
    _check(statistics)
    if statistics == "fermions" and number_of_states(q_max) < N:
        raise ValueError("Hay menos estados que fermiones: aumente q_max.")
    levels = np.arange(q_max + 1)
    g = level_degeneracy(levels).astype(int)
    L = levels.size
    coeffs = [_level_log_coefficients(e, gi, beta, N, statistics) for e, gi in zip(levels, g)]
    unit = np.full(N + 1, -np.inf)
    unit[0] = 0.0
    prefix = [unit]
    for c in coeffs:
        prefix.append(_log_poly_mul(prefix[-1], c))
    suffix = [unit] * (L + 1)
    for l in range(L - 1, -1, -1):
        suffix[l] = _log_poly_mul(coeffs[l], suffix[l + 1])
    log_Z = prefix[L]

    shift = -1 if statistics == "fermions" else +1
    occupation = np.empty(L)
    for l in range(L):
        modified = _level_log_coefficients(levels[l], g[l] + shift, beta, N, statistics)
        P = _log_poly_mul(_log_poly_mul(prefix[l], suffix[l + 1]), modified)
        occupation[l] = np.exp(-beta * levels[l] + P[N - 1] - log_Z[N])

    population = g * occupation
    mean_energy = float(np.sum(levels * population))
    return {
        "N": N,
        "beta": beta,
        "q_max": q_max,
        "statistics": statistics,
        "levels": levels,
        "degeneracy": g,
        "occupation": occupation,
        "level_population": population,
        "mean_energy": mean_energy,
        "energy_per_particle": mean_energy / N,
        "n0": float(occupation[0]),
        "condensate_fraction": float(occupation[0]) / N,
        "log_Z": float(log_Z[N]),
        "particle_number_check": float(population.sum()),
    }


def truncation_error(N: int, q_max: int, beta: float, statistics: str, q_ref: int | None = None) -> float:
    """Error de truncamiento en ⟨E⟩/N: E(q_max)/N - E(q_ref)/N (exacto, ≤ 0)."""
    q_ref = q_ref if q_ref is not None else reference_cutoff(N, 1.0 / beta, statistics)
    a = exact_canonical(N, q_max, beta, statistics)["energy_per_particle"]
    b = exact_canonical(N, q_ref, beta, statistics)["energy_per_particle"]
    return a - b


def reference_cutoff(N: int, T: float, statistics: str) -> int:
    """Corte "prácticamente infinito": la ocupación del último nivel es ~e^{-40}."""
    base = fermi_level(N) if statistics == "fermions" else 0
    return int(np.ceil(base + 3 + 40.0 * T))


def choose_q_max(N: int, T: float, statistics: str, tol: float = 1e-4, q_min: int = 4) -> int:
    """Corte mínimo tal que el error de truncamiento en ⟨E⟩/N sea ≤ tol·T.

    Usa la solución canónica exacta como referencia y una búsqueda binaria,
    porque ⟨E⟩ crece de forma monótona con q_max. Así el error de truncamiento queda
    muy por debajo del error estadístico y se distingue del efecto de tamaño finito.
    """
    beta = 1.0 / T
    q_ref = reference_cutoff(N, T, statistics)
    E_ref = exact_canonical(N, q_ref, beta, statistics)["energy_per_particle"]
    lo = max(q_min, (fermi_level(N) + 2) if statistics == "fermions" else q_min)
    if lo >= q_ref:
        return lo
    hi = q_ref

    def ok(q):
        return abs(exact_canonical(N, q, beta, statistics)["energy_per_particle"] - E_ref) <= tol * T

    if ok(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# 4. Ensemble gran canónico (Fermi–Dirac / Bose–Einstein)
# ---------------------------------------------------------------------------
def grand_canonical_occupation(eps, beta: float, mu: float, statistics: str):
    """⟨n(ε)⟩ = 1 / (e^{β(ε-μ)} ∓ 1)   (− Bose–Einstein, + Fermi–Dirac)."""
    _check(statistics)
    x = beta * (np.asarray(eps, dtype=float) - mu)
    if statistics == "fermions":
        return expit(-x)
    return 1.0 / np.expm1(x)


def grand_canonical_particle_number(mu: float, beta: float, q_max: int, statistics: str) -> float:
    eps = np.arange(q_max + 1)
    return float(np.sum(level_degeneracy(eps) * grand_canonical_occupation(eps, beta, mu, statistics)))


def solve_chemical_potential(N: int, beta: float, q_max: int, statistics: str) -> float:
    """μ tal que ⟨N⟩_GC = N con el mismo espectro truncado."""
    _check(statistics)
    if statistics == "bosons":
        # μ < ε_0 = 0. Se resuelve en s = ln(-βμ) para resolver bien el régimen μ → 0⁻.
        def f(s):
            return grand_canonical_particle_number(-np.exp(s) / beta, beta, q_max, statistics) - N

        s = brentq(f, np.log(1e-13), np.log(200.0), xtol=1e-14, rtol=1e-14, maxiter=500)
        return float(-np.exp(s) / beta)

    def h(mu):
        return grand_canonical_particle_number(mu, beta, q_max, statistics) - N

    if number_of_states(q_max) <= N:
        raise ValueError("Con fermiones se necesita M > N.")
    return float(brentq(h, -60.0 / beta - 1.0, q_max + 60.0 / beta + 1.0, xtol=1e-13, maxiter=500))


def grand_canonical_observables(N: int, q_max: int, beta: float, statistics: str) -> dict:
    """Observables del ensemble gran canónico con ⟨N⟩ = N (mismo corte)."""
    mu = solve_chemical_potential(N, beta, q_max, statistics)
    levels = np.arange(q_max + 1)
    g = level_degeneracy(levels)
    occ = grand_canonical_occupation(levels, beta, mu, statistics)
    E = float(np.sum(g * levels * occ))
    return {
        "mu": mu,
        "levels": levels,
        "degeneracy": g,
        "occupation": occ,
        "mean_energy": E,
        "energy_per_particle": E / N,
        "n0": float(occ[0]),
        "condensate_fraction": float(occ[0]) / N,
    }


# ---------------------------------------------------------------------------
# 5. Límite termodinámico
# ---------------------------------------------------------------------------
def bec_critical_temperature(N):
    """T_c = (N / ζ(3))^{1/3} (unidades de ħω/k_B), límite N → ∞."""
    return (np.asarray(N, dtype=float) / ZETA3) ** (1.0 / 3.0)


def bec_critical_temperature_finite_size(N):
    """T_c con la primera corrección de tamaño finito: T_c (1 - C_FS N^{-1/3})."""
    N = np.asarray(N, dtype=float)
    return bec_critical_temperature(N) * (1.0 - C_FS * N ** (-1.0 / 3.0))


def condensate_fraction_thermodynamic(T, N):
    """N_0/N = 1 - (T/T_c)^3 para T < T_c (0 en otro caso)."""
    t = np.asarray(T, dtype=float) / bec_critical_temperature(N)
    return np.clip(1.0 - t**3, 0.0, None)


def condensate_fraction_finite_size(T, N):
    """N_0/N ≈ 1 - t^3 - (3ζ(2)/(2ζ(3)^{2/3})) t^2 N^{-1/3},  t = T/T_c."""
    N = np.asarray(N, dtype=float)
    t = np.asarray(T, dtype=float) / bec_critical_temperature(N)
    return np.clip(1.0 - t**3 - 3.0 * C_FS * t**2 * N ** (-1.0 / 3.0), 0.0, None)


def fermi_energy(N: int) -> int:
    """Energía de Fermi de N fermiones en el estado base (nivel ε_F más alto ocupado)."""
    return fermi_level(N)


def ground_state_energy_per_particle(N: int, statistics: str) -> float:
    """E_0/N a T = 0: 0 para bosones; mar de Fermi lleno para fermiones (→ 3E_F/4 si N ≫ 1)."""
    return ground_state_energy(N, statistics) / N


def fermi_energy_semiclassical(N):
    """E_F ≈ (6N)^{1/3} ħω (aproximación continua, N ≫ 1)."""
    return (6.0 * np.asarray(N, dtype=float)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# 6. Límite termodinámico semiclásico (para comparar con experimentos, N ~ 10⁴–10⁶)
# ---------------------------------------------------------------------------
# Densidad de estados de la trampa g(ε) = ε²/[2(ħω̄)³], con ω̄ la media geométrica de las
# frecuencias. Así los resultados valen también en trampas anisótropas.
def fermi_dirac_integral(j: float, eta: float) -> float:
    """F_j(η) = ∫₀^∞ x^j / (e^{x-η} + 1) dx."""
    def f(x):
        return x**j * expit(eta - x)

    a = max(eta, 0.0)
    head = quad(f, 0.0, a, limit=200)[0] if a > 0 else 0.0
    return head + quad(f, a, np.inf, limit=200)[0]


def bose_einstein_function(n: float, z: float) -> float:
    """g_n(z) = Σ_k z^k / k^n para 0 < z ≤ 1 (polilogaritmo)."""
    if z >= 1.0:
        return float(zeta(n))
    integrand = lambda x: x ** (n - 1) * z * np.exp(-x) / (1.0 - z * np.exp(-x))  # noqa: E731
    return quad(integrand, 0.0, np.inf, limit=200)[0] / float(gamma_fn(n))


def fermi_excess_energy_thermodynamic(t):
    """Exceso de energía δU/U_cl = U/(3NkT) − 1 del gas de Fermi ideal en trampa, t = T/T_F.

    k_B T_F = ħω̄ (6N)^{1/3}. Con η = μ/k_BT: N fija F_2(η) = 1/(3t³) y U/(3NkT) = F_3(η) / [3 F_2(η)].
    Es la curva teórica de DeMarco y Jin (1999), Fig. 4.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    for k, tk in enumerate(t):
        target = 1.0 / (3.0 * tk**3)
        eta = brentq(lambda e: fermi_dirac_integral(2, e) - target, -60.0, 1.0 / tk + 20.0, xtol=1e-12)
        out[k] = fermi_dirac_integral(3, eta) / (3.0 * fermi_dirac_integral(2, eta)) - 1.0
    return out


def bose_release_energy_thermodynamic(t):
    """Energía cinética (de liberación) por partícula del gas de Bose ideal, en unidades de k_B T_0.

    t = T/T_0 con k_B T_0 = ħω̄ (N/ζ(3))^{1/3}. Por el teorema del virial la energía cinética es
    la mitad de la total: E_cin/(N k T_0) = (3/2) t⁴ g_4(z)/ζ(3), con g_3(z) = ζ(3)/t³ si t > 1
    y z = 1 (condensado) si t ≤ 1. A T alta tiende a la recta clásica (3/2) t.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    for k, tk in enumerate(t):
        if tk <= 1.0:
            z = 1.0
        else:
            target = ZETA3 / tk**3
            s = brentq(lambda s: bose_einstein_function(3, np.exp(-np.exp(s))) - target, -30.0, 5.0, xtol=1e-12)
            z = float(np.exp(-np.exp(s)))
        out[k] = 1.5 * tk**4 * bose_einstein_function(4, z) / ZETA3
    return out


def _check(statistics: str) -> None:
    if statistics not in STATISTICS:
        raise ValueError(f"statistics debe ser uno de {STATISTICS}")
