"""Modelo microscópico: gas ideal en una trampa armónica isotrópica 3D.

Hamiltoniano de una partícula (energía de punto cero restada):

    h = ħω (q_x + q_y + q_z),      q_x, q_y, q_z ∈ {0, 1, 2, ...}

Hamiltoniano de N partículas no interactuantes en representación de ocupaciones:

    H(n) = Σ_i n_i ε_i,      Σ_i n_i = N,
    n_i ∈ {0, 1, ..., N} (bosones),   n_i ∈ {0, 1} (fermiones, una componente de espín).

Unidades reducidas: ħω = k_B = 1. Así ε_i es un entero, T se mide en unidades de
ħω/k_B y β = 1/T. Las funciones de conversión de este módulo pasan del sistema
internacional (ω en rad/s, T en K, E en J) al reducido y viceversa.

Ensemble canónico: N, T y ω fijos. Cada microestado es un vector de ocupaciones n
con peso P(n) ∝ exp[-β H(n)] (las partículas son indistinguibles, así que cada
vector de ocupaciones es un único microestado).
"""

from __future__ import annotations

import numpy as np
from scipy import constants

HBAR = constants.hbar  # J·s
K_B = constants.k  # J/K

STATISTICS = ("bosons", "fermions")
INIT_MODES = ("ground", "excited")


# ---------------------------------------------------------------------------
# Unidades: sistema internacional <-> unidades reducidas (ħω = k_B = 1)
# ---------------------------------------------------------------------------
def reduced_temperature(T_kelvin: float, omega: float) -> float:
    """Temperatura reducida T* = k_B T / (ħω). ``omega`` en rad/s."""
    return K_B * T_kelvin / (HBAR * omega)


def kelvin_temperature(T_reduced: float, omega: float) -> float:
    """Temperatura en kelvin a partir de la reducida: T = T* ħω / k_B."""
    return T_reduced * HBAR * omega / K_B


def energy_to_joules(energy_reduced, omega: float):
    """Energía en J a partir de la reducida: E = E* ħω."""
    return np.asarray(energy_reduced, dtype=float) * HBAR * omega


def energy_to_reduced(energy_joules, omega: float):
    """Energía reducida E* = E / (ħω)."""
    return np.asarray(energy_joules, dtype=float) / (HBAR * omega)


def beta_from_temperature(T_reduced: float) -> float:
    """β = 1 / T en unidades reducidas."""
    if T_reduced <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    return 1.0 / T_reduced


# ---------------------------------------------------------------------------
# Espectro de una partícula
# ---------------------------------------------------------------------------
def single_particle_energy(qx, qy, qz, omega: float | None = None, zero_point: bool = False):
    """Energía del estado (q_x, q_y, q_z) del oscilador 3D.

    Con ``omega=None`` devuelve unidades reducidas (ħω = 1); con ``omega`` en rad/s
    devuelve joules. ``zero_point=True`` suma la energía de punto cero 3ħω/2, que en
    el resto del proyecto se omite porque sólo desplaza E en 3Nħω/2.
    """
    eps = np.asarray(qx) + np.asarray(qy) + np.asarray(qz) + (1.5 if zero_point else 0.0)
    return eps if omega is None else energy_to_joules(eps, omega)


def level_degeneracy(eps):
    """Degeneración del nivel ε del oscilador 3D isotrópico: g(ε) = (ε+1)(ε+2)/2."""
    eps = np.asarray(eps)
    return (eps + 1) * (eps + 2) // 2


def number_of_states(q_max: int) -> int:
    """Número de estados con ε ≤ q_max: M = (q+1)(q+2)(q+3)/6."""
    return (q_max + 1) * (q_max + 2) * (q_max + 3) // 6


def build_state_catalog(q_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Lista explícita de los estados de una partícula con ε = q_x + q_y + q_z ≤ q_max.

    Cada estado degenerado aparece por separado. Los estados quedan ordenados por
    energía, de modo que el índice 0 es el estado fundamental (0, 0, 0).

    Returns
    -------
    energies : ndarray int64, forma (M,)
        Energía reducida ε_i de cada estado (también es el índice de su nivel).
    quantum_numbers : ndarray int64, forma (M, 3)
        Ternas (q_x, q_y, q_z) correspondientes.
    """
    if q_max < 0:
        raise ValueError("q_max debe ser >= 0")
    states = [
        (qx, qy, eps - qx - qy)
        for eps in range(q_max + 1)
        for qx in range(eps, -1, -1)
        for qy in range(eps - qx, -1, -1)
    ]
    quantum_numbers = np.array(states, dtype=np.int64)
    energies = quantum_numbers.sum(axis=1)
    return energies, quantum_numbers


# ---------------------------------------------------------------------------
# Configuraciones de N partículas
# ---------------------------------------------------------------------------
def compute_total_energy(n: np.ndarray, energies: np.ndarray, omega: float | None = None):
    """Hamiltoniano E(n) = Σ_i n_i ε_i (reducido, o en J si se da ``omega``)."""
    E = int(np.dot(n, energies))
    return E if omega is None else float(energy_to_joules(E, omega))


def initialize_configuration(
    N: int, M: int, statistics: str, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Vector de ocupaciones inicial con Σ_i n_i = N.

    mode="ground"  : estado base (bosones: todos en ε = 0; fermiones: los N estados
                     más bajos llenos, mar de Fermi a T = 0).
    mode="excited" : configuración "caliente" (T = ∞): cada partícula se coloca en un
                     estado elegido uniformemente (fermiones: N estados distintos al azar).
                     Su energía (≈ 3 q_max/4 por partícula) queda muy por encima del
                     equilibrio, lo que permite verificar la equilibración desde arriba.
    """
    _check_statistics(statistics)
    if mode not in INIT_MODES:
        raise ValueError(f"mode debe ser uno de {INIT_MODES}")
    if N < 1:
        raise ValueError("N debe ser >= 1")
    if statistics == "fermions" and N >= M:
        raise ValueError(
            f"Para fermiones se necesita M > N (M={M}, N={N}); aumente q_max."
        )
    if M < 2:
        raise ValueError("Se necesitan al menos dos estados (q_max >= 1).")

    n = np.zeros(M, dtype=np.int64)
    if mode == "ground":
        if statistics == "bosons":
            n[0] = N
        else:
            n[:N] = 1  # los estados están ordenados por energía
    else:
        if statistics == "bosons":
            n += np.bincount(rng.integers(0, M, size=N), minlength=M)
        else:
            n[rng.choice(M, size=N, replace=False)] = 1
    return n


def ground_state_energy(N: int, statistics: str) -> int:
    """Energía del estado base de N partículas (reducida)."""
    _check_statistics(statistics)
    if statistics == "bosons":
        return 0
    E, remaining, eps = 0, N, 0
    while remaining > 0:
        take = min(remaining, int(level_degeneracy(eps)))
        E += take * eps
        remaining -= take
        eps += 1
    return E


def fermi_level(N: int) -> int:
    """Nivel ε_F más alto ocupado en el estado base de N fermiones."""
    eps, filled = 0, 0
    while True:
        filled += int(level_degeneracy(eps))
        if filled >= N:
            return eps
        eps += 1


def is_closed_shell(N: int) -> bool:
    """True si N fermiones llenan exactamente sus capas (N = 1, 4, 10, 20, 35, 56, ...)."""
    eps = fermi_level(N)
    return number_of_states(eps) == N


def _check_statistics(statistics: str) -> None:
    if statistics not in STATISTICS:
        raise ValueError(f"statistics debe ser uno de {STATISTICS}, no {statistics!r}")
