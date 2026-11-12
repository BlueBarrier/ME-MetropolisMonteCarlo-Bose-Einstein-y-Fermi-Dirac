"""Algoritmo de Metropolis–Hastings sobre configuraciones de ocupación.

Distribución objetivo (ensemble canónico, partículas indistinguibles):

    P(n) ∝ exp[-β E(n)]   si n es admisible (Σ n_i = N; además n_i ≤ 1 para fermiones),
    P(n) = 0              en otro caso.

Movimiento: se elige una partícula al azar (el estado de origen i sale con probabilidad
n_i/N) y un estado de destino j ≠ i uniforme entre los M − 1 restantes; la partícula pasa
de i a j, con ΔE = ε_j − ε_i y N conservado exactamente. Las probabilidades de proponer el
movimiento y su inverso son

    g(a→b) = (n_i / N) · 1/(M-1),     g(b→a) = ((n_j + 1) / N) · 1/(M-1),

y la aceptación de Metropolis–Hastings A = min{1, [P_b g(b→a)] / [P_a g(a→b)]} queda

    bosones  : A = min{1, (n_j + 1)/n_i · e^{-βΔE}}
    fermiones: A = min{1, e^{-βΔE}}   (n_i = 1 y n_j = 0: la propuesta es simétrica;
                                       si n_j = 1 el movimiento viola Pauli y se rechaza).

Balance detallado: con esa aceptación, P_a g(a→b) A(a→b) = P_b g(b→a) A(b→a).
El factor (n_j + 1)/n_i es el que impone la indistinguibilidad: sin él, elegir una partícula
y un destino al azar muestrearía partículas distinguibles (Maxwell–Boltzmann).

Ergodicidad: cualquier configuración admisible se alcanza desde cualquier otra moviendo una
partícula a la vez (los fermiones necesitan M > N). Los rechazos dejan la cadena en su
sitio, lo que la hace aperiódica.

El núcleo está compilado con numba y usa ``np.random.Generator`` (PCG64) con semilla fija,
así que cada corrida es reproducible.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit

from .model import (
    INIT_MODES,
    STATISTICS,
    build_state_catalog,
    compute_total_energy,
    initialize_configuration,
)

ALGORITHM = "metropolis-hastings"


# ---------------------------------------------------------------------------
# Núcleo compilado (un paso de la cadena)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _uniform_index(rng, m):
    """Entero uniforme en {0, ..., m-1} a partir de u ~ U[0, 1).

    Es unas 20 veces más rápido que ``rng.integers`` dentro de numba. El sesgo es
    ≤ m·2⁻⁵³ (~10⁻¹¹ con m = 10⁵), despreciable frente a los errores estadísticos.
    """
    k = int(rng.random() * m)
    return k if k < m else m - 1


@njit(cache=True)
def _step(n, owner, energies, beta, fermions, rng):
    """Un paso de Metropolis–Hastings. Devuelve (aceptado, i, j).

    ``owner[k]`` es el estado que ocupa la "partícula" k. Es sólo una etiqueta
    auxiliar que permite escoger el estado i con probabilidad n_i/N. El estado
    físico sigue siendo el vector de ocupaciones n.
    """
    N = owner.shape[0]
    M = n.shape[0]
    k = _uniform_index(rng, N)
    i = owner[k]
    j = _uniform_index(rng, M - 1)
    if j >= i:
        j += 1
    if fermions:
        if n[j] == 1:  # Pauli
            return False, i, j
        log_q = 0.0  # g(b→a)/g(a→b) = 1
    else:
        log_q = np.log((n[j] + 1.0) / n[i])  # factor de Hastings bosónico
    log_r = log_q - beta * (energies[j] - energies[i])
    if log_r >= 0.0 or rng.random() < np.exp(log_r):
        n[i] -= 1
        n[j] += 1
        owner[k] = j
        return True, i, j
    return False, i, j


@njit(cache=True)
def _run_kernel(n, owner, energies, beta, fermions, n_steps, sample_interval,
                level_occ, energy, rng, E_trace, level_trace):
    """Ejecuta ``n_steps`` pasos y registra E y las ocupaciones por nivel cada
    ``sample_interval`` pasos (si ``sample_interval`` > 0). Devuelve (aceptados, E)."""
    accepted = 0
    s = 0
    for t in range(n_steps):
        acc, i, j = _step(n, owner, energies, beta, fermions, rng)
        if acc:
            accepted += 1
            ei = energies[i]
            ej = energies[j]
            energy += ej - ei
            level_occ[ei] -= 1
            level_occ[ej] += 1
        if sample_interval > 0 and (t + 1) % sample_interval == 0:
            E_trace[s] = energy
            level_trace[s, :] = level_occ
            s += 1
    return accepted, energy


# ---------------------------------------------------------------------------
# Interfaz en Python
# ---------------------------------------------------------------------------
def metropolis_hastings_step(n, owner, energies, beta, statistics, rng) -> tuple[bool, float]:
    """Un paso de Metropolis–Hastings. Modifica ``n`` y ``owner`` en sitio.

    ``owner`` debe ser consistente con ``n`` (créelo con ``particle_labels(n)``).
    Devuelve (aceptado, ΔE); si se rechaza, ΔE = 0 y nada cambia.
    """
    acc, i, j = _step(n, owner, energies, float(beta), statistics == "fermions", rng)
    return bool(acc), float(energies[j] - energies[i]) if acc else 0.0


def particle_labels(n: np.ndarray) -> np.ndarray:
    """Arreglo ``owner`` (estado de cada partícula) consistente con las ocupaciones n."""
    return np.repeat(np.arange(n.size, dtype=np.int64), n).astype(np.int64)


def run_mcmc_chain(
    N: int,
    beta: float,
    q_max: int,
    statistics: str,
    n_steps: int,
    n_burn: int,
    sample_interval: int,
    seed: int,
    init_mode: str = "ground",
    burn_in_interval: int = 0,
) -> dict:
    """Corre una cadena de Metropolis–Hastings completa con semilla fija.

    Parameters
    ----------
    N, beta, q_max, statistics : definición del sistema (β = 1/T reducido).
    n_steps : pasos de producción (cada paso es una propuesta).
    n_burn : pasos de termalización que se descartan antes de medir.
    sample_interval : se mide cada ``sample_interval`` pasos de producción.
    seed : semilla de ``np.random.default_rng``; se guarda en los metadatos.
    init_mode : "ground" o "excited" (ver ``model.initialize_configuration``).
    burn_in_interval : si > 0, también registra E durante la termalización
        (útil para las figuras de equilibración).

    Returns
    -------
    dict con las trazas ``E_trace`` (energía total), ``level_trace`` (partículas
    por nivel ε = 0..q_max), ``burn_E_trace``, la tasa de aceptación,
    la configuración final y un diccionario ``metadata`` con todos los parámetros.
    """
    if statistics not in STATISTICS:
        raise ValueError(f"statistics debe ser uno de {STATISTICS}")
    if init_mode not in INIT_MODES:
        raise ValueError(f"init_mode debe ser uno de {INIT_MODES}")
    if sample_interval < 1:
        raise ValueError("sample_interval debe ser >= 1")

    rng = np.random.default_rng(seed)
    energies, _ = build_state_catalog(q_max)
    M = energies.size
    n = initialize_configuration(N, M, statistics, init_mode, rng)
    owner = particle_labels(n)
    level_occ = np.bincount(energies, weights=n, minlength=q_max + 1).astype(np.int64)
    E0 = compute_total_energy(n, energies)
    fermions = statistics == "fermions"
    beta = float(beta)

    t0 = time.perf_counter()
    # --- Termalización (se descarta) ---------------------------------------
    n_burn_samples = n_burn // burn_in_interval if burn_in_interval > 0 else 0
    burn_E = np.zeros(n_burn_samples, dtype=np.int64)
    burn_levels = np.zeros((n_burn_samples, q_max + 1), dtype=np.int32)
    acc_burn, energy = _run_kernel(
        n, owner, energies, beta, fermions, int(n_burn), int(burn_in_interval),
        level_occ, int(E0), rng, burn_E, burn_levels,
    )
    # --- Producción (de aquí salen todas las mediciones) --------------------
    n_samples = n_steps // sample_interval
    E_trace = np.zeros(n_samples, dtype=np.int64)
    level_trace = np.zeros((n_samples, q_max + 1), dtype=np.int32)
    acc_prod, energy = _run_kernel(
        n, owner, energies, beta, fermions, int(n_steps), int(sample_interval),
        level_occ, int(energy), rng, E_trace, level_trace,
    )
    elapsed = time.perf_counter() - t0

    # Verificaciones de consistencia (vínculo y Pauli).
    assert n.sum() == N, "Se violó la conservación de N"
    assert energy == compute_total_energy(n, energies), "Energía acumulada inconsistente"
    if fermions:
        assert n.max() <= 1, "Se violó el principio de exclusión"

    metadata = {
        "N": int(N),
        "beta": beta,
        "T": 1.0 / beta,
        "q_max": int(q_max),
        "M": int(M),
        "statistics": statistics,
        "algorithm": ALGORITHM,
        "n_steps": int(n_steps),
        "n_burn": int(n_burn),
        "sample_interval": int(sample_interval),
        "burn_in_interval": int(burn_in_interval),
        "seed": int(seed),
        "init_mode": init_mode,
        "E_initial": int(E0),
        "acceptance_rate": acc_prod / n_steps if n_steps else float("nan"),
        "acceptance_rate_burn": acc_burn / n_burn if n_burn else float("nan"),
        "elapsed_seconds": elapsed,
    }
    return {
        "E_trace": E_trace,
        "level_trace": level_trace,
        "burn_E_trace": burn_E,
        "burn_level_trace": burn_levels,
        "final_n": n,
        "metadata": metadata,
    }
