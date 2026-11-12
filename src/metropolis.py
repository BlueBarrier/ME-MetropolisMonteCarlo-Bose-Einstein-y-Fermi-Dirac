"""Núcleo Markov Chain Monte Carlo sobre configuraciones de ocupación.

Distribución objetivo (ensemble canónico, partículas indistinguibles):

    P(n) ∝ exp[-β E(n)]   si n es admisible (Σ n_i = N; además n_i ≤ 1 para fermiones),
    P(n) = 0              en otro caso.

Un movimiento transfiere una partícula de un estado i a otro j ≠ i, de modo que
ΔE = ε_j - ε_i y N se conserva exactamente. Hay dos maneras de proponer (i, j):

1. ``proposal="state"`` (Metropolis, el núcleo de la propuesta aprobada).
   Se eligen i y j ≠ i uniformemente entre los M estados:
       g(a→b) = g(b→a) = 1 / [M (M-1)]     (propuesta simétrica).
   Si n_i = 0, o si es fermión y n_j = 1, el movimiento se rechaza (la cadena se
   queda en a y el paso cuenta). Si no, se acepta con A = min(1, e^{-βΔE}).
   Balance detallado: P_a g A(a→b) = P_b g A(b→a), porque A(a→b)/A(b→a) = e^{-βΔE}.

2. ``proposal="particle"`` (Metropolis–Hastings).
   Se elige una partícula al azar (el estado i sale con probabilidad n_i/N) y un
   destino j ≠ i uniforme:
       g(a→b) = (n_i / N) · 1/(M-1),     g(b→a) = ((n_j + 1) / N) · 1/(M-1).
   La aceptación de Hastings es A = min{1, [P_b g(b→a)] / [P_a g(a→b)]}, es decir
       bosones  : A = min{1, (n_j + 1)/n_i · e^{-βΔE}}
       fermiones: A = min{1, e^{-βΔE}}   (n_i = 1 y n_j = 0, la propuesta es simétrica).
   Nunca se propone desde un estado vacío, así que es mucho más eficiente cuando
   M ≫ N. Muestrea exactamente la misma distribución que el núcleo 1.

Ergodicidad: cualquier configuración admisible se alcanza desde cualquier otra
moviendo una partícula a la vez (los fermiones necesitan M > N). Los rechazos
dejan la cadena en su sitio, lo que la hace aperiódica.

Los núcleos están compilados con numba y usan ``np.random.Generator`` (PCG64)
con semilla fija, así que cada corrida es reproducible.
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

PROPOSALS = ("state", "particle")


# ---------------------------------------------------------------------------
# Núcleos compilados (un paso de la cadena)
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
def _step_symmetric(n, energies, beta, fermions, rng):
    """Un paso del núcleo simétrico. Devuelve (aceptado, i, j)."""
    M = n.shape[0]
    i = _uniform_index(rng, M)  # origen: uniforme en {0, ..., M-1}
    j = _uniform_index(rng, M - 1)  # destino: uniforme en los M-1 estados restantes
    if j >= i:
        j += 1
    # Filtro de validez física: estas propuestas tienen P_b = 0 -> A = 0.
    if n[i] == 0:
        return False, i, j
    if fermions and n[j] == 1:  # principio de exclusión de Pauli
        return False, i, j
    dE = energies[j] - energies[i]
    if dE <= 0 or rng.random() < np.exp(-beta * dE):
        n[i] -= 1
        n[j] += 1
        return True, i, j
    return False, i, j


@njit(cache=True)
def _step_hastings(n, owner, energies, beta, fermions, rng):
    """Un paso del núcleo de Metropolis–Hastings. Devuelve (aceptado, i, j).

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
def _run_kernel(
    n, owner, energies, beta, fermions, proposal, n_steps, sample_interval,
    level_occ, energy, rng, E_trace, level_trace,
):
    """Ejecuta ``n_steps`` pasos y registra E y las ocupaciones por nivel cada
    ``sample_interval`` pasos (si ``sample_interval`` > 0). Devuelve (aceptados, E)."""
    accepted = 0
    s = 0
    for t in range(n_steps):
        if proposal == 0:
            acc, i, j = _step_symmetric(n, energies, beta, fermions, rng)
        else:
            acc, i, j = _step_hastings(n, owner, energies, beta, fermions, rng)
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
def metropolis_step(n, energies, beta, statistics, rng) -> tuple[bool, float]:
    """Un paso del núcleo simétrico (propuesta aprobada). Modifica ``n`` en sitio.

    Devuelve (aceptado, ΔE). Si se rechaza, ΔE = 0 y ``n`` no cambia.
    """
    acc, i, j = _step_symmetric(n, energies, float(beta), statistics == "fermions", rng)
    return bool(acc), float(energies[j] - energies[i]) if acc else 0.0


def metropolis_hastings_step(n, owner, energies, beta, statistics, rng) -> tuple[bool, float]:
    """Un paso del núcleo de Metropolis–Hastings (partícula al azar). Modifica ``n`` y ``owner``.

    ``owner`` debe ser consistente con ``n`` (créelo con ``particle_labels(n)``). Por eso no se
    debe alternar con ``metropolis_step`` sobre la misma configuración, que no actualiza ``owner``.
    """
    acc, i, j = _step_hastings(n, owner, energies, float(beta), statistics == "fermions", rng)
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
    proposal: str = "state",
    burn_in_interval: int = 0,
) -> dict:
    """Corre una cadena de Markov completa con semilla fija.

    Parameters
    ----------
    N, beta, q_max, statistics : definición del sistema (β = 1/T reducido).
    n_steps : pasos de producción (cada paso es una propuesta).
    n_burn : pasos de termalización que se descartan.
    sample_interval : se mide cada ``sample_interval`` pasos de producción.
    seed : semilla de ``np.random.default_rng``; se guarda en los metadatos.
    init_mode : "ground" o "excited" (ver ``model.initialize_configuration``).
    proposal : "state" (Metropolis simétrico) o "particle" (Metropolis–Hastings).
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
    if proposal not in PROPOSALS:
        raise ValueError(f"proposal debe ser uno de {PROPOSALS}")
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
    code = PROPOSALS.index(proposal)
    beta = float(beta)

    t0 = time.perf_counter()
    # --- Termalización -----------------------------------------------------
    n_burn_samples = n_burn // burn_in_interval if burn_in_interval > 0 else 0
    burn_E = np.zeros(n_burn_samples, dtype=np.int64)
    burn_levels = np.zeros((n_burn_samples, q_max + 1), dtype=np.int32)
    acc_burn, energy = _run_kernel(
        n, owner, energies, beta, fermions, code, int(n_burn), int(burn_in_interval),
        level_occ, int(E0), rng, burn_E, burn_levels,
    )
    # --- Producción --------------------------------------------------------
    n_samples = n_steps // sample_interval
    E_trace = np.zeros(n_samples, dtype=np.int64)
    level_trace = np.zeros((n_samples, q_max + 1), dtype=np.int32)
    acc_prod, energy = _run_kernel(
        n, owner, energies, beta, fermions, code, int(n_steps), int(sample_interval),
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
        "proposal": proposal,
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
