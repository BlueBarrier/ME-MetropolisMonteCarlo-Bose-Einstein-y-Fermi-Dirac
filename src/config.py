"""Parámetros globales de las corridas (un solo lugar para reproducirlas).

Todas las semillas se derivan de ``BASE_SEED`` con ``make_seed``, de forma
determinista. La semilla entera de cada corrida queda en sus metadatos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analytics import bec_critical_temperature, fermi_energy

ROOT = Path(__file__).resolve().parents[1]

BASE_SEED = 20261112  # fecha de entrega: 12-11-2026

# Tamaños: capas cerradas para fermiones (ε_F = 2, 3, 4, 5).
N_VALUES = (10, 20, 35, 56)

# Malla común de temperaturas reducidas (unidades de ħω/k_B), igual para las dos estadísticas.
T_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)

# Núcleos: el simétrico (propuesta aprobada) valida; el de Hastings produce los barridos.
VALIDATION_PROPOSAL = "state"
PRODUCTION_PROPOSAL = "particle"

N_SAMPLES = 20_000  # mediciones por corrida
BURN_FRACTION = 0.10  # termalización = 10 % de los pasos de producción
STEPS_DEFAULT = 60_000_000
STEPS_SLOW = 240_000_000  # bosones cerca de T_c (dinámica lenta del condensado)
Q_TOL = 1e-4  # error de truncamiento máximo en E/N, en unidades de T
FAST_FACTOR = 0.05  # modo --fast: fracción de los pasos (sólo para verificar que todo corre)


def make_seed(*keys: int) -> int:
    """Semilla entera determinista a partir de BASE_SEED y una tupla de enteros."""
    ss = np.random.SeedSequence([BASE_SEED, *[int(k) for k in keys]])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def output_dirs(fast: bool = False) -> tuple[Path, Path]:
    """(directorio de datos crudos, directorio de figuras). El modo rápido no pisa los datos finales."""
    if fast:
        return ROOT / "data" / "raw_fast", ROOT / "figures" / "fast"
    return ROOT / "data" / "raw", ROOT / "figures"


def reference_temperature(N: int, statistics: str) -> float:
    """Escala natural de temperatura: T_c (bosones) o T_F = ε_F (fermiones)."""
    return float(bec_critical_temperature(N)) if statistics == "bosons" else float(fermi_energy(N))


def steps_for(N: int, T: float, statistics: str, fast: bool = False) -> int:
    """Número de pasos de producción según el régimen.

    Por debajo de T_c el intercambio de partículas con el condensado es lento,
    así que esas corridas se hacen 4 veces más largas.
    """
    steps = STEPS_DEFAULT
    if statistics == "bosons" and 0.4 <= T / reference_temperature(N, statistics) <= 1.2:
        steps = STEPS_SLOW
    if fast:
        steps = int(steps * FAST_FACTOR)
    return max(steps - steps % N_SAMPLES, N_SAMPLES)
