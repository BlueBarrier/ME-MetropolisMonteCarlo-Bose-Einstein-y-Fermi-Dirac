"""Prueba automatizada del caso de validación (algoritmo de Metropolis–Hastings).

Simulaciones cortas con semilla fija (seed = 42) que se comparan con resultados exactos:
  - N = 1 contra ⟨E⟩ = 3/(e^β − 1) (trampa infinita, corte con truncamiento < 10⁻⁴ T),
  - N = 2 (bosones y fermiones) contra enumeración directa de todas las configuraciones.
Se exige una discrepancia z < 3σ en ⟨E⟩/N, en ⟨n_0⟩/N y en el calor específico C/N
(medido por fluctuaciones de la energía).

Ejecutar:  python -m pytest tests/test_validation.py -v
"""

import numpy as np
import pytest

from src import analytics as an
from src.config import BURN_FRACTION
from src.metropolis import run_mcmc_chain
from src.model import build_state_catalog
from src.observables import compute_observables, heat_capacity
from src.stats import discrepancy_in_sigmas

SEED = 42


def _simulate(N, T, q_max, statistics, n_steps):
    res = run_mcmc_chain(N, 1.0 / T, q_max, statistics, n_steps, int(BURN_FRACTION * n_steps),
                         n_steps // 10_000, SEED, "ground")
    return compute_observables(res["E_trace"], res["level_trace"], N)


@pytest.mark.parametrize("T", [0.5, 1.0, 3.0])
def test_single_particle_energy(T):
    q = an.choose_q_max(1, T, "bosons")
    obs = _simulate(1, T, q, "bosons", 4_000_000)
    exact = float(an.exact_single_particle_energy(1.0 / T))
    # El corte elegido hace despreciable el truncamiento frente al error estadístico.
    assert abs(an.exact_single_particle_energy_truncated(1.0 / T, q) - exact) < 0.1 * obs["energy_per_particle_err"]
    z = discrepancy_in_sigmas(obs["energy_per_particle"], obs["energy_per_particle_err"], exact)
    assert z < 3.0, f"N=1, T={T}: z = {z:.2f}"


@pytest.mark.parametrize("statistics", ["bosons", "fermions"])
def test_two_particles_against_enumeration(statistics):
    N, T, q = 2, 1.0, 5
    energies, _ = build_state_catalog(q)
    exact = an.exact_canonical_enumeration(N, energies, 1.0 / T, statistics)
    n0_exact = exact["occupation_per_state"][0] / N
    obs = _simulate(N, T, q, statistics, 3_000_000)
    zE = discrepancy_in_sigmas(obs["energy_per_particle"], obs["energy_per_particle_err"], exact["mean_energy"] / N)
    zf = discrepancy_in_sigmas(obs["condensate_fraction"], obs["condensate_fraction_err"], n0_exact)
    assert zE < 3.0, f"E/N: z = {zE:.2f}"
    assert zf < 3.0, f"<n_0>/N: z = {zf:.2f}"


def test_both_statistics_coincide_for_one_particle():
    """Con N = 1 nunca actúa el principio de exclusión: misma semilla ⇒ misma cadena."""
    a = run_mcmc_chain(1, 1.0, 8, "bosons", 200_000, 0, 100, SEED, "ground")
    b = run_mcmc_chain(1, 1.0, 8, "fermions", 200_000, 0, 100, SEED, "ground")
    np.testing.assert_array_equal(a["E_trace"], b["E_trace"])


@pytest.mark.parametrize("statistics", ["bosons", "fermions"])
def test_heat_capacity_from_energy_fluctuations(statistics):
    """C/N = Var(E)/(N T²) de la simulación frente a la derivada exacta d⟨E⟩/dT."""
    N, T, q = 10, 1.5, 12
    n_steps = 4_000_000
    res = run_mcmc_chain(N, 1.0 / T, q, statistics, n_steps, int(BURN_FRACTION * n_steps),
                         n_steps // 20_000, SEED, "ground")
    C, C_err = heat_capacity(res["E_trace"], T, N)
    exact = an.heat_capacity_canonical(N, q, T, statistics)
    z = discrepancy_in_sigmas(C, C_err, exact)
    assert z < 3.0, f"C/N: {C:.3f} ± {C_err:.3f} vs {exact:.3f} (z = {z:.2f})"
