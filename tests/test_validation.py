"""Prueba automatizada del caso de validación (rúbrica A4).

Simulaciones cortas con semilla fija (seed = 42) que se comparan con resultados exactos:
  - N = 1 contra ⟨E⟩ = 3/(e^β − 1) (trampa infinita, corte con truncamiento < 10⁻⁴ T),
  - N = 2 (bosones y fermiones) contra enumeración directa de todas las configuraciones,
con los dos núcleos. Se exige una discrepancia z < 3σ en ⟨E⟩/N y en ⟨n_0⟩/N.

Ejecutar:  python -m pytest tests/test_validation.py -v
"""

import numpy as np
import pytest

from src import analytics as an
from src.metropolis import run_mcmc_chain
from src.model import build_state_catalog
from src.observables import compute_observables
from src.stats import discrepancy_in_sigmas

SEED = 42


def _simulate(N, T, q_max, statistics, proposal, n_steps):
    res = run_mcmc_chain(N, 1.0 / T, q_max, statistics, n_steps, n_steps // 10, n_steps // 10_000,
                         SEED, "ground", proposal)
    return compute_observables(res["E_trace"], res["level_trace"], N)


@pytest.mark.parametrize(
    "proposal, T, n_steps",
    [("state", 1.0, 40_000_000), ("particle", 1.0, 4_000_000), ("particle", 3.0, 4_000_000)],
)
def test_single_particle_energy(proposal, T, n_steps):
    q = an.choose_q_max(1, T, "bosons")
    obs = _simulate(1, T, q, "bosons", proposal, n_steps)
    exact = float(an.exact_single_particle_energy(1.0 / T))
    # El corte elegido hace despreciable el truncamiento frente al error estadístico.
    assert abs(an.exact_single_particle_energy_truncated(1.0 / T, q) - exact) < 0.1 * obs["energy_per_particle_err"]
    z = discrepancy_in_sigmas(obs["energy_per_particle"], obs["energy_per_particle_err"], exact)
    assert z < 3.0, f"N=1 {proposal}: z = {z:.2f}"


@pytest.mark.parametrize("statistics", ["bosons", "fermions"])
@pytest.mark.parametrize("proposal", ["state", "particle"])
def test_two_particles_against_enumeration(statistics, proposal):
    N, T, q = 2, 1.0, 5
    energies, _ = build_state_catalog(q)
    exact = an.exact_canonical_enumeration(N, energies, 1.0 / T, statistics)
    n0_exact = exact["occupation_per_state"][0] / N
    obs = _simulate(N, T, q, statistics, proposal, 10_000_000 if proposal == "state" else 3_000_000)
    zE = discrepancy_in_sigmas(obs["energy_per_particle"], obs["energy_per_particle_err"], exact["mean_energy"] / N)
    zf = discrepancy_in_sigmas(obs["condensate_fraction"], obs["condensate_fraction_err"], n0_exact)
    assert zE < 3.0, f"E/N: z = {zE:.2f}"
    assert zf < 3.0, f"<n_0>/N: z = {zf:.2f}"


def test_both_statistics_coincide_for_one_particle():
    """Con N = 1 nunca actúa el principio de exclusión: misma semilla ⇒ misma cadena."""
    for proposal in ("state", "particle"):
        a = run_mcmc_chain(1, 1.0, 8, "bosons", 200_000, 0, 100, SEED, "ground", proposal)
        b = run_mcmc_chain(1, 1.0, 8, "fermions", 200_000, 0, 100, SEED, "ground", proposal)
        np.testing.assert_array_equal(a["E_trace"], b["E_trace"])
