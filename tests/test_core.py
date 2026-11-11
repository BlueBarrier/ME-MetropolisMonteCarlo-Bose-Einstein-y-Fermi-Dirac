"""Pruebas de los componentes: modelo, soluciones exactas, núcleos MCMC y estadística."""

from itertools import combinations, combinations_with_replacement

import numpy as np
import pytest

from src import analytics as an
from src import model
from src.metropolis import metropolis_hastings_step, metropolis_step, particle_labels, run_mcmc_chain
from src.stats import (
    blocking_analysis,
    discrepancy_in_sigmas,
    integrated_autocorrelation_time,
    statistical_uncertainty,
)


# ---------------------------------------------------------------- modelo
@pytest.mark.parametrize("q_max", range(0, 9))
def test_catalog_size_and_degeneracy(q_max):
    energies, qn = model.build_state_catalog(q_max)
    assert energies.size == model.number_of_states(q_max)
    assert np.all(np.diff(energies) >= 0) and energies[0] == 0
    np.testing.assert_array_equal(qn.sum(axis=1), energies)
    assert len({tuple(r) for r in qn}) == energies.size  # sin estados repetidos
    for eps in range(q_max + 1):
        assert np.sum(energies == eps) == model.level_degeneracy(eps)


def test_units_roundtrip():
    omega = 2 * np.pi * 100.0
    T = 37e-9
    assert np.isclose(model.kelvin_temperature(model.reduced_temperature(T, omega), omega), T)
    assert np.isclose(model.energy_to_reduced(model.energy_to_joules(3.5, omega), omega), 3.5)
    assert np.isclose(model.single_particle_energy(1, 2, 0, omega=omega), 3 * model.HBAR * omega)


@pytest.mark.parametrize("statistics", model.STATISTICS)
@pytest.mark.parametrize("mode", model.INIT_MODES)
def test_initial_configuration(statistics, mode):
    rng = np.random.default_rng(0)
    n = model.initialize_configuration(12, 56, statistics, mode, rng)
    assert n.sum() == 12 and n.min() >= 0
    if statistics == "fermions":
        assert n.max() <= 1


def test_fermi_shells():
    assert [model.fermi_level(N) for N in (1, 4, 10, 20, 35, 56)] == [0, 1, 2, 3, 4, 5]
    assert all(model.is_closed_shell(N) for N in (1, 4, 10, 20, 35, 56))
    assert not model.is_closed_shell(25)
    assert model.ground_state_energy(10, "fermions") == 0 * 1 + 1 * 3 + 2 * 6
    assert model.ground_state_energy(35, "fermions") == 105


# ---------------------------------------------------------------- exactos
@pytest.mark.parametrize("statistics", model.STATISTICS)
@pytest.mark.parametrize("N", [1, 2, 3, 4])
@pytest.mark.parametrize("beta", [0.3, 1.0, 5.0])
def test_generating_function_matches_enumeration(statistics, N, beta):
    q = 4 if N == 4 else 5
    e, _ = model.build_state_catalog(q)
    bf = an.exact_canonical_enumeration(N, e, beta, statistics)
    gf = an.exact_canonical(N, q, beta, statistics)
    occ_bf = np.array([bf["occupation_per_state"][e == l].mean() for l in range(q + 1)])
    assert np.isclose(bf["mean_energy"], gf["mean_energy"], rtol=1e-10, atol=1e-12)
    assert np.isclose(bf["log_Z"], gf["log_Z"], rtol=1e-10)
    np.testing.assert_allclose(occ_bf, gf["occupation"], rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("statistics", model.STATISTICS)
def test_exact_particle_number_and_limits(statistics):
    ex = an.exact_canonical(35, 30, 0.7, statistics)
    assert np.isclose(ex["particle_number_check"], 35.0, rtol=1e-10)
    if statistics == "fermions":
        assert ex["occupation"].max() <= 1.0 + 1e-12
    # N = 1: límite de trampa infinita
    assert np.isclose(an.exact_canonical(1, 80, 1.3, statistics)["mean_energy"],
                      an.exact_single_particle_energy(1.3), rtol=1e-12)


def test_exact_low_temperature_limits():
    assert an.exact_canonical(20, 10, 30.0, "bosons")["condensate_fraction"] > 0.999
    assert np.isclose(an.exact_canonical(20, 10, 30.0, "fermions")["mean_energy"],
                      model.ground_state_energy(20, "fermions"), rtol=1e-6)


@pytest.mark.parametrize("statistics", model.STATISTICS)
def test_grand_canonical_particle_number(statistics):
    for T in (0.3, 1.0, 4.0):
        gc = an.grand_canonical_observables(20, 60, 1.0 / T, statistics)
        assert np.isclose(np.sum(gc["degeneracy"] * gc["occupation"]), 20.0, rtol=1e-9)
        if statistics == "bosons":
            assert gc["mu"] < 0


def test_choose_q_max_bounds_truncation():
    for statistics in model.STATISTICS:
        q = an.choose_q_max(10, 2.0, statistics, tol=1e-4)
        assert abs(an.truncation_error(10, q, 0.5, statistics)) <= 1e-4 * 2.0


# ---------------------------------------------------------------- núcleos
@pytest.mark.parametrize("statistics", model.STATISTICS)
@pytest.mark.parametrize("proposal", ["state", "particle"])
def test_kernels_conserve_particles_and_pauli(statistics, proposal):
    # Cada cadena usa un solo núcleo (las etiquetas ``owner`` sólo las mantiene el de Hastings).
    energies, _ = model.build_state_catalog(6)
    rng = np.random.default_rng(3)
    n = model.initialize_configuration(8, energies.size, statistics, "excited", rng)
    owner = particle_labels(n)
    E = model.compute_total_energy(n, energies)
    for _ in range(20_000):
        if proposal == "state":
            acc, dE = metropolis_step(n, energies, 0.8, statistics, rng)
        else:
            acc, dE = metropolis_hastings_step(n, owner, energies, 0.8, statistics, rng)
        E += dE
        assert n.min() >= 0
        if statistics == "fermions":
            assert n.max() <= 1
    assert n.sum() == 8
    assert E == model.compute_total_energy(n, energies)
    if proposal == "particle":
        np.testing.assert_array_equal(np.bincount(owner, minlength=n.size), n)


@pytest.mark.parametrize("statistics", model.STATISTICS)
@pytest.mark.parametrize("proposal", ["state", "particle"])
def test_stationary_distribution_over_microstates(statistics, proposal):
    """La frecuencia de cada microestado converge a e^{-βE}/Z (balance detallado).

    Sistema mínimo: N = 2, q_max = 1 (4 estados con ε = 0, 1, 1, 1).
    """
    N, beta = 2, 0.7
    energies, _ = model.build_state_catalog(1)
    M = energies.size
    gen = combinations_with_replacement if statistics == "bosons" else combinations
    configs = [np.bincount(c, minlength=M) for c in gen(range(M), N)]
    index = {tuple(c): k for k, c in enumerate(configs)}
    w = np.array([np.exp(-beta * np.dot(c, energies)) for c in configs])
    p_exact = w / w.sum()

    rng = np.random.default_rng(11)
    n = model.initialize_configuration(N, M, statistics, "ground", rng)
    owner = particle_labels(n)
    n_steps = 120_000
    visits = np.empty(n_steps, dtype=np.int64)
    for t in range(n_steps):
        if proposal == "state":
            metropolis_step(n, energies, beta, statistics, rng)
        else:
            metropolis_hastings_step(n, owner, energies, beta, statistics, rng)
        visits[t] = index[tuple(n)]
    for k, p in enumerate(p_exact):
        mean, _, err = statistical_uncertainty((visits == k).astype(float))
        assert discrepancy_in_sigmas(mean, err, p) < 4.0, (configs[k], mean, p)


def test_seed_reproducibility():
    a = run_mcmc_chain(10, 0.7, 10, "bosons", 300_000, 1000, 100, 123, "excited", "particle")
    b = run_mcmc_chain(10, 0.7, 10, "bosons", 300_000, 1000, 100, 123, "excited", "particle")
    c = run_mcmc_chain(10, 0.7, 10, "bosons", 300_000, 1000, 100, 124, "excited", "particle")
    np.testing.assert_array_equal(a["E_trace"], b["E_trace"])
    np.testing.assert_array_equal(a["level_trace"], b["level_trace"])
    assert not np.array_equal(a["E_trace"], c["E_trace"])


# ---------------------------------------------------------------- estadística
def test_autocorrelation_time_of_ar1_process():
    """Proceso AR(1) x_t = φ x_{t-1} + ruido: τ_int = (1+φ) / (2(1-φ)) = 4.5 para φ = 0.8."""
    phi, n = 0.8, 400_000
    rng = np.random.default_rng(5)
    noise = rng.normal(size=n)
    x = np.empty(n)
    x[0] = noise[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    tau = integrated_autocorrelation_time(x)
    assert abs(tau - 4.5) / 4.5 < 0.1
    _, _, err = statistical_uncertainty(x)
    assert abs(blocking_analysis(x)["plateau_error"] / err - 1.0) < 0.3


def test_discrepancy_edge_cases():
    assert discrepancy_in_sigmas(1.0, 0.5, 2.0) == pytest.approx(2.0)
    assert discrepancy_in_sigmas(1.0, 0.0, 1.0) == 0.0
    assert discrepancy_in_sigmas(1.0, 0.0, 2.0) == float("inf")
    mean, tau, err = statistical_uncertainty(np.ones(1000), resolution=0.1)
    assert mean == 1.0 and tau == 0.5 and err == pytest.approx(3e-4)
    # Serie constante cuyo promedio en coma flotante no es exacto (p. ej. n_0/N = 1/56).
    mean, tau, err = statistical_uncertainty(np.full(20_000, 1 / 56), resolution=1 / 56)
    assert err == pytest.approx(3 / 56 / 20_000)


# ---------------------------------------------------------------- límite termodinámico
def test_thermodynamic_limit_curves():
    """Límites conocidos de las curvas usadas para comparar con los experimentos."""
    # Fermi: T → 0 da U = (3/4) N k T_F, o sea δU/U_cl → 1/(4t) − 1; T alta da δ → 0.
    t = 0.02
    assert an.fermi_excess_energy_thermodynamic(t)[0] == pytest.approx(1 / (4 * t) - 1, rel=1e-2)
    assert abs(an.fermi_excess_energy_thermodynamic(20.0)[0]) < 1e-3
    # Bose: E_cin/(N k T_0) = (3/2) ζ(4)/ζ(3) t⁴ por debajo de T_0; recta clásica (3/2)t a T alta.
    assert an.bose_release_energy_thermodynamic(0.5)[0] == pytest.approx(1.5 * 1.0823232337 / an.ZETA3 * 0.5**4, rel=1e-8)
    assert an.bose_release_energy_thermodynamic(20.0)[0] == pytest.approx(30.0, rel=2e-3)
    # g_n(z) por integral vs serie
    z = 0.6
    assert an.bose_einstein_function(3, z) == pytest.approx(sum(z**k / k**3 for k in range(1, 400)), rel=1e-10)


def test_semiclassical_limit_of_exact_fermi_energy():
    """Con N grande, la solución canónica exacta (con punto cero) se acerca a la curva semiclásica."""
    N, t = 120, 0.6  # capa cerrada (ε_F = 7)
    TF = (6 * N) ** (1 / 3)
    T = t * TF
    E = an.exact_canonical(N, an.reference_cutoff(N, T, "fermions"), 1 / T, "fermions")["energy_per_particle"]
    delta = (E + 1.5 - 3 * T) / (3 * T)
    assert delta == pytest.approx(an.fermi_excess_energy_thermodynamic(t)[0], abs=5e-3)
