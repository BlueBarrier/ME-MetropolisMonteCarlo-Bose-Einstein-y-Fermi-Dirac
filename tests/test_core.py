"""Pruebas de los componentes: modelo, soluciones exactas, calor específico, Metropolis–Hastings y estadística."""

from itertools import combinations, combinations_with_replacement

import numpy as np
import pytest

from src import analytics as an
from src import model
from src.metropolis import metropolis_hastings_step, particle_labels, run_mcmc_chain
from src.stats import (
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


# ---------------------------------------------------------------- Metropolis–Hastings
@pytest.mark.parametrize("statistics", model.STATISTICS)
def test_kernel_conserves_particles_and_pauli(statistics):
    energies, _ = model.build_state_catalog(6)
    rng = np.random.default_rng(3)
    n = model.initialize_configuration(8, energies.size, statistics, "excited", rng)
    owner = particle_labels(n)
    E = model.compute_total_energy(n, energies)
    for _ in range(20_000):
        acc, dE = metropolis_hastings_step(n, owner, energies, 0.8, statistics, rng)
        E += dE
        assert n.min() >= 0
        if statistics == "fermions":
            assert n.max() <= 1
    assert n.sum() == 8
    assert E == model.compute_total_energy(n, energies)
    np.testing.assert_array_equal(np.bincount(owner, minlength=n.size), n)


@pytest.mark.parametrize("statistics", model.STATISTICS)
def test_stationary_distribution_over_microstates(statistics):
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
        metropolis_hastings_step(n, owner, energies, beta, statistics, rng)
        visits[t] = index[tuple(n)]
    for k, p in enumerate(p_exact):
        mean, _, err = statistical_uncertainty((visits == k).astype(float))
        assert discrepancy_in_sigmas(mean, err, p) < 4.0, (configs[k], mean, p)


def test_seed_reproducibility():
    a = run_mcmc_chain(10, 0.7, 10, "bosons", 300_000, 1000, 100, 123, "excited")
    b = run_mcmc_chain(10, 0.7, 10, "bosons", 300_000, 1000, 100, 123, "excited")
    c = run_mcmc_chain(10, 0.7, 10, "bosons", 300_000, 1000, 100, 124, "excited")
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


def test_discrepancy_edge_cases():
    assert discrepancy_in_sigmas(1.0, 0.5, 2.0) == pytest.approx(2.0)
    assert discrepancy_in_sigmas(1.0, 0.0, 1.0) == 0.0
    assert discrepancy_in_sigmas(1.0, 0.0, 2.0) == float("inf")
    mean, tau, err = statistical_uncertainty(np.ones(1000), resolution=0.1)
    assert mean == 1.0 and tau == 0.5 and err == pytest.approx(3e-4)
    # Serie constante cuyo promedio en coma flotante no es exacto (p. ej. n_0/N = 1/56).
    mean, tau, err = statistical_uncertainty(np.full(20_000, 1 / 56), resolution=1 / 56)
    assert err == pytest.approx(3 / 56 / 20_000)


# ---------------------------------------------------------------- comparación con teoría del usuario
def test_theory_comparison_against_exact_values():
    """Si la "teoría" del usuario es la solución exacta con el mismo q_max, la simulación es compatible."""
    from src.theory_compare import compare_with_theory, parse_pairs

    x, y = parse_pairs("# T E/N\n1.0, 0 ; 2.0 0")  # admite comentarios, comas, espacios y ';'
    assert list(x) == [1.0, 2.0]
    theory = [an.exact_canonical(10, 12, 1 / T, "fermions")["energy_per_particle"] for T in x]
    res = compare_with_theory("fermions", 10, 12, "energy", x, theory, n_steps=2_000_000, seed=5, verbose=False)
    assert np.all(np.abs(res["z"]) < 3.5)
    assert res["seeds"] == [5, 6]



# ---------------------------------------------------------------- calor específico y potencial químico
def test_heat_capacity_single_particle():
    """N = 1: C = d/dT [3/(e^β − 1)] = 3 β² e^β / (e^β − 1)²."""
    for T in (0.5, 1.0, 3.0):
        b = 1.0 / T
        exact = 3 * b**2 * np.exp(b) / np.expm1(b) ** 2
        q = an.reference_cutoff(1, T, "bosons")
        assert an.heat_capacity_canonical(1, q, T, "bosons") == pytest.approx(exact, rel=1e-6)


def test_bose_lambda_point_thermodynamic_limit():
    z3, z2, z4 = an.ZETA3, an.ZETA2, an.ZETA4
    assert an.bose_heat_capacity_thermodynamic(0.5) == pytest.approx(12 * z4 / z3 * 0.125)
    below = an.bose_heat_capacity_thermodynamic(1.0)
    above = an.bose_heat_capacity_thermodynamic(1.0 + 1e-7)
    assert below == pytest.approx(12 * z4 / z3)              # ≈ 10.8
    assert below - above == pytest.approx(9 * z3 / z2, rel=1e-3)  # salto ≈ 6.6
    assert an.bose_heat_capacity_thermodynamic(20.0) == pytest.approx(3.0, abs=2e-3)


def test_fermi_thermodynamic_limits():
    assert an.fermi_heat_capacity_thermodynamic(0.01) == pytest.approx(np.pi**2 * 0.01, rel=5e-3)
    assert an.fermi_heat_capacity_thermodynamic(10.0) == pytest.approx(3.0, abs=2e-3)
    assert an.fermi_chemical_potential_thermodynamic(0.02) == pytest.approx(1 - np.pi**2 / 3 * 0.02**2, abs=1e-6)
    t0 = (1 / (6 * 0.75 * an.ZETA3)) ** (1 / 3)  # μ = 0 cuando f_3(1) = 3ζ(3)/4
    assert an.fermi_chemical_potential_thermodynamic(t0) == pytest.approx(0.0, abs=1e-8)


def test_fit_chemical_potential_recovers_fermi_dirac():
    from src.observables import fit_chemical_potential

    eps = np.arange(15.0)
    T, mu = 0.8, 4.3
    occ = 1 / (np.exp((eps - mu) / T) + 1)
    m, err = fit_chemical_potential(eps, occ, np.full(eps.size, 1e-3), T)
    assert m == pytest.approx(mu, abs=1e-6)
    assert err < 1e-2
