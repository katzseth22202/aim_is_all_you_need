import astropy.units as u
import numpy as np
import pytest

from src.carbon_hydrogen_equilibrium import b_prime_table, graphite_equilibrium


def test_chamber_peak_pressure_values():
    """The two figures the companion reply of 2026-10-04 quotes."""
    cold = graphite_equilibrium(2500 * u.K, 818 * u.bar)
    hot = graphite_equilibrium(3900 * u.K, 818 * u.bar)
    assert cold.b_prime == pytest.approx(0.697, abs=0.005)
    assert hot.b_prime == pytest.approx(5.28, abs=0.03)


def test_methane_carries_the_carbon_at_2500_k_and_acetylene_at_3900_k():
    cold = graphite_equilibrium(2500 * u.K, 818 * u.bar)
    hot = graphite_equilibrium(3900 * u.K, 818 * u.bar)
    assert cold.mole_fractions["CH4"] > 5 * cold.mole_fractions["C2H2,acetylene"]
    hot_carbon = [n for n, _ in hot.leading_species() if n != "H2"]
    assert hot_carbon[0] == "C2H2,acetylene"


def test_mole_fractions_sum_to_one():
    eq = graphite_equilibrium(3400 * u.K, 100 * u.bar)
    assert sum(eq.mole_fractions.values()) == pytest.approx(1.0, rel=1e-9)


def test_pressure_matters_at_2500_k_but_not_at_3900_k():
    """Methane formation removes gas moles; acetylene formation does not."""
    assert graphite_equilibrium(2500 * u.K, 62 * u.bar).b_prime < 0.3
    flat = [graphite_equilibrium(3900 * u.K, p * u.bar).b_prime for p in (30, 818)]
    assert flat[0] == pytest.approx(flat[1], rel=0.06)


def test_one_atmosphere_matches_low_pressure_hydrogen_ablation():
    """At 1 atm, 3000 K, acetylene and atomic H carry ~0.7 kg C per kg H."""
    eq = graphite_equilibrium(3000 * u.K, 1.01325 * u.bar)
    assert eq.b_prime == pytest.approx(0.71, abs=0.02)


def test_above_sublimation_raises_and_tabulates_as_nan():
    with pytest.raises(ValueError):
        graphite_equilibrium(4500 * u.K, 1 * u.bar)
    table = b_prime_table([3900, 4500] * u.K, [1, 818] * u.bar)
    assert np.isnan(table.loc[4500.0, 1.0])
    assert table.loc[3900.0, 818.0] == pytest.approx(5.28, abs=0.03)
