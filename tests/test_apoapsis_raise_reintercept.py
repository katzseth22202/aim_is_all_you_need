"""Tests for src/apoapsis_raise_reintercept.py."""

import pytest
from astropy import units as u

from src.apoapsis_raise_reintercept import (
    apoapsis_raise_economics,
    apoapsis_raise_finite_burn,
    apoapsis_raise_reintercept,
)
from src.heliocentric_reintercept import millionfold_scaling_time
from tests.test_helpers import is_nearly_equal


def test_apoapsis_raise_reintercept_matches_design_point() -> None:
    # Apoapsis-raise design doc Sec. 4 (locked design point): raising heliocentric
    # aphelion to a phasing-exact Q ~ 2.26 AU with a methalox Oberth burn, then a
    # 4 km/s retrograde argon-SEP burn at apoapsis, lets the craft fall back to
    # intercept Earth at a ~24 km/s closing speed after ~1.69 yr. Every value is
    # pinned from the repo's primitives against the design doc's table.
    r = apoapsis_raise_reintercept()
    # The single free knob Q is root-solved to phasing-exact (residual ~ 0).
    assert is_nearly_equal(r.aphelion, 2.2579 * u.AU, percent=0.001)  # doc 2.258 AU
    assert is_nearly_equal(
        r.leg1_semimajor_axis, 1.6289 * u.AU, percent=0.001
    )  # doc 1.629 AU
    assert abs(r.phasing_residual) < 1e-6 * u.deg  # doc 0.036 deg; root-solved to ~0
    # Burns: methalox Oberth departure and the argon SEP apoapsis kick.
    assert is_nearly_equal(
        r.departure_burn, 1.2014 * u.km / u.s, percent=0.001
    )  # doc 1.202 km/s
    assert is_nearly_equal(r.sep_burn, 4.0 * u.km / u.s, percent=1e-9)  # doc 4.000
    # Rocket-equation mass fractions retained and their product.
    assert r.methalox_mass_fraction == pytest.approx(0.7244, rel=1e-3)  # doc 0.724
    assert r.sep_mass_fraction == pytest.approx(0.8155, rel=1e-3)  # doc 0.816
    assert r.combined_dry_fraction == pytest.approx(0.5908, rel=1e-3)  # doc 0.591
    # Trajectory: truncated perihelion (never reached), SOI excess, closing speed.
    assert is_nearly_equal(
        r.truncated_perihelion, 0.4598 * u.AU, percent=0.002
    )  # doc 0.460 AU
    assert is_nearly_equal(
        r.v_infinity_arrival, 21.393 * u.km / u.s, percent=0.001
    )  # doc 21.394 km/s
    assert is_nearly_equal(
        r.closing_speed, 24.059 * u.km / u.s, percent=0.001
    )  # doc 24.060 km/s
    assert is_nearly_equal(
        r.transit_time, 1.69196 * u.year, percent=0.001
    )  # doc 1.692 yr
    # The 2x escape-velocity design bar is met with the doc's ~9% margin.
    assert is_nearly_equal(
        r.twice_escape_target, 22.017 * u.km / u.s, percent=0.001
    )  # doc 22.017 km/s
    assert r.closing_speed > r.twice_escape_target
    assert float((r.closing_speed / r.twice_escape_target).value) == pytest.approx(
        1.093, rel=1e-2
    )  # doc margin 9.3%


def test_apoapsis_raise_economics_matches_design_point() -> None:
    # Apoapsis-raise design doc Sec. 5 (economics): eq:PuffSat_ratio at f=0.8 with
    # v_rf = 200 km Earth escape and v_p = the closing speed gives m_r/m_p ~ 2.62,
    # a ~+54.7% net growth per 1.69 yr cycle, and ~54 yr to a millionfold. (The
    # doc's 2.619 uses v_rf=11.0 flat; the repo uses the self-consistent 11.009 km/s
    # Earth escape speed, which rounds to the same 2.62.)
    econ = apoapsis_raise_economics()
    assert econ.payload_puffsat_mass_ratio == pytest.approx(2.6157, rel=1e-3)  # ~2.62
    assert econ.net_growth_per_cycle == pytest.approx(1.5453, rel=1e-3)  # doc 1.5469
    # net_growth_percent is just net_growth_per_cycle as a percentage gain --
    # doc's "~+54.7%".
    assert econ.net_growth_percent == pytest.approx(
        100.0 * (econ.net_growth_per_cycle - 1.0)
    )
    assert is_nearly_equal(econ.cycle_time, 1.69196 * u.year, percent=0.001)  # 1.692
    assert econ.cycles_to_millionfold == pytest.approx(31.75, rel=1e-2)  # doc 31.7
    assert is_nearly_equal(
        econ.time_to_millionfold, 53.71 * u.year, percent=0.01
    )  # doc ~53.6 yr
    assert econ.doublings_per_year == pytest.approx(0.371, rel=1e-2)  # doc 0.372
    # The whole point: the gentle 24 km/s cycle grows, but ~3x slower than the
    # ~17 yr solar-dive loop -- the price of the far lower closing speed.
    assert econ.net_growth_per_cycle > 1.0
    assert econ.time_to_millionfold > millionfold_scaling_time()


def test_apoapsis_raise_finite_burn_confirms_impulsive() -> None:
    # Apoapsis-raise design doc Sec. 3: a finite-thrust SEP burn centered on
    # apoapsis reproduces the impulsive kick -- a 60-90 day burn to within ~1% in
    # closing speed (0.3-0.7% in the doc), a 180-day burn within 2.8%. Apoapsis is
    # the slowest part of the orbit, so the thrust stays near-tangential throughout.
    r = apoapsis_raise_reintercept()
    finite_60 = apoapsis_raise_finite_burn(burn_duration=60 * u.day, reintercept=r)
    finite_90 = apoapsis_raise_finite_burn(burn_duration=90 * u.day, reintercept=r)
    finite_180 = apoapsis_raise_finite_burn(burn_duration=180 * u.day, reintercept=r)
    # Closing speed reproduced to within the doc's stated windows.
    assert finite_60.closing_speed_error < 0.005  # doc ~0.3%
    assert finite_90.closing_speed_error < 0.01  # doc ~0.7%
    assert finite_180.closing_speed_error < 0.03  # doc 2.8%
    # For the 60-90 day window the doc claims ~1 day in transit, ~0.005 AU in
    # perihelion, and under 1 deg in phasing -- all hold here.
    assert abs(finite_90.transit_time - r.transit_time) < 0.01 * u.year
    assert abs(finite_90.truncated_perihelion - r.truncated_perihelion) < 0.01 * u.AU
    assert abs(finite_90.phasing_residual) < 1.0 * u.deg
    # Longer burns drift monotonically further from the impulsive kick.
    assert (
        finite_60.closing_speed_error
        < finite_90.closing_speed_error
        < finite_180.closing_speed_error
    )
