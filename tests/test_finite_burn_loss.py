"""Tests for src/finite_burn_loss.py, pinned to the parent's integrated finite-burn losses.

``sec:ntr_departure`` integrates a burn centred on the 600 km periapsis of the
20-day cycle orbit, steered along the velocity, for the reference reactor's
5.39 km/s at 906 s.  Its first 60 m/s is a methalox kick that is not modelled
here, so the figures are checked to 5%.
"""

import pytest
from astropy import units as u
from astropy.constants import g0

from src.finite_burn_loss import finite_burn_loss, fixed_direction_loss

REACTOR_BURN = 5.39 * u.km / u.s
REACTOR_EXHAUST = (906.0 * u.s * g0).to(u.km / u.s)


@pytest.mark.parametrize(
    "seconds, printed", [(320.0, 22.0), (1200.0, 226.0), (2100.0, 486.0)]
)
def test_a_steered_burn_reproduces_the_reactor_losses(
    seconds: float, printed: float
) -> None:
    loss = finite_burn_loss(REACTOR_BURN, REACTOR_EXHAUST, seconds * u.s, steered=True)
    assert loss.to_value(u.m / u.s) == pytest.approx(printed, rel=0.05)


@pytest.mark.parametrize("seconds, printed", [(320.0, 27.0), (640.0, 101.0)])
def test_a_fixed_direction_burn_reproduces_the_pulsed_losses(
    seconds: float, printed: float
) -> None:
    """``sec:jovian_meeting_altitudes``: 27 m/s over 320 s, 101 m/s over 640 s, on the
    three-synodic 5.4 km/s burn."""
    exhaust = (1100.0 * u.s * g0).to(u.km / u.s)
    loss = finite_burn_loss(5.4 * u.km / u.s, exhaust, seconds * u.s)
    assert loss.to_value(u.m / u.s) == pytest.approx(printed, rel=0.05)


def test_the_loss_follows_the_burn_not_the_exhaust_speed() -> None:
    """At a fixed burn time the exhaust speed barely matters (under 1% from 800 s to
    1431 s), while a 7.0 km/s two-synodic burn loses ~30% more than 5.4 km/s."""
    burn, t = 5.4 * u.km / u.s, 640.0 * u.s
    low, high = (
        finite_burn_loss(burn, (isp * u.s * g0).to(u.km / u.s), t).to_value(u.m / u.s)
        for isp in (800.0, 1431.0)
    )
    assert high == pytest.approx(low, rel=0.01)
    bigger = finite_burn_loss(7.0 * u.km / u.s, (1100.0 * u.s * g0).to(u.km / u.s), t)
    assert bigger.to_value(u.m / u.s) / low == pytest.approx(7.0 / 5.4, rel=0.03)


@pytest.mark.slow
@pytest.mark.parametrize(
    "burn_km_s, seconds", [(5.33, 470.0), (7.17, 1130.0), (8.4, 2650.0)]
)
def test_the_tabulated_loss_matches_direct_integration(
    burn_km_s: float, seconds: float
) -> None:
    """The chamber-count search reads the loss off a cached table; off its nodes it
    must agree with the integration to 1%."""
    burn, t = burn_km_s * u.km / u.s, seconds * u.s
    direct = finite_burn_loss(burn, 11.0 * u.km / u.s, t).to_value(u.m / u.s)
    assert fixed_direction_loss(burn, t).to_value(u.m / u.s) == pytest.approx(
        direct, rel=0.01
    )


def test_the_table_refuses_a_burn_it_does_not_cover() -> None:
    with pytest.raises(ValueError):
        fixed_direction_loss(5.4 * u.km / u.s, 5000.0 * u.s)
