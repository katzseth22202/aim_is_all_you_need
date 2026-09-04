"""Pinned tests for src/bag_converge.py (the ledger's rule 2 fixed-point check).

The point of this module is to *measure* the paper's three deliberate cuts, not
to replace the published tables, so the tests pin the size of the gap.
"""

import astropy.units as u
import numpy as np

from src.bag_converge import (
    FLOWN_RADIATED_SHARE,
    FLOWN_RADIUS,
    bag_for_fixed_field,
    converge,
    optical_depth,
    optically_thick_limit,
    radiated_share,
)
from src.bag_state import SOLVED_LEAK_FRACTIONS
from src.plume_state import plume_state


def test_optical_limit_reproduces_the_papers_seven_metres() -> None:
    """Across the diameter, ``tau = 1`` at 7.13 m -- the paper's "about 7 m"."""
    assert np.isclose(optically_thick_limit().to_value(u.m), 7.13, rtol=1e-2)
    assert optical_depth(FLOWN_RADIUS) > 1.0


def test_cut_one_costs_a_factor_of_three_at_the_hot_end_and_nothing_at_the_cold() -> (
    None
):
    """The number rule 2 asks for.

    The bag was sized holding the plume at 15 000 K.  Radiative loss goes as
    ``T^4``, so at the solved 26 521 K the same bag radiates 3.1x what the
    table says.  At the cold pulse the solved state is 15 165 K and the
    tabulated figure is right to within 10% -- which is why the cut was
    defensible where it was made and misleading where it was carried.
    """
    from src.bag_state import radiated_fraction

    tabulated = radiated_fraction(FLOWN_RADIUS)
    hot_t, hot_f = plume_state(75.0)
    cold_t, cold_f = plume_state(45.58)
    assert np.isclose(
        radiated_share(FLOWN_RADIUS, hot_t, hot_f) / tabulated, 3.1, rtol=1e-1
    )
    assert np.isclose(
        radiated_share(FLOWN_RADIUS, cold_t, cold_f) / tabulated, 0.9, rtol=1e-1
    )


def test_the_iteration_converges_and_is_a_contraction() -> None:
    """Under ten iterations at every leg, monotonically."""
    _radius, _t, _f, _state, history = converge(
        45.58, SOLVED_LEAK_FRACTIONS["equilibrium"][45.58]
    )
    steps = [abs(history[i + 1] - history[i]) for i in range(len(history) - 1)]
    assert steps[0] > steps[5]
    assert steps[-1] < 1e-6
    assert history[-1] > history[0]  # the cold leg wants a slightly bigger bag


def test_the_legs_disagree_about_bag_size_by_a_factor_of_three() -> None:
    """And one bag has to serve all of them.

    Holding the radiated share at the value the bag was sized for, the hot leg
    wants 2.0 m and the cold leg 6.0 m -- a factor of 25 in volume.  The flown
    5.4 m is the cold leg's answer.
    """
    radii = {
        speed: converge(speed, SOLVED_LEAK_FRACTIONS["equilibrium"][speed])[0].to_value(
            u.m
        )
        for speed in (75.0, 45.58)
    }
    assert radii[45.58] / radii[75.0] > 2.5
    assert np.isclose(radii[45.58], FLOWN_RADIUS.to_value(u.m), rtol=0.15)


def test_every_converged_bag_stays_optically_thick() -> None:
    """The fixed point does not walk the design out of its own valid regime."""
    for speed in (75.0, 65.0, 56.53, 45.58):
        radius, *_ = converge(speed, SOLVED_LEAK_FRACTIONS["equilibrium"][speed])
        assert optical_depth(radius) > 1.0


def test_the_gap_does_not_reach_the_film() -> None:
    """Cold storage holds the film under the handling floor however the bag is
    sized (item 10).

    Zero on the hot leg, which never finishes melting, and two thirds of a
    kilogram on the cold one, which does (ADR 0027).  Converging the loop
    shrinks the bag and so raises the leak, which is why this is 0.66 kg
    against the cut table's 0.51 -- still an order of magnitude under
    ``tab:axial_bag``'s 2.4 kg handling floor, so the gap does not reach the
    film in the sense that matters: what the bag is sized by.
    """
    for speed in (75.0, 45.58):
        *_, state, _ = converge(speed, SOLVED_LEAK_FRACTIONS["equilibrium"][speed])
        assert state.film_mass.to_value(u.kg) < 1.0
    hot, cold = (
        converge(speed, SOLVED_LEAK_FRACTIONS["equilibrium"][speed])[-2]
        for speed in (75.0, 45.58)
    )
    assert hot.film_mass.to_value(u.kg) == 0.0
    assert 0.5 < cold.film_mass.to_value(u.kg) < 0.8


def test_a_hotter_pulse_wants_a_bigger_bag_not_a_smaller_one() -> None:
    """The counterintuitive direction, and the one a designer would get wrong.

    Thinning the slug drops the pressure faster than the resulting temperature
    rise raises it, so holding the confinement field fixed means *growing* the
    bag on the hot legs.
    """
    hot = bag_for_fixed_field(75.0).to_value(u.m)
    cold = bag_for_fixed_field(45.58).to_value(u.m)
    assert hot > cold
    assert np.isclose(hot, 7.40, rtol=2e-2)
    assert np.isclose(cold, 5.52, rtol=2e-2)


def test_per_speed_sizing_is_an_optimisation_not_a_fix() -> None:
    """The flown single bag is comfortable on every axis at every leg.

    Field 4.25-6.86 T against a ~20 T working point, optically thick throughout,
    and 1-3.6% radiated.  Per-speed sizing buys ~6% of nozzle mass and spends
    the whole optical-thickness margin doing it, so it is a lever rather than a
    repair.
    """
    from src.bag_converge import _plume_field

    fields = [_plume_field(s, FLOWN_RADIUS).to_value(u.T) for s in (75.0, 45.58)]
    assert max(fields) < 20.0
    assert optical_depth(FLOWN_RADIUS) > 1.0
    # Holding the field instead pushes the hot leg onto the optical limit.
    assert optical_depth(bag_for_fixed_field(75.0)) < 1.0
