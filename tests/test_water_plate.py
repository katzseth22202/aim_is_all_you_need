"""Tests for src/water_plate.py: the water-injected plate on the overtake leg."""

import numpy as np
import pytest
from astropy import units as u

from src.plume_thermal import chemistry_efficiency
from src.water_plate import (
    ARGON_SLUG,
    NO_BONDS,
    PLATE_MAX_SLUG_RATIO,
    WATER_SLUG,
    PlatePush,
    PlateSlug,
    chemistry_ceiling,
    optimal_plate_push,
    plate_push,
)

KM_S = u.km / u.s


@pytest.mark.parametrize("k, printed", [(9.0, 19.37), (10.0, 19.80)])
def test_a_constant_loading_reproduces_the_parents_delivery(
    k: float, printed: float
) -> None:
    """``eq:water_plate_delivery``: at the ideal ceiling, closing 68 -> 57 km/s through
    an 11 km/s push, the stack delivered per kilogram of PuffSat is 19.37 at k = 9
    and 19.80 at k = 10.  The parent's ceiling pays no bonds, so neither may we."""
    push = plate_push(
        68.0 * KM_S, 11.0 * KM_S, efficiency=1.0, slug_ratio=k,
        slug=ARGON_SLUG, impactor_bond_energy=NO_BONDS,
    )  # fmt: skip
    assert push.delivered_per_puffsat == pytest.approx(printed, abs=5e-3)
    assert push.slug_per_puffsat == pytest.approx(k, rel=1e-9)


def test_waters_bonds_cost_it_delivery_at_every_loading() -> None:
    """At equal loading water pays 50.9 MJ/kg more per kilogram than argon, a toll
    that grows with ``k`` and bites hardest as the wave slows."""
    w0, gain = 57.4 * KM_S, 10.786 * KM_S
    for k in (2.0, 6.0, 10.0):
        water = plate_push(w0, gain, 0.7, k, slug=WATER_SLUG)
        argon = plate_push(w0, gain, 0.7, k, slug=ARGON_SLUG)
        assert water.delivered_per_puffsat < argon.delivered_per_puffsat


@pytest.mark.parametrize("slug", [WATER_SLUG, ARGON_SLUG], ids=["water", "argon"])
@pytest.mark.parametrize("efficiency", [0.5, 0.7, 1.0])
def test_the_optimal_schedule_sprays_more_early_and_beats_every_constant_loading(
    efficiency: float, slug: PlateSlug
) -> None:
    """Minimising PuffSats + lambda x water, Pontryagin's principle picks at every
    pulse the ``k`` minimising ``(1 + c k) / beta(k)``, where the water price ``c``
    rises through the push and ends at ``lambda``.  So ``k`` falls pulse by pulse,
    and no constant loading costs less on the same objective."""
    w0, gain = 57.4 * KM_S, 10.786 * KM_S
    best = optimal_plate_push(
        w0, gain, efficiency, initial_water_price=0.03, max_slug_ratio=None, slug=slug
    )
    assert best.slug_ratio_start > best.slug_ratio_end > 0.0
    price = best.final_water_price

    def cost(push: PlatePush) -> float:
        return push.puffsat_fraction + price * push.slug_fraction

    # Both pushes start from the same craft mass, as the control problem does.
    for k in np.linspace(1.0, 40.0, 79):
        fixed = plate_push(w0, gain, efficiency, float(k), slug=slug)
        assert cost(best) <= cost(fixed) + 1e-9


@pytest.mark.parametrize("efficiency", [0.5, 0.7, 1.0])
def test_the_capped_schedule_stays_inside_the_validated_loadings(
    efficiency: float,
) -> None:
    """The ideal-ceiling ``beta`` is unvalidated far above the parent's k = 9-10, so
    the schedule is clamped there.  The per-pulse cost is single-peaked in ``k``, so
    clamping is still Pontryagin-optimal under the bound, and it still beats every
    constant loading the bound allows."""
    w0, gain = 57.4 * KM_S, 10.786 * KM_S
    capped = optimal_plate_push(w0, gain, efficiency, initial_water_price=0.01)
    assert capped.slug_ratio_start == pytest.approx(PLATE_MAX_SLUG_RATIO)
    assert capped.slug_per_puffsat <= PLATE_MAX_SLUG_RATIO
    uncapped = optimal_plate_push(
        w0, gain, efficiency, initial_water_price=0.01, max_slug_ratio=None
    )
    assert uncapped.slug_ratio_start > PLATE_MAX_SLUG_RATIO
    price = capped.final_water_price
    for k in np.linspace(1.0, PLATE_MAX_SLUG_RATIO, 37):
        fixed = plate_push(w0, gain, efficiency, float(k))
        assert capped.puffsat_fraction + price * capped.slug_fraction <= (
            fixed.puffsat_fraction + price * fixed.slug_fraction + 1e-9
        )


@pytest.mark.parametrize("speed, printed", [(45.58, 0.730), (75.0, 0.910)])
def test_water_on_ice_pays_the_parents_chemistry_toll(
    speed: float, printed: float
) -> None:
    """``eq:eta_chem`` at k = 8.52: 0.730 at the growth push's 45.58 km/s cold end and
    0.910 at 75 km/s, and the same as :func:`src.plume_thermal.chemistry_efficiency`."""
    ceiling = chemistry_ceiling(speed * KM_S, 8.52, WATER_SLUG)
    assert ceiling == pytest.approx(printed, abs=5e-4)
    assert ceiling == pytest.approx(chemistry_efficiency(speed * KM_S, 8.52), rel=1e-9)


def test_argon_pays_only_the_bonds_the_impactor_brings() -> None:
    """Every kilogram of blob pays its own bonds.  An argon slug has none, so on an
    ice PuffSat only the impactor's are paid, whatever the loading; all-argon pays
    nothing."""
    w = 50.0 * KM_S
    on_ice = [chemistry_ceiling(w, k, ARGON_SLUG) for k in (2.0, 10.0)]
    expected = np.sqrt(1.0 - 2.0 * 50.9e6 / 50.0e3**2)
    assert on_ice == pytest.approx([expected, expected], rel=2e-3)
    assert (
        chemistry_ceiling(w, 10.0, ARGON_SLUG, impactor_bond_energy=0.0 * u.J / u.kg)
        == 1.0
    )
    assert chemistry_ceiling(w, 10.0, WATER_SLUG) < on_ice[1]
