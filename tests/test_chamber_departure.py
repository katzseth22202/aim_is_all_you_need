"""Tests for src/chamber_departure.py: the walled chamber's departure, hardware charged."""

import pytest
from astropy import units as u

from src.chamber_departure import (
    best_departure,
    doubling_time,
    growth_per_cycle,
    price_chain_departures,
    price_departure,
    square_law_loss,
)
from src.chamber_isp import (
    HYDROGEN_5500K,
    METHANE_7000K,
    PLUG_RATIO,
    ROD_MASS,
    ChamberPairing,
    chamber_departure_burn,
)
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.two_wave_growth import TwoWaveCycle, adaptive_two_wave_cycles

KM_S = u.km / u.s
#: A 2S-like departure: 200 km cycle periapsis, 7.0 km/s burn, 61 km/s wave.
V_PERI = 10.95 * KM_S
BURN = 7.0 * KM_S
WAVE = 61.0 * KM_S


def test_the_stack_delivered_is_net_of_tanks_and_chambers() -> None:
    """Tanks are charged on the gas only (plug and pitch ride without tanks), and
    the chambers' wall and nozzle ride the whole burn."""
    stack = 1000.0 * u.t
    dep = price_departure(
        stack,
        V_PERI,
        BURN,
        WAVE,
        HYDROGEN_5500K,
        0.858,
        chambers=6,
        loss_model=square_law_loss,
    )
    gas = (1.0 - dep.delivered_fraction) - PLUG_RATIO * dep.rod_mass_fraction
    assert dep.tank_share == pytest.approx(
        HYDROGEN_5500K.tank_fraction * gas, rel=1e-12
    )
    assert dep.chamber_share == pytest.approx(
        float((6 * dep.chamber_unit_mass / stack).to_value(u.one)), rel=1e-12
    )
    assert dep.delivered_net == pytest.approx(
        dep.delivered_fraction - dep.tank_share - dep.chamber_share, rel=1e-12
    )


@pytest.mark.parametrize("chambers", [3, 6, 12])
def test_the_fixed_direction_loss_is_paid_on_the_burn_it_lengthens(
    chambers: int,
) -> None:
    """The loss lengthens the burn and the burn sets the loss, so the priced burn is
    a fixed point: the loss paid is the model's loss at the burn's own length.
    Checked on the parent's square law, 27 m/s at 320 s, which is cheap to call."""
    stack = 1000.0 * u.t
    dep = price_departure(
        stack, V_PERI, BURN, WAVE, HYDROGEN_5500K, 0.858, chambers=chambers,
        loss_model=square_law_loss,
    )  # fmt: skip
    pulses = dep.rod_mass_fraction * float((stack / ROD_MASS).to_value(u.one))
    assert dep.pulses == pytest.approx(pulses, rel=1e-12)
    assert dep.burn_time.to_value(u.s) == pytest.approx(
        pulses / (4.0 * chambers), rel=1e-12
    )
    expected = 27.0 * (dep.burn_time.to_value(u.s) / 320.0) ** 2
    assert dep.finite_burn_loss.to_value(u.m / u.s) == pytest.approx(expected, rel=1e-6)
    priced = chamber_departure_burn(
        V_PERI, V_PERI + BURN + dep.finite_burn_loss, WAVE, HYDROGEN_5500K, 0.858
    )
    assert dep.delivered_fraction == pytest.approx(priced.delivered_fraction, rel=1e-6)


@pytest.mark.parametrize("pairing", [HYDROGEN_5500K, METHANE_7000K])
@pytest.mark.parametrize("stack_t", [100.0, 1000.0])
def test_the_chamber_count_trades_wall_mass_against_the_loss(
    pairing: ChamberPairing, stack_t: float
) -> None:
    """Each chamber is tens of tonnes, and each one removed lengthens the burn and
    its loss.  The count chosen delivers more than one fewer or one more."""
    stack = stack_t * u.t
    best = best_departure(
        stack, V_PERI, BURN, WAVE, pairing, 0.7, loss_model=square_law_loss
    )
    for neighbour in (best.chambers - 1, best.chambers + 1):
        if neighbour < 1:
            continue
        other = price_departure(
            stack, V_PERI, BURN, WAVE, pairing, 0.7, chambers=neighbour,
            loss_model=square_law_loss,
        )  # fmt: skip
        assert other.delivered_net < best.delivered_net


def test_growth_per_cycle_reproduces_the_parents_hydrogen_ledger() -> None:
    """``tab:h2_breakdown``, hydrogen 2 m^2: push ratio 8.43, and at ``k = 7.77`` the
    0.46 of stack spent needs 0.46/7.77 of rods.  The departure wave is that share of
    each pushed kilogram, so a share 1/(1 + rods x push) pushes: 0.667.  With 0.446
    delivered net that is 2.51 per cycle, doubling in 1.646 yr on the 2.18 yr clock."""
    rods = (1.0 - 0.446 - 0.094) / 7.77
    growth = growth_per_cycle(8.43, rods, 0.446)
    assert 1.0 / (1.0 + rods * 8.43) == pytest.approx(0.667, abs=5e-4)
    assert growth == pytest.approx(2.51, abs=5e-3)
    assert doubling_time(growth, 2.184 * u.yr).to_value(u.yr) == pytest.approx(
        1.646, abs=5e-3
    )


def _cycle(multiple: int, burn: float, v_b: float) -> TwoWaveCycle:
    period = {2: 2.184, 3: 3.276}[multiple]
    return TwoWaveCycle(
        index=0, departure_jd=0.0, return_jd=0.0, synodic_multiple=multiple,
        period_years=period, departure_burn=burn, nozzle_wave_v_b=v_b,
        nozzle_wave_dsm=0.0, split_days=10.0, growth_wave_arrival_jd=0.0,
        growth_wave_v_b=v_b + 2.0, growth_wave_burn=0.0,
    )  # fmt: skip


def test_each_cycle_departs_on_its_own_burn_into_its_own_wave() -> None:
    """A cycle's burn starts at the 200 km cycle periapsis and meets the nozzle
    wave head-on, so ignition closes at ``v_b`` plus the periapsis speed."""
    stack = 1000.0 * u.t
    cycles = [_cycle(2, 7.0, 61.0), _cycle(3, 5.33, 55.4)]
    priced = price_chain_departures(
        cycles, stack, METHANE_7000K, 0.7, loss_model=square_law_loss
    )
    v_peri = puffsat_cycle_periapsis_speed()
    for cycle, ledger in zip(cycles, priced):
        alone = best_departure(
            stack, v_peri, cycle.departure_burn * KM_S, cycle.nozzle_wave_v_b * KM_S,
            METHANE_7000K, 0.7, loss_model=square_law_loss,
        )  # fmt: skip
        assert ledger == alone
    assert priced[0].delivered_net < priced[1].delivered_net


@pytest.mark.slow
def test_the_flown_chain_departs_every_cycle() -> None:
    """All eleven flown cycles (seven 2S, four 3S) close with the methane chamber."""
    cycles = adaptive_two_wave_cycles()
    priced = price_chain_departures(cycles, 1000.0 * u.t, METHANE_7000K, 0.7)
    assert [c.synodic_multiple for c in cycles].count(2) == 7
    assert len(priced) == 11 and all(p.delivered_net > 0.0 for p in priced)


def test_the_loss_model_is_asked_about_the_cycles_own_burn() -> None:
    """The parent's square law ignores the burn's size; the integrated loss does
    not, so the model must be handed the impulsive burn, not the priced one."""
    asked = []

    def model(burn: u.Quantity, burn_time: u.Quantity) -> u.Quantity:
        asked.append(burn.to_value(u.km / u.s))
        return square_law_loss(burn, burn_time)

    price_departure(
        1000.0 * u.t,
        V_PERI,
        BURN,
        WAVE,
        METHANE_7000K,
        0.7,
        chambers=2,
        loss_model=model,
    )
    assert asked and all(b == pytest.approx(7.0) for b in asked)


@pytest.mark.slow
def test_the_integrated_loss_still_leaves_a_best_chamber_count() -> None:
    stack = 1000.0 * u.t
    best = best_departure(stack, V_PERI, BURN, WAVE, METHANE_7000K, 0.7)
    for neighbour in (best.chambers - 1, best.chambers + 1):
        if neighbour >= 1:
            other = price_departure(
                stack, V_PERI, BURN, WAVE, METHANE_7000K, 0.7, chambers=neighbour
            )
            assert other.delivered_net < best.delivered_net
