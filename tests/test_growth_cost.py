"""Tests for src/growth_cost.py: the growth-charged cash flows, on hand-built chains.

These run in the fast suite.  Every chain here is built by hand, so each
behaviour can be checked against a closed form; the real chain is pinned in
tests/test_growth_cost_inputs.py, under the slow marker.
"""

from dataclasses import replace

import pytest

from src.growth_cost import (
    CycleHardware,
    DeliveryOption,
    Departure,
    DesignInputs,
    DiscountSchedule,
    PriceBook,
    RouteSeed,
    break_even_price,
    delayed,
    legacy_prices,
    present_value,
    route_break_even,
    run_program,
    steady_state_cost,
    value_per_seed_dollar,
)
from src.learning_curve import LearningCurve

UNIT = 1.5e6
SEED = 80.0e3
#: A launch unit's growth push: consumes the seed, sends 2.5 seeds onward.
CYCLE = CycleHardware(
    consumed=SEED,
    growth=2.0,
    onward_puffsats=160.0e3,
    onward_rods=40.0e3,
    plate_puffsats=65.0e3,
    argon=580.0e3,
    departure_units=1,
    tanks=12.0e3,
    gas=400.0e3,
    cryostats=0.0,
    plugs=26.0e3,
    pitch=33.0e3,
)
#: A delivery: 80 t of PuffSats push 800 t of cargo and spray 530 t of argon.
DELIVERY = DeliveryOption(puffsats=80.0e3, cargo=800.0e3, slug=530.0e3)


def chain(
    cycles: int = 3, period: float = 3.0, harvest: int = 2, arrival: float = 1.0
) -> DesignInputs:
    """``cycles`` identical cycles, ``period`` years apart."""
    return DesignInputs(
        label="Methane, test",
        departure=Departure.METHANE,
        launch_unit=UNIT,
        seed_puffsats=0.8 * SEED,
        seed_rods=0.2 * SEED,
        cycles=(CYCLE,) * cycles,
        times=tuple(period * (n + 1) for n in range(cycles)),
        arrivals=(arrival,) * cycles,
        harvest=harvest,
        deliveries=((DELIVERY,),) * cycles,
    )


def test_a_stepped_rate_discounts_at_the_early_rate_until_the_switch() -> None:
    schedule = DiscountSchedule.stepped(early=0.30, late=0.10, switch_years=5.46)
    assert schedule.factor(0.0) == 1.0
    assert schedule.factor(5.46) == pytest.approx(1.30**-5.46)


def test_a_stepped_rate_discounts_at_the_late_rate_after_the_switch() -> None:
    schedule = DiscountSchedule.stepped(early=0.30, late=0.10, switch_years=5.46)
    assert schedule.factor(15.46) == pytest.approx(1.30**-5.46 * 1.10**-10.0)


def test_a_flat_rate_is_a_stepped_rate_with_one_rate() -> None:
    assert DiscountSchedule.flat(0.076).factor(9.83) == pytest.approx(1.076**-9.83)


def test_with_growth_uncharged_liquidation_pays_the_batch_as_cargo() -> None:
    # Two cycles of G = 2 grow the seed fourfold; 0.9 of it survives to deliver.
    inputs = chain(arrival=0.9)
    prices = legacy_prices(plate_per_kg=114.0)
    flows = run_program(inputs, prices, seed_price=337.0, l1_price=500.0, steady=False)
    delivered = 4.0 * 0.9 * SEED
    units = delivered / DELIVERY.puffsats
    per_unit = 25.0 * UNIT + 114.0 * 156.0e3
    expected = 500.0 * units * DELIVERY.cargo - units * per_unit
    assert flows.flows[0] == (0.0, pytest.approx(-337.0 * SEED))
    assert flows.flows[-1] == (9.0, pytest.approx(expected))
    assert sum(amount for _, amount in flows.flows[1:-1]) == 0.0


#: Every curve flat, so a launch unit's charge has a closed form.
FLAT = PriceBook(
    label="flat",
    plate=LearningCurve.flat(30.0e6),
    methane_chamber=LearningCurve.flat(40.0e6),
    package=LearningCurve.flat(100.0),
)


def test_a_growth_launch_unit_pays_every_line() -> None:
    # One growth return (harvest index 1) lofts exactly one launch unit.
    flows = run_program(chain(harvest=1), FLAT, 337.0, 500.0, steady=False).flows
    fleet = 160.0e3 * (100.0 * 2 / 60.0 + 3.0) + 40.0e3 * (100.0 * 3 / 2.5 + 4.0)
    expected = (
        25.0 * UNIT  # lob
        + 30.0e6
        + 3.0e6
        + 0.001 * 10.0 * 65.0e3  # plate, spray, film
        + 1.0 * 580.0e3  # argon
        + fleet
        + 40.0e6
        + 2.0e6  # chamber and its sprayers
        + 40.0 * 12.0e3  # tanks
        + 1.0 * 400.0e3
        + 5.0 * 26.0e3
        + 2.0 * 33.0e3  # methane, plugs, pitch
    )
    assert flows[1] == (3.0, pytest.approx(-expected))


def test_the_seed_is_bought_and_built_at_the_start() -> None:
    flows = run_program(chain(), FLAT, 337.0, 500.0, steady=False).flows
    built = 0.8 * SEED * (100.0 * 2 / 60.0 + 3.0) + 0.2 * SEED * (100.0 * 3 / 2.5 + 4.0)
    assert flows[0] == (0.0, pytest.approx(-(337.0 * SEED + built)))


def test_the_papers_flat_fleet_price_replaces_packages_and_bodies() -> None:
    paper = replace(FLAT, fleet_flat=20.0)
    flows = run_program(chain(), paper, 337.0, 500.0, steady=False).flows
    assert flows[0] == (0.0, pytest.approx(-(337.0 + 20.0) * SEED))


def test_with_growth_uncharged_the_steady_state_is_adr_0036s_perpetuity() -> None:
    # Held level from the harvest (t = 9) on, G = 2 every 3 years: each return
    # delivers (1 - 1/G) of a 4-seed batch, forever.
    inputs = chain()
    prices = legacy_prices(plate_per_kg=114.0)
    rate = 0.10
    program = run_program(inputs, prices, 337.0, 500.0, steady=True)
    units = 4.0 * 0.5 * SEED / DELIVERY.puffsats
    per_return = units * (500.0 * DELIVERY.cargo - 25.0 * UNIT - 114.0 * 156.0e3)
    step = 1.10**-3.0
    perpetuity = per_return * 1.10**-9.0 / (1.0 - step)
    after_seed = present_value(program.flows[1:], DiscountSchedule.flat(rate))
    assert after_seed == pytest.approx(perpetuity, rel=1e-9)


def test_the_steady_state_overhead_is_the_growth_charge_over_g_minus_one() -> None:
    # Per delivered kilogram: the delivery's own lob, plate and argon, plus
    # C_g / (G - 1) spread over the cargo one consumed seed's worth pushes.
    growth_unit = -run_program(chain(harvest=1), FLAT, 337.0, 500.0, False).flows[1][1]
    delivery_unit = 25.0 * UNIT + 30.0e6 + 3.0e6 + 0.001 * 10.0 * 80.0e3 + 1.0 * 530.0e3
    per_puffsat = DELIVERY.cargo / DELIVERY.puffsats
    overhead = growth_unit / ((2.0 - 1.0) * SEED * per_puffsat)
    total, lines = steady_state_cost(chain(), FLAT, l1_price=500.0)
    assert total == pytest.approx(delivery_unit / DELIVERY.cargo + overhead, rel=1e-9)
    assert sum(lines.values()) == pytest.approx(total)


STEPPED = DiscountSchedule.stepped(0.30, 0.10, switch_years=6.0)


def test_the_break_even_price_repays_the_seed_and_every_growth_cost() -> None:
    price = break_even_price(chain(), FLAT, 337.0, STEPPED, steady=True)
    assert price is not None
    program = run_program(chain(), FLAT, 337.0, price, steady=True)
    assert present_value(program.flows, STEPPED) == pytest.approx(
        0.0, abs=1e-3 * program.seed_outlay
    )
    assert value_per_seed_dollar(chain(), FLAT, 337.0, price, STEPPED) == pytest.approx(
        1.0, abs=1e-3
    )


def test_value_per_seed_dollar_clears_one_only_above_break_even() -> None:
    price = break_even_price(chain(), FLAT, 337.0, STEPPED, steady=True)
    assert price is not None
    assert value_per_seed_dollar(chain(), FLAT, 337.0, price + 20.0, STEPPED) > 1.0
    assert value_per_seed_dollar(chain(), FLAT, 337.0, price - 20.0, STEPPED) < 1.0


def test_a_break_even_above_the_cap_is_none() -> None:
    dear = replace(FLAT, lob=500.0)
    assert (
        break_even_price(chain(), dear, 9293.0, STEPPED, steady=False, cap=500.0)
        is None
    )


def test_each_delivery_flies_the_schedule_that_nets_most_per_puffsat() -> None:
    # A heavier argon load buys 10% more cargo for 400 t more argon: worth it
    # at $500/kg, not at $50/kg once argon costs $100/kg.
    lean = DeliveryOption(puffsats=80.0e3, cargo=800.0e3, slug=130.0e3)
    loaded = DeliveryOption(puffsats=80.0e3, cargo=880.0e3, slug=530.0e3)
    inputs = replace(chain(), deliveries=((lean, loaded),) * 3)
    dear_argon = replace(FLAT, argon=100.0)

    def net_at(price: float) -> float:
        return run_program(inputs, dear_argon, 337.0, price, steady=False).flows[-1][1]

    units = 4.0 * SEED / 80.0e3
    common = units * (25.0 * UNIT + 30.0e6 + 3.0e6) + 0.001 * 10.0 * 4.0 * SEED
    assert net_at(500.0) == pytest.approx(
        units * (500.0 * 880.0e3 - 100.0 * 530.0e3) - common
    )
    assert net_at(50.0) == pytest.approx(
        units * (50.0 * 800.0e3 - 100.0 * 130.0e3) - common
    )


def test_growth_and_delivery_plates_share_one_learning_tally() -> None:
    learning = replace(FLAT, plate=LearningCurve(30.0e6, 6.0e6, 0.8))
    program = run_program(chain(), learning, 337.0, 500.0, steady=False)
    growth_units = 1.0 + 2.0  # returns 0 and 1 loft 1 and 2 launch units
    delivery_units = 4.0 * SEED / DELIVERY.puffsats
    assert program.plates == pytest.approx(growth_units + delivery_units)


def test_a_slower_seed_delays_every_return_and_the_proof() -> None:
    # A seed route that comes back 2.5 years later shifts the whole program; the
    # seed is still paid at t = 0, so the 30% rate runs 2.5 years longer.
    late = delayed(chain(), 2.5)
    assert late.times == pytest.approx((5.5, 8.5, 11.5))
    assert late.proof_years == pytest.approx(8.5)
    # A lap of the repeating chain is still 9 years long, not 11.5.
    assert late.chain_years == pytest.approx(chain().chain_years)
    on_time = run_program(chain(), FLAT, 337.0, 500.0, steady=True).flows
    slipped = run_program(late, FLAT, 337.0, 500.0, steady=False).flows
    assert slipped[0] == on_time[0]
    assert [t for t, _ in slipped[1:]] == pytest.approx([5.5, 8.5, 11.5])


def test_a_late_cheap_seed_is_the_on_time_seed_at_its_discounted_price() -> None:
    # Delaying the program by dt while the rate is still 30% (the proof slides
    # with it) scales every later flow by 1.3^-dt, so a route k times cheaper
    # and dt later breaks even where the direct seed would at S / (k 1.3^-dt).
    # Exact only when the seed's own manufacture, paid at t = 0 and not scaled
    # by k, is free, so the report applies delayed() rather than this shortcut.
    k, dt, price = 2.9, 2.3, 9293.0
    book = replace(FLAT, fleet_flat=0.0)
    late_chain = delayed(chain(), dt)

    def schedule(inputs: DesignInputs) -> DiscountSchedule:
        return DiscountSchedule.stepped(0.30, 0.10, inputs.proof_years)

    late = break_even_price(late_chain, book, price / k, schedule(late_chain), True)
    equivalent = break_even_price(
        chain(), book, price / (k * 1.3**-dt), schedule(chain()), True
    )
    assert late is not None and equivalent is not None
    assert late == pytest.approx(equivalent, rel=1e-6)


def test_a_route_with_no_saving_and_no_delay_is_the_direct_seed() -> None:
    def schedule(inputs: DesignInputs) -> DiscountSchedule:
        return DiscountSchedule.stepped(0.30, 0.10, inputs.proof_years)

    direct = break_even_price(chain(), FLAT, 9293.0, schedule(chain()), True)
    same = route_break_even(
        chain(), FLAT, 9293.0, RouteSeed("direct", True, 1.0, 0.0), schedule
    )
    assert same == pytest.approx(direct)


def test_a_cheaper_later_route_pays_only_when_k_beats_the_wait() -> None:
    # At 30% until the (delayed) proof, a route 2.2x cheaper and 2 years late
    # beats direct (2.2 > 1.3^2 = 1.69); one 1.5x cheaper does not. The seed's
    # own manufacture is zeroed so the k (1 + r)^-dt test is exact.
    book = replace(FLAT, fleet_flat=0.0)

    def schedule(inputs: DesignInputs) -> DiscountSchedule:
        return DiscountSchedule.stepped(0.30, 0.10, inputs.proof_years)

    direct = break_even_price(chain(), book, 9293.0, schedule(chain()), True)
    good = route_break_even(
        chain(), book, 9293.0, RouteSeed("good", True, 1 / 2.2, 2.0), schedule
    )
    poor = route_break_even(
        chain(), book, 9293.0, RouteSeed("poor", True, 1 / 1.5, 2.0), schedule
    )
    assert direct is not None and good is not None and poor is not None
    assert good < direct < poor


def test_risk_is_charged_once_at_the_proof_and_time_at_the_bond_rate() -> None:
    # ADR 0040: 10% a year throughout; flows from the proof on happen only if
    # the cycle works (50%); flows before it are spent regardless.
    risked = DiscountSchedule.risked(0.10, 0.5, proof_years=5.46)
    assert risked.factor(3.0) == pytest.approx(1.1**-3.0)
    assert risked.factor(5.46) == pytest.approx(0.5 * 1.1**-5.46)
    assert risked.factor(12.0) == pytest.approx(0.5 * 1.1**-12.0)
    # Certain success is the plain 10% schedule.
    sure = DiscountSchedule.risked(0.10, 1.0, proof_years=5.46)
    assert sure.factor(12.0) == pytest.approx(DiscountSchedule.flat(0.10).factor(12.0))
    # The old stepped schedule is unchanged by the new fields.
    assert STEPPED.factor(8.0) == pytest.approx(1.3**-6.0 * 1.1**-2.0)
