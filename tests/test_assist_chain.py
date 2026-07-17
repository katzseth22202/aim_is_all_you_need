"""Tests for src/assist_chain.py."""

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Earth

from src import conic_kernel
from src.assist_chain import (
    _powered_jovian_terminal,
    assist_chain_return,
    assist_chain_window_cadence,
    jovian_return_phasing_envelope,
    minimum_departure_burn_assist_chain,
    venus_reach_departure_floor,
)
from src.astro_constants import (
    ASSIST_CHAIN_MAX_FLYBYS,
    ASSIST_CHAIN_MAX_TRIP_TIME,
    ASSIST_CHAIN_PHASING_BUDGET,
    JUPITER_A,
    JUPITER_FLYBY_MAX_TOF,
)
from src.orbit_utils import speed_with_escape_energy
from src.propulsion import payload_mass_ratio, retrograde_jovian_hohmann_transfer
from src.retrograde_return_legs import (
    _ASSIST_MIN_LEG_TIME,
    _assist_chain_params,
    _flyby_return_leg,
)
from src.scenario_catalog import hohmann_v_infinity, lunar_transfer_periapsis_speed
from tests.test_helpers import is_nearly_equal


def test_venus_reach_departure_floor() -> None:
    # The analytic floor of the assist chain: below ~279.4 m/s even the best
    # (anti-tangential) aim leaves the transfer perihelion above Venus's orbit.
    # In particular the hoped-for 250 m/s cannot reach Venus at all.
    floor = venus_reach_departure_floor()
    assert 0.279 * u.km / u.s < floor < 0.280 * u.km / u.s
    assert floor > 0.250 * u.km / u.s


def test_venus_only_pump_ceiling_blocks_vve_at_300_mps(flyby_params) -> None:
    # ADR 0004: a V->V leg preserves the Venus-relative excess speed (the
    # Tisserand invariant), so any number of Venus-only interior flybys caps
    # the Earth-return excess at the single-Venus-pump ceiling. At a 300 m/s
    # departure that ceiling (~5.24 km/s) sits below even the Hohmann
    # Jupiter-reach excess (~8.79), so Cassini-style E-V..V-E-J chains cannot
    # close at low burn -- the pump ladder's V<->E alternations are mandatory.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies

    def max_excess_at(body_from, excess, body_to):
        best = 0.0
        for aim in np.linspace(-np.pi, np.pi, 721):
            v_t = body_from.v_circ + excess * np.cos(aim)
            v_r = excess * np.sin(aim)
            for _, v_t1, v_r1, _, _ in conic_kernel.conic_radius_crossings(
                params.flyby.mu_sun,
                body_from.orbit_radius,
                float(v_t),
                float(v_r),
                body_to.orbit_radius,
                min_leg_time=_ASSIST_MIN_LEG_TIME,
            ):
                best = max(best, float(np.hypot(v_t1 - body_to.v_circ, v_r1)))
        return best

    v_esc = flyby_params.v_esc_leo
    launch_excess = np.sqrt((v_esc + 0.300) ** 2 - v_esc**2)
    max_w_venus = max_excess_at(earth, launch_excess, venus)
    ceiling = max_excess_at(venus, max_w_venus, earth)
    assert 5.0 < ceiling < 5.5  # the ~5.24 km/s single-Venus-pump ceiling
    jupiter_reach = hohmann_v_infinity(JUPITER_A).to_value(u.km / u.s)
    assert ceiling < jupiter_reach
    # The retrograde hard floor is stricter still: even a Hohmann arrival's
    # excess at Jupiter is far below Jupiter's orbital speed, so no unpowered
    # bend of any size can turn it retrograde.
    assert jupiter_reach < flyby_params.v_jupiter_orbit


def test_assist_chain_window_cadence() -> None:
    # ADR 0005: the synodic scaffolding behind "how often can the chain fly".
    # Pins the textbook periods -- Earth-Venus synodic ~1.60 yr (583.9 d),
    # Earth-Jupiter ~1.09 yr, the 8-yr Earth-Venus near-cycle, the ~24 yr full
    # V-E-J realignment (also ~2 Jupiter years) -- and the growth arithmetic
    # at the calibrated 300 m/s chain's trip time and end-to-end ratio.
    cadence = assist_chain_window_cadence(
        total_time=3.46 * u.year, end_to_end_mass_ratio=5.72
    )
    assert is_nearly_equal(cadence.venus_window, 1.599 * u.year, percent=0.1)
    assert is_nearly_equal(cadence.jupiter_window, 1.092 * u.year, percent=0.1)
    assert is_nearly_equal(cadence.earth_venus_cycle, 7.99 * u.year, percent=0.1)
    assert 23.5 * u.year < cadence.triple_realignment < 24.5 * u.year
    # The realignment is also ~22 Earth-Jupiter synodics and ~2 Jupiter years.
    assert is_nearly_equal(
        cadence.triple_realignment, 22.0 * cadence.jupiter_window, percent=0.3
    )
    # Effective cycle spans trip time to trip time plus one Venus window.
    assert cadence.effective_cycle_floor == 3.46 * u.year
    assert is_nearly_equal(
        cadence.effective_cycle_ceiling - cadence.effective_cycle_floor,
        cadence.venus_window,
        percent=0.001,
    )
    # Doubling every ~1.4-2.0 yr at ~5.7x per cycle.
    assert 1.3 * u.year < cadence.doubling_time_floor < 1.5 * u.year
    assert 1.9 * u.year < cadence.doubling_time_ceiling < 2.1 * u.year
    # No growth cycle exists at ratio <= 1.
    with pytest.raises(ValueError, match="must exceed 1"):
        assist_chain_window_cadence(3.46 * u.year, 0.9)


@pytest.mark.slow
def test_assist_chain_return_at_300_mps(flyby_optimum, flyby_params):
    # The replay-validated headline chain: 300 m/s of departure burn (vs the
    # powered flyby's 4.45 km/s) reaches the same-target retrograde return via
    # an E-V pump ladder in well under the 10 yr cap.
    result = assist_chain_return(
        departure_burn=0.300 * u.km / u.s,
        target_collision_speed=flyby_optimum.collision_speed,
    )
    assert result is not None
    assert result.collision_speed >= flyby_optimum.collision_speed * 0.9999
    assert result.total_time <= ASSIST_CHAIN_MAX_TRIP_TIME
    assert result.total_time < 4.0 * u.year  # calibrated chain: ~3.46 yr
    assert result.sequence.startswith("E-V") and result.sequence.endswith("-J")
    assert 1 <= result.flyby_count <= ASSIST_CHAIN_MAX_FLYBYS
    # Departure is a coast state, not a flyby: its rotation must be zero.
    assert result.steps[0].rotation_angle == 0.0 * u.deg
    # Bookkeeping consistency: legs sum to the chain time, chain + return to
    # the total, and elapsed is monotone.
    legs_total = sum((step.leg_time for step in result.steps), start=0.0 * u.year)
    assert is_nearly_equal(legs_total, result.chain_time, percent=0.001)
    assert is_nearly_equal(
        result.chain_time + result.return_time, result.total_time, percent=0.001
    )
    elapsed = [step.elapsed.to_value(u.year) for step in result.steps]
    assert elapsed == sorted(elapsed)
    # Same v_b convention as the powered flyby: 1 AU closing speed folded
    # through Earth's well to the surface.
    assert is_nearly_equal(
        result.collision_speed,
        speed_with_escape_energy(result.closing_speed_1au, Earth),
        percent=0.001,
    )
    # Mass accounting: burn + phasing budget through the rocket equation, times
    # the mass ratio at the achieved v_b against the same push as the flyby.
    assert result.phasing_budget == ASSIST_CHAIN_PHASING_BUDGET
    assert result.end_to_end_mass_ratio == pytest.approx(
        result.delivered_fraction * result.payload_puffsat_mass_ratio, rel=1e-9
    )
    assert result.payload_puffsat_mass_ratio == pytest.approx(
        payload_mass_ratio(
            v_rf=lunar_transfer_periapsis_speed(), v_b=result.collision_speed
        ),
        rel=1e-6,
    )
    # The point of the chain: with propellant nearly free, the end-to-end mass
    # ratio beats the powered flyby's optimum.
    assert result.end_to_end_mass_ratio > flyby_optimum.end_to_end_mass_ratio
    # ADR 0004 bend margin: retrograde requires the Jovian arrival excess to
    # exceed Jupiter's own orbital speed (the hard floor VVE cannot clear at
    # low burn), and the bend actually used must fit inside the unpowered
    # turning limit at the ADR 0002 perijove floor -- with room to spare
    # (~96 of ~122 deg at the ~15.4 km/s arrival).
    w_jupiter = result.v_infinity_jupiter.to_value(u.km / u.s)
    assert w_jupiter > flyby_params.v_jupiter_orbit
    ecc = 1.0 + flyby_params.periapsis_floor * w_jupiter**2 / flyby_params.mu_jupiter
    bend_limit_deg = 2.0 * np.degrees(np.arcsin(1.0 / ecc))
    bend_used_deg = abs(result.jovian_bend_angle.to_value(u.deg))
    assert bend_used_deg <= bend_limit_deg
    assert bend_limit_deg - bend_used_deg > 10.0  # not scraping the limit


def test_jovian_return_phasing_envelope_closes(chain_at_300):
    # ADR 0006: retuning the unpowered Jovian bend walks the Earth arrival
    # across more than a full Earth orbit, so every launch phase admits a
    # phased intercept with no propellant and no powered flyby.
    env = jovian_return_phasing_envelope(chain_at_300)
    assert env.closes
    assert env.wraps > 3.0  # ~3.09: coverage ~1113 deg
    assert env.earth_phase_coverage > 360 * u.deg
    # It still closes under the powered flyby's own 7 yr cap, and under a 5 yr
    # one -- the cap trims the slow outbound branch but not past one wrap.
    assert jovian_return_phasing_envelope(
        chain_at_300, max_total_time=JUPITER_FLYBY_MAX_TOF
    ).closes
    capped = jovian_return_phasing_envelope(chain_at_300, max_total_time=5.0 * u.year)
    assert capped.closes
    assert capped.wraps > 1.0  # ~1.58
    # The chain's own return is inside the reachable span, and the span is a
    # superset of it -- the search just keeps the earliest arrival.
    assert env.return_time_min <= chain_at_300.return_time <= env.return_time_max


def test_jovian_return_phasing_span_is_connected(chain_at_300):
    # The span is only authority if it has no holes. Quoting max-minus-min over
    # a disconnected set would count an unreachable hole as coverage, so pin
    # the evidence of connectivity: the largest step between sampled arrivals
    # is resolution-limited and falls as 1/samples. A real hole would plateau.
    gaps = [
        jovian_return_phasing_envelope(chain_at_300, samples=n).largest_gap.to_value(
            u.day
        )
        for n in (1000, 4000, 16000)
    ]
    assert gaps[0] > gaps[1] > gaps[2]
    # Each 4x refinement should shrink the gap ~4x; allow slack for where on
    # the curve the widest step lands.
    for coarse, fine in zip(gaps, gaps[1:]):
        assert 2.0 < coarse / fine < 8.0
    assert gaps[-1] < 1.0  # sub-day at 16000 samples


def test_jovian_return_phasing_cannot_reach_the_catalog_69_km_s(
    chain_at_300, flyby_params
):
    # Phasing is free but v_b is not a choice: the bend that phases Earth
    # dictates v_b, and its ceiling is full reversal of the Tisserand-fixed
    # excess -- no timing or periapsis reaches the catalog's 69.27 km/s.
    env = jovian_return_phasing_envelope(chain_at_300)
    ceiling = env.collision_speed_max
    assert ceiling < retrograde_jovian_hohmann_transfer()
    assert is_nearly_equal(ceiling, 56.27 * u.km / u.s, percent=0.1)
    # The ceiling IS the full-reversal state: excess exactly anti-parallel to
    # Jupiter's motion, i.e. v_r = 0 and v_t = v_jupiter - v_inf.
    w = env.v_infinity.to_value(u.km / u.s)
    reversed_leg = _flyby_return_leg(
        flyby_params.v_jupiter_orbit - w, 0.0, flyby_params
    )
    assert reversed_leg is not None
    assert is_nearly_equal(
        reversed_leg.collision_speed * u.km / u.s, ceiling, percent=0.01
    )
    # Every v_b phasing can force on you is an acceptable loop, which is the
    # only reason the lottery is benign.
    assert env.collision_speed_min > lunar_transfer_periapsis_speed()


@pytest.mark.slow
def test_minimum_departure_burn_assist_chain(flyby_optimum):
    # The default probe grid brackets the Venus-reach floor: 250 and 280 m/s
    # find no chain (280 clears the floor but is Tisserand-locked too tightly
    # for the beam), while 290 m/s closes -- an order of magnitude below the
    # powered flyby's departure burn.
    scan = minimum_departure_burn_assist_chain(
        target_collision_speed=flyby_optimum.collision_speed
    )
    assert scan.minimum is not None
    assert is_nearly_equal(
        scan.minimum.departure_burn, 0.290 * u.km / u.s, percent=0.001
    )
    infeasible = [burn.to_value(u.km / u.s) for burn in scan.infeasible_burns]
    assert infeasible == [0.250, 0.280]
    assert scan.minimum.total_time <= ASSIST_CHAIN_MAX_TRIP_TIME
    assert scan.minimum.collision_speed >= scan.target_collision_speed * 0.9999
    assert scan.minimum.departure_burn < flyby_optimum.departure_burn / 10.0


def test_powered_jovian_terminal_breaks_the_vb_lottery() -> None:
    # An unpowered Jupiter has ONE knob: perijove radius IS the bend. Demanding
    # both a v_b and an arrival time over-determines it. A powered flyby has two
    # (r_p, dv), so it can hold v_b while the geometry varies. Show the second
    # knob doing real work: at a FIXED perijove, the burn moves v_b.
    params = _assist_chain_params(target_collision_speed=51.134)
    earth = params.bodies[params.earth_index]
    v_t0, v_r0 = earth.v_circ + 10.0, 0.0
    r_p = params.flyby.periapsis_floor

    def reach(burn: float):
        return _powered_jovian_terminal(
            v_t0, v_r0, earth.orbit_radius, 0.0, r_p, burn, 1.0, params
        )

    # The lottery: unpowered, this arrival cannot make the target v_b AT ALL.
    # One knob (perijove radius is the bend) against two demands.
    assert reach(0.0) is None
    # Powered, the same arrival makes it comfortably -- the burn is the second
    # knob, and it is what buys the return, not a marginal saving.
    hot = reach(4.0)
    hotter = reach(6.0)
    assert hot is not None and hotter is not None
    assert hot.return_leg.collision_speed == pytest.approx(55.872, rel=1e-3)
    assert hotter.return_leg.collision_speed == pytest.approx(69.424, rel=1e-3)
    # Perijove Oberth is why: ~7 km/s of v_b per km/s of burn.
    leverage = (
        hotter.return_leg.collision_speed - hot.return_leg.collision_speed
    ) / 2.0
    assert leverage > 5.0
    # Speeding up costs bend, so the turn must shrink as the burn grows.
    assert hotter.turn_angle < hot.turn_angle
    assert hotter.v_infinity_out > hot.v_infinity_out > hot.v_infinity_in
