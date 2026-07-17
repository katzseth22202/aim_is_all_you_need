"""Tests for src/retrograde_return_legs.py (the shared substrate)."""

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Jupiter
from boinor.core.iod import izzo
from boinor.threebody.flybys import compute_flyby

from src import conic_kernel
from src.astro_constants import EARTH_A, JUPITER_A
from src.orbit_utils import apoapsis_speed, hyperbolic_eccentricity, orbit_from_rp_ra
from src.propulsion import retrograde_jovian_hohmann_transfer
from src.retrograde_return_legs import (
    _ASSIST_MIN_LEG_TIME,
    _assist_chain_params,
    _body_state,
    _earth_phase_mismatch,
    _flyby_mismatch_burn,
    _flyby_return_leg,
    _jupiter_assist_body,
    _mean_motion,
    _phased_jovian_flyby,
    _phased_ladder_burn,
    _phased_leg_rotations,
    _powered_flyby_leg,
    _powered_flyby_params,
    _powered_node_burn,
)
from tests.test_helpers import is_nearly_equal


def test_flyby_return_leg_reproduces_retrograde_hohmann(flyby_params):
    # The v_b convention pin: fed the retrograde Jupiter->Earth Hohmann state
    # (tangential retrograde at Jupiter's orbit radius, apoapsis speed of the
    # 1 AU x 5.2028 AU transfer ellipse), the float return leg must reproduce
    # retrograde_jovian_hohmann_transfer() -- the ~69.27 km/s the catalog's
    # three Jovian rows assume.
    v_apo = apoapsis_speed(
        orbit_from_rp_ra(apoapsis_radius=JUPITER_A, periapsis_radius=EARTH_A)
    ).to_value(u.km / u.s)
    leg = _flyby_return_leg(-v_apo, 0.0, flyby_params)
    assert leg is not None
    reference = retrograde_jovian_hohmann_transfer()
    assert is_nearly_equal(leg.collision_speed * u.km / u.s, reference, percent=0.001)
    # Tangential arrival: the return perihelion is 1 AU itself.
    assert is_nearly_equal(leg.perihelion * u.km, EARTH_A.to(u.km), percent=0.001)
    # Half the transfer ellipse period, ~2.73 yr.
    assert is_nearly_equal((leg.tof * u.s).to(u.year), 2.731 * u.year, percent=0.01)


def test_flyby_return_leg_rejects_prograde(flyby_params):
    # Retrograde is orbit-level: a prograde tangential state must be rejected,
    # not scored (the retrograde-plunge degeneracy guard, ADR 0002).
    assert _flyby_return_leg(+5.0, 0.0, flyby_params) is None
    assert _flyby_return_leg(0.0, -10.0, flyby_params) is None


def test_powered_flyby_unpowered_limit(flyby_params):
    # With dv2 = 0 the split hyperbola collapses to a single one: the excess
    # speed is conserved and the turn equals the unpowered 2*asin(1/e), with e
    # from the cross-module hyperbolic_eccentricity. Pin point verified
    # feasible: v_inf = 11 km/s, tangential, periapsis 6x the floor.
    periapsis_radius = 6.0 * flyby_params.periapsis_floor
    leg = _powered_flyby_leg(
        v_infinity_earth=11.0,
        aim_angle=0.0,
        periapsis_radius=periapsis_radius,
        flyby_burn=0.0,
        descending_arrival=False,
        bend_sign=1.0,
        params=flyby_params,
    )
    assert leg is not None
    assert leg.v_infinity_out == pytest.approx(leg.v_infinity_in, rel=1e-9)
    ecc = hyperbolic_eccentricity(
        periapsis_radius * u.km, leg.v_infinity_in * u.km / u.s, Jupiter
    )
    assert leg.turn_angle == pytest.approx(2.0 * np.arcsin(1.0 / ecc), rel=1e-9)


def test_powered_flyby_burn_grows_v_infinity(flyby_params):
    # A periapsis burn pumps the outgoing excess speed above the incoming one
    # (Oberth) while shrinking the outbound bend contribution: the powered
    # trajectory's turn must be smaller than the unpowered one at the same
    # periapsis, since the faster outbound hyperbola bends less.
    periapsis_radius = 2.0 * flyby_params.periapsis_floor
    powered = _powered_flyby_leg(
        12.0, 0.0, periapsis_radius, 2.0, False, 1.0, flyby_params
    )
    assert powered is not None
    assert powered.v_infinity_out > powered.v_infinity_in
    ecc_in = 1.0 + periapsis_radius * powered.v_infinity_in**2 / (
        flyby_params.mu_jupiter
    )
    unpowered_turn = 2.0 * np.arcsin(1.0 / ecc_in)
    assert powered.turn_angle < unpowered_turn


def test_bend_limit_matches_boinor_flyby(flyby_params) -> None:
    # The turning formula (e = 1 + r_p*w^2/mu, delta = 2*arcsin(1/e)) carries a
    # lot of weight: _jovian_terminal bounds its bend with it, the phased leg
    # solver bounds its root search with it, _flyby_mismatch_burn prices nodes
    # with it, and ADR 0006's 56.27 km/s Tisserand ceiling and 122.5 deg bend
    # limit both follow from it. Check it against an implementation we did not
    # write, at the excesses the chain actually reaches.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    cases = [
        (venus, 3.349),
        (venus, 11.114),
        (earth, 5.210),
        (earth, 17.842),
    ]
    for body, excess in cases:
        v_body = np.array([0.0, body.v_circ, 0.0]) * (u.km / u.s)
        v_sc = np.array([excess, body.v_circ, 0.0]) * (u.km / u.s)
        v_out, delta = compute_flyby(
            v_sc, v_body, body.mu * u.km**3 / u.s**2, body.min_periapsis * u.km
        )
        ecc = conic_kernel.hyperbolic_eccentricity(body.mu, body.min_periapsis, excess)
        mine = float(np.degrees(conic_kernel.unpowered_bend_angle(ecc)))
        assert delta.to_value(u.deg) == pytest.approx(mine, abs=1e-9)
        # And the flyby is unpowered: it rotates the excess without rescaling
        # it. That is the Tisserand invariant the whole assist chain rests on.
        excess_out = np.linalg.norm((v_out - v_body).to_value(u.km / u.s))
        assert excess_out == pytest.approx(excess, abs=1e-9)
    # The same formula at the Jovian arrival gives ADR 0006's bend limit.
    jovian_ecc = conic_kernel.hyperbolic_eccentricity(
        flyby_params.mu_jupiter, flyby_params.periapsis_floor, 15.3685
    )
    jovian = float(np.degrees(conic_kernel.unpowered_bend_angle(jovian_ecc)))
    v_body = np.array([0.0, flyby_params.v_jupiter_orbit, 0.0]) * (u.km / u.s)
    v_sc = np.array([15.3685, flyby_params.v_jupiter_orbit, 0.0]) * (u.km / u.s)
    _, delta = compute_flyby(
        v_sc,
        v_body,
        flyby_params.mu_jupiter * u.km**3 / u.s**2,
        flyby_params.periapsis_floor * u.km,
    )
    assert delta.to_value(u.deg) == pytest.approx(jovian, abs=1e-9)
    assert jovian == pytest.approx(122.48, abs=0.01)


def test_lambert_reproduces_the_conic_propagator(flyby_params) -> None:
    # The phased ladder prices legs with Lambert while the phasing-free chain
    # propagates them as conics. Both must describe the same arc, or the price
    # is of a different trajectory than the one being priced. Generate a leg by
    # propagation, then Lambert between its endpoints: the velocities must
    # match. This also pins the swept angle and the (tangential, radial-out)
    # to Cartesian convention that _body_state encodes.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    excess = 2.5875  # the 300 m/s chain's departure
    v_t0, v_r0 = earth.v_circ - excess, 0.0  # aim -180: brake to fall inward
    crossings = conic_kernel.conic_radius_crossings(
        params.flyby.mu_sun,
        earth.orbit_radius,
        v_t0,
        v_r0,
        venus.orbit_radius,
        min_leg_time=_ASSIST_MIN_LEG_TIME,
    )
    assert crossings
    for leg_tof, v_t1, v_r1, _, swept in crossings:
        r0, _ = _body_state(earth.orbit_radius, 0.0, earth.v_circ)
        r1, _ = _body_state(venus.orbit_radius, swept, venus.v_circ)
        depart = np.array([-v_t0 * np.sin(0.0), v_t0 * np.cos(0.0), 0.0]) + np.array(
            [v_r0 * np.cos(0.0), v_r0 * np.sin(0.0), 0.0]
        )
        arrive = np.array(
            [-v_t1 * np.sin(swept), v_t1 * np.cos(swept), 0.0]
        ) + np.array([v_r1 * np.cos(swept), v_r1 * np.sin(swept), 0.0])
        lambert_depart, lambert_arrive = izzo(
            params.flyby.mu_sun, r0, r1, leg_tof, 0, True, True, 35, 1e-8
        )
        assert np.linalg.norm(lambert_depart - depart) < 1e-6
        assert np.linalg.norm(lambert_arrive - arrive) < 1e-6


def test_phased_ladder_burn_is_free_when_the_planets_cooperate() -> None:
    # The pricing model's calibration: hand it planets placed exactly where the
    # phasing-free chain imagines them, and it must charge nothing beyond the
    # departure -- that chain is perfectly phased for its own invented
    # ephemeris, and every turn it uses is inside the bend limit. Any node cost
    # here would mean the model charges for phasing that is already satisfied.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    excess = 2.5875
    leg1 = [
        c
        for c in conic_kernel.conic_radius_crossings(
            params.flyby.mu_sun,
            earth.orbit_radius,
            earth.v_circ - excess,
            0.0,
            venus.orbit_radius,
            min_leg_time=_ASSIST_MIN_LEG_TIME,
        )
        if not c.outbound
    ]
    assert leg1
    tof1, v_t1, v_r1, _, swept1 = leg1[0]
    # Rotate at Venus by the real chain's recorded bend, then run on to Earth.
    ex_t, ex_r = v_t1 - venus.v_circ, v_r1
    w2 = float(np.hypot(ex_t, ex_r))
    angle = float(np.arctan2(ex_r, ex_t)) + np.radians(44.08122687)
    leg2 = [
        c
        for c in conic_kernel.conic_radius_crossings(
            params.flyby.mu_sun,
            venus.orbit_radius,
            venus.v_circ + w2 * np.cos(angle),
            w2 * np.sin(angle),
            earth.orbit_radius,
            min_leg_time=_ASSIST_MIN_LEG_TIME,
        )
        if c.outbound
    ]
    assert leg2
    tof2, _, _, _, swept2 = leg2[0]
    # Place each planet exactly at its leg's arrival point.
    longitudes = [
        0.0,
        swept1 - _mean_motion(venus) * tof1,
        (swept1 + swept2) - _mean_motion(earth) * (tof1 + tof2),
    ]
    priced = _phased_ladder_burn(
        0.0, [tof1, tof2], (earth, venus, earth), longitudes, params
    )
    assert priced is not None
    # The departure is the real chain's 300 m/s, recovered through Lambert.
    assert priced.departure_burn == pytest.approx(0.300, abs=1e-4)
    # And the flyby needs no help at all.
    assert priced.node_total < 1e-6
    assert all(burn < 1e-6 for burn in priced.node_burns)


def test_powered_and_retired_nodes_agree_on_a_pure_rotation() -> None:
    # The two node models disagree about what a speed change costs, but a node
    # that only *rotates* the excess within the bend limit is free under both.
    # Any daylight here would mean one of them charges for geometry the flyby
    # supplies for nothing, which would make the ADR 0007 re-price unreadable.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, _, _ = params.bodies
    excess_in = np.array([4.0, 0.0, 0.0])
    for turn_deg in (0.0, 10.0, 30.0):
        turn = np.radians(turn_deg)
        excess_out = 4.0 * np.array([np.cos(turn), np.sin(turn), 0.0])
        powered, _ = _powered_node_burn(excess_in, excess_out, venus)
        retired = _flyby_mismatch_burn(excess_in, excess_out, venus)
        assert powered < 1e-9
        assert retired < 1e-9


def test_powered_node_burn_gets_no_oberth_discount_for_a_zero_turn() -> None:
    # A node that needs no turn has no reason to go anywhere near the planet:
    # the bend is strictly positive at every finite periapsis, so zero turn is
    # only approached as r_p -> infinity, where the Oberth discount vanishes and
    # the speed change costs its full value at infinity. Charging burn(floor)
    # here would award the *deepest* discount to the turn that earned none --
    # a free 2x that an optimizer will drive the whole chain into.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    for body in (venus, earth):
        excess_in = np.array([5.0, 0.0, 0.0])
        excess_out = np.array([7.0, 0.0, 0.0])
        powered, periapsis = _powered_node_burn(excess_in, excess_out, body)
        # The whole 2 km/s, not the ~1 km/s a grazing pass would cost.
        assert powered == pytest.approx(2.0, abs=1e-6)
        assert periapsis > 1e6 * body.min_periapsis
    # And the discount must arrive smoothly as the turn grows, not in a jump:
    # more turn forces a lower periapsis, which deepens the well.
    previous = 2.0
    for turn_deg in (10.0, 20.0, 40.0, 60.0):
        turn = np.radians(turn_deg)
        excess_out = 7.0 * np.array([np.cos(turn), np.sin(turn), 0.0])
        burn, periapsis = _powered_node_burn(
            np.array([5.0, 0.0, 0.0]), excess_out, venus
        )
        assert burn < previous
        previous = burn


def test_phased_leg_solves_onto_the_moving_target() -> None:
    # The phased ladder's primitive: solve the flyby rotation so the leg lands
    # where the planet actually is, rather than sampling the rotation and
    # letting the planet be wherever the leg lands (the phasing-free model).
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    n_venus = _mean_motion(venus)
    # A departure aims its excess for free -- no flyby, so no bend limit.
    solutions = []
    for outbound in (False, True):
        solutions += _phased_leg_rotations(
            earth,
            0.0,
            2.5875,
            0.0,
            venus,
            np.radians(300.0),
            outbound,
            params,
            samples=721,
            rotation_limit=np.pi,
        )
    assert solutions
    # Every solved rotation must actually put the craft on Venus: recompute the
    # arrival longitude independently and check Venus is there at that time.
    for rotation, leg_tof, _, _ in solutions:
        assert abs(rotation) <= np.pi
        assert leg_tof > 0.0
        arrival_is = np.radians(300.0) + n_venus * leg_tof
        v_t0 = earth.v_circ + 2.5875 * np.cos(rotation)
        v_r0 = 2.5875 * np.sin(rotation)
        swept = [
            c.swept
            for c in conic_kernel.conic_radius_crossings(
                params.flyby.mu_sun,
                earth.orbit_radius,
                v_t0,
                v_r0,
                venus.orbit_radius,
                min_leg_time=_ASSIST_MIN_LEG_TIME,
            )
            if abs(c.tof - leg_tof) < 1.0
        ]
        assert swept, "solved leg did not replay"
        assert conic_kernel.wrap_pi(swept[0] - arrival_is) == pytest.approx(
            0.0, abs=1e-6
        )


def test_phased_leg_windows_recur_at_the_earth_venus_synodic() -> None:
    # The check that separates a working phasing model from a plausible one:
    # the launch-window cadence must *emerge*, never be supplied. Advancing the
    # launch epoch by one Earth-Venus synodic period restores the same relative
    # geometry (Venus gains exactly one lap on Earth), so the identical
    # rotations must solve. An earlier draft advanced the target from t=0
    # instead of from the flyby, double-counting the epoch and inventing a
    # spurious 162 d cadence -- this pins the real one.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    n_earth, n_venus = _mean_motion(earth), _mean_motion(venus)
    synodic = 2.0 * np.pi / abs(n_venus - n_earth)
    assert is_nearly_equal((synodic * u.s).to(u.day), 583.9 * u.day, percent=0.1)

    def solve_at(epoch: float):
        out = []
        for outbound in (False, True):
            out += _phased_leg_rotations(
                earth,
                n_earth * epoch,
                2.5875,
                0.0,
                venus,
                n_venus * epoch,
                outbound,
                params,
                samples=721,
                rotation_limit=np.pi,
            )
        return sorted(r for r, _, _, _ in out)

    inside_window = 496.0 * 86400.0
    now = solve_at(inside_window)
    assert now, "the measured E-V window near day 496 should admit a phased leg"
    later = solve_at(inside_window + synodic)
    assert len(later) == len(now)
    assert later == pytest.approx(now, abs=1e-6)
    # And the geometry is genuinely a window, not a permanent state: half a
    # synodic period later the relative geometry is opposite and it shuts.
    assert not solve_at(inside_window + synodic / 2.0)


def test_powered_jovian_flyby_burn_changes_the_turn_not_just_the_speed() -> None:
    # The property that makes a powered flyby two knobs instead of one: the burn
    # fires at perijove, so the OUTGOING hyperbola has its own eccentricity and
    # the total bend is asin(1/e_in) + asin(1/e_out). Speeding up raises e_out
    # and so COSTS bend; slowing down buys it. Charging the burn against the
    # incoming eccentricity twice -- 2*asin(1/e_in), as the retired node model
    # did -- overstates what the flyby can point at.
    params = _assist_chain_params(target_collision_speed=51.134)
    flyby = params.flyby
    r_p = flyby.periapsis_floor
    w_in = 8.0
    mu_j = flyby.mu_jupiter
    ecc_in = 1.0 + r_p * w_in**2 / mu_j
    unpowered_turn = 2.0 * np.arcsin(1.0 / ecc_in)

    def turn_for(burn: float) -> float:
        v_peri_in = np.sqrt(w_in**2 + 2.0 * mu_j / r_p)
        w_out = np.sqrt((v_peri_in + burn) ** 2 - 2.0 * mu_j / r_p)
        ecc_out = 1.0 + r_p * w_out**2 / mu_j
        return float(np.arcsin(1.0 / ecc_in) + np.arcsin(1.0 / ecc_out))

    # Zero burn recovers the unpowered bend exactly.
    assert turn_for(0.0) == pytest.approx(unpowered_turn, rel=1e-12)
    # Speeding up costs bend, monotonically -- the turn is NOT independent of dv.
    speeding = [turn_for(b) for b in (0.0, 1.0, 2.0, 4.0)]
    assert speeding == sorted(speeding, reverse=True)
    assert speeding[-1] < speeding[0]
    # Slowing down buys bend back -- but only just. Perijove speed clears
    # Jupiter escape (57.94 km/s at the floor) by 0.55 km/s at this w_in, so a
    # retro burn past that CAPTURES the craft rather than bending it. "Slow down
    # for more turn" is barely available at Jupiter, which is why the useful
    # powered flyby here speeds up and pays for it in bend.
    assert turn_for(-0.3) > unpowered_turn
    v_esc_jupiter = float(np.sqrt(2.0 * mu_j / r_p))
    v_peri_in = float(np.sqrt(w_in**2 + 2.0 * mu_j / r_p))
    assert v_peri_in - v_esc_jupiter < 0.6
    assert np.isnan(turn_for(-1.0))


def test_phased_ladder_lands_the_jovian_leg_on_jupiter_itself() -> None:
    # The model's largest hole: _jovian_terminal reaches Jupiter through
    # _conic_radius_crossings, which finds crossings of Jupiter's orbit RADIUS
    # and assumes Jupiter is there. Handing Jupiter to _phased_ladder_burn as an
    # ordinary body closes it -- the Jovian leg becomes a Lambert arc onto
    # Jupiter's true position. Verify Jupiter is actually THERE on arrival.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    jupiter = _jupiter_assist_body(params)
    assert jupiter.symbol == "J"
    assert jupiter.min_periapsis == params.flyby.periapsis_floor

    year = 365.25 * 86400.0
    ladder = (earth, venus, earth, jupiter)
    # Jupiter placed so it IS where a 2.0 yr final leg arrives.
    longitudes0 = [0.0, 1.0, 2.0, 2.7]
    legs = [0.4 * year, 0.9 * year, 2.0 * year]
    priced = _phased_ladder_burn(0.0, legs, ladder, longitudes0, params, True)
    assert priced is not None

    # Jupiter's longitude at arrival, computed independently of the pricing.
    arrival = sum(legs)
    expected_lon = longitudes0[3] + _mean_motion(jupiter) * arrival
    assert priced.arrival_longitude == pytest.approx(expected_lon, rel=1e-12)
    # And the arrival excess is Jupiter-relative, so it must be modest -- if the
    # leg were merely crossing 5.2 AU with Jupiter elsewhere, this would be the
    # raw heliocentric speed instead.
    assert 0.0 < priced.arrival_excess < 40.0
    # The last Earth is now a charged node, not a free rotation: two nodes for a
    # four-body ladder (Venus and the returning Earth).
    assert len(priced.node_burns) == 2


def test_phased_jovian_flyby_bends_a_known_arrival_into_a_return() -> None:
    # With the leg phased, the flyby needs no radius search: it bends and
    # rescales a known arrival excess. A zero burn must leave the excess SPEED
    # untouched (an unpowered flyby only rotates), and the burn must raise it.
    params = _assist_chain_params(target_collision_speed=20.0)
    r_p = params.flyby.periapsis_floor
    lon = 0.7
    # A retrograde return requires the post-flyby heliocentric tangential speed
    # to go NEGATIVE, i.e. v_jupiter + w_out*cos(angle) < 0. No bend angle can
    # do that unless w_out exceeds Jupiter's own orbital speed -- you cannot
    # rotate a 6 km/s excess into cancelling 13.058 km/s of Jupiter's motion.
    # So arrival v_inf > 13.058 km/s is a NECESSARY condition for an unpowered
    # Jovian flyby, and the only way round it is to burn.
    assert params.flyby.v_jupiter_orbit == pytest.approx(13.058, abs=0.01)
    cold = np.array([-5.0, 4.0, 0.0])  # w = 6.40, far under the threshold
    assert float(np.linalg.norm(cold)) < params.flyby.v_jupiter_orbit
    assert _phased_jovian_flyby(cold, lon, r_p, 0.0, 1.0, params) is None
    # ... but a burn lifts w_out over the threshold and the return appears.
    assert _phased_jovian_flyby(cold, lon, r_p, 3.0, 1.0, params) is not None

    # A hot enough arrival returns unpowered, as ADR 0002's direct flyby does
    # (it reaches Jupiter at w_in ~ 15.1 km/s and spends no Jovian burn).
    excess_in = np.array([-14.0, 6.0, 0.0])
    assert float(np.linalg.norm(excess_in)) > params.flyby.v_jupiter_orbit
    unpowered = _phased_jovian_flyby(excess_in, lon, r_p, 0.0, 1.0, params)
    powered = _phased_jovian_flyby(excess_in, lon, r_p, 3.0, 1.0, params)
    assert unpowered is not None and powered is not None
    # Powering the flyby buys a hotter return.
    assert powered.collision_speed > unpowered.collision_speed
    # Below the perijove floor is refused outright.
    assert _phased_jovian_flyby(excess_in, lon, r_p * 0.5, 0.0, 1.0, params) is None
    # A retro burn past the escape margin captures rather than bends.
    w_in = float(np.linalg.norm(excess_in))
    margin = np.sqrt(w_in**2 + 2 * params.flyby.mu_jupiter / r_p) - np.sqrt(
        2 * params.flyby.mu_jupiter / r_p
    )
    assert (
        _phased_jovian_flyby(excess_in, lon, r_p, -(margin + 0.5), 1.0, params) is None
    )


def test_return_leg_sweep_angle_approaches_a_hohmann_half_turn() -> None:
    # The retrograde Jupiter->Earth Hohmann is the one return whose geometry is
    # known in closed form: it leaves Jupiter's radius at aphelion and arrives at
    # 1 AU at perihelion, so it sweeps exactly half a turn in half a period.
    #
    # The exact Hohmann cannot be evaluated here, and that is a property of the
    # problem rather than a defect: its perihelion lands 4 ulp ABOVE 1 AU (0.12 mm
    # on 1.5e8 km), so the leg correctly reports no crossing. The tangent case has
    # measure zero. Approach the limit from inside instead, which pins the branch,
    # the sign convention AND continuity into the closed-form answer.
    params = _powered_flyby_params(max_total_tof=10 * u.year)
    mu = params.mu_sun
    r_j, r_e = params.r_jupiter_orbit, params.r_earth_orbit

    def leg_with_perihelion(q: float) -> object:
        # Aphelion at Jupiter's radius, perihelion at q, flown retrograde. An
        # aphelion state is purely tangential by definition, so v_r = 0.
        a = 0.5 * (r_j + q)
        v_aph = float(np.sqrt(mu * (2.0 / r_j - 1.0 / a)))
        return _flyby_return_leg(-v_aph, 0.0, params)

    sweeps = []
    for shortfall in (1e-4, 1e-6, 1e-8):
        leg = leg_with_perihelion(r_e * (1.0 - shortfall))
        assert leg is not None
        # Crossing 1 AU strictly before perihelion, so strictly under a half turn.
        assert leg.sweep_angle < np.pi
        sweeps.append(leg.sweep_angle)
    # Monotone, and converging on the half turn.
    assert sweeps[0] < sweeps[1] < sweeps[2]
    assert sweeps[2] == pytest.approx(np.pi, abs=1e-3)

    # At the closest approach to the limit, tof must match half the transfer
    # ellipse's period -- the independent check that sweep and tof describe the
    # SAME arc, since they are derived in the same branch from the same anomalies.
    q = r_e * (1.0 - 1e-8)
    leg = leg_with_perihelion(q)
    assert leg is not None
    a = 0.5 * (r_j + q)
    period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
    # Strictly under half a period, and for the same reason the sweep is strictly
    # under a half turn: 1 AU is crossed just BEFORE perihelion. Converging to the
    # half-period limit from below is the check; landing exactly on it would mean
    # the branch had silently run past perihelion.
    assert leg.tof < 0.5 * period
    assert leg.tof == pytest.approx(0.5 * period, rel=1e-4)
    assert leg.perihelion == pytest.approx(q, rel=1e-9)


def test_return_leg_sweeps_further_when_it_climbs_before_falling() -> None:
    # An outbound state (v_r > 0) must go out to aphelion before coming back in,
    # so it sweeps MORE than the same-energy inbound state, and takes longer. The
    # two branches are separately derived in _flyby_return_leg; this pins that
    # they stay ordered.
    params = _powered_flyby_params(max_total_tof=10 * u.year)
    inbound = _flyby_return_leg(-6.0, -2.0, params)
    outbound = _flyby_return_leg(-6.0, 2.0, params)
    assert inbound is not None and outbound is not None
    assert outbound.sweep_angle > inbound.sweep_angle
    assert outbound.tof > inbound.tof
    # Same speed and radius, so same energy -> same conic, hence same perihelion
    # and the same arrival speed. Only the arc flown differs.
    assert outbound.perihelion == pytest.approx(inbound.perihelion, rel=1e-9)
    assert outbound.collision_speed == pytest.approx(inbound.collision_speed, rel=1e-9)
    # Every sweep is a real angle in (0, 2*pi).
    for leg in (inbound, outbound):
        assert 0.0 < leg.sweep_angle < 2.0 * np.pi


def test_earth_phase_mismatch_vanishes_exactly_where_earth_is() -> None:
    # The mismatch is the constraint the whole "real doubling time" question
    # turns on, so it must be zero ONLY when Earth is truly at the crossing.
    params = _powered_flyby_params(max_total_tof=10 * u.year)
    leg = _flyby_return_leg(-6.0, -2.0, params)
    assert leg is not None
    n_earth = params.v_earth_orbit / params.r_earth_orbit
    jupiter_lon, flyby_time = 1.3, 0.0

    # Place Earth at t=0 exactly where the crossing will be at arrival, running
    # Earth's motion backwards over the flight. By construction the mismatch is 0.
    crossing = jupiter_lon - leg.sweep_angle
    earth_0 = crossing - n_earth * (flyby_time + leg.tof)
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0, params
    ) == pytest.approx(0.0, abs=1e-12)

    # Nudge Earth and the mismatch tracks it one-for-one...
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0 + 0.25, params
    ) == pytest.approx(-0.25, abs=1e-12)
    # ... and is wrapped, so a half-turn error never reports as a near miss.
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0 + np.pi, params
    ) == pytest.approx(-np.pi, abs=1e-9)
    # A whole-turn offset IS the same place: Earth is a body on a ring, not a
    # point on a line.
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0 + 2.0 * np.pi, params
    ) == pytest.approx(0.0, abs=1e-9)
