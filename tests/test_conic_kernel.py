"""Tests for the float two-body conic geometry in src/conic_kernel.py.

Includes parity checks against boinor's own (semi-internal) float layer --
``boinor.core.angles`` and ``boinor.threebody.flybys.compute_flyby`` -- as
oracles: an implementation we did not write, agreeing with ours, is stronger
evidence than either alone. The kernel keeps its own closed forms rather than
delegating to boinor's (arctan2 half-angle / clamped arccosh are exact at
nu = pi, where boinor's tan(nu/2) forms overflow), so these are cross-checks,
not a dependency.
"""

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Earth, Jupiter, Sun, Venus
from boinor.core.angles import E_to_M, F_to_M, nu_to_E, nu_to_F
from boinor.threebody.flybys import compute_flyby

from src import conic_kernel
from src.astro_constants import EARTH_A, JUPITER_A, MARS_A
from src.orbit_utils import (
    apoapsis_speed,
    elliptic_time_of_flight,
    hyperbolic_time_of_flight,
    orbit_from_rp_ra,
    periapsis_speed,
    true_anomaly_at_radius,
)

_MU_SUN = float(Sun.k.to_value(u.km**3 / u.s**2))
_MU_EARTH = float(Earth.k.to_value(u.km**3 / u.s**2))
_MU_JUPITER = float(Jupiter.k.to_value(u.km**3 / u.s**2))
_R_EARTH_ORBIT = float(EARTH_A.to_value(u.km))
_R_JUPITER_ORBIT = float(JUPITER_A.to_value(u.km))


def test_hyperbolic_eccentricity_at_zero_excess_is_one() -> None:
    # v_infinity = 0 is the degenerate parabola/circle boundary: e = 1 exactly.
    assert conic_kernel.hyperbolic_eccentricity(_MU_EARTH, 7000.0, 0.0) == 1.0


def test_eccentricity_from_energy_and_h_matches_hyperbolic_eccentricity() -> None:
    # Two routes to the same eccentricity: hyperbolic_eccentricity works from
    # (r_p, v_infinity); eccentricity_from_energy_and_h works from the orbit's
    # conserved (energy, h). For a hyperbola launched *at* periapsis (v_radial
    # = 0), energy = v_p^2/2 - mu/r_p and h = r_p*v_p must agree with it.
    mu, r_p, v_inf = _MU_JUPITER, 90000.0, 12.5
    v_p = float(np.sqrt(v_inf * v_inf + 2.0 * mu / r_p))
    energy = v_p * v_p / 2.0 - mu / r_p
    h = r_p * v_p
    expected = conic_kernel.hyperbolic_eccentricity(mu, r_p, v_inf)
    assert conic_kernel.eccentricity_from_energy_and_h(mu, energy, h) == pytest.approx(
        expected, rel=1e-12
    )


def test_bend_angle_identities() -> None:
    # unpowered_bend_angle is the e_in == e_out special case of the powered
    # split; half_turn_angle is the atomic half-angle both compose from.
    for ecc in (1.05, 1.5, 3.0, 20.0):
        half = conic_kernel.half_turn_angle(ecc)
        assert conic_kernel.unpowered_bend_angle(ecc) == pytest.approx(2.0 * half)
        assert conic_kernel.powered_bend_angle(ecc, ecc) == pytest.approx(2.0 * half)
    # A split hyperbola that slows down (e_out > e_in) bends less than a
    # symmetric pass at the faster eccentricity: half_turn_angle is decreasing.
    assert conic_kernel.half_turn_angle(1.2) > conic_kernel.half_turn_angle(3.0)


def test_semimajor_axis_and_periapsis_radius_round_trip() -> None:
    # Build an ellipse from (r_p, r_a), derive its energy and semi-latus
    # rectum, and check both round-trip helpers recover the inputs.
    r_p, r_a = _R_EARTH_ORBIT, _R_JUPITER_ORBIT
    a = (r_p + r_a) / 2.0
    ecc = (r_a - r_p) / (r_a + r_p)
    energy = -_MU_SUN / (2.0 * a)
    assert conic_kernel.semimajor_axis_from_energy(_MU_SUN, energy) == pytest.approx(
        a, rel=1e-12
    )
    p = a * (1.0 - ecc * ecc)
    assert conic_kernel.periapsis_radius_of_conic(p, ecc) == pytest.approx(
        r_p, rel=1e-9
    )


def test_conic_state_at_radius_is_orbit_invariant() -> None:
    # energy, h, p, ecc are conserved along the orbit: sampling the state at
    # r0 and evaluating it back out at r0 (via speed_components_at_radius)
    # must reproduce the original (v_t0, v_r0), and re-deriving the state from
    # a different point on the same orbit must agree exactly.
    # Use the Earth->Jupiter Hohmann periapsis state, so r1 below is guaranteed
    # reachable (it is the transfer's apoapsis) -- an unreachable r1 would
    # silently clamp speed_components_at_radius's radial term to zero and
    # break the round trip on a state that was never really on the orbit.
    mu, r0 = _MU_SUN, _R_EARTH_ORBIT
    transfer = orbit_from_rp_ra(periapsis_radius=EARTH_A, apoapsis_radius=JUPITER_A)
    v_t0 = periapsis_speed(transfer).to_value(u.km / u.s)
    v_r0 = 0.0
    state = conic_kernel.conic_state_at_radius(mu, r0, v_t0, v_r0)
    v_t_back, v_r_back = conic_kernel.speed_components_at_radius(state, mu, r0)
    assert v_t_back == pytest.approx(v_t0, rel=1e-12)
    assert v_r_back == pytest.approx(abs(v_r0), abs=1e-9)

    r1 = _R_JUPITER_ORBIT
    v_t1, v_r1 = conic_kernel.speed_components_at_radius(state, mu, r1)
    state_from_r1 = conic_kernel.conic_state_at_radius(mu, r1, v_t1, v_r1)
    assert state_from_r1.energy == pytest.approx(state.energy, rel=1e-9)
    assert state_from_r1.h == pytest.approx(state.h, rel=1e-9)
    assert state_from_r1.p == pytest.approx(state.p, rel=1e-9)
    assert state_from_r1.ecc == pytest.approx(state.ecc, rel=1e-9)


def test_true_anomaly_at_radius_rad_matches_orbit_utils() -> None:
    # The kernel float and orbit_utils' Quantity interface must agree exactly
    # -- orbit_utils delegates to this function, so this also pins the wiring.
    # p=1.4e8, ecc=0.5 gives periapsis ~9.3e7 km and apoapsis 2.8e8 km; r must
    # fall inside that band to be reachable.
    p, ecc, r = 1.4e8, 0.5, 2.0e8
    nu = conic_kernel.true_anomaly_at_radius_rad(p, ecc, r)
    assert nu is not None
    r_p = (p / (1.0 + ecc)) * u.km
    nu_quantity = true_anomaly_at_radius(r_p, ecc, r * u.km)
    assert np.degrees(nu) == pytest.approx(nu_quantity.to_value(u.deg), abs=1e-9)


def test_true_anomaly_at_radius_rad_rejects_unreachable_radius() -> None:
    assert conic_kernel.true_anomaly_at_radius_rad(1.0, 0.5, 0.1) is None


def test_elliptic_tof_matches_orbit_utils() -> None:
    r_p, ecc = 6.6e8, 0.5
    a = r_p / (1.0 - ecc)
    for nu_deg in (10.0, 90.0, 179.0):
        nu = np.radians(nu_deg)
        tof = conic_kernel.elliptic_tof_seconds(_MU_SUN, a, ecc, nu)
        expected = elliptic_time_of_flight(r_p * u.km, ecc, nu_deg * u.deg)
        assert (tof * u.s).to_value(u.day) == pytest.approx(
            expected.to_value(u.day), rel=1e-9
        )


def test_hyperbolic_tof_matches_orbit_utils() -> None:
    r_p, v_inf = 9.0e4, 12.5
    a_abs = _MU_JUPITER / (v_inf * v_inf)
    ecc = conic_kernel.hyperbolic_eccentricity(_MU_JUPITER, r_p, v_inf)
    for nu_deg in (10.0, 60.0, 100.0):
        nu = np.radians(nu_deg)
        tof = conic_kernel.hyperbolic_tof_seconds(_MU_JUPITER, a_abs, ecc, nu)
        expected = hyperbolic_time_of_flight(
            r_p * u.km, v_inf * u.km / u.s, nu_deg * u.deg, attractor=Jupiter
        )
        assert (tof * u.s).to_value(u.day) == pytest.approx(
            expected.to_value(u.day), rel=1e-9
        )


def test_elliptic_tof_matches_boinor_angle_converters() -> None:
    # Oracle: an implementation we did not write. Avoid nu = pi, where
    # boinor's tan(nu/2) eccentric-anomaly form diverges (the kernel's
    # arctan2 form is exact there, which is exactly why it is not delegated
    # to); this only checks the two agree away from that singular point.
    mu, a, ecc = _MU_SUN, 2.0e8, 0.3
    for nu_deg in (15.0, 80.0, 150.0):
        nu = np.radians(nu_deg)
        mean_anomaly = E_to_M(nu_to_E(nu, ecc), ecc)
        expected = float(mean_anomaly / np.sqrt(mu / a**3))
        assert conic_kernel.elliptic_tof_seconds(mu, a, ecc, nu) == pytest.approx(
            expected, rel=1e-9
        )


def test_hyperbolic_tof_matches_boinor_angle_converters() -> None:
    mu, a_abs, ecc = _MU_JUPITER, 6.0e4, 1.8
    for nu_deg in (10.0, 60.0, 100.0):
        nu = np.radians(nu_deg)
        mean_anomaly = F_to_M(nu_to_F(nu, ecc), ecc)
        expected = float(mean_anomaly / np.sqrt(mu / a_abs**3))
        assert conic_kernel.hyperbolic_tof_seconds(mu, a_abs, ecc, nu) == pytest.approx(
            expected, rel=1e-9
        )


def test_conic_radius_crossings_reproduces_hohmann_leg() -> None:
    # Fed the periapsis state of the Earth->Jupiter Hohmann ellipse, the
    # crossing finder must reproduce the textbook transfer: the Jupiter
    # crossing at aphelion after the half period (~2.731 yr) at the apoapsis
    # speed, and the Mars-radius crossings as an outbound/inbound pair
    # sharing h/r.
    transfer = orbit_from_rp_ra(periapsis_radius=EARTH_A, apoapsis_radius=JUPITER_A)
    v_peri = periapsis_speed(transfer).to_value(u.km / u.s)
    jupiter_crossings = conic_kernel.conic_radius_crossings(
        _MU_SUN, _R_EARTH_ORBIT, v_peri, 0.0, _R_JUPITER_ORBIT
    )
    assert jupiter_crossings
    v_apo = apoapsis_speed(transfer).to_value(u.km / u.s)
    crossing = jupiter_crossings[0]
    assert (crossing.tof * u.s).to(u.year).to_value(u.year) == pytest.approx(
        2.731, rel=0.1
    )
    assert crossing.v_tangential == pytest.approx(v_apo, rel=1e-6)
    assert crossing.v_radial == pytest.approx(0.0, abs=1e-3)
    # A Hohmann runs perihelion to aphelion, so it sweeps exactly half a turn.
    # Tolerance is 1e-4 deg (0.36 arcsec), not tighter: recovering the true
    # anomaly at an apsis inverts a cosine through arccos(-1), ill-conditioned
    # there.
    assert np.degrees(crossing.swept) == pytest.approx(180.0, abs=1e-4)

    r_mars = float(MARS_A.to_value(u.km))
    mars_crossings = conic_kernel.conic_radius_crossings(
        _MU_SUN, _R_EARTH_ORBIT, v_peri, 0.0, r_mars
    )
    assert len(mars_crossings) == 2
    outbound = [c for c in mars_crossings if c.outbound]
    inbound = [c for c in mars_crossings if not c.outbound]
    assert len(outbound) == 1 and len(inbound) == 1
    assert outbound[0].tof < inbound[0].tof  # outward crossing comes first
    assert outbound[0].v_radial > 0.0 > inbound[0].v_radial
    h = _R_EARTH_ORBIT * v_peri
    assert outbound[0].v_tangential == pytest.approx(h / r_mars, rel=1e-9)
    # NamedTuple fields are also positionally indexable, matching the
    # (tof, v_tangential, v_radial, outbound, swept) tuple contract callers
    # unpack against.
    assert outbound[0][3] is True


def test_conic_radius_crossings_min_leg_time_filters_degenerate_legs() -> None:
    transfer = orbit_from_rp_ra(periapsis_radius=EARTH_A, apoapsis_radius=JUPITER_A)
    v_peri = periapsis_speed(transfer).to_value(u.km / u.s)
    # A target radius one km beyond periapsis has two crossings: the outbound
    # one is reached almost instantly (degenerate), the inbound one only after
    # a full orbit (a legitimate multi-year leg that min_leg_time must not
    # touch).
    near = conic_kernel.conic_radius_crossings(
        _MU_SUN, _R_EARTH_ORBIT, v_peri, 0.0, _R_EARTH_ORBIT + 1.0
    )
    assert any(c.outbound and c.tof < 86400.0 for c in near)
    filtered = conic_kernel.conic_radius_crossings(
        _MU_SUN, _R_EARTH_ORBIT, v_peri, 0.0, _R_EARTH_ORBIT + 1.0, min_leg_time=86400.0
    )
    assert not any(c.tof <= 86400.0 for c in filtered)
    assert any(not c.outbound for c in filtered)  # the legitimate leg survives


def test_mean_motion() -> None:
    v_circ, r = 29.78, _R_EARTH_ORBIT
    assert conic_kernel.mean_motion(v_circ, r) == pytest.approx(v_circ / r)


def test_wrap_pi() -> None:
    assert conic_kernel.wrap_pi(0.0) == pytest.approx(0.0)
    # Odd multiples of pi are the branch cut: the modulo resolves them to -pi,
    # not +pi (both represent the same angle).
    assert conic_kernel.wrap_pi(np.pi) == pytest.approx(-np.pi)
    assert conic_kernel.wrap_pi(3.0 * np.pi) == pytest.approx(-np.pi)
    assert conic_kernel.wrap_pi(-3.0 * np.pi) == pytest.approx(-np.pi)
    assert conic_kernel.wrap_pi(np.pi - 1e-9) == pytest.approx(np.pi - 1e-9)
    assert conic_kernel.wrap_pi(2.5 * np.pi) == pytest.approx(0.5 * np.pi)


def test_speed_with_escape_energy_matches_quadrature() -> None:
    # Vis-viva for a hyperbolic orbit: excess kinetic energy and escape energy
    # add in quadrature, not linearly.
    assert conic_kernel.speed_with_escape_energy(3.0, 4.0) == pytest.approx(5.0)
    # Zero excess speed leaves exactly the escape speed.
    assert conic_kernel.speed_with_escape_energy(0.0, 11.2) == pytest.approx(11.2)
    # Symmetric in its two arguments -- it's a hypot, not a directional fold.
    assert conic_kernel.speed_with_escape_energy(5.0, 12.0) == pytest.approx(
        conic_kernel.speed_with_escape_energy(12.0, 5.0)
    )


def test_bend_limit_matches_boinor_flyby() -> None:
    # The turning formula (e = 1 + r_p*w^2/mu, delta = 2*asin(1/e)) is the
    # foundation of every bend limit the assist chain and the Jovian terminal
    # use. Check it against an implementation we did not write, self-contained
    # (no mission-specific altitude floors -- see test_retrograde_return_legs.py's
    # test_bend_limit_matches_boinor_flyby for the chain's actual numbers).
    cases = [
        (
            Venus,
            float(Venus.k.to_value(u.km**3 / u.s**2)),
            float(Venus.R.to_value(u.km)) + 300.0,
            3.5,
            60.0,
        ),
        (Earth, _MU_EARTH, float(Earth.R.to_value(u.km)) + 300.0, 5.2, 30.0),
        (Jupiter, _MU_JUPITER, float(Jupiter.R.to_value(u.km)) + 2000.0, 15.4, 0.0),
    ]
    for body, mu, r_p, excess, v_circ in cases:
        v_body = np.array([0.0, v_circ, 0.0]) * (u.km / u.s)
        v_sc = np.array([excess, v_circ, 0.0]) * (u.km / u.s)
        v_out, delta = compute_flyby(v_sc, v_body, mu * u.km**3 / u.s**2, r_p * u.km)
        ecc = conic_kernel.hyperbolic_eccentricity(mu, r_p, excess)
        mine = float(np.degrees(conic_kernel.unpowered_bend_angle(ecc)))
        assert delta.to_value(u.deg) == pytest.approx(mine, abs=1e-9)
        # And the flyby is unpowered: it rotates the excess without rescaling
        # it -- the Tisserand invariant the whole assist chain rests on.
        excess_out = np.linalg.norm((v_out - v_body).to_value(u.km / u.s))
        assert excess_out == pytest.approx(excess, abs=1e-9)
