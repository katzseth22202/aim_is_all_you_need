"""Lighter plates on smaller pulses, flown against gravity above 400 km (ADR 0062).

**Abandoned** (author, 2026-10-14): it moves deliveries about 8%, and a growth
push cannot fly the delivery optimum's plate at all.  Kept as the record behind
ADR 0062's delivery curve.

The spray cup pushes the 1500 t launch unit with 12 MN s pulses at 4 Hz: 48 MN,
3.3 g at the start and about 5.4 g at the end, on a 150 t plate.  The author
asked (2026-10-14) whether smaller pulses, about 40 kg of PuffSat instead of
74-92, would deliver more: the plate scales with its pulse (parent
``eq:plate_pulse_size``, ``J = 8 m_s nu s``), so a smaller pulse means a lighter
plate and a lower peak acceleration.  The catch is time.  A weaker push lasts
longer, and the stack must stay at or above the **growth-wave floor** for all of
it, because no PuffSat may pass Earth below 400 km and a PuffSat that meets the
stack at altitude ``h`` has its perigee at or below ``h``.

**The push** (:func:`fly_push`) is integrated in the orbit plane, with gravity:

* The stack starts at 400 km climbing at the lob's ``rise`` speed, at rest
  horizontally (Earth's rotation ignored, as in :func:`src.lob_rise.push_track`).
* Each PuffSat is aimed on its own: it meets the stack at a flight-path angle
  ``gamma`` on a hyperbola of the wave's excess speed, and the thrust points
  along the closing velocity.  Upward ``gamma`` holds the stack up, but the
  hyperbola's perigee must stay at or above 400 km, which caps ``|gamma|`` at
  ``gamma_max(r)``: zero at the floor, about 5 deg at 430 km.  ``gamma`` is
  ``gamma_max`` times a profile interpolated over the push's progress in
  orbital energy, whose nodes are optimised.
* Thrust is ``J nu`` at the plate's pulse size ``J`` and 4 Hz.  A delivery is
  held to the **delivery acceleration cap**, 4 g felt by the payload, which is
  a mean of 4 / 1.1 g under the springs' assumed 10% ripple: past it, the
  pulses taper (lighter PuffSats along the wave).
* Mass leaves as argon, ``k`` per kilogram of PuffSat, and as film; the
  PuffSats themselves leave in the exhaust.  The impulse law is
  :func:`src.water_plate._impulse_factor`, at the spray cup's efficiency.
  Deliveries load ``k = 10``.  With ``slug_ratio=None`` the push flies the
  ledger's Pontryagin schedule instead, its price carried through the
  integration: that is how ADR 0062 found the growth push's smallest plate.
* The push ends when the orbit's energy reaches the 20-day parking orbit's,
  and its perigee must also be at or above 400 km.

**The plate** is 150 t times ``J / 12 MN s`` (:func:`plate_mass`), with a
sensitivity that holds 20 t of it fixed.

**The booster** is fixed in size: 1500 t at 400 km climbing at 0.75 km/s, the
parent's lob, and less for a faster climb (:func:`lofted`, from
:func:`src.lob_rise.braked_lob`).

**The objective** (CONTEXT.md, **delivery push objective**): at each climb
speed, the pulse size and aim that deliver the most data-centre tonnes to the
parking orbit per booster flight; reported as a curve over climb speed.
"""

import argparse
import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from tabulate import tabulate

from src.growth_ledger import (
    LAUNCH_UNIT,
    PLATE_MASS,
    PULSE_IMPULSE,
    SPRAY_CUP_SHIELDED,
    PlateDesign,
)
from src.lob_rise import BASELINE_RISE_SPEED, OPERATING_RISE_SPEED, braked_lob
from src.water_plate import (
    ARGON_SLUG,
    _best_loading,
    _impulse_factor,
)

_MU = 3.986004418e14  # m^3/s^2
_R_EARTH = 6378.137e3  # m
_G0 = 9.80665
#: The growth-wave floor: no PuffSat passes Earth lower (parent
#: ``sec:jovian_meeting_altitudes``), so the stack stays at or above it too.
FLOOR_ALTITUDE = 400.0e3  # m
FLOOR_RADIUS = _R_EARTH + FLOOR_ALTITUDE
#: The spray cup's pulse rate and the plate it sizes (``eq:plate_pulse_size``).
PULSE_RATE = 4.0  # Hz
REFERENCE_PULSE = float(PULSE_IMPULSE.to_value(u.N * u.s))
REFERENCE_PLATE = float(PLATE_MASS.to_value(u.kg))
#: The plate share held fixed in the sensitivity (skirt frame, seals, spray
#: system): a guess, not a sourced figure.
FIXED_PLATE_SHARE = 20.0e3  # kg
#: The delivery acceleration cap, felt by the payload, and the springs' ripple
#: assumed on top of the cycle mean (CONTEXT.md).
DELIVERY_CAP_G = 4.0
SPRING_RIPPLE = 0.10
#: Deliveries load the plate to the validated cap on every pulse.
DELIVERY_SLUG_RATIO = 10.0
#: The plate every push flies: the spray cup, its film carried (ADR 0043).
PLATE: PlateDesign = SPRAY_CUP_SHIELDED
#: Growth-wave speeds at 400 km bracketing the flown chain (parent
#: ``sec:two_leg_nozzle``): the slowest three-synodic return and the fastest
#: two-synodic one.  The plate is chosen on the slowest, where pushes are longest.
WAVE_SPEEDS_400 = {"3S slowest": 56.53e3, "2S fastest": 65.13e3}  # m/s
#: The 20-day parking orbit the push ends on.
PARKING_DAYS = 20.0
#: Climb speeds at 400 km the curve is drawn at (m/s).  The braked lob's root
#: fails above 2.0 km/s.
CLIMB_SPEEDS = (0.75e3, 1.0e3, 1.25e3, 1.5e3, 1.75e3, 2.0e3)
#: Search bounds: the plate's pulse (N s) and the aim profile's nodes.
PULSE_BOUNDS = (2.0e6, 16.0e6)
AIM_NODES = 4
_MAX_TIME = 4000.0  # s
_INFEASIBLE = 1.0e9


def plate_mass(pulse: float, fixed: float = 0.0) -> float:
    """The plate for a pulse of ``pulse`` N s at 4 Hz and 2.8 m (kg).

    Args:
        pulse: Impulse per pulse.
        fixed: Plate mass that does not scale with the pulse.

    Returns:
        ``fixed + (150 t - fixed) J / 12 MN s``.
    """
    return fixed + (REFERENCE_PLATE - fixed) * pulse / REFERENCE_PULSE


@lru_cache(maxsize=64)
def lofted(rise: float) -> float:
    """Mass the fixed booster lofts to 400 km climbing at ``rise`` m/s (kg).

    The booster lofts the 1500 t launch unit on the parent's 0.75 km/s lob;
    a faster climb lofts less, in the braked lob's proportion.
    """
    ratio = braked_lob(rise).lofted / braked_lob(BASELINE_RISE_SPEED).lofted
    return float(LAUNCH_UNIT.to_value(u.kg)) * ratio


def parking_energy(days: float = PARKING_DAYS) -> float:
    """Specific orbital energy of the parking orbit (J/kg)."""
    period = days * 86400.0
    axis = (_MU * period**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    return float(-_MU / (2.0 * axis))


def excess_speed(wave_speed_400: float) -> float:
    """The wave's hyperbolic excess speed from its speed at 400 km (m/s)."""
    return float(np.sqrt(wave_speed_400**2 - 2.0 * _MU / FLOOR_RADIUS))


def max_aim(radius: float, excess: float) -> float:
    """Largest ``|gamma|`` a PuffSat can meet the stack at, perigee >= 400 km.

    A hyperbola's perigee rises with its angular momentum, so the floor is a
    floor on ``r V cos(gamma)``: the perigee speed at 400 km times 400 km.

    Args:
        radius: The stack's radius (m).
        excess: The wave's excess speed (m/s).

    Returns:
        The angle (rad); zero at or below the floor.
    """
    if radius <= FLOOR_RADIUS:
        return 0.0
    speed = np.sqrt(excess * excess + 2.0 * _MU / radius)
    floor_momentum = FLOOR_RADIUS * np.sqrt(excess * excess + 2.0 * _MU / FLOOR_RADIUS)
    return float(np.arccos(min(1.0, floor_momentum / (radius * speed))))


@dataclass(frozen=True)
class FlownPush:
    """One push flown against gravity (kg, s, m/s).

    Attributes:
        feasible: It reached the parking orbit without crossing the floor.
        start_mass: The unit at the intercept.
        end_mass: Left at the parking orbit, plate and tanks included.
        puffsats: PuffSat mass spent.
        argon: Argon sprayed.
        film: Film burned.
        duration: Push time.
        min_altitude: Lowest altitude reached (m).
        final_perigee: The parking orbit's perigee altitude (m).
        peak_g: Largest mean acceleration, in g.
        first_puffsat: PuffSat mass in the first pulse (kg).
        last_puffsat: PuffSat mass in the last pulse (kg).
        loading_start: ``k`` on the first pulse.
        loading_end: ``k`` on the last pulse.
        final_price: The loading schedule's price at the end.
    """

    feasible: bool
    start_mass: float
    end_mass: float
    puffsats: float
    argon: float
    film: float
    duration: float
    min_altitude: float
    final_perigee: float
    peak_g: float
    first_puffsat: float
    last_puffsat: float
    loading_start: float
    loading_end: float
    final_price: float


def fly_push(
    wave_speed_400: float,
    pulse: float,
    rise: float,
    aim: Sequence[float],
    cap_g: Optional[float],
    slug_ratio: Optional[float] = DELIVERY_SLUG_RATIO,
    initial_price: float = 0.0,
    start_mass: Optional[float] = None,
    plate: PlateDesign = PLATE,
    days: float = PARKING_DAYS,
) -> FlownPush:
    """Fly one push from the lob to the parking orbit, against gravity.

    Args:
        wave_speed_400: The growth wave's speed at 400 km (m/s).
        pulse: The plate's impulse per pulse (N s).
        rise: The lob's climb speed at 400 km (m/s).
        aim: Profile nodes in [-1, 1], times ``gamma_max``, spaced evenly over
            the push's progress in orbital energy.
        cap_g: The payload's acceleration cap (g), or None; the mean is held
            to ``cap_g / (1 + SPRING_RIPPLE)``.
        slug_ratio: A fixed loading, or None for the Pontryagin schedule.
        initial_price: The schedule's starting price, when ``slug_ratio`` is None.
        start_mass: The unit at the intercept; the fixed booster's by default.
        plate: The plate's efficiency, cap and film.
        days: The parking orbit's period.

    Returns:
        The push.
    """
    excess = excess_speed(wave_speed_400)
    jet = float(np.sqrt(plate.efficiency))
    bonds = (
        plate.impactor_bond_j_kg,
        float(ARGON_SLUG.bond_energy.to_value(u.J / u.kg)),
    )
    film_per_ns = float(plate.film_per_impulse.to_value(u.kg / (u.N * u.s)))
    mass0 = lofted(rise) if start_mass is None else start_mass
    thrust_full = pulse * PULSE_RATE
    cap = None if cap_g is None else cap_g * _G0 / (1.0 + SPRING_RIPPLE)
    energy0 = 0.5 * rise * rise - _MU / FLOOR_RADIUS
    target = parking_energy(days)
    nodes = np.asarray(aim, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, nodes.size)

    def state(
        y: npt.NDArray[np.float64],
    ) -> Tuple[float, float, float, float, float, float]:
        """(thrust, closing speed, beta, k, thrust direction radial, tangential)."""
        r, _, vr, vt, mass = y[:5]
        energy = 0.5 * (vr * vr + vt * vt) - _MU / r
        progress = min(max((energy - energy0) / (target - energy0), 0.0), 1.0)
        gamma = float(np.interp(progress, grid, nodes)) * max_aim(r, excess)
        speed = np.sqrt(excess * excess + 2.0 * _MU / r)
        rel_r, rel_t = speed * np.sin(gamma) - vr, speed * np.cos(gamma) - vt
        closing = float(np.hypot(rel_r, rel_t))
        k = (
            slug_ratio
            if slug_ratio is not None
            else _best_loading(y[8], closing / 1e3, jet, plate.max_slug_ratio, bonds)
        )
        beta = float(_impulse_factor(k, closing / 1e3, jet, bonds))
        thrust = thrust_full if cap is None else min(thrust_full, cap * mass)
        return thrust, closing, beta, float(k), rel_r / closing, rel_t / closing

    def rhs(t: float, y: npt.NDArray[np.float64]) -> List[float]:
        r, _, vr, vt, mass = y[:5]
        thrust, closing, beta, k, dr, dt = state(y)
        accel = thrust / mass
        puffsat_rate = thrust / (beta * closing)
        film_rate = film_per_ns * thrust
        price = y[8]
        price_rate = (accel / 1e3) * (
            (1.0 + price * k) / (beta * closing / 1e3) + price * film_per_ns * 1e3
        )
        return [
            vr,
            vt / r,
            vt * vt / r - _MU / r**2 + accel * dr,
            -vr * vt / r + accel * dt,
            -k * puffsat_rate - film_rate,
            puffsat_rate,
            k * puffsat_rate,
            film_rate,
            price_rate,
        ]

    def floor(t: float, y: npt.NDArray[np.float64]) -> float:
        return float(y[0] - FLOOR_RADIUS + 1.0)

    def done(t: float, y: npt.NDArray[np.float64]) -> float:
        return float(0.5 * (y[2] ** 2 + y[3] ** 2) - _MU / y[0] - target)

    floor.terminal = True  # type: ignore[attr-defined]
    floor.direction = -1  # type: ignore[attr-defined]
    done.terminal = True  # type: ignore[attr-defined]
    done.direction = 1  # type: ignore[attr-defined]
    start = np.array(
        [FLOOR_RADIUS, 0.0, rise, 0.0, mass0, 0.0, 0.0, 0.0, initial_price]
    )
    flown = solve_ivp(
        rhs, (0.0, _MAX_TIME), start, events=(floor, done), max_step=2.0, rtol=1e-7,
        atol=1e-6,
    )  # fmt: skip
    end = flown.y[:, -1]
    reached = flown.status == 1 and len(flown.t_events[1]) > 0
    r, vr, vt = end[0], end[2], end[3]
    momentum = r * vt
    axis = -_MU / (2.0 * target)
    ecc = float(np.sqrt(max(0.0, 1.0 - momentum * momentum / (_MU * axis))))
    perigee = axis * (1.0 - ecc) - _R_EARTH
    accel = [state(flown.y[:, i])[0] / flown.y[4, i] for i in range(flown.y.shape[1])]
    first, last = state(start), state(end)
    return FlownPush(
        feasible=bool(reached and perigee >= FLOOR_ALTITUDE - 1.0),
        start_mass=mass0,
        end_mass=float(end[4]),
        puffsats=float(end[5]),
        argon=float(end[6]),
        film=float(end[7]),
        duration=float(flown.t[-1]),
        min_altitude=float(flown.y[0].min() - _R_EARTH),
        final_perigee=float(perigee),
        peak_g=float(max(accel) / _G0),
        first_puffsat=float(first[0] / PULSE_RATE / (first[2] * first[1])),
        last_puffsat=float(last[0] / PULSE_RATE / (last[2] * last[1])),
        loading_start=float(first[3]),
        loading_end=float(last[3]),
        final_price=float(end[8]),
    )


def delivered(push: FlownPush, pulse: float, fixed: float = 0.0) -> float:
    """Data-centre mass at the parking orbit: what is left, less plate and tank (kg)."""
    tank = ARGON_SLUG.tank_fraction * push.argon
    return push.end_mass - plate_mass(pulse, fixed) - tank


@dataclass(frozen=True)
class Delivery:
    """The best delivery push at one climb speed.

    Attributes:
        rise: The lob's climb speed at 400 km (m/s).
        pulse: The plate's pulse (N s).
        aim: The aim profile's nodes.
        push: The push.
        plate: The plate (kg).
        payload: Data-centre mass at the parking orbit (kg).
    """

    rise: float
    pulse: float
    aim: Tuple[float, ...]
    push: FlownPush
    plate: float
    payload: float

    @property
    def per_puffsat(self) -> float:
        """Data-centre kilograms per PuffSat kilogram."""
        return self.payload / self.push.puffsats


def _delivery_objective(
    x: Sequence[float], wave_speed_400: float, rise: float, fixed: float
) -> float:
    pulse, aim = float(x[0]), tuple(float(v) for v in x[1:])
    push = fly_push(wave_speed_400, pulse, rise, aim, DELIVERY_CAP_G)
    if not push.feasible:
        return _INFEASIBLE - push.min_altitude
    return -delivered(push, pulse, fixed)


def best_delivery(
    wave_speed_400: float,
    rise: float,
    fixed: float = 0.0,
    seed: int = 7,
    maxiter: int = 40,
    popsize: int = 10,
) -> Optional[Delivery]:
    """The pulse and aim that deliver most at one climb speed (scipy DE).

    Args:
        wave_speed_400: The growth wave's speed at 400 km (m/s).
        rise: The lob's climb speed (m/s).
        fixed: Plate mass that does not scale with the pulse (kg).
        seed: Random seed.
        maxiter: Generations.
        popsize: Population per parameter.

    Returns:
        The best feasible delivery, or None.
    """
    found = differential_evolution(
        _delivery_objective,
        bounds=[PULSE_BOUNDS] + [(-1.0, 1.0)] * AIM_NODES,
        args=(wave_speed_400, rise, fixed),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        polish=False,
    )
    pulse, aim = float(found.x[0]), tuple(float(v) for v in found.x[1:])
    push = fly_push(wave_speed_400, pulse, rise, aim, DELIVERY_CAP_G)
    if not push.feasible:
        return None
    return Delivery(
        rise, pulse, aim, push, plate_mass(pulse, fixed), delivered(push, pulse, fixed)
    )


def todays_delivery(wave_speed_400: float, cap_g: Optional[float]) -> Delivery:
    """Today's plate (12 MN s, 1.1 km/s climb) at k = 10, its aim optimised."""
    best: Optional[Delivery] = None
    for aim in itertools.product((-1.0, 0.0, 0.5, 1.0), repeat=AIM_NODES):
        push = fly_push(
            wave_speed_400, REFERENCE_PULSE, OPERATING_RISE_SPEED, aim, cap_g
        )
        if not push.feasible:
            continue
        payload = delivered(push, REFERENCE_PULSE)
        if best is None or payload > best.payload:
            best = Delivery(
                OPERATING_RISE_SPEED,
                REFERENCE_PULSE,
                aim,
                push,
                REFERENCE_PLATE,
                payload,
            )
    assert best is not None, "today's plate cannot hold the floor on any aim"
    return best


def delivery_row(label: str, found: Delivery) -> Dict[str, object]:
    """One table row."""
    p = found.push
    return {
        "case": label,
        "climb km/s": found.rise / 1e3,
        "lofted t": p.start_mass / 1e3,
        "pulse MN s": found.pulse / 1e6,
        "PuffSat/pulse kg": p.first_puffsat,
        "last kg": p.last_puffsat,
        "plate t": found.plate / 1e3,
        "push s": p.duration,
        "peak g": p.peak_g,
        "min alt km": p.min_altitude / 1e3,
        "argon t": p.argon / 1e3,
        "PuffSat t": p.puffsats / 1e3,
        "data centre t": found.payload / 1e3,
        "kg per PuffSat kg": found.per_puffsat,
    }


def _curve_point(task: Tuple[str, float, float]) -> Optional[Delivery]:
    name, rise, fixed = task
    return best_delivery(WAVE_SPEEDS_400[name], rise, fixed)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the delivery curve over climb speed and the comparison with today.

    Args:
        argv: ``--workers``, ``--fixed`` (also run the fixed-share sensitivity).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args(argv)
    from concurrent.futures import ProcessPoolExecutor

    rows = []
    for name, speed in WAVE_SPEEDS_400.items():
        for cap in (None, DELIVERY_CAP_G):
            today = todays_delivery(speed, cap)
            rows.append(
                delivery_row(
                    f"{name}: today, {'4 g cap' if cap else 'uncapped'}", today
                )
            )
    print("Today's plate at k = 10, aim optimised on a coarse grid:")
    print(tabulate(rows, headers="keys", floatfmt=".4g"), flush=True)
    fixed_shares = (0.0, FIXED_PLATE_SHARE) if args.fixed else (0.0,)
    for name, fixed in itertools.product(WAVE_SPEEDS_400, fixed_shares):
        points = [(name, rise, fixed) for rise in CLIMB_SPEEDS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            found = list(pool.map(_curve_point, points))
        curve = [
            delivery_row(f"{name}, fixed {fixed / 1e3:g} t", f)
            for f in found
            if f is not None
        ]
        print(
            f"\nBest delivery per booster flight at each climb ({name}, plate "
            f"fixed share {fixed / 1e3:g} t), 4 g cap, k = 10:"
        )
        print(tabulate(curve, headers="keys", floatfmt=".4g"), flush=True)
        for f in found:
            if f is not None:
                print(f"  climb {f.rise:g}: pulse {f.pulse!r} N s, aim {f.aim!r}")


if __name__ == "__main__":
    main()
