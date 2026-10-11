"""Verify/reoptimize a recorded zero-ablation hydrogen departure.

Float substrate uses km, s, and unit ignition mass, as orbit_turn_analysis does.
All mission inputs are explicit; no scratch files or mutated module constants.
"""

import numpy as np
from astropy import units as u
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from src import orbit_turn_analysis as ot


def evaluate(parameters, geometry, fine=True):
    """Propagate the signed projectile-momentum law and check the Earth corridor."""
    angle, before, spent, altitude_scale = parameters
    duration = geometry["duration_s"]
    front = geometry["front"]
    mu, earth, radius, speed = ot._orbit(
        altitude_scale * 10000 * u.km, geometry["parking_days"] * u.day
    )
    start = ot._coast_with_sense(
        mu, radius, speed, before * duration, -1 if front else 1
    )
    rotation = ot._rotation(angle)
    start[:2] = rotation @ start[:2]
    start[2:] = rotation @ start[2:]
    incoming = ot._rotation(geometry["aim_rad"]) @ np.array([1.0, 0.0])
    excess = geometry["return_vinf_km_s"]
    flow = spent / duration
    loading = 26.76 * 5500 / geometry["temperature_K"]

    def impulse(position, velocity):
        route = ot._uncanted_route(
            ot._incoming_routes(mu, earth, position, excess, incoming), front
        )
        relative = route.velocity - velocity
        closing = np.linalg.norm(relative)
        carried = loading * (closing / 75) ** 2 - 1
        exhaust = closing * np.sqrt(0.85**2 / (carried + 1))
        value = (1 + 1 / carried) * exhaust * ot._unit(velocity) + relative / carried
        return route, carried, value

    def rhs(time, state):
        position, velocity, mass = state[:2], state[2:4], state[4]
        _, _, value = impulse(position, velocity)
        return np.r_[
            velocity,
            -mu * position / np.linalg.norm(position) ** 3 + flow / mass * value,
            -flow,
        ]

    times = np.linspace(0, duration, 1201 if fine else 121)
    solution = solve_ivp(
        rhs,
        (0, duration),
        np.r_[start, 1.0],
        t_eval=times,
        rtol=1e-10 if fine else 3e-8,
        atol=1e-9 if fine else 3e-8,
        max_step=2 if fine else 20,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    vinf, direction = ot._outgoing_asymptote(mu, solution.y[:, -1])
    target = geometry["departure_vinf_km_s"]
    clearance, missed, rods = float("inf"), float("inf"), []
    for position, velocity in zip(solution.y[:2].T, solution.y[2:4].T):
        route, carried, _ = impulse(position, velocity)
        visible = np.linalg.norm(position) - earth
        if route.radial_velocity > 0:
            visible = min(visible, route.periapsis_radius - earth)
        clearance = min(clearance, visible)
        missed = min(missed, route.periapsis_radius - earth)
        rods.append(flow / carried)
    return {
        "residual": np.array([(vinf - target) / target, direction]),
        "clearance": clearance,
        "miss": missed,
        "spent": spent,
        "rod": float(np.trapezoid(rods, times)),
        "peak_rod_rate": max(rods),
    }


def optimize(geometry):
    """Repeat the original SLSQP search box, without a cached trajectory."""
    mu, _, radius, _ = ot._orbit(5000 * u.km, geometry["parking_days"] * u.day)
    half = np.arcsin(1 / (1 + radius * geometry["departure_vinf_km_s"] ** 2 / mu))
    sense = -1 if geometry["front"] else 1
    initial = [-sense * (np.pi / 2 + half), 0.5, 0.53, 0.5]
    cache = {}

    def ev(values):
        key = tuple(values)
        if key not in cache:
            cache.clear()
            cache[key] = evaluate(values, geometry, fine=False)
        return cache[key]

    result = minimize(
        lambda values: values[2],
        initial,
        method="SLSQP",
        bounds=[(-np.pi, np.pi), (0, 1), (0.05, 0.9), (0.06, 2)],
        constraints=[
            {"type": "eq", "fun": lambda values: ev(values)["residual"]},
            {"type": "ineq", "fun": lambda values: ev(values)["clearance"] / 1000},
            {"type": "ineq", "fun": lambda values: (ev(values)["miss"] - 600) / 1000},
        ],
        options={"ftol": 1e-10, "maxiter": 150},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, evaluate(result.x, geometry)
