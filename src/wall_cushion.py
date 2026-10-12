"""A chamber wall tile on a gas cushion, the Orion shock-absorber trick (ADR 0054).

The survivable chamber's wall sees two loads (parent ``sec:dry_wrap_wall``): a
spike of a few kPa s arriving in 2-4 microseconds, and the steady ~94 bar that
rises over about a millisecond and decays with the blowdown.  The spike sets
the hoop overshoot ``D`` (1.8-5.7, ``tab:dry_wrap_zones``), the solved shell's
peak strain over its static strain.

Here an inner tile of areal mass ``mu`` rides on a sealed gas layer of
thickness ``d`` against a strong backwall.  The cushion is precharged to
``p0`` and holds the tile on inner stops until the chamber pressure beats it.
The spike only gives the tile a speed ``I / mu``; the cushion stops it over
centimetres and milliseconds, so the backwall never sees the microseconds.
Orifices between cushion cells add a damping ``c v`` whose force also reaches
the backwall.

One degree of freedom per unit area, adiabatic cushion, uniform spike.  The
companion impact simulation is where a real answer lives.
"""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from scipy.integrate import solve_ivp

#: The survivable methane chamber's steady peak, its rise and its blowdown
#: e-fold (``chamber_isp.blowdown_time`` on ``METHANE_SURVIVABLE_GAS``).
STEADY_PEAK = 94.0 * u.bar
RISE_TIME = 1.0 * u.ms
BLOWDOWN_EFOLD = 64.0 * u.ms


@dataclass(frozen=True)
class CushionResponse:
    """How a cushioned tile answers one pulse.

    Attributes:
        travel: Largest outward travel of the tile.
        backwall_ratio: Largest backwall load (cushion plus damping) over the
            steady peak; the cushioned wall's counterpart of ``D``.
    """

    travel: u.Quantity
    backwall_ratio: float


def cushion_response(
    tile: u.Quantity,
    thickness: u.Quantity,
    precharge: u.Quantity,
    spike: u.Quantity,
    damping: u.Quantity = 0.0 * u.Pa * u.s / u.m,
    gamma: float = 1.4,
    duration: u.Quantity = 60.0 * u.ms,
) -> CushionResponse:
    """Integrate one pulse on a cushioned tile.

    Args:
        tile: Tile mass per unit area.
        thickness: Cushion gas layer thickness.
        precharge: Cushion pressure with the tile on its stops.
        spike: Spike impulse per unit area, delivered at once.
        damping: Orifice damping, force per area per unit tile speed.
        gamma: The cushion gas's ratio of heat capacities.
        duration: How long to follow the pulse.

    Returns:
        The tile's travel and the backwall's peak load ratio.

    Raises:
        ValueError: If an input is nonphysical.
    """
    mu = float(tile.to_value(u.kg / u.m**2))
    d = float(thickness.to_value(u.m))
    p0 = float(precharge.to_value(u.Pa))
    c = float(damping.to_value(u.Pa * u.s / u.m))
    peak = float(STEADY_PEAK.to_value(u.Pa))
    rise = float(RISE_TIME.to_value(u.s))
    efold = float(BLOWDOWN_EFOLD.to_value(u.s))
    end = float(duration.to_value(u.s))
    if mu <= 0.0 or d <= 0.0 or p0 <= 0.0 or c < 0.0:
        raise ValueError("tile, thickness and precharge must be positive")

    def chamber(t: float) -> float:
        return peak * min(t / rise, 1.0) * float(np.exp(-max(t - rise, 0.0) / efold))

    def rhs(t: float, y: np.ndarray) -> list[float]:
        x, v = float(y[0]), float(y[1])
        force = chamber(t) - p0 * (d / (d - x)) ** gamma - c * v
        if x <= 0.0 and v <= 0.0 and force < 0.0:
            return [0.0, 0.0]
        return [v, force / mu]

    times = np.linspace(0.0, end, 60001)
    speed = float(spike.to_value(u.Pa * u.s)) / mu
    sol = solve_ivp(
        rhs,
        (0.0, end),
        [0.0, speed],
        max_step=5e-6,
        rtol=1e-9,
        atol=1e-12,
        t_eval=times,
    )
    x = np.maximum(sol.y[0], 0.0)
    load = p0 * (d / (d - x)) ** gamma + c * np.maximum(sol.y[1], 0.0)
    return CushionResponse(float(np.max(x)) * u.m, float(np.max(load)) / peak)


def main() -> None:
    """Print the cushion table behind ADR 0054."""
    print(
        "10 mm steel tile (80 kg/m^2) on a gas cushion, 94 bar steady peak.\n"
        "cushion  precharge  spike    damping     travel   backwall / 94 bar"
    )
    for thickness, precharge in ((6.0, 80.0), (8.0, 60.0)):
        for spike in (2.0, 5.0):
            for damping in (0.0, 1.0e5):
                r = cushion_response(
                    80.0 * u.kg / u.m**2,
                    thickness * u.cm,
                    precharge * u.bar,
                    spike * u.kPa * u.s,
                    damping * u.Pa * u.s / u.m,
                )
                print(
                    f"{thickness:4.0f} cm  {precharge:5.0f} bar  {spike:3.0f} kPa s  "
                    f"{damping:8.0e}  {r.travel.to_value(u.cm):6.2f} cm  "
                    f"{r.backwall_ratio:6.2f}"
                )


if __name__ == "__main__":
    main()
