"""Exploratory 100-tonne rotating-tether sizing; not a closed mission design.

Run: PYTHONPATH=. .venv/bin/python docs/tether_plate_analysis.py
Physical inputs carry Astropy units. Dimensionless taper assumes constant
allowable stress, neglects Earth tides, and sizes one radial payload-bearing arm.
"""

import math

from astropy import units as u
from astropy.constants import GM_earth, R_earth, g0
from scipy.integrate import quad
from scipy.optimize import brentq


def main():
    payload = 100 * u.t
    radius = 10 * u.km
    tip = 3 * u.km / u.s
    catch = R_earth + 400 * u.km
    hub = catch + radius
    release = catch + 2 * radius
    circular = (GM_earth / catch) ** 0.5
    hub_speed = circular + tip
    outgoing = circular + 2 * tip
    axis = 1 / (2 / hub - hub_speed**2 / GM_earth)
    period = 2 * math.pi * (axis**3 / GM_earth) ** 0.5
    excess = (outgoing**2 - 2 * GM_earth / release) ** 0.5
    print(f"Throw mass: {payload}; ideal heavy-hub approximation")
    print(f"Tip acceleration: {(tip**2 / radius / g0).decompose():.3f} g")
    print(f"Half rotation: {(math.pi * radius / tip).to(u.s):.3f}")
    print(f"Hub periapsis speed: {hub_speed.to(u.km/u.s):.4f}")
    print(f"Hub apogee radius: {(2*axis-hub).to(u.km):.1f}")
    print(f"Hub period: {period.to(u.day):.4f}")
    print(f"Release excess speed: {excess.to(u.km/u.s):.4f}")
    # Float Kepler kernel: seconds, km, km/s. Unthrusted reference orbit only.
    eccentricity = (1 - hub / axis).decompose().value
    mean_motion = (2 * math.pi / period).to_value(1 / u.s)
    axis_km = axis.to_value(u.km)
    mu = GM_earth.to_value(u.km**3 / u.s**2)

    def coast(time_s):
        anomaly = brentq(
            lambda value: value - eccentricity * math.sin(value) - mean_motion * time_s,
            0,
            math.pi,
        )
        distance = axis_km * (1 - eccentricity * math.cos(anomaly))
        return distance, math.sqrt(mu * (2 / distance - 1 / axis_km))

    mean_speed = quad(lambda time: coast(time)[1], 0, 2700)[0] / 2700
    print(
        f"At +/-45 min: altitude {coast(2700)[0]-R_earth.to_value(u.km):.1f} km; "
        f"90-min mean coast speed {mean_speed:.3f} km/s"
    )
    for target in (11.564279430721319, 13.830549988691214):
        burn = ((target * u.km / u.s) ** 2 + 2 * GM_earth / release) ** 0.5 - outgoing
        retained = math.exp(-(burn / (380 * u.s * g0)).decompose().value)
        print(
            f"Target excess {target:.4f} km/s: burn {burn.to(u.km/u.s):.4f}; "
            f"ideal mass retained {retained:.4f}"
        )
    density = 970 * u.kg / u.m**3
    for stress in (1, 1.5, 1.75, 2, 3):
        strength = stress * u.GPa
        k = (density * tip**2 / (2 * strength)).decompose().value
        ratio = math.sqrt(math.pi * k) * math.exp(k) * math.erf(math.sqrt(k))
        area = (payload * tip**2 / (radius * strength)).to(u.m**2)
        print(
            f"Allowable stress {stress:g} GPa: taper {math.exp(k):.3f}; "
            f"one arm {(ratio*payload).to(u.t):.1f}; "
            f"two arms {(2*ratio*payload).to(u.t):.1f}; tip area {area:.4f}"
        )
    exhaust = 2000 * u.s * g0
    efficiency = 0.5
    impulse = payload * 2 * tip
    energy = (impulse * exhaust / (2 * efficiency)).to(u.kW * u.hour)
    battery_specific = 200 * u.W * u.hour / u.kg
    print(
        f"Leading-order reboost propellant per throw: {(impulse/exhaust).to(u.t):.2f}"
    )
    print(f"Electrical energy per throw: {energy.to(u.GW*u.hour):.3f}")
    for burns in (1, 10, 100):
        power = (energy / (burns * 90 * u.min)).to(u.MW)
        battery = (energy / burns / battery_specific).to(u.t)
        average = (energy / (burns * period)).to(u.MW)
        array = (average / (150 * u.W / u.kg)).to(u.t)
        print(
            f"{burns} burns/throw: {power:.2f}; battery {battery:.1f}; "
            f"nominal coast/recharge interval {(burns*period).to(u.day):.1f}; "
            f"average array {average:.3f}, {array:.1f}"
        )
    pulse_bound = (2 * efficiency * battery_specific / exhaust).to(u.m / u.s)
    print(
        f"Deployment pulse delta-v bound: {pulse_bound:.2f} times battery mass fraction"
    )
    for ratio in (20, 100):
        gain = (2 * tip * ratio / (ratio + 1)).to(u.km / u.s)
        recoil = (2 * tip / (ratio + 1)).to(u.m / u.s)
        print(
            f"Point-mass elastic analogy, hub/payload {ratio}: gain {gain:.4f}, recoil {recoil:.2f}"
        )
    print("Tip-speed sensitivity: fixed 10 km radius, 1.75 GPa, 10 recovery burns")
    for speed in (1, 1.5, 2, 2.5, 3):
        alternative_tip = speed * u.km / u.s
        k = (density * alternative_tip**2 / (2 * 1.75 * u.GPa)).decompose().value
        arm_ratio = math.sqrt(math.pi * k) * math.exp(k) * math.erf(math.sqrt(k))
        dv = (
            ((11.564279430721319 * u.km / u.s) ** 2 + 2 * GM_earth / release) ** 0.5
            - circular
            - 2 * alternative_tip
        )
        alternative_axis = 1 / (2 / hub - (circular + alternative_tip) ** 2 / GM_earth)
        alternative_period = 2 * math.pi * (alternative_axis**3 / GM_earth) ** 0.5
        alternative_energy = payload * 2 * alternative_tip * exhaust / (2 * efficiency)
        print(
            f"Tip {speed:.1f} km/s: two-arm fibre {(2*arm_ratio*payload).to(u.t):.1f}; "
            f"burn {dv.to(u.km/u.s):.3f}; retained {math.exp(-(dv/(380*u.s*g0)).decompose().value):.3f}; "
            f"10-orbit interval {(10*alternative_period).to(u.day):.2f}; "
            f"battery {(alternative_energy/10/battery_specific).to(u.t):.1f}"
        )
    # Sanity checks, not engineering validation.
    assert 90 < (tip**2 / radius / g0).decompose().value < 100
    assert axis > hub and hub_speed < (2 * GM_earth / hub) ** 0.5
    assert 8 < excess.to_value(u.km / u.s) < 9
    assert 3 < energy.to_value(u.GW * u.hour) < 4


if __name__ == "__main__":
    main()
