"""Aiming the Sun-wave disk train: re-aim, tolerance, drift and thrusters.

Answers the child's thick-disk request (ADR 0086): a 2.5 kg, 1 cm, 38 cm
graphite disk at one pulse a second, against the headline's 1 kg, 3.6 mm,
40 cm disk at two, at the 9.86 R_sun baseline (head-on closing ~392 km/s).
The thin disk's charges are the simulation's Result 24 (@ 1eb4115, the
parent's S62; ADR 0087): its best charge is 1.2x its radius, and the thin
trains are priced again at 4 R_sun, where no coupling ratio changes.

The impact simulation's Result 23 (``puffsat_impact_simulation`` @ 3581fbd,
``docs/thick_disk_answers.md``) sets the tolerance.  A disk is caught whole
until its rim clips the bowl's 0.25 m bore, **6 cm** off centre on the
recommended charge 1.3x the disk's radius.  An off-centre hit pushes the ship
sideways by a share of the pulse that grows with the offset.  So the
tolerance is centimetres, not the "meters" the parent assumed, and four
things follow:

1. **Re-aim.**  Each pulse kicks the ship sideways by ``g r`` of its kick,
   ``r`` the disk's offset and ``g`` the side-impulse slope.  A disk steered
   on the network's prediction cannot see the kicks of its last ``B`` seconds,
   so it misses by ``g sigma a sqrt(dt B^3 / 3)`` on each axis
   (:func:`blind_miss`).  That is the parent's scheme, and at ``B`` of tens
   of seconds it is unstable: the miss feeds the next offset
   (:func:`loop_gain`).  Two fixes close it.  In the first, the ship holds
   its track by cancelling each kick's side impulse and torque with its own
   rockets (:func:`hold_fraction`, charged in the ledger as
   ``Design.sun_hold``).  In the second, the ship broadcasts each measured
   kick straight to the disks a second or two out (:func:`short_loop_window`).
   Either way the ship must know each kick to a fraction of a mm/s
   (:func:`kick_knowledge`).
2. **Tolerance.**  The dispersion each axis may carry
   (:func:`dispersion_requirement`), against the budget the network, the
   disk's own drift and the ship carry (:func:`dispersion_budget`).
3. **Drift.**  The simulation's CME drift law (``cme_drift.py`` @ 3581fbd),
   re-flown on each disk's geometry (:func:`cme_drift`), and edge-on sunlight
   integrated over the finite Sun (:func:`edge_sunlight_share`).
4. **Thrusters.**  The thrust and cold gas each disk needs (:func:`thruster`).

Everything is float arithmetic (m, s, kg) except :func:`priced_trains`,
which flies each train through :mod:`src.solar_dive_growth`.  Run with
``make disk-train-aim``.
"""

import argparse
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tabulate import tabulate

from src import solar_dive_growth as sdg

_G0 = 9.80665
_M_P = 1.67262192e-27  # kg, the simulation's conversion of number density
_R_SUN = 6.957e8  # m
_L_SUN = 3.828e26  # W
_C = 2.99792458e8  # m/s

#: The closure every row flies: Parker's depth, the slowest return.
CLOSURE = (9.86, 60.0)
#: The CME cases below are the simulation's at this depth only.
DIVE_SOLAR_RADII = CLOSURE[0]
#: ADR 0085's deep closure ($85 / $89).  Result 24: the 2-D gas has no speed
#: scale, so every coupling ratio, clearance and side slope carries to 616.6
#: km/s unchanged.
DEEP_CLOSURE = (4.0, 90.0)

#: The Sun plate's radius, from its area: the attitude thrusters' lever arm.
PLATE_RADIUS = math.sqrt(sdg.SUN_PLATE_AREA / math.pi)
#: The bowl's bore radius (Result 23).
BORE = 0.25  # m

#: The network's sideways error at the train's ends: 1 cm ranges times 1.3,
#: the parent's figure for the 0.99 h train (sec:sun_wave_ranging).  The
#: 1 Hz train is shorter, so this is the larger of the two.
NETWORK_SIDEWAYS = 0.013  # m
#: How old a network fix is when it reaches a disk: the parent's 15-20 s.
FIX_LATENCY = 20.0  # s
#: The parent's last correction, a minute or two before impact.
PARENT_BLIND = (60.0, 120.0)  # s
#: The disk's final move: the network's 3-sigma correction, in this time.
FINAL_MOVE_TIME = 10.0  # s
#: Cold gas, the parent's 60 s (the simulation's cme_drift.COLD_GAS_ISP).
COLD_GAS_ISP = 60.0  # s
#: A disk clipping the bore is lost and strikes its lip: one in a thousand.
CLIP_ODDS = 1.0e-3
#: Sunlight's modelling error, the simulation's "1%" (cme_drift.md item 5).
SUNLIGHT_MODEL_ERROR = 0.01
#: The disks fall from Jupiter: at most half an orbit of sunlight before the
#: train (true anomaly swept, rad).  A bound.
INBOUND_ANOMALY = math.pi


@dataclass(frozen=True)
class Charge:
    """A Sun bowl charge under a disk (the simulation's Results 23 and 24).

    Attributes:
        name: Label.
        coupling: Exhaust speed over the thin disk on its matched charge, k = 20.
        clearance: Offset at which the disk's rim clips the bore (m).
        side_slope: Side impulse per unit axial impulse per metre of offset.
        lens_radius: The charge's radius when it is matched to the disk, so
            that an offset disk loses the lens ``2 d / (pi R)`` of itself
            (the low bracket); 0 when the disk is caught whole to the bore.
        charge_radius: A wide charge's radius when the bore is wider still,
            so the disk is caught whole only to ``charge_radius -
            disk_radius`` and loses its overhang beyond (Result 24's x1.2,
            the lone-lens bound); 0 when caught whole to the bore.
        disk_radius: The disk's radius, with ``charge_radius``.
    """

    name: str
    coupling: float
    clearance: float
    side_slope: float
    lens_radius: float = 0.0
    charge_radius: float = 0.0
    disk_radius: float = 0.0


@dataclass(frozen=True)
class Disk:
    """A Sun-wave disk and the train it flies in.

    Attributes:
        name: Label.
        mass: Struck mass (kg), the ledger's disk floor.
        rate: Pulses a second.
        diameter: m.
        thickness: m.
        charge: The charge it strikes.
    """

    name: str
    mass: float
    rate: float
    diameter: float
    thickness: float
    charge: Charge

    @property
    def face(self) -> float:
        """Face area (m^2)."""
        return math.pi * self.diameter**2 / 4.0

    @property
    def edge(self) -> float:
        """Edge's projected area, ``D t`` (m^2)."""
        return self.diameter * self.thickness


# Result 23's side impulses are 0.4% at 6.0 cm on the matched charge, 0.8% at
# 6.0 cm on x1.3 and 1.3% at 9.5 cm on x1.5 (thick_disk_offset.csv): odd in
# the offset, so linear near the axis.
_MATCHED_SLOPE = 0.004185855026520384 / 0.05981173018445443
_X13_SLOPE = 0.008054976293215412 / 0.05981173018445443
_X15_SLOPE = 0.01316158239279586 / 0.09509413490777277

#: Result 23's 2.5 kg disk at 22 kg/m^2.
_THICK_RADIUS = 0.19018826981554557
#: Result 24's 1 kg disk at 8 kg/m^2.
_THIN_RADIUS = 0.19947114020071635
# Result 24 (thin_disk_side.csv, 0.0125 m grid): the side impulse is linear
# in offset to 3% from 3 cm out, 0.136 per metre on any wide charge (the
# bowl's, for both disks) and half that on the matched charge, where the
# burst sits at the struck lens's centroid.
_THIN_MATCHED_SLOPE = 0.07169657941112233
_THIN_WIDE_SLOPE = 0.13574341283697652
# Its couplings over the thin disk's matched charge, at the widths the
# finest grid resolved (thin_disk_width.csv): x1.19 and x1.31.
THIN_MATCHED = Charge(
    "matched", 1.0, BORE - _THIN_RADIUS, _THIN_MATCHED_SLOPE, _THIN_RADIUS
)
THIN_X12 = Charge(
    "x1.2",
    1.0741317672617798,
    BORE - _THIN_RADIUS,
    _THIN_WIDE_SLOPE,
    charge_radius=0.2393653682408596,
    disk_radius=_THIN_RADIUS,
)
THIN_X13 = Charge(
    "x1.3", 1.0466250631108969, 0.2593124822609313 - _THIN_RADIUS, _THIN_WIDE_SLOPE
)
#: ADR 0086's assumed x1.3: Result 23's 1.08 (a x1.25 charge, Result 24 found),
#: a 5 cm clearance and the thick disk's slope.  Kept to show the change.
ASSUMED_X13 = Charge("x1.3 (ADR 0086, assumed)", 1.08, BORE - 0.20, _X13_SLOPE)
THICK_MATCHED = Charge(
    "matched", 0.94, BORE - _THICK_RADIUS, _MATCHED_SLOPE, _THICK_RADIUS
)
THICK_X13 = Charge("x1.3", 1.11, BORE - _THICK_RADIUS, _X13_SLOPE)
THICK_X15 = Charge("x1.5", 1.05, 0.28528240472331834 - _THICK_RADIUS, _X15_SLOPE)

HEADLINE = Disk("1 kg, 2 Hz (headline)", 1.0, 2.0, 0.40, 3.6e-3, THIN_MATCHED)
HEADLINE_X12 = replace(HEADLINE, name="1 kg, 2 Hz, x1.2 charge", charge=THIN_X12)
HEADLINE_X13 = replace(HEADLINE, name="1 kg, 2 Hz, x1.3 charge", charge=THIN_X13)
HEADLINE_ASSUMED = replace(
    HEADLINE, name="1 kg, 2 Hz, x1.3 assumed (ADR 0086)", charge=ASSUMED_X13
)
THICK = Disk(
    "2.5 kg, 1 Hz, matched charge", 2.5, 1.0, 2.0 * _THICK_RADIUS, 0.01, THICK_MATCHED
)
PROPOSAL = replace(THICK, name="2.5 kg, 1 Hz, x1.3 charge (proposal)", charge=THICK_X13)
THICK_WIDE = replace(THICK, name="2.5 kg, 1 Hz, x1.5 charge", charge=THICK_X15)
DISKS = (
    HEADLINE,
    HEADLINE_X12,
    HEADLINE_X13,
    HEADLINE_ASSUMED,
    THICK,
    PROPOSAL,
    THICK_WIDE,
)
#: The trains priced again at :data:`DEEP_CLOSURE`: the 1 kg disk's.
DEEP_DISKS = (HEADLINE, HEADLINE_X12, HEADLINE_X13)


# --- 1. Re-aim -------------------------------------------------------------


def blind_miss(side: float, accel: float, dt: float, blind: float) -> float:
    """One axis's miss from the kicks a disk cannot see (m, rms).

    Kicks of ``accel dt`` arrive every ``dt``, each with a random side impulse
    of rms ``side`` of itself.  The kick ``k dt`` before impact displaces the
    ship by its side impulse times ``k dt``.  The disk saw the kick at its
    last correction and the struck one moves nothing yet, so the unseen kicks
    are ``k = 1 .. n - 1`` with ``n = blind / dt``: for long windows,
    ``side accel sqrt(dt blind^3 / 3)``.

    Args:
        side: Side impulse per unit kick, rms on one axis.
        accel: The ship's mean acceleration (m/s^2).
        dt: Seconds between pulses.
        blind: Seconds before impact the disk last corrects.

    Returns:
        The miss (m).
    """
    m = max(math.ceil(blind / dt - 1e-9) - 1, 0)
    return side * accel * dt * dt * math.sqrt(m * (m + 1.0) * (2.0 * m + 1.0) / 6.0)


def loop_gain(disk: Disk, accel: float, blind: float) -> float:
    """Miss per unit offset when the ship's unseen kicks set the next offset.

    The side impulse is ``g r``, so a dispersion ``sigma`` makes a side rms
    ``g sigma`` and a miss ``G sigma``.  The dispersion then solves
    ``sigma^2 = sigma_0^2 + G^2 sigma^2``: it has no steady value at
    ``G >= 1``.

    Args:
        disk: The train.
        accel: The ship's mean acceleration (m/s^2).
        blind: Seconds before impact the disk last corrects.

    Returns:
        ``G``.
    """
    return blind_miss(disk.charge.side_slope, accel, 1.0 / disk.rate, blind)


def short_loop_window(disk: Disk, accel: float, gain: float = 0.5) -> float:
    """The longest blind window, in whole pulses, at loop gain ``gain`` or less (s).

    The ship broadcasts each measured kick to the disks a few seconds out,
    which correct in the time left.  At ``G = 0.5`` the ship's share
    amplifies the rest of the dispersion by 1.15.  The window is a pulse or
    two, so it is counted in pulses, not by the long-window formula.
    """
    dt = 1.0 / disk.rate
    pulses = 1
    while loop_gain(disk, accel, (pulses + 1) * dt) <= gain:
        pulses += 1
    return pulses * dt


def hold_fraction(disk: Disk, sigma: float, lever: float = PLATE_RADIUS) -> float:
    """Ship propellant to hold track and attitude, as a share of its Sun boost.

    The ship cancels each pulse's side impulse ``g r`` of the kick, and its
    torque ``J r``, with thrusters ``lever`` from the axis.  With ``r`` the
    2-D offset at ``sigma`` an axis, ``E|r| = sigma sqrt(pi/2)``, so the side
    impulses sum to ``g E|r|`` of the boost and the torques to ``E|r| / lever``
    of it, whatever the pulse size or cadence.  Per-pulse cancelling is the
    upper bound: any slack lets the tilt and side velocity walk.

    Args:
        disk: The train.
        sigma: Arrival dispersion, one axis (m).
        lever: The attitude thrusters' arm (m).

    Returns:
        The ``Design.sun_hold`` fraction.
    """
    mean_offset = sigma * math.sqrt(math.pi / 2.0)
    return mean_offset * (disk.charge.side_slope + 1.0 / lever)


def kick_knowledge(budget: float, dt: float, horizon: float = FIX_LATENCY) -> float:
    """How well the ship must know each kick (m/s, rms an axis).

    An error in a measured kick is a velocity nobody corrects until the
    network sees the ship move, ``horizon`` later.  Each pulse's error then
    walks the ship ``sqrt(horizon^3 / (3 dt))`` times itself.

    Args:
        budget: The ship's share of the dispersion (m).
        dt: Seconds between pulses.
        horizon: Seconds until the network sees the ship's motion.

    Returns:
        The rms error each kick may carry.
    """
    return budget / blind_miss(1.0, 1.0, dt, horizon) * dt


def simulate_blind_miss(
    side: float,
    kicks: Sequence[float],
    dt: float,
    blind: float,
    hold: bool,
    trials: int = 50,
    seed: int = 1,
) -> Tuple[float, float]:
    """Monte Carlo of one axis: the disks' miss and the ship's hold propellant.

    Each pulse kicks the ship sideways by a Gaussian ``side`` of itself.  A
    disk's last correction aims at the ship's position predicted ``blind``
    seconds out from its state then.  With ``hold`` the ship cancels each
    side impulse before the next pulse.

    Returns:
        (rms miss in m, mean hold delta-v in m/s), one axis.
    """
    rng = np.random.default_rng(seed)
    kick = np.asarray(kicks, dtype=float)
    n = kick.size
    lag = int(round(blind / dt))
    misses, spent = [], []
    for _ in range(trials):
        side_v = rng.normal(0.0, side, n) * kick
        if hold:
            spent.append(float(np.sum(np.abs(side_v))))
            misses.append(np.zeros(n - lag))
            continue
        vel = np.cumsum(side_v)
        pos = np.concatenate(([0.0], np.cumsum(vel[:-1] * dt)))
        j = np.arange(lag, n)
        misses.append(pos[j] - (pos[j - lag] + vel[j - lag] * blind))
        spent.append(0.0)
    m = np.concatenate(misses)
    return float(np.sqrt(np.mean(m * m))), float(np.mean(spent))


# --- 2. Tolerance ----------------------------------------------------------


def dispersion_requirement(charge: Charge, clip_odds: float = CLIP_ODDS) -> float:
    """The arrival dispersion an axis may carry (m, rms).

    The rim must clear the bore at all but ``clip_odds`` of pulses:
    ``P(r > c) = exp(-c^2 / 2 sigma^2)`` for a 2-D Gaussian at ``sigma`` an
    axis.  A matched charge also loses coupling well inside that
    (:func:`mean_coupling_loss`), which the ledger charges rather than forbids.
    """
    return charge.clearance / math.sqrt(2.0 * math.log(1.0 / clip_odds))


def struck_share(offset: float, disk_radius: float, charge_radius: float) -> float:
    """The share of a disk's face over the charge at an offset (circle overlap).

    The simulation's lone-lens bound (``thick_disk.lens_share``): the
    overhang goes unstruck.
    """
    r, c, d = disk_radius, charge_radius, offset
    if d <= c - r:
        return 1.0
    if d >= c + r:
        return 0.0
    area = (
        r * r * math.acos((d * d + r * r - c * c) / (2.0 * d * r))
        + c * c * math.acos((d * d + c * c - r * r) / (2.0 * d * c))
        - 0.5 * math.sqrt((-d + r + c) * (d + r - c) * (d - r + c) * (d + r + c))
    )
    return area / (math.pi * r * r)


def mean_coupling_loss(charge: Charge, sigma: float, n: int = 400) -> float:
    """Mean coupling lost to offset at a dispersion.

    An offset ``d`` on a matched charge leaves the lens ``2 d / (pi R)`` of
    the disk unstruck (Result 23's low bracket, the struck lens alone), and
    ``E d = sigma sqrt(pi/2)``.  A wide charge inside a wider bore catches
    the disk whole to ``charge_radius - disk_radius`` and loses the overhang
    beyond, averaged over the 2-D Gaussian out to the bore (past it the disk
    clips, counted in :data:`CLIP_ODDS`).  Zero on a charge that catches the
    disk whole to the bore.
    """
    if charge.lens_radius > 0.0:
        return 2.0 * sigma * math.sqrt(math.pi / 2.0) / (math.pi * charge.lens_radius)
    if charge.charge_radius <= 0.0:
        return 0.0
    whole = charge.charge_radius - charge.disk_radius
    if whole >= charge.clearance:
        return 0.0
    r = np.linspace(whole, charge.clearance, n)
    lost = np.array(
        [1.0 - struck_share(x, charge.disk_radius, charge.charge_radius) for x in r]
    )
    pdf = r / sigma**2 * np.exp(-(r**2) / (2.0 * sigma**2))
    return float(np.trapezoid(lost * pdf, r))


@dataclass(frozen=True)
class Budget:
    """One axis's arrival dispersion, by source (m, rms).

    Attributes:
        required: :func:`dispersion_requirement`.
        network: The ranging network's sideways error.
        drift: The disk's unseen drift between its last fix and impact.
        ship: What is left for the ship's unpredicted motion.
    """

    required: float
    network: float
    drift: float
    ship: float

    @property
    def closes(self) -> bool:
        """Whether anything is left for the ship."""
        return self.ship > 0.0


def dispersion_budget(disk: Disk, drift_accel: float) -> Budget:
    """The dispersion budget at a disk's unseen acceleration (m/s^2).

    The disk drifts ``a t^2 / 2`` over the fix's latency.  The ship gets what
    is left in quadrature, or a negative number when nothing is.
    """
    required = dispersion_requirement(disk.charge)
    drift = 0.5 * drift_accel * FIX_LATENCY**2
    left = required**2 - NETWORK_SIDEWAYS**2 - drift**2
    return Budget(
        required, NETWORK_SIDEWAYS, drift, math.copysign(math.sqrt(abs(left)), left)
    )


# --- 3. Drift --------------------------------------------------------------

#: The simulation's CME cases at 9.86 R_sun (cme_drift.py @ 3581fbd):
#: coronagraph densities with 2%/R_sun mass growth, its median speed, and
#: the densest of 29 times Parker's shock compression at 2500 km/s.
CME_MEDIAN_N = 46659.28470220401  # cm^-3
CME_DENSEST_N = 171287.95814396706 * 2.0331294597349645  # cm^-3
CME_CASES = (
    ("coronagraph median, 680 km/s", CME_MEDIAN_N, 680.0e3),
    ("coronagraph median, 1000 km/s", CME_MEDIAN_N, 1000.0e3),
    ("densest x 2.03, 2500 km/s", CME_DENSEST_N, 2500.0e3),
)
#: Out-of-plane flow tilts (deg): radial, ejecta at 1 au, the largest 1 au
#: sheath, Parker's 2022 sheath.
CME_TILTS = (0.0, 0.9115622203461408, 18.4, 23.43864527899927)
#: The simulation's passage at 9.86 R_sun.
CME_PASSAGE = 9000.0  # s
#: The request's own figures, for comparison.
REQUEST_CME_PRESSURE = 1.0e-3  # Pa
REQUEST_SUNLIGHT_PRESSURE = 2.0e-3  # Pa


@dataclass(frozen=True)
class Drift:
    """A CME's push on a disk held in it for the passage.

    Attributes:
        pressure: ``rho v_rel^2`` (Pa).
        accel: m/s^2.
        dv: m/s over the passage.
        drift: m over the passage, unseen.
    """

    pressure: float
    accel: float
    dv: float
    drift: float


def cme_drift(
    disk: Disk,
    n: float,
    speed: float,
    tilt_deg: float,
    disk_speed: float,
    passage: float = CME_PASSAGE,
    face_on: bool = False,
) -> Drift:
    """The simulation's CME law on this disk's geometry.

    The disk flies face normal out of the orbital plane at ``disk_speed``,
    tangentially; the plasma flows at ``speed`` tilted ``tilt_deg`` out of
    the plane.  It is absorbing, so the force is
    ``rho v_rel^2 (A_face |sin tilt_rel| + D t cos tilt_rel)``.  ``face_on``:
    a tumbled disk presenting its whole face.

    Args:
        disk: The disk.
        n: Plasma density (cm^-3).
        speed: Plasma speed (m/s).
        tilt_deg: Its tilt out of the orbital plane.
        disk_speed: The disk's speed (m/s).
        passage: Seconds held in the flow.
        face_on: Whole face presented.

    Returns:
        The drift.
    """
    v_rel = math.hypot(speed, disk_speed)
    rel = math.asin(speed * math.sin(math.radians(tilt_deg)) / v_rel)
    pressure = n * 1e6 * _M_P * v_rel**2
    area = (
        disk.face
        if face_on
        else disk.face * abs(math.sin(rel)) + disk.edge * math.cos(rel)
    )
    accel = pressure * area / disk.mass
    return Drift(pressure, accel, accel * passage, 0.5 * accel * passage**2)


def sunlight_pressure(solar_radii: float) -> float:
    """Sunlight's pressure on an absorbing face at a depth (Pa)."""
    r = solar_radii * _R_SUN
    return _L_SUN / (4.0 * math.pi * r * r * _C)


def edge_sunlight_share(
    disk: Disk, solar_radii: float, limb_u: float = 0.6, n: int = 400
) -> float:
    """The outward push on an edge-on disk as a share of its face-on push.

    The simulation's ``edge_sunlight`` integral over the finite solar disk
    (linear limb darkening, absorbing faces), with this disk's edge.
    """
    alpha = math.asin(1.0 / solar_radii)
    rho = (np.arange(n) + 0.5) / n * alpha
    psi = (np.arange(n) + 0.5) / n * 2.0 * math.pi
    rr, pp = np.meshgrid(rho, psi, indexing="ij")
    mu = np.sqrt(np.clip(1.0 - (np.sin(rr) / math.sin(alpha)) ** 2, 0.0, 1.0))
    weight = (1.0 - limb_u * (1.0 - mu)) * np.sin(rr)
    radial = np.cos(rr)
    normal = np.abs(np.sin(rr) * np.sin(pp))
    face_on = float(np.sum(weight * radial * radial))
    faces = float(np.sum(weight * normal * radial))
    return (faces + face_on * disk.edge / disk.face) / face_on


def sunlight_accel(disk: Disk, solar_radii: float, face_on: bool = False) -> float:
    """Sunlight's outward acceleration on the disk at a depth (m/s^2)."""
    share = 1.0 if face_on else edge_sunlight_share(disk, solar_radii)
    return sunlight_pressure(solar_radii) * disk.face * share / disk.mass


def sunlight_dv(
    disk: Disk,
    periapsis_radius: float,
    periapsis_speed: float,
    anomaly: float = INBOUND_ANOMALY,
    face_on: bool = False,
) -> float:
    """Sunlight's summed push over the inbound dive (m/s).

    ``a ~ 1/r^2`` and ``dt = r^2 dtheta / h``, so ``integral a dt`` is
    ``a_p r_p^2 dtheta / h`` with ``h = r_p v_p``, the edge-on share held at
    its perihelion value.  It is modelled, not unseen: only its modelling
    error needs gas.

    Args:
        disk: The disk.
        periapsis_radius: m.
        periapsis_speed: m/s.
        anomaly: True anomaly swept inbound (rad).
        face_on: Whole face presented.
    """
    a_p = sunlight_accel(disk, periapsis_radius / _R_SUN, face_on)
    return a_p * periapsis_radius * anomaly / periapsis_speed


# --- 4. Thrusters ----------------------------------------------------------


@dataclass(frozen=True)
class Thruster:
    """One disk's cold-gas thrusters.

    Attributes:
        accel: Acceleration needed (m/s^2).
        thrust: N.
        dv: Gas delta-v over the dive (m/s).
        gas: kg at :data:`COLD_GAS_ISP`.
        sized_by: What sets the thrust.
    """

    accel: float
    thrust: float
    dv: float
    gas: float
    sized_by: str


def thruster(
    disk: Disk,
    worst_cme: Drift,
    sunlight: float,
    move: float,
    move_time: float,
    follow: float = 0.0,
) -> Thruster:
    """Size a disk's thrusters.

    Thrust is the larger of twice the worst CME's push (cancelled as it
    builds) and the final move ``move`` made in ``move_time`` (accelerate half
    the time, brake half: ``a = 4 d / t^2``).  The gas covers the worst CME,
    sunlight's modelling error, the final move twice over, and ``follow``:
    the ship's own wander a disk must match when the ship does not hold.

    Args:
        disk: The disk.
        worst_cme: The worst CME's drift.
        sunlight: Sunlight's summed push over the dive (m/s).
        move: The final correction (m).
        move_time: Its time (s).
        follow: Delta-v to follow the ship (m/s).
    """
    cme = 2.0 * worst_cme.accel
    final = 4.0 * move / move_time**2
    accel = max(cme, final)
    dv = (
        worst_cme.dv
        + SUNLIGHT_MODEL_ERROR * sunlight
        + 2.0 * 2.0 * math.sqrt(move * final)
        + follow
    )
    return Thruster(
        accel,
        accel * disk.mass,
        dv,
        disk.mass * dv / (COLD_GAS_ISP * _G0),
        "CME" if cme >= final else "final move",
    )


# --- The trains through the ledger ------------------------------------------


@dataclass(frozen=True)
class Train:
    """A disk train as the departure-wave ship flies it at the Sun.

    Attributes:
        disk: The disk.
        arc_hours: The arc.
        pulses: Pulses in the train.
        ship: The ship at the Sun (kg).
        boost: Its Sun boost (m/s).
        kicks: Each pulse's kick (m/s), in order.
        periapsis_radius: m.
        periapsis_speed: m/s.
        closing: m/s.
    """

    disk: Disk
    arc_hours: float
    pulses: int
    ship: float
    boost: float
    kicks: Tuple[float, ...]
    periapsis_radius: float
    periapsis_speed: float
    closing: float

    @property
    def accel(self) -> float:
        """Mean acceleration (m/s^2)."""
        return sum(self.kicks) * self.disk.rate / len(self.kicks)

    @property
    def length(self) -> float:
        """The train's length along the disks' path when the boost begins (m)."""
        return self.closing * self.arc_hours * 3600.0

    @property
    def sweep_deg(self) -> float:
        """Arc swept around the Sun over the boost."""
        return math.degrees(
            self.arc_hours * 3600.0 * self.periapsis_speed / self.periapsis_radius
        )


def flown_coupling(disk: Disk) -> float:
    """The charge's coupling less its offset loss at the required dispersion."""
    sigma = dispersion_requirement(disk.charge)
    return disk.charge.coupling * (1.0 - mean_coupling_loss(disk.charge, sigma))


def ledger_design(
    disk: Disk,
    hold: float = 0.0,
    offset_loss: bool = True,
    closure: Tuple[float, float] = CLOSURE,
) -> sdg.Design:
    """The headline design re-dialled for this disk (the optimizer's start).

    ``offset_loss`` charges the charge's offset loss at the required
    dispersion; False is ADR 0085's ledger, which centres every disk.
    """
    start = sdg.HEADLINE_STARTS[closure]
    return replace(
        start,
        disk_floor=disk.mass,
        sun_rate=disk.rate,
        sun_coupling=flown_coupling(disk) if offset_loss else disk.charge.coupling,
        sun_hold=hold,
        arc_hours=start.arc_hours / disk.mass / disk.rate * 2.0,
    )


def fly_train(
    disk: Disk, design: sdg.Design, closure: Tuple[float, float] = CLOSURE
) -> Train:
    """The departure-wave ship's train at a design."""
    departure = sdg.departure_route(*closure)
    push = sdg.push_route(*closure, sdg.PARKING_DAYS)
    assert departure is not None and push is not None
    lines = sdg.fly_unit(departure, push, departure, design).lines
    ve = lines["ve_sun"] * 1e3
    pulses = int(round(design.sun_rate * design.arc_hours * 3600.0))
    ship = lines["ship_at_sun"]
    charge = lines["pulse_disk"] * design.k_sun
    masses = ship - charge * np.arange(pulses)
    kicks = tuple(float(k) for k in charge * ve / masses)
    return Train(
        disk,
        design.arc_hours,
        pulses,
        ship,
        departure.periapsis_boost * 1e3,
        kicks,
        departure.periapsis_radius * 1e3,
        departure.periapsis_speed * 1e3,
        departure.sun_closing * 1e3,
    )


@dataclass(frozen=True)
class Priced:
    """A train re-optimized through the ledger.

    Attributes:
        disk: The disk.
        hold: ``Design.sun_hold``.
        optimum: The optimizer's answer.
        lines: Departure-wave and push-wave ships' ledgers.
        closure: (perihelion R_sun, return excess km/s).
    """

    disk: Disk
    hold: float
    optimum: sdg.Optimum
    lines: Tuple[Dict[str, float], Dict[str, float]]
    closure: Tuple[float, float] = CLOSURE


def price(
    disk: Disk,
    hold: float,
    offset_loss: bool = True,
    closure: Tuple[float, float] = CLOSURE,
) -> Priced:
    """Re-optimize the dear-seed break-even with this disk and hold."""
    departure = sdg.departure_route(*closure)
    push = sdg.push_route(*closure, sdg.PARKING_DAYS)
    assert departure is not None and push is not None
    best = sdg.optimize_design(
        departure, push, ledger_design(disk, hold, offset_loss, closure)
    )
    lines = tuple(
        sdg.fly_unit(route, push, departure, best.design).lines
        for route in (departure, push)
    )
    return Priced(disk, hold, best, (lines[0], lines[1]), closure)


def headline_hold(disk: Disk) -> float:
    """The hold fraction at the disk's required dispersion."""
    return hold_fraction(disk, dispersion_requirement(disk.charge))


def priced_trains(
    closure: Tuple[float, float] = CLOSURE, disks: Sequence[Disk] = DISKS
) -> List[Priced]:
    """ADR 0085's headline, then every disk with its hold charged."""
    out = [price(HEADLINE, 0.0, offset_loss=False, closure=closure)]
    out += [price(disk, headline_hold(disk), closure=closure) for disk in disks]
    return out


def _closure_label(closure: Tuple[float, float]) -> str:
    return f"{closure[0]:g} R_sun, X {closure[1]:g}"


# --- Report ------------------------------------------------------------------


def _cm(x: float) -> str:
    return f"{100.0 * x:.2f}"


def aim_report(trains: Sequence[Train], closure: Tuple[float, float] = CLOSURE) -> str:
    """Items 1-4 for each train, without the optimizer.

    Drift, the budget against a CME and the thrusters use the simulation's
    CME cases at 9.86 R_sun, so they print at :data:`CLOSURE` only.
    """
    out: List[str] = []
    rows = []
    for t in trains:
        rows.append(
            [
                t.disk.name,
                f"{t.arc_hours:.3f}",
                t.pulses,
                f"{t.kicks[0]:.2f}-{t.kicks[-1]:.2f}",
                f"{t.accel / _G0:.3f}",
                f"{t.length / 1e9:.2f}",
                f"{t.sweep_deg:.1f}",
            ]
        )
    out.append(f"The trains (departure-wave ship, {_closure_label(closure)})")
    out.append(
        tabulate(
            rows,
            [
                "disk",
                "arc h",
                "pulses",
                "kick m/s",
                "mean g",
                "length 1e6 km",
                "sweep deg",
            ],
        )
    )

    out.append("\n2. Tolerance: the dispersion an axis may carry")
    rows = []
    for t in trains:
        c = t.disk.charge
        req = dispersion_requirement(c)
        rows.append(
            [
                t.disk.name,
                f"{c.coupling:.2f}",
                _cm(c.clearance),
                f"{100 * c.side_slope:.2f}",
                _cm(req),
                f"{100 * mean_coupling_loss(c, req):.1f}",
                f"{flown_coupling(t.disk):.3f}",
            ]
        )
    out.append(
        tabulate(
            rows,
            [
                "disk",
                "coupling",
                "bore clear cm",
                "side %/m",
                "sigma req cm",
                "offset loss %",
                "flown coupling",
            ],
        )
    )

    out.append("\n1. Re-aim: one axis, rms; side impulse at the required dispersion")
    rows = []
    for t in trains:
        req = dispersion_requirement(t.disk.charge)
        side = t.disk.charge.side_slope * req
        dt = 1.0 / t.disk.rate
        hold = hold_fraction(t.disk, req)
        lateral = req * math.sqrt(math.pi / 2.0) * t.disk.charge.side_slope
        rows.append(
            [
                t.disk.name,
                f"{100 * side:.2f}",
                *(f"{blind_miss(side, t.accel, dt, b):.1f}" for b in PARENT_BLIND),
                f"{loop_gain(t.disk, t.accel, PARENT_BLIND[0]):.0f}",
                f"{short_loop_window(t.disk, t.accel):.2f}",
                f"{lateral * t.boost:.0f} + {(hold - lateral) * t.boost:.0f}",
                f"{t.ship * hold * t.boost / (sdg.STORABLE_ISP * _G0) / 1e3:.1f}",
                f"{1e3 * kick_knowledge(max(dispersion_budget(t.disk, 0.0).ship, 0.0), dt):.3f}",
            ]
        )
    out.append(
        tabulate(
            rows,
            [
                "disk",
                "side %",
                "miss B=60 m",
                "B=120 m",
                "gain G B=60",
                "short loop B s",
                "hold dv m/s side+torque",
                "hold t",
                "kick known mm/s",
            ],
        )
    )

    if closure != CLOSURE:
        return "\n".join(out)

    out.append("\n3. Drift at 9.86 R_sun over a 2.5 h passage (dv m/s / drift m)")
    rows = []
    seen = set()
    for t in trains:
        geo = (t.disk.mass, t.disk.diameter, t.disk.thickness)
        if geo in seen:
            continue
        seen.add(geo)
        for case, n, v in CME_CASES:
            cells = []
            for tilt in CME_TILTS:
                d = cme_drift(t.disk, n, v, tilt, t.periapsis_speed)
                cells.append(f"{d.dv:.3f} / {d.drift:.0f}")
            face = cme_drift(t.disk, n, v, 0.0, t.periapsis_speed, face_on=True)
            cells.append(f"{face.dv:.2f} / {face.drift:.0f}")
            p = cme_drift(t.disk, n, v, 0.0, t.periapsis_speed).pressure
            rows.append([f"{t.disk.mass:g} kg", case, f"{p * 1e3:.2f}", *cells])
    out.append(
        tabulate(
            rows,
            [
                "disk",
                "case",
                "mPa",
                *(f"{x:.1f} deg" for x in CME_TILTS),
                "tumbled face-on",
            ],
        )
    )
    rows = []
    seen = set()
    for t in trains:
        geo = (t.disk.mass, t.disk.diameter, t.disk.thickness)
        if geo in seen:
            continue
        seen.add(geo)
        share = edge_sunlight_share(t.disk, DIVE_SOLAR_RADII)
        rows.append(
            [
                f"{t.disk.mass:g} kg",
                f"{sunlight_pressure(DIVE_SOLAR_RADII) * 1e3:.2f}",
                f"{share:.3f} (1/{1 / share:.0f})",
                f"{sunlight_accel(t.disk, DIVE_SOLAR_RADII):.2e}",
                f"{sunlight_dv(t.disk, t.periapsis_radius, t.periapsis_speed):.2f}",
                f"{sunlight_dv(t.disk, t.periapsis_radius, t.periapsis_speed, face_on=True):.1f}",
            ]
        )
    out.append(
        tabulate(
            rows,
            [
                "disk",
                "sunlight mPa",
                "edge-on share",
                "a m/s^2",
                "dv inbound m/s",
                "face-on m/s",
            ],
        )
    )

    out.append("\n2. The budget (one axis, cm): worst CME edge-on in the fix's latency")
    rows = []
    for t in trains:
        worst = cme_drift(
            t.disk, CME_DENSEST_N, 2500.0e3, CME_TILTS[-1], t.periapsis_speed
        )
        typical = cme_drift(
            t.disk, CME_MEDIAN_N, 1000.0e3, CME_TILTS[-1], t.periapsis_speed
        )
        for label, d in (("typical", typical), ("worst", worst)):
            b = dispersion_budget(t.disk, d.accel)
            rows.append(
                [
                    t.disk.name,
                    label,
                    _cm(b.required),
                    _cm(b.network),
                    _cm(b.drift),
                    _cm(b.ship) if b.closes else f"short {_cm(-b.ship)}",
                ]
            )
    out.append(
        tabulate(rows, ["disk", "CME", "required", "network", "drift", "left for ship"])
    )

    out.append("\n4. Thrusters (ship holds track; short loop in brackets)")
    rows = []
    for t in trains:
        req = dispersion_requirement(t.disk.charge)
        worst = cme_drift(
            t.disk, CME_DENSEST_N, 2500.0e3, CME_TILTS[-1], t.periapsis_speed
        )
        sun = sunlight_dv(t.disk, t.periapsis_radius, t.periapsis_speed)
        held = thruster(t.disk, worst, sun, 3.0 * NETWORK_SIDEWAYS, FINAL_MOVE_TIME)
        window = short_loop_window(t.disk, t.accel)
        side = t.disk.charge.side_slope * req
        wander = (
            side * math.sqrt(sum(k * k for k in t.kicks)) * math.sqrt(math.pi / 2.0)
        )
        loop = thruster(t.disk, worst, sun, 3.0 * req, window, follow=wander)
        rows.append(
            [
                t.disk.name,
                f"{1e3 * held.thrust:.2f} ({1e3 * loop.thrust:.0f})",
                held.sized_by,
                f"{held.dv:.2f} ({loop.dv:.2f})",
                f"{1e3 * held.gas:.1f} ({1e3 * loop.gas:.1f})",
            ]
        )
    out.append(tabulate(rows, ["disk", "thrust mN", "sized by", "gas dv m/s", "gas g"]))
    return "\n".join(out)


def price_report(priced: Sequence[Priced]) -> str:
    """The trains through the ledger."""
    rows = []
    for p in priced:
        dep, push = p.lines
        rows.append(
            [
                p.disk.name + ("" if p.hold else " (ADR 0085, no hold)"),
                f"{p.optimum.design.arc_hours:.3f}",
                f"{dep['pulse_disk']:.2f} / {push['pulse_disk']:.2f}",
                f"{(dep['sun_plate'] + dep['sun_bag']) / 1e3:.1f} / {(push['sun_plate'] + push['sun_bag']) / 1e3:.1f}",
                f"{dep['sun_stroke']:.2f} / {push['sun_stroke']:.2f}",
                f"{dep['sun_hold'] / 1e3:.1f} / {push['sun_hold'] / 1e3:.1f}",
                f"{p.optimum.valuation.growth:.3f}",
                f"{p.optimum.valuation.break_even:.1f} / {p.optimum.dear_break_even:.1f}",
            ]
        )
    label = _closure_label(priced[0].closure) if priced else ""
    return f"Priced ({label}, re-optimized for the dear seed)\n" + tabulate(
        rows,
        [
            "disk",
            "arc h",
            "kg/pulse dep/push",
            "Sun plate+bag t",
            "stroke m",
            "hold t",
            "growth",
            "BE cheap / dear",
        ],
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the report; ``--no-price`` skips the optimizer (~25 s of ~1 min)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-price", action="store_true")
    parser.add_argument(
        "--no-deep", action="store_true", help="skip the 4 R_sun restatement"
    )
    args = parser.parse_args(argv)
    runs: List[Tuple[Tuple[float, float], Sequence[Disk]]] = [(CLOSURE, DISKS)]
    if not args.no_deep:
        runs.append((DEEP_CLOSURE, DEEP_DISKS))
    for closure, disks in runs:
        departure = sdg.departure_route(*closure)
        push = sdg.push_route(*closure, sdg.PARKING_DAYS)
        assert departure is not None and push is not None
        trains = []
        for disk in disks:
            design = ledger_design(disk, closure=closure)
            design = replace(design, arc_hours=sdg.floor_arc(departure, push, design))
            trains.append(fly_train(disk, design, closure))
        print(aim_report(trains, closure))
        if not args.no_price:
            print()
            print(price_report(priced_trains(closure, disks)))
        print()


if __name__ == "__main__":
    main()
