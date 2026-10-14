"""Pinned tests for src/chamber_isp.py: the parent paper's ``eq:eta_isp``.

Every figure here is closed-form algebra, and each one is pinned to a table of
``Balloon-Pulse-Propulsion``'s ``templateArxiv.tex``, printed rounded to the
second.  The paper computed them from unrounded inputs and prints ``eta`` to three
decimals and ``k`` to one, so a cell is checked against the band its own printed
inputs span (``_printed_band``), widened by the half second of output rounding.
"""

import numpy as np
import pytest
from astropy import units as u

from src.chamber_isp import (
    GATE_THRUST_COST,
    HYDROGEN_3000K,
    HYDROGEN_5500K,
    HYDROGEN_BULGED,
    HYDROGEN_BULGED_IMPULSE,
    HYDROGEN_SPHERE_GAS,
    LIQUID_HYDROGEN,
    METHANE_7000K,
    METHANE_SURVIVABLE_GAS,
    PLUG_RATIO,
    ROD_MASS,
    SLUSH_HYDROGEN,
    SPHERE_THROAT_AREA,
    SPHERE_VOLUME,
    SURVIVABLE_IMPULSE,
    SURVIVABLE_IMPULSE_CARBON_FROZEN,
    SURVIVABLE_IMPULSE_S17,
    SURVIVABLE_IMPULSE_SUDDEN_FREEZE,
    SURVIVABLE_THROAT_AREA,
    SURVIVABLE_THROAT_AREA_S17,
    SURVIVABLE_VOLUME,
    ChamberPairing,
    absolute_efficiency,
    blowdown_time,
    chamber_departure_burn,
    effective_isp,
    efficiency_for_impulse,
    momentum_debit_share,
    slug_ratio_at,
    stored_as,
    survivable_cycle,
    survivable_methane,
)

W_REF = 75.0 * u.km / u.s


def _printed_band(eta: float, k: float, **kwargs: float) -> tuple[float, float]:
    """Isp range over the rounding of printed ``eta`` (+-0.0005) and ``k`` (+-0.05), +-0.5 s."""
    cells = [
        float(effective_isp(W_REF, eta + de, k + dk, **kwargs).to_value(u.s))
        for de in (-0.0005, 0.0005)
        for dk in (-0.05, 0.05)
    ]
    return min(cells) - 0.5, max(cells) + 0.5


# (A/A*, eta, printed I_eff): every cell of ``tab:nozzle_area_ratio``.  Methane
# carries two edges per row, carbon frozen and carbon in equilibrium.
HYDROGEN_ROWS = [
    (14, 0.642, 934),
    (30, 0.715, 1002),
    (100, 0.797, 1075),
    (300, 0.858, 1126),
    (1000, 0.901, 1162),
]
METHANE_ROWS = [
    (14, 0.360, 602), (14, 0.389, 635),
    (30, 0.406, 655), (30, 0.441, 693),
    (100, 0.466, 720), (100, 0.511, 766),
    (300, 0.485, 740), (300, 0.538, 793),
    (1000, 0.499, 754), (1000, 0.562, 817),
]  # fmt: skip


@pytest.mark.parametrize(
    "pairing, eta, printed",
    [(HYDROGEN_5500K, eta, isp) for _, eta, isp in HYDROGEN_ROWS]
    + [(METHANE_7000K, eta, isp) for _, eta, isp in METHANE_ROWS],
)
def test_every_raw_area_ratio_cell_reproduces(
    pairing: ChamberPairing, eta: float, printed: float
) -> None:
    """The raw table leaves out the gate and the pitch, so ``eq:eta_isp`` alone must match it."""
    low, high = _printed_band(eta, pairing.reference_slug_ratio)
    assert low <= printed <= high


@pytest.mark.parametrize("eta, printed", [(0.858, 1112), (0.642, 922)])
def test_the_gate_takes_one_percent_of_exhaust_momentum(
    eta: float, printed: float
) -> None:
    """``tab:wall_pairing_doubling``, hydrogen with the gate: the 1% comes off
    ``(k+1+P) v_e`` before the head-on debit, not off the finished Isp."""
    low, high = _printed_band(
        eta, HYDROGEN_5500K.reference_slug_ratio, gate_thrust_cost=GATE_THRUST_COST
    )
    assert low <= printed <= high


@pytest.mark.parametrize(
    "eta, most_pitch_isp, least_pitch_isp",
    [(0.538, 762, 777), (0.485, 711, 725), (0.389, 612, 623)],
)
def test_methane_pitch_rides_as_carried_propellant(
    eta: float, most_pitch_isp: float, least_pitch_isp: float
) -> None:
    """``tab:wall_pairing_doubling``, methane with the gate: the 1.4-5.6 kg of pitch
    lost per pulse leaves with the exhaust at the same ``eta`` and is charged as
    carried mass, spanning each printed range from its high-pitch to its low-pitch end.
    """
    k = METHANE_7000K.reference_slug_ratio
    per_rod = [float((pitch * u.kg / ROD_MASS).to_value(u.one)) for pitch in (5.6, 1.4)]
    for pitch_ratio, printed in zip(per_rod, (most_pitch_isp, least_pitch_isp)):
        low, high = _printed_band(
            eta, k, gate_thrust_cost=GATE_THRUST_COST, pitch_ratio=pitch_ratio
        )
        assert low <= printed <= high


@pytest.mark.parametrize("pairing", [HYDROGEN_5500K, METHANE_7000K])
def test_slug_ratio_at_the_reference_speed_is_the_tabulated_one(
    pairing: ChamberPairing,
) -> None:
    assert slug_ratio_at(W_REF, pairing) == pytest.approx(
        pairing.reference_slug_ratio, rel=1e-12
    )


@pytest.mark.parametrize("pairing", [HYDROGEN_5500K, METHANE_7000K])
@pytest.mark.parametrize("speed_km_s", [60.0, 68.0, 90.0])
def test_each_pulse_resizes_its_charge_to_hold_the_chamber_temperature(
    pairing: ChamberPairing, speed_km_s: float
) -> None:
    """The closing speed changes pulse to pulse, and the gas load follows it so that
    ``(k + 1 + P) u(T) = w^2 / 2`` keeps the energy per kilogram, hence ``T``, fixed."""

    def energy_per_kg(w_km_s: float, k: float) -> float:
        return (w_km_s * 1e3) ** 2 / 2.0 / (k + 1.0 + PLUG_RATIO)

    reference = energy_per_kg(75.0, pairing.reference_slug_ratio)
    k = slug_ratio_at(speed_km_s * u.km / u.s, pairing)
    assert energy_per_kg(speed_km_s, k) == pytest.approx(reference, rel=1e-12)
    assert (k > pairing.reference_slug_ratio) == (speed_km_s > 75.0)


def test_a_pulse_too_slow_to_reach_the_chamber_temperature_is_refused() -> None:
    """Hydrogen needs about 24 km/s just to heat the rod and plug to 5500 K with no charge."""
    assert slug_ratio_at(25.0 * u.km / u.s, HYDROGEN_5500K) >= 0.0
    with pytest.raises(ValueError):
        slug_ratio_at(23.0 * u.km / u.s, HYDROGEN_5500K)


KM_S = u.km / u.s
G0 = 9.80665e-3  # km/s^2


def test_a_short_burn_is_the_rocket_equation_at_the_ignition_isp() -> None:
    """Over 1 m/s the closing speed barely moves, so the burn is Tsiolkovsky at ``I_eff(w0)``."""
    burn = chamber_departure_burn(
        10.9 * KM_S, 10.901 * KM_S, 64.1 * KM_S, HYDROGEN_5500K, 0.858
    )
    isp = effective_isp(
        75.0 * KM_S, 0.858, slug_ratio_at(75.0 * KM_S, HYDROGEN_5500K)
    ).to_value(u.s)
    assert burn.delivered_fraction == pytest.approx(
        np.exp(-0.001 / (isp * G0)), rel=1e-9
    )


def test_the_rods_spent_are_the_propellant_over_each_pulses_carried_ratio() -> None:
    """Propellant per rod is ``k + P``, and ``k`` grows through the burn, so the rod
    count falls between the propellant priced at the ignition ``k`` and at cutoff's."""
    burn = chamber_departure_burn(
        10.9 * KM_S, 16.3 * KM_S, 64.1 * KM_S, HYDROGEN_5500K, 0.858
    )
    spent = 1.0 - burn.delivered_fraction
    k_start = slug_ratio_at(75.0 * KM_S, HYDROGEN_5500K)
    k_end = slug_ratio_at(80.4 * KM_S, HYDROGEN_5500K)
    assert (
        spent / (k_end + PLUG_RATIO)
        < burn.rod_mass_fraction
        < spent / (k_start + PLUG_RATIO)
    )


@pytest.mark.parametrize("pairing", [HYDROGEN_5500K, METHANE_7000K])
def test_lost_efficiency_bites_harder_head_on_because_the_debit_does_not_shrink(
    pairing: ChamberPairing,
) -> None:
    """The rod's own momentum ``w`` is debited in full whatever ``eta`` is, while the
    exhaust's momentum falls as ``sqrt(eta)``.  So the debit's share of the gross
    momentum grows as ``eta`` falls, and the Isp falls faster than ``sqrt(eta)``."""
    k = pairing.reference_slug_ratio
    shares = [momentum_debit_share(eta, k) for eta in (1.0, 0.7, 0.5)]
    assert shares == sorted(shares)
    assert shares[-1] == pytest.approx(
        1.0 / np.sqrt(0.5 * (k + 1.0 + PLUG_RATIO)), rel=1e-12
    )
    full, half = (effective_isp(W_REF, eta, k).to_value(u.s) for eta in (1.0, 0.5))
    assert half / full < np.sqrt(0.5)


def test_a_burn_that_loses_speed_is_refused() -> None:
    with pytest.raises(ValueError):
        chamber_departure_burn(
            16.0 * KM_S, 10.9 * KM_S, 64.1 * KM_S, HYDROGEN_5500K, 0.858
        )


@pytest.mark.parametrize(
    "pairing, bond_share, recovered",
    [(HYDROGEN_5500K, 0.36, 0.94), (METHANE_7000K, 0.69, 0.52)],
)
def test_the_chemistry_ceiling_is_the_pulse_net_of_bonds_that_never_return(
    pairing: ChamberPairing, bond_share: float, recovered: float
) -> None:
    """``tab:wall_pairings`` and ``tab:nozzle_area_ratio`` at the A/A* = 300 design:
    the share of the pulse held in bonds at peak, and the share of that the nozzle
    returns at equilibrium (methane's carbon never re-bonds).  The rest is out of
    reach of any chamber."""
    assert pairing.chemistry_ceiling == pytest.approx(
        1.0 - bond_share * (1.0 - recovered)
    )


@pytest.mark.parametrize(
    "pairing, solved, share",
    [(HYDROGEN_5500K, 0.858, 0.877), (METHANE_7000K, 0.538, 0.804)],
)
def test_efficiency_is_quoted_as_a_share_of_that_ceiling(
    pairing: ChamberPairing, solved: float, share: float
) -> None:
    """The simulation's solved chambers reach about 88% (H2) and 80% (CH4) of their
    ceilings, and a sweep's 50% means half the ceiling, not half the pulse."""
    assert solved / pairing.chemistry_ceiling == pytest.approx(share, abs=1e-3)
    assert absolute_efficiency(pairing, 0.5) == pytest.approx(
        0.5 * pairing.chemistry_ceiling
    )


@pytest.mark.parametrize(
    "area_ratio, impulse_kn_s, isp_s",
    [(100, 885.0, 771.0), (150, 908.0, 791.0), (300, 941.0, 820.0)],
)
def test_the_survivable_chambers_eta_gives_the_impact_sims_pulse(
    area_ratio: int, impulse_kn_s: float, isp_s: float
) -> None:
    """ADR 0056: 885 / 908 / 941 kN s per 5 kg pulse (gate and finite-rate
    recombination charged) is 771 / 791 / 820 s on the 117 kg of charge and
    plug."""
    pairing = survivable_methane(area_ratio)
    eta = efficiency_for_impulse(SURVIVABLE_IMPULSE[area_ratio], pairing)
    isp = effective_isp(
        W_REF,
        eta,
        pairing.reference_slug_ratio,
        plug_ratio=pairing.plug_ratio,
        gate_thrust_cost=GATE_THRUST_COST,
    )
    carried = (pairing.reference_slug_ratio + pairing.plug_ratio) * pairing.rod_mass
    assert carried.to_value(u.kg) == pytest.approx(117.0)
    assert (isp * carried * 9.80665 * u.m / u.s**2).to_value(
        u.kN * u.s
    ) == pytest.approx(impulse_kn_s, rel=1e-9)
    assert isp.to_value(u.s) == pytest.approx(isp_s, abs=2.0)


def test_finite_rate_sits_between_the_carbon_floor_and_the_sudden_freeze() -> None:
    """ADR 0056: the sudden freeze tripled hydrogen's loss, so the finite-rate
    impulse beats ADR 0055's; carbon frozen is the floor under both."""
    for ar in (100, 300):
        assert (
            SURVIVABLE_IMPULSE_CARBON_FROZEN[ar]
            < SURVIVABLE_IMPULSE_SUDDEN_FREEZE[ar]
            < SURVIVABLE_IMPULSE[ar]
        )
    assert SURVIVABLE_IMPULSE_CARBON_FROZEN[300] < SURVIVABLE_IMPULSE[150]


def test_s17s_impulse_charged_neither_gate_nor_freeze() -> None:
    """ADR 0044's 906 / 971 kN s gave A/A* = 300 the sphere's solved 0.538;
    charging the finite-rate freeze costs the chamber about 5% of its eta."""
    pairing = survivable_methane(300)
    old = efficiency_for_impulse(SURVIVABLE_IMPULSE_S17[300], pairing)
    new = efficiency_for_impulse(SURVIVABLE_IMPULSE[300], pairing)
    assert old == pytest.approx(0.538, abs=0.001)
    assert 0.92 < new / old < 0.97


def test_the_bulged_hydrogen_chamber_is_the_spheres_charge_in_the_bulge() -> None:
    """ADR 0055: the 60 kg charge on the 2.5 kg rod, 37 t wrap, 2 Hz, and an
    extension scaled to its 0.84 m^2 throat; its finite-rate eta (ADR 0064) sits below the sphere's.
    """
    pairing = HYDROGEN_BULGED
    assert pairing.hydrogen
    charge = pairing.reference_slug_ratio * pairing.rod_mass
    assert charge.to_value(u.kg) == pytest.approx(60.0)
    assert pairing.pulse_rate == 2.0 / u.s
    assert pairing.extension_mass.to_value(u.t) == pytest.approx(9.3, abs=0.05)
    bulged = efficiency_for_impulse(HYDROGEN_BULGED_IMPULSE, pairing)
    sphere = efficiency_for_impulse(697.0 * u.kN * u.s, HYDROGEN_5500K)
    assert bulged < sphere
    assert bulged == pytest.approx(0.822, abs=0.002)


def test_a_split_survivable_chamber_shares_the_rod_wall_and_extension() -> None:
    whole = survivable_methane(100)
    half = survivable_methane(100, 49.8 * u.t, rods_split=2)
    assert half.rod_mass == whole.rod_mass / 2
    assert half.wall_mass.to_value(u.t) == pytest.approx(24.9)
    assert half.extension_mass == whole.extension_mass / 2
    assert half.pulse_rate == whole.pulse_rate
    assert (half.plug_ratio, half.reference_slug_ratio) == (
        whole.plug_ratio,
        whole.reference_slug_ratio,
    )


def test_the_slug_ratio_holds_each_pairings_own_plug() -> None:
    """k(w) keeps k + 1 + P proportional to w^2 with the pairing's own P."""
    pairing = survivable_methane(300)
    k = slug_ratio_at(2.0 * W_REF, pairing)
    assert k + 1.0 + pairing.plug_ratio == pytest.approx(
        4.0 * (pairing.reference_slug_ratio + 1.0 + pairing.plug_ratio)
    )


def test_slush_shrinks_the_tank_by_its_density_and_keeps_the_chamber() -> None:
    slush = stored_as(HYDROGEN_5500K, SLUSH_HYDROGEN)
    assert slush.tank_fraction == pytest.approx(0.205 / 1.15)
    assert slush.wall_mass == HYDROGEN_5500K.wall_mass
    assert slush.chemistry_ceiling == HYDROGEN_5500K.chemistry_ceiling
    assert slush.hydrogen and slush.storage is SLUSH_HYDROGEN
    assert SLUSH_HYDROGEN.boil_off_per_day == 0.0
    assert HYDROGEN_5500K.storage is LIQUID_HYDROGEN


def test_only_a_hydrogen_pairing_has_a_storage_to_change() -> None:
    assert not METHANE_7000K.hydrogen
    with pytest.raises(ValueError, match="hydrogen"):
        stored_as(METHANE_7000K, SLUSH_HYDROGEN)


def test_the_slush_heat_sink_outlasts_the_hold() -> None:
    """Melting half the solid and warming to 20.3 K absorbs ~18% of boiling.

    Fusion 58.2 kJ/kg on half the mass, plus about 52 kJ/kg warming the
    liquid from 13.8 K, against 446 kJ/kg to boil it: at the liquid's
    0.3%/day that is about 60 days with nothing boiled.
    """
    sink = 0.5 * 58.2 + 52.0
    days = sink / 446.0 / LIQUID_HYDROGEN.boil_off_per_day
    assert sink / 446.0 == pytest.approx(0.18, abs=0.01)
    assert days > 20.0 * 2.5


def test_the_3000_k_charge_is_sized_by_the_equilibrium_energy() -> None:
    """``(k + 1 + P) u = w^2 / 2`` at 75 km/s, with ``u`` solved at the
    charge's own density (ADR 0052)."""
    from src.orbit_turn_analysis import hydrogen_hot_energy

    density = HYDROGEN_3000K.reference_slug_ratio * ROD_MASS / (20.0 * u.m**3)
    energy = hydrogen_hot_energy(3000.0 * u.K, density)
    sized = (75.0 * u.km / u.s) ** 2 / (2.0 * energy) - 1.0 - PLUG_RATIO
    assert HYDROGEN_3000K.reference_slug_ratio == pytest.approx(
        float(sized.to_value(u.one)), abs=0.1
    )
    assert HYDROGEN_3000K.hydrogen


def test_the_hydrogen_sphere_empties_on_the_papers_31_ms() -> None:
    """The parent's e-fold time for hydrogen at V/A* ~ 100 m is ~31 ms."""
    e = blowdown_time(
        SPHERE_VOLUME, SPHERE_THROAT_AREA, HYDROGEN_SPHERE_GAS, np.exp(-1)
    )
    assert e.to_value(u.ms) == pytest.approx(31.0, abs=2.0)


def test_two_hertz_leaves_the_methane_chamber_time_to_refill_and_four_does_not() -> (
    None
):
    """Emptying the 212 m^3 chamber to its ~1 bar warm fill takes 278 ms at
    this solve's sound speed and 336 ms at the parent's 3.7 km/s through
    ADR 0053's 1.33 m^2 throat.  The impact sim's real-gas solve gives 281 ms."""
    gas = METHANE_SURVIVABLE_GAS
    to_fill = (1.0 / 94.0) ** (1.0 / gas.gamma)
    throat = SURVIVABLE_THROAT_AREA_S17
    fast = blowdown_time(SURVIVABLE_VOLUME, throat, gas, to_fill)
    slow = blowdown_time(SURVIVABLE_VOLUME, throat, gas, to_fill, 3.7 * u.km / u.s)
    assert fast.to_value(u.ms) == pytest.approx(278.0, abs=3.0)
    assert slow.to_value(u.ms) == pytest.approx(336.0, abs=3.0)
    assert 250.0 < fast.to_value(u.ms) < slow.to_value(u.ms) < 500.0 - 150.0


def test_the_doubled_throat_halves_the_blowdown() -> None:
    """ADR 0055: time goes as V/A*, so 2.66 m^2 empties in half the time
    (139 ms to fill pressure in the impact sim's real-gas solve)."""
    gas = METHANE_SURVIVABLE_GAS
    to_fill = (1.0 / 94.0) ** (1.0 / gas.gamma)
    old = blowdown_time(SURVIVABLE_VOLUME, SURVIVABLE_THROAT_AREA_S17, gas, to_fill)
    new = blowdown_time(SURVIVABLE_VOLUME, SURVIVABLE_THROAT_AREA, gas, to_fill)
    assert (new / old).to_value(u.one) == pytest.approx(0.5)
    assert new.to_value(u.ms) == pytest.approx(139.0, abs=3.0)


@pytest.mark.parametrize(
    "throat, seal, period_ms",
    [(1.33, 1.11, 439.0), (1.33, 0.3, 548.0), (2.66, 1.11, 297.0), (2.66, 0.3, 351.0)],
)
def test_the_cycle_reproduces_the_impact_sims_periods(
    throat: float, seal: float, period_ms: float
) -> None:
    """ADR 0055/0056: the impact sim's 439 / 548 ms and 297 / 351 ms."""
    period = survivable_cycle(throat * u.m**2, seal)
    assert period.to_value(u.ms) == pytest.approx(period_ms, abs=2.0)


def test_the_doubled_throat_fits_three_hertz_only_with_a_faster_fill() -> None:
    """ADR 0056: a polymer film at 0.3 bar and the 1 t/s fill run 352 ms,
    past 3 Hz's 333; a 2 t/s fill brings it to 298 ms.  4 Hz never fits."""
    throat = SURVIVABLE_THROAT_AREA
    assert survivable_cycle(throat, 0.3) > (1.0 / 3.0) * u.s
    assert survivable_cycle(throat, 0.3, 2.0 * u.t / u.s) < (1.0 / 3.0) * u.s
    assert survivable_cycle(throat, 0.3, 2.0 * u.t / u.s) > 0.25 * u.s


def test_blowdown_rejects_an_impossible_target() -> None:
    with pytest.raises(ValueError, match="mass_left"):
        blowdown_time(SPHERE_VOLUME, SPHERE_THROAT_AREA, HYDROGEN_SPHERE_GAS, 1.0)
