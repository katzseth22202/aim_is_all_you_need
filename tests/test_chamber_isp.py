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
    HYDROGEN_5500K,
    METHANE_7000K,
    PLUG_RATIO,
    ROD_MASS,
    ChamberPairing,
    chamber_departure_burn,
    effective_isp,
    momentum_debit_share,
    slug_ratio_at,
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
