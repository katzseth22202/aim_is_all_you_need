"""What state the plume is actually in, and what that does to the bag.

The paper's `sec:watering_it_down` works this out by hand and reverses one of
its own claims in the process: dividing the leftover 180 MJ/kg by water's
219 MJ/kg of first-ionisation cost suggests the plume ends up four fifths
ionised, and that is wrong, because the gas does not choose its ionisation
fraction independently of its temperature. Saha fixes the fraction once
temperature and density are given, and it is stingy at these densities -- 5.9%
at 15 000 K. The plume cannot pour 180 MJ/kg into a channel that narrow, so the
temperature climbs until the channel opens.

**This module does not re-solve that.** ``puffsat_impact_simulation``'s
``eos_water`` already does it properly -- dissociation by law of mass action,
the full ``O+ .. O8+`` Saha ladder, real degeneracies and potentials -- and its
answer agrees with the paper's hand solve to 1-3% in temperature. What stays
here is what this repository owns: the **burn envelope** (which closing speeds
the chain actually flies, and at what slug ratio), and the **bag consequence**
that follows from the solved state.

    burn envelope        eps = w^2 k / (2 (1+k)^2)      <- here
    equilibrium solve    (rho, eps) -> (T, f, P)        <- eos_water, vendored
    bag consequence      P/P0, B/B0, E_B                <- here

The solved table is `data/plume_state.csv`; see `data/README.md` for its
provenance and for the one trap in re-deriving it.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from astropy import units as u

from src.bag_state import (
    COLD_PULSE_TEMPERATURE,
    stored_field_energy,
)
from src.plume_thermal import specific_thermal_energy

#: Vendored solve from ``puffsat_impact_simulation``.
PLUME_STATE_CSV = Path(__file__).resolve().parent.parent / "data" / "plume_state.csv"

#: Bag density the paper's own rows are worked at, ``213 kg / 659.6 m^3``.
FLOWN_BAG_DENSITY = 0.323

#: The four closing speeds the paper quotes plume states for.  75 and 65 are
#: not grid points in the vendored table and are interpolated; 56.53 and 45.58
#: were inserted exactly, being the top and bottom of the slowest cycle's burn.
QUOTED_SPEEDS = (75.0, 65.0, 56.53, 45.58)


@lru_cache(maxsize=1)
def _grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the vendored solve as ``(speeds, densities, temperature, ionised)``.

    Returns:
        Sorted axes and two ``(n_speed, n_density)`` value arrays.
    """
    rows: List[Dict[str, str]] = list(csv.DictReader(PLUME_STATE_CSV.open()))
    speeds = np.array(sorted({float(r["closing_speed_km_s"]) for r in rows}))
    densities = np.array(sorted({float(r["rho_kg_m3"]) for r in rows}))
    temperature = np.zeros((len(speeds), len(densities)))
    ionised = np.zeros_like(temperature)
    for row in rows:
        i = int(np.searchsorted(speeds, float(row["closing_speed_km_s"])))
        j = int(np.searchsorted(densities, float(row["rho_kg_m3"])))
        temperature[i, j] = float(row["temp_K"])
        ionised[i, j] = float(row["ionised_fraction"])
    return speeds, densities, temperature, ionised


def plume_state(
    closing_speed: float, density: float = FLOWN_BAG_DENSITY
) -> Tuple[u.Quantity, float]:
    """Solved plume temperature and ionisation fraction.

    Bilinear on the vendored grid, linear in closing speed and in ``log(rho)``
    -- the second because the table's density axis is log-spaced and the state
    varies far more smoothly in the log.

    Args:
        closing_speed: Impactor speed relative to the vehicle (km/s).
        density: Slug density of the bag (kg/m^3).

    Returns:
        ``(temperature, ionisation_fraction)``.

    Raises:
        ValueError: If either coordinate is outside the solved grid, which is
            44-76 km/s and 0.05-2.0 kg/m^3.  Extrapolating a Saha solve is not
            something this module will do silently.
    """
    speeds, densities, temperature, ionised = _grid()
    if not speeds[0] <= closing_speed <= speeds[-1]:
        raise ValueError(f"{closing_speed} km/s is outside the solved grid")
    if not densities[0] <= density <= densities[-1]:
        raise ValueError(f"rho = {density} is outside the solved grid")

    def interpolate(values: np.ndarray) -> float:
        per_speed = [
            float(np.interp(np.log(density), np.log(densities), values[i]))
            for i in range(len(speeds))
        ]
        return float(np.interp(closing_speed, speeds, per_speed))

    return interpolate(temperature) * u.K, interpolate(ionised)


def pressure_ratio(
    closing_speed: float,
    density: float = FLOWN_BAG_DENSITY,
    baseline: u.Quantity = COLD_PULSE_TEMPERATURE,
) -> float:
    """Plume pressure against the dissociated-neutral 15 000 K assumption.

    **Ionisation changes the particle count, and pressure counts particles
    rather than mass.** Water goes from three particles per molecule to
    ``3(1+f)``, so the ratio is ``(1+f) T / T0`` -- the temperature rise and the
    extra particles multiply.

    Args:
        closing_speed: Impactor speed relative to the vehicle (km/s).
        density: Slug density of the bag (kg/m^3).
        baseline: Temperature the original assumption held.

    Returns:
        ``P / P0``.
    """
    temperature, ionised = plume_state(closing_speed, density)
    return float((1.0 + ionised) * temperature / baseline)


def field_ratio(closing_speed: float, density: float = FLOWN_BAG_DENSITY) -> float:
    """Field against the baseline column, ``B/B0 = sqrt(P/P0)``.

    Args:
        closing_speed: Impactor speed relative to the vehicle (km/s).
        density: Slug density of the bag (kg/m^3).

    Returns:
        ``B / B0``.
    """
    return float(np.sqrt(pressure_ratio(closing_speed, density)))


def burn_stored_energy(
    closing_speed: float, density: float = FLOWN_BAG_DENSITY
) -> u.Quantity:
    """Field energy the bag confines at this point in the burn.

    ``E_B`` scales with the pressure, since the volume is fixed by the bag, so
    this is the 4.43 GJ baseline times :func:`pressure_ratio`.

    Args:
        closing_speed: Impactor speed relative to the vehicle (km/s).
        density: Slug density of the bag (kg/m^3).

    Returns:
        Stored field energy (astropy Quantity, J).
    """
    return stored_field_energy() * pressure_ratio(closing_speed, density)


def burn_envelope(
    slug_ratio: float = 8.5, density: float = FLOWN_BAG_DENSITY
) -> List[Tuple[float, u.Quantity, u.Quantity, float, float, float, u.Quantity]]:
    """The quoted burn, from dissipated energy through to stored field energy.

    Args:
        slug_ratio: ``k``.
        density: Slug density of the bag (kg/m^3).

    Returns:
        One tuple per quoted speed:
        ``(w, dissipated, T, f, P/P0, B/B0, E_B)``.
    """
    rows = []
    for speed in QUOTED_SPEEDS:
        dissipated = specific_thermal_energy(speed * u.km / u.s, slug_ratio)
        temperature, ionised = plume_state(speed, density)
        ratio = pressure_ratio(speed, density)
        rows.append(
            (
                speed,
                dissipated.to(u.MJ / u.kg),
                temperature,
                ionised,
                ratio,
                float(np.sqrt(ratio)),
                burn_stored_energy(speed, density),
            )
        )
    return rows


# --- Item 3: tab:seed_window ----------------------------------------------
#
# The seed's job is conductivity, not thrust.  Potassium ionises at 4.34 eV
# against water's 13.6, so it keeps supplying electrons at 3000-6000 K where
# the water has already re-formed -- and conductivity is what decides whether
# the field stays put.  The table's `Rm` column is therefore a **leak schedule
# written backwards**: the field diffuses over `mu0 sigma L^2` against an
# expansion clock of `L/v`, and the ratio is `mu0 sigma v L`, which is `Rm`.

#: Electrical conductivity of the seeded plume [S/m] against temperature [K],
#: at the flown bag density and a 1% potassium seed.
#:
#: **Provenance.**  ``puffsat_impact_simulation`` @ ``0216a09``,
#: ``puffsat.conductivity.sigma(T, rho=0.323, x_k=0.01)``, regenerated there
#: with ``make analysis-conductivity``.  A Drude/Lorentz form with Coulomb and
#: electron-neutral scattering added as rates, whose electron density comes
#: from ``eos_water`` for the water plus an exact trace-layer closure for the
#: seed.  It reproduces Spitzer's conductivity to 2% with electron-neutral
#: scattering switched off.
_SEED_WINDOW_SIGMA: Dict[int, float] = {
    2000: 1.05,
    3000: 63.28,
    4000: 286.30,
    5000: 492.17,
    6000: 599.59,
    15000: 6987.10,
}

#: Potassium ionised fraction at the same points, same source.
_SEED_WINDOW_IONISED: Dict[int, float] = {
    2000: 0.0002,
    3000: 0.0200,
    4000: 0.1846,
    5000: 0.5627,
    6000: 0.8555,
    15000: 0.9878,
}

#: ``Rm`` as ``tab:seed_window`` prints it, for comparison.
_PAPER_RM: Dict[int, float] = {
    2000: 0.1,
    3000: 9.2,
    4000: 76.5,
    5000: 238.0,
    6000: 400.0,
    15000: 361.0,
}

#: Flow speed times flux-tube radius [m^2/s].  **The paper states neither `v`
#: nor `L`**, and they enter only as this product.  The value is the middle of
#: the 5.5e4-9.7e4 that ``puffsat_impact_simulation``'s solved expansion
#: reports (Q-L), which is the first time either quantity has been an output
#: rather than a guess.
SOLVED_V_L = 7.4e4

_MU_0 = 4.0e-7 * np.pi


def magnetic_reynolds(temperature: float, v_l: float = SOLVED_V_L) -> float:
    """``Rm = mu0 sigma v L`` at a tabulated temperature.

    Args:
        temperature: Plume temperature (K); must be a tabulated point.
        v_l: Flow speed times flux-tube radius (m^2/s).

    Returns:
        The magnetic Reynolds number.

    Raises:
        KeyError: If the temperature is not one of the tabulated points.
    """
    return _MU_0 * _SEED_WINDOW_SIGMA[int(temperature)] * v_l


def leak_fraction(temperature: float, v_l: float = SOLVED_V_L) -> float:
    """Share of the stored field energy that diffuses out, ``~1/Rm``.

    Capped at 1.0: below the conductivity cliff the field is simply gone, and a
    "leak" above 100% is a statement that confinement has ended rather than a
    larger number.

    Args:
        temperature: Plume temperature (K); must be a tabulated point.
        v_l: Flow speed times flux-tube radius (m^2/s).

    Returns:
        Leak fraction in [0, 1].
    """
    return min(1.0, 1.0 / magnetic_reynolds(temperature, v_l))


def implied_v_l(temperature: float) -> float:
    """The ``v L`` the paper's own ``Rm`` implies at this temperature.

    **This is the diagnostic that matters for item 3.**  If ``tab:seed_window``
    were a single expansion sampled at several temperatures, every row would
    imply the same ``v L``.  They do not: the spread is a factor of three, so
    the column cannot be reproduced at any one value and is not an isotherm
    sweep of one flow.

    Args:
        temperature: Plume temperature (K); must be a tabulated point.

    Returns:
        The implied ``v L`` (m^2/s).
    """
    return _PAPER_RM[int(temperature)] / (_MU_0 * _SEED_WINDOW_SIGMA[int(temperature)])


def seed_window() -> List[Tuple[int, float, float, float, float, float]]:
    """Regenerate ``tab:seed_window``.

    Returns:
        One tuple per row: ``(T, seed_ionised, sigma, Rm, leak, implied_v_L)``.
    """
    return [
        (
            temperature,
            _SEED_WINDOW_IONISED[temperature],
            _SEED_WINDOW_SIGMA[temperature],
            magnetic_reynolds(temperature),
            leak_fraction(temperature),
            implied_v_l(temperature),
        )
        for temperature in sorted(_SEED_WINDOW_SIGMA)
    ]


def main() -> None:
    """Print the burn envelope and the regenerated seed window."""
    print("=== Item 1: the burn envelope and what it does to the bag ===")
    print(
        f"{'w [km/s]':>9}{'dissipated':>13}{'T [K]':>9}{'f':>9}"
        f"{'P/P0':>8}{'B/B0':>8}{'E_B [GJ]':>11}"
    )
    for (
        speed,
        dissipated,
        temperature,
        ionised,
        ratio,
        field,
        energy,
    ) in burn_envelope():
        print(
            f"{speed:>9.2f}{dissipated.to_value(u.MJ / u.kg):>10.1f} MJ/kg"
            f"{temperature.to_value(u.K):>9.0f}{ionised:>9.4f}"
            f"{ratio:>8.2f}{field:>8.2f}{energy.to_value(u.GJ):>11.2f}"
        )
    print(
        "\nSolved on eos_water, so these run 1-3% warmer than the paper's hand\n"
        "figures: the hand solve charged 54 MJ/kg for vaporisation plus\n"
        "dissociation where the real bond energy is 50.9, and the difference\n"
        "stays in the thermal pool."
    )

    print("\n=== Item 3: tab:seed_window, regenerated ===")
    print(
        f"{'T [K]':>7}{'K ionised':>11}{'sigma [S/m]':>13}{'Rm':>9}"
        f"{'leak':>9}{'paper Rm':>10}{'implied vL':>13}"
    )
    for temperature, ionised, conductivity, reynolds, leak, v_l in seed_window():
        print(
            f"{temperature:>7d}{100 * ionised:>10.2f}%{conductivity:>13.2f}"
            f"{reynolds:>9.1f}{100 * leak:>8.1f}%{_PAPER_RM[temperature]:>10.1f}"
            f"{v_l:>13.2e}"
        )
    spread = max(implied_v_l(t) for t in _SEED_WINDOW_SIGMA) / min(
        implied_v_l(t) for t in _SEED_WINDOW_SIGMA
    )
    print(
        f"\nThe paper's Rm column implies a v L that varies by {spread:.1f}x across\n"
        "its own rows, so it cannot be one expansion sampled at several\n"
        "temperatures. The solved expansion gives v L = 5.5e4-9.7e4 (Q-L)."
    )


if __name__ == "__main__":
    main()
