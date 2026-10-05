"""How much carbon hot hydrogen carries off a graphite wall at equilibrium.

The ablation literature calls it B': kilograms of carbon held in the gas per
kilogram of hydrogen, when the gas next to the wall is in equilibrium with
solid graphite.  It sets the chemical loss of a pitch coat in the hydrogen
chamber, ``m = g ln(1 + B') t``, where ``g`` is the gas the boundary layer
brings to the wall (parent ``sec:methane_7000_near_term``, companion reply of
2026-10-04).

Each gas species C_a H_b holds a partial pressure set by unit-activity
graphite and the free hydrogen::

    p_i = K_i p_H2^(b/2),   ln K_i = -(g_i - a g_C(gr) - (b/2) g_H2) / RT

and ``p_H2`` is root-found so that the partial pressures sum to the total.
Thermochemistry is Cantera's bundled NASA Glenn data: every C/H species in
``nasa_gas.yaml`` and ``C(gr)`` from ``nasa_condensed.yaml``.  The gas is ideal.
For H2 at 818 bar and 3900 K the compressibility is about 1.07.

Two regimes matter for the chamber.  At 2500 K and 818 bar the carbon leaves
mostly as **methane**, which pressure favours, so B' falls as the blowdown
empties.  At 3900 K it leaves mostly as acetylene, which conserves gas moles,
so B' holds near 5.3 from 818 bar down to about 30 bar.  Above graphite's
sublimation line no equilibrium with a solid wall exists, and
``graphite_equilibrium`` raises.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import astropy.units as u
import cantera as ct
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq

_CARBON_MOLAR_MASS = 12.011  # g/mol
_HYDROGEN_MOLAR_MASS = 1.008  # g/mol
#: Bracket on ln(p_H2 / 1 atm) for the root-find.
_LN_P_FLOOR = -60.0

#: Wall temperatures and pressures the companion reply tabulates.
TABLE_TEMPERATURES = (2500, 2800, 3100, 3400, 3700, 3900, 4100) * u.K
TABLE_PRESSURES = (10, 30, 62, 100, 300, 818) * u.bar


@dataclass(frozen=True)
class GraphiteEquilibrium:
    """Gas in equilibrium with graphite at one temperature and pressure.

    Attributes:
        temperature: Wall temperature.
        pressure: Total gas pressure.
        b_prime: Kilograms of carbon in the gas per kilogram of hydrogen.
        mole_fractions: Every gas species' mole fraction, keyed by Cantera name.
    """

    temperature: u.Quantity
    pressure: u.Quantity
    b_prime: float
    mole_fractions: dict[str, float]

    def leading_species(self, count: int = 4) -> list[tuple[str, float]]:
        """The ``count`` largest mole fractions, largest first.

        Args:
            count: How many species to return.

        Returns:
            ``(name, mole fraction)`` pairs.
        """
        ranked = sorted(self.mole_fractions.items(), key=lambda kv: -kv[1])
        return ranked[:count]


@lru_cache(maxsize=1)
def _phases() -> tuple[Any, Any]:
    """Cantera's C/H gas phase and solid graphite, built once."""
    species = [
        s
        for s in ct.Species.list_from_file("nasa_gas.yaml")
        if set(s.composition) <= {"C", "H"}
    ]
    gas = ct.Solution(thermo="ideal-gas", species=species)
    graphite = ct.Solution(
        thermo="fixed-stoichiometry",
        species=[
            s
            for s in ct.Species.list_from_file("nasa_condensed.yaml")
            if s.name == "C(gr)"
        ],
    )
    return gas, graphite


def _atom_counts(gas: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Carbon and hydrogen atoms per molecule of each gas species."""
    carbon = np.array([s.composition.get("C", 0.0) for s in gas.species()])
    hydrogen = np.array([s.composition.get("H", 0.0) for s in gas.species()])
    return carbon, hydrogen


def graphite_equilibrium(
    temperature: u.Quantity, pressure: u.Quantity
) -> GraphiteEquilibrium:
    """Solve the gas over a graphite wall for pure-hydrogen edge gas.

    Carbon already in the edge gas (the rod and plug's, in the hydrogen chamber)
    would lower B', so this is an upper bound for that chamber.

    Args:
        temperature: Wall temperature.
        pressure: Total gas pressure.

    Returns:
        The equilibrium state, with B' and the species split.

    Raises:
        ValueError: Above graphite's sublimation line, where carbon vapor alone
            exceeds the total pressure and no solid wall can be in equilibrium.
    """
    gas, graphite = _phases()
    temp_k = float(temperature.to_value(u.K))
    p_atm = float(pressure.to_value(u.Pa)) / ct.one_atm
    gas.TP = temp_k, ct.one_atm
    graphite.TP = temp_k, ct.one_atm
    g_gas = gas.standard_gibbs_RT
    g_carbon = graphite.standard_gibbs_RT[0]
    g_h2 = g_gas[gas.species_index("H2")]
    carbon, hydrogen = _atom_counts(gas)
    ln_k = -(g_gas - carbon * g_carbon - 0.5 * hydrogen * g_h2)

    def excess(ln_p_h2: float) -> float:
        return float(np.exp(ln_k + 0.5 * hydrogen * ln_p_h2).sum() - p_atm)

    ln_p_ceiling = float(np.log(p_atm))
    if excess(_LN_P_FLOOR) > 0.0:
        raise ValueError(
            f"{temp_k:.0f} K is above graphite's sublimation line at "
            f"{pressure.to(u.bar):.3g}"
        )
    ln_p_h2 = brentq(excess, _LN_P_FLOOR, ln_p_ceiling)
    partial = np.exp(ln_k + 0.5 * hydrogen * ln_p_h2)
    carbon_mass = float((partial * carbon).sum()) * _CARBON_MOLAR_MASS
    hydrogen_mass = float((partial * hydrogen).sum()) * _HYDROGEN_MOLAR_MASS
    fractions = partial / partial.sum()
    return GraphiteEquilibrium(
        temperature=temperature,
        pressure=pressure,
        b_prime=carbon_mass / hydrogen_mass,
        mole_fractions={
            name: float(x) for name, x in zip(gas.species_names, fractions)
        },
    )


def b_prime_table(
    temperatures: u.Quantity = TABLE_TEMPERATURES,
    pressures: u.Quantity = TABLE_PRESSURES,
) -> pd.DataFrame:
    """B' over a grid of wall temperatures and pressures.

    Args:
        temperatures: Rows.
        pressures: Columns.

    Returns:
        B' indexed by temperature in K, with a column per pressure in bar.
        Points above the sublimation line are NaN.
    """
    rows: dict[float, list[float]] = {}
    for temp in temperatures:
        row: list[float] = []
        for p in pressures:
            try:
                row.append(graphite_equilibrium(temp, p).b_prime)
            except ValueError:
                row.append(float("nan"))
        rows[float(temp.to_value(u.K))] = row
    columns: Sequence[float] = [float(p.to_value(u.bar)) for p in pressures]
    table = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    table.index.name = "T (K)"
    table.columns.name = "p (bar)"
    return table


def main() -> None:
    """Print the B' table and the species behind the two chamber cases."""
    print("B' (kg C per kg H) over graphite, pure H2 edge gas, ideal gas")
    print(b_prime_table().round(2).to_string())
    print()
    for temp in (2500, 3900) * u.K:
        eq = graphite_equilibrium(temp, 818 * u.bar)
        lead = ", ".join(f"{n} {x:.3f}" for n, x in eq.leading_species())
        print(f"{temp:.0f} at 818 bar: B' = {eq.b_prime:.2f}; mole fractions {lead}")


if __name__ == "__main__":
    main()
