"""Replay the fixed-flight hardware recovery sensitivity.

Run from the repository root: PYTHONPATH=. .venv/bin/python docs/hydrogen_recovery_analysis.py
Inputs and outputs live in data/; no scratch files are required.
"""

import json
from dataclasses import replace
from pathlib import Path

from src.growth_cost import (
    CycleHardware,
    DeliveryOption,
    Departure,
    DesignInputs,
    DiscountSchedule,
    PriceBook,
    break_even_price,
    steady_state_cost,
)
from src.learning_curve import LearningCurve


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/hydrogen_curvature_cost_2026-10-10.json"
    snapshot = json.loads(source.read_text())
    rows = []
    # Source's RL10 extension area-scaling proxy; the chamber wall is 32 t.
    extension_kg = 131 * (8.7 / 2.13) ** 2
    for name in ("hydrogen_3000", "hydrogen_5500"):
        record = snapshot["financial_input_records"]["common_3S/" + name]
        fields = dict(record["inputs"])
        fields["departure"] = Departure(fields["departure"])
        fields["cycles"] = tuple(CycleHardware(**c) for c in fields["cycles"])
        fields["deliveries"] = tuple(
            tuple(DeliveryOption(**d) for d in ds) for ds in fields["deliveries"]
        )
        fields["times"] = tuple(fields["times"])
        fields["arrivals"] = tuple(fields["arrivals"])
        base = DesignInputs(**fields)
        prices = dict(record["prices"])
        for key, value in prices.items():
            if isinstance(value, dict):
                prices[key] = LearningCurve(**value)
        book = PriceBook(**prices)
        hardware = record["hardware"]
        seeds = next(
            r["seed_prices"]
            for r in snapshot["common_route_results"]
            if r["case"] == name
        )
        old = base.cycles[0]
        original = old.onward_puffsats + old.onward_rods
        assert abs(original - old.growth * old.consumed) < 1e-6
        assert old.departure_units == 1
        for scope in ("baseline", "extension_only", "chamber_and_nozzle"):
            recovered_tank = 0 if scope == "baseline" else 0.9 * old.tanks
            recovered_engine = (
                0
                if scope == "baseline"
                else 0.6
                * (
                    extension_kg
                    if scope == "extension_only"
                    else hardware["hardware_t"] * 1000
                )
            )
            equipment = 0 if scope == "baseline" else 5000
            onward = original - equipment + recovered_tank + recovered_engine
            ratio = onward / original
            cycle = replace(
                old,
                growth=onward / old.consumed,
                onward_puffsats=old.onward_puffsats * ratio,
                onward_rods=old.onward_rods * ratio,
            )
            # All recovered material is available by the next return.
            # Ordinary body and steering-package prices apply to all of it.
            # No credit for buying less virgin body material is taken.
            unrecovered = (
                old.tanks
                + hardware["hardware_t"] * 1000
                - recovered_tank
                - recovered_engine
            )
            assert (
                abs(
                    onward
                    + equipment
                    + unrecovered
                    - original
                    - old.tanks
                    - hardware["hardware_t"] * 1000
                )
                < 1e-6
            )
            for synodics in (15, 21):
                current = replace(
                    base, cycles=(cycle,) * len(base.cycles), harvest=synodics // 3 - 1
                )
                schedule = DiscountSchedule.risked(0.1, 0.5, current.proof_years)
                for machine_price in (
                    (0,) if scope == "baseline" else (0, 1e6, 5e6, 20e6)
                ):
                    # One chamber and one processor per growth launch. The existing
                    # flat per-chamber allowance supplies exactly the extra cash flow.
                    priced = replace(
                        book, chamber_spray=book.chamber_spray + machine_price
                    )
                    cost, lines = steady_state_cost(current, priced)
                    rows.append(
                        dict(
                            case=name,
                            scope=scope,
                            horizon_synodics=synodics,
                            harvest_year=current.times[current.harvest],
                            machine_purchase_dollars=machine_price,
                            equipment_kg=equipment,
                            recovered_tank_kg=recovered_tank,
                            recovered_engine_kg=recovered_engine,
                            onward_kg=onward,
                            growth=cycle.growth,
                            steady_cost=cost,
                            cost_lines=lines,
                            be_steady=[
                                break_even_price(current, priced, s, schedule, True)
                                for s in seeds
                            ],
                        )
                    )
                    if synodics == 15 and machine_price == 1e6:
                        print(
                            name,
                            scope,
                            machine_price,
                            "onward",
                            round(onward / 1000, 2),
                            "growth",
                            round(cycle.growth, 4),
                            "operating",
                            round(cost, 2),
                            "BE",
                            [round(v, 2) for v in rows[-1]["be_steady"]],
                        )
        previous = next(
            r
            for r in snapshot["common_route_results"]
            if r["case"] == name and r["horizon_years"] > 15
        )
        replay = next(
            r
            for r in rows
            if r["case"] == name
            and r["scope"] == "baseline"
            and r["horizon_synodics"] == 15
        )
        assert abs(previous["steady_cost"] - replay["steady_cost"]) < 1e-8
        assert all(
            abs(a - b) < 1e-8
            for a, b in zip(previous["be_steady"], replay["be_steady"])
        )
    output = dict(
        source_snapshot=source.name,
        extension_kg=extension_kg,
        method="Fixed flight and launch push; processor displaces payload; "
        "recovered material allocated in original PuffSat/rod ratio; "
        "unchanged exhaust and impact performance; no recovery of processor; "
        "normal body/package prices charged on recovered mass; machine "
        "price charged on every growth launch without learning.",
        results=rows,
    )
    (root / "data/hydrogen_recovery_cost_2026-10-10.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
