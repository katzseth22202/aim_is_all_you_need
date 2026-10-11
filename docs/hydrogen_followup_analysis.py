"""Replay the recorded 3500 K, early-wave, and adaptive-chain sensitivities.

Run: PYTHONPATH=. .venv/bin/python docs/hydrogen_followup_analysis.py
Add --verify-trajectories to independently integrate the saved departure inputs.
Uses only tracked JSON inputs and repository modules; never reads scratch state.
This replays the selected financial designs, not a new global optimization.
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src import growth_cost as gc
from src.learning_curve import LearningCurve


def restore(record):
    """Restore pure financial records with their physical mass fields intact."""
    fields = dict(record["inputs"])
    fields["departure"] = gc.Departure(fields["departure"])
    fields["cycles"] = tuple(gc.CycleHardware(**c) for c in fields["cycles"])
    fields["deliveries"] = tuple(
        tuple(gc.DeliveryOption(**d) for d in ds) for ds in fields["deliveries"]
    )
    fields["times"] = tuple(fields["times"])
    fields["arrivals"] = tuple(fields["arrivals"])
    prices = {
        key: LearningCurve(**value) if isinstance(value, dict) else value
        for key, value in record["prices"].items()
    }
    return gc.DesignInputs(**fields), gc.PriceBook(**prices)


@contextmanager
def equipment_charge(dollars):
    """Charge a fixed machine bill per launch, independently of chamber count.

    Older fixed-count scenarios put this bill in chamber_spray and pass zero.
    The adaptive snapshots set chamber_spray to zero and pass $2M here. The
    temporary hook is restored even on failure; no production default changes.
    """
    original = gc._grow

    def grow(inputs, n, units, years, prices, program, tally):
        cost = original(inputs, n, units, years, prices, program, tally)
        extra = dollars * units if prices.charge_growth else 0.0
        if tally is not None and dollars:
            tally["recovery_equipment"] = tally.get("recovery_equipment", 0.0) + extra
        return cost + extra

    gc._grow = grow
    try:
        yield
    finally:
        gc._grow = original


def verify_selected_hardware(snapshot):
    """Check cadence, stream balance, and adaptive selection rules."""
    profiles = {p["id"]: p for p in snapshot["departure_profiles"]}
    for name, design in snapshot["selected_designs"].items():
        if name.startswith("T"):
            assert design["hardware"]["peak_hz"] <= 4 + 1e-8, name
        elif name.startswith("early_"):
            for index in (0, 1):
                peak = profiles[f"early_stage{index}"]["trajectory"]["peak_rod_rate"]
                cadence = (
                    peak
                    * design["stacks_kg"][index]
                    / (2.5 * design["chambers"][index])
                )
                assert cadence <= 4 + 1e-8, name
            assert (
                abs(
                    design["useful_kg"][0] / design["seed_puffsats"]
                    - design["useful_kg"][1] / design["seed_rods"]
                )
                < 1e-8
            ), name
        elif name.startswith("adaptive_"):
            rows = design["cycles"]
            for index, row in enumerate(rows):
                for stream in (0, 1):
                    peak = profiles[f"{name}_{index}_{stream}"]["trajectory"][
                        "peak_rod_rate"
                    ]
                    cadence = (
                        peak
                        * row["stacks_kg"][stream]
                        / (2.5 * row["chambers"][stream])
                    )
                    assert cadence <= 4 + 1e-8, name
                next_row = rows[(index + 1) % len(rows)]
                assert (
                    abs(
                        row["useful_kg"][1] / row["useful_kg"][0]
                        - next_row["consumed_kg"][1] / next_row["consumed_kg"][0]
                    )
                    < 1e-8
                ), name
            for cycle in snapshot["adaptive_routes"][name]["cycles"]:
                if cycle["synodic_multiple"] == 2:
                    assert cycle["nozzle_wave_dsm"] < 0.05, name
                    if name.endswith("_both"):
                        assert cycle["growth_wave_burn"] < 0.05, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-trajectories", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads(
        (root / "data/hydrogen_followups_2026-10-10.json").read_text()
    )
    verify_selected_hardware(snapshot)
    for case in snapshot["financial_cases"]:
        inputs, prices = restore(case["financial_record"])
        for cycle in inputs.cycles:
            assert (
                abs(
                    cycle.growth * cycle.consumed
                    - cycle.onward_puffsats
                    - cycle.onward_rods
                )
                < 1e-6
            )
        schedule = gc.DiscountSchedule.risked(0.1, 0.5, inputs.proof_years)
        with equipment_charge(case["financial_record"]["equipment_per_flight_dollars"]):
            cost, lines = gc.steady_state_cost(inputs, prices)
            values = [
                gc.break_even_price(inputs, prices, seed, schedule, True)
                for seed in case["seed_prices"]
            ]
        assert abs(cost - case["expected_operating"]) < 1e-7, case["id"]
        assert abs(sum(lines.values()) - cost) < 1e-8
        assert all(
            a is not None and abs(a - b) < 1e-6
            for a, b in zip(values, case["expected_break_even"])
        ), case["id"]
        print(
            f"{case['id']}: operating ${cost:.2f}; break-even ${values[0]:.2f} / ${values[1]:.2f}"
        )
    if args.verify_trajectories:
        from hydrogen_followup_trajectory import evaluate

        for profile in snapshot["departure_profiles"]:
            expected = profile["trajectory"]
            found = evaluate(expected["x"], profile["geometry"])
            assert np.linalg.norm(found["residual"]) < 1e-6, profile["id"]
            assert found["clearance"] >= -1e-3 and found["miss"] >= 600 - 1e-3
            assert abs(found["rod"] - expected["rod"]) < 1e-8
            assert abs(found["peak_rod_rate"] - expected["peak_rod_rate"]) < 1e-10
        print(f"Verified {len(snapshot['departure_profiles'])} finite-burn profiles.")


if __name__ == "__main__":
    main()
