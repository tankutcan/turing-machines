"""Checks of the alignment census: encoding, table structure, dynamics, outcomes."""

import argparse
from collections import Counter
import csv
import json
from math import factorial
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from turing_machines.alignment_census import (
    alignment_level, alignment_mechanism, census_row, entry_pattern, index_from_table,
    simulate_gas, static_structure, summarize, table_from_index, table_space_size, tape_class,
)
from turing_machines.analyze_two_state_structure import state_structure
from turing_machines.many_turing_machines import make_tm, simulate
from turing_machines.order_two_state_zoo import rule_observables

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(tape_length=200, copies=200, steps=800)


def two_state(entries):
    return {1: {0: entries[0], 1: entries[1]}, 2: {0: entries[2], 1: entries[3]}}


# State 1 is Langton's rule and absorbing; state 2 always enters it.
ABSORBING = two_state([[1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1]])
# Everything erases and moves right; (1,0) is the only fixed entry.
FIXED_ENTRY = two_state([[0, 1, 1], [0, 1, 2], [0, 1, 1], [0, 1, 1]])
# Heads never change state: two absorbing states.
FROZEN_STATES = two_state([[0, 1, 1], [1, 1, 1], [0, 1, 2], [1, 1, 2]])
# States swap every step whatever is read.
SWAP_CLOCK = two_state([[0, 1, 2], [1, 1, 2], [0, 1, 1], [1, 1, 1]])


class TableEncodingTests(unittest.TestCase):
    def test_enumeration_is_a_bijection(self):
        self.assertEqual([table_space_size(n) for n in (1, 2, 3)], [16, 4096, 12**6])
        for n in (1, 2):
            tables = [table_from_index(i, n) for i in range(table_space_size(n))]
            self.assertEqual(len({json.dumps(tm) for tm in tables}), table_space_size(n))
            for i, tm in enumerate(tables):
                self.assertEqual(index_from_table(tm), i)
                for row in tm.values():
                    for write_bit, move_bit, next_state in row.values():
                        self.assertIn(write_bit, (0, 1))
                        self.assertIn(move_bit, (0, 1))
                        self.assertIn(next_state, range(1, n + 1))

    def test_round_trip_for_sampled_tables(self):
        rng = np.random.default_rng(5)
        for n in (3, 4, 6):
            for _ in range(50):
                tm = make_tm(n, rng)
                self.assertEqual(table_from_index(index_from_table(tm), n), tm)

    def test_index_bounds(self):
        for index in (-1, 4096):
            with self.assertRaises(ValueError):
                table_from_index(index, 2)


class StaticStructureTests(unittest.TestCase):
    def test_all_4096_two_state_tables(self):
        counts = Counter()
        for index in range(4096):
            tm = table_from_index(index, 2)
            s = static_structure(tm)
            absorbing, fixed = bool(s["absorbing_states"]), bool(s["fixed_entries"])
            counts["absorbing"] += absorbing
            counts["reachable"] += bool(s["reachable_absorbing_states"])
            counts["fixed_only"] += fixed and not absorbing
            counts["neither"] += not fixed and not absorbing
            for key in ("symbol_blind", "synchronizing", "weakly_mergeable", "reversible"):
                counts[key] += s[key]
            # Independent classification from the saved-zoo analysis.
            tag = state_structure(rule_observables(tm))["structure_class"]
            self.assertEqual(bool(s["reachable_absorbing_states"]), tag in ("S1", "S2"))
            self.assertEqual(len(s["absorbing_states"]) == 2, tag == "I")
            if s["reachable_absorbing_states"]:
                self.assertTrue(s["synchronizing"])
            if s["synchronizing"]:
                self.assertTrue(s["weakly_mergeable"])
        self.assertEqual(counts, {
            "absorbing": 1792,          # 4096 * (1 - (3/4)^2)
            "reachable": 1536,          # classes S1 and S2
            "fixed_only": 1280, "neither": 1024,
            "symbol_blind": 1024,       # S_q(0)=S_q(1) for both states
            "synchronizing": 3072,      # fails only when both read maps are permutations
            "weakly_mergeable": 3584,
            "reversible": factorial(4) * 2**2,
        })

    def test_named_tables(self):
        s = static_structure(ABSORBING)
        self.assertEqual((s["absorbing_states"], s["reachable_absorbing_states"]), ([1], [1]))
        s = static_structure(FIXED_ENTRY)
        self.assertEqual((s["absorbing_states"], s["fixed_entries"]), ([], [[1, 0]]))
        s = static_structure(FROZEN_STATES)
        self.assertEqual((s["absorbing_states"], s["reachable_absorbing_states"]), ([1, 2], []))
        self.assertFalse(s["weakly_mergeable"])
        s = static_structure(SWAP_CLOCK)
        self.assertTrue(s["symbol_blind"])
        self.assertFalse(s["synchronizing"])


class DynamicsTests(unittest.TestCase):
    def test_matches_the_gas_simulator_with_conflicting_writes(self):
        conflicts = 0
        for seed in range(6):
            n = 2 + seed % 3
            tm = make_tm(n, np.random.default_rng(100 + seed))
            args = argparse.Namespace(
                seed=seed, states=n, copies=40, steps=60, spacing=1, one_density=0.5,
                collision_policy="majority", plot_width=24, record_every=1,
                burn_in_fraction=0.5, boundary="periodic", tape_length=24,
                machine_ensemble="identical", initial_state_mode="random",
                update_scheme="synchronous")
            reference = simulate(args, tm_override=tm)
            conflicts += reference["conflicting_cells"]
            run = simulate_gas(
                tm, tape_length=24, copies=40, steps=60, seed=None,
                initial=(reference["spacetime"][0], reference["initial_tape_positions"],
                         reference["initial_states"]))
            np.testing.assert_array_equal(run["final_tape"], reference["tape"])
            # Heads are interchangeable: compare the occupation of (position, state).
            self.assertEqual(
                Counter(zip(run["final_positions"].tolist(), run["final_states"].tolist())),
                Counter(zip(reference["final_positions"].tolist(),
                            reference["final_states"].tolist())))
        self.assertGreater(conflicts, 0)

    def test_seeded_runs_are_reproducible(self):
        tm = table_from_index(1234, 2)
        first = simulate_gas(tm, seed=[0, 1, 1234, 0], **SMALL)
        second = simulate_gas(tm, seed=[0, 1, 1234, 0], **SMALL)
        np.testing.assert_array_equal(first["final_tape"], second["final_tape"])
        self.assertEqual(first["entry_occupation"], second["entry_occupation"])
        self.assertAlmostEqual(sum(first["entry_occupation"]), 1, places=4)


class OutcomeTests(unittest.TestCase):
    def test_absorbing_state_aligns_at_once(self):
        run = simulate_gas(ABSORBING, seed=1, **SMALL)
        self.assertEqual(alignment_level(run, 0.99), "exact")
        self.assertLessEqual(run["alignment_time"], 1)
        self.assertEqual(run["dominant_state"], 1)
        self.assertEqual(alignment_mechanism(static_structure(ABSORBING), run), "absorbing_state")
        self.assertEqual(entry_pattern(run), "2 entries, same state")

    def test_fixed_entry_aligns_on_a_uniform_tape(self):
        run = simulate_gas(FIXED_ENTRY, seed=2, **SMALL)
        self.assertEqual(alignment_level(run, 0.99), "exact")
        self.assertEqual(alignment_mechanism(static_structure(FIXED_ENTRY), run), "fixed_entry")
        self.assertEqual(run["entry_occupation"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual((run["tape_density"], run["velocity"], run["flip_rate"]), (0.0, 1.0, 0.0))
        self.assertEqual(tape_class(run["tape_density"]), "uniform")
        self.assertEqual(entry_pattern(run), "1 entry")

    def test_conserved_states_do_not_align(self):
        run = simulate_gas(FROZEN_STATES, seed=3, **SMALL)
        self.assertEqual(alignment_level(run, 0.99), "none")
        self.assertAlmostEqual(run["alignment_mean"], run["alignment_min"], places=12)
        fixed = simulate_gas(FROZEN_STATES, seed=3, initial_state_mode="fixed", **SMALL)
        self.assertEqual(alignment_level(fixed, 0.99), "exact")

    def test_clock_keeps_phase_differences(self):
        run = simulate_gas(SWAP_CLOCK, seed=4, **SMALL)
        self.assertEqual(alignment_level(run, 0.99), "none")
        fixed = simulate_gas(SWAP_CLOCK, seed=4, initial_state_mode="fixed", **SMALL)
        self.assertEqual(alignment_level(fixed, 0.99), "exact")
        self.assertEqual(fixed["dominant_states_late"], 2)
        self.assertEqual(alignment_mechanism(static_structure(SWAP_CLOCK), fixed), "cycling")

    def test_tape_class_boundaries(self):
        self.assertEqual([tape_class(m) for m in (0, 0.01, 0.011, 0.39, 0.4, 0.6, 0.61, 0.99, 1)],
                         ["uniform", "uniform", "intermediate", "intermediate", "near_half",
                          "near_half", "intermediate", "uniform", "uniform"])


class CensusTests(unittest.TestCase):
    PARAMETERS = dict(samples=None, seed=0, tape_length=120, copies=60, steps=300,
                      replicates=2, one_density=0.5, threshold=0.99, late_fraction=0.25)

    def test_one_state_tables_are_trivially_aligned(self):
        rows = [census_row((1, i, self.PARAMETERS))[0] for i in range(16)]
        summary = summarize(rows)
        self.assertEqual((summary["tables"], summary["aligned"], summary["exactly_aligned"]),
                         (16, 16, 16))
        self.assertEqual(summary["mechanisms"], {"absorbing_state": 16})
        self.assertEqual(summary["aligned_by_structure"]["reachable_absorbing_state"], [16, 16])
        self.assertEqual(sum(total for _, total in summary["aligned_by_tape_class"].values()), 16)

    def test_row_is_reproducible_and_matches_its_table(self):
        index = index_from_table(ABSORBING)
        row, tm = census_row((2, index, self.PARAMETERS))
        self.assertEqual(tm, ABSORBING)
        self.assertEqual(row, census_row((2, index, self.PARAMETERS))[0])
        self.assertTrue(row["aligned"] and row["exactly_aligned"])
        self.assertEqual(row["mechanism"], "absorbing_state")
        sampled = dict(self.PARAMETERS, samples=3)
        self.assertEqual(census_row((3, 1, sampled))[1],
                         make_tm(3, np.random.default_rng([0, 0, 1])))

    def test_command_writes_a_complete_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "census"
            command = [sys.executable, "-m", "scripts.alignment_census", "--states", "1",
                       "--copies", "20", "--tape-length", "50", "--steps", "40",
                       "--workers", "2", "--output-dir", str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"census.csv", "transition_tables.json", "summary.json", "README.md"})
            with (output / "census.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["table_id"]) for row in rows], list(range(16)))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["summary"]["aligned"], 16)
            self.assertEqual(len(json.loads((output / "transition_tables.json").read_text())), 16)
            # A nonempty destination is refused rather than overwritten.
            again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("not empty", again.stderr)


if __name__ == "__main__":
    unittest.main()
