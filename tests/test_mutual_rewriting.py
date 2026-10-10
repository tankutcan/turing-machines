"""Checks of the mutual-rewriting model: encoding, dynamics, cycles, bookkeeping."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from turing_machines.alignment_census import table_from_index
from turing_machines.mutual_rewriting import (
    MACHINES, TAPE_LENGTH, analyze, decode_tape, encode_table, find_cycles, pack, step,
    summarize, table_ids, tape_string, unpack, write_class,
)

ROOT = Path(__file__).resolve().parents[1]


def reference_step(state):
    """Scalar update written directly from the model description."""
    word1, word2, x1, x2, q1, q2 = (int(v) for v in unpack(state))
    tape1 = [(word1 >> cell) & 1 for cell in range(TAPE_LENGTH)]
    tape2 = [(word2 >> cell) & 1 for cell in range(TAPE_LENGTH)]
    write1, move1, next1 = decode_tape(word1)[q1 + 1][tape2[x1]]
    write2, move2, next2 = decode_tape(word2)[q2 + 1][tape1[x2]]
    tape2[x1], tape1[x2] = write1, write2
    word1 = sum(bit << cell for cell, bit in enumerate(tape1))
    word2 = sum(bit << cell for cell, bit in enumerate(tape2))
    return pack(word1, word2, (x1 + 2 * move1 - 1) % TAPE_LENGTH,
                (x2 + 2 * move2 - 1) % TAPE_LENGTH, next1 - 1, next2 - 1)


def random_states(count, seed):
    rng = np.random.default_rng(seed)
    return pack(rng.integers(0, MACHINES, count), rng.integers(0, MACHINES, count),
                rng.integers(0, TAPE_LENGTH, count), rng.integers(0, TAPE_LENGTH, count),
                rng.integers(0, 2, count), rng.integers(0, 2, count))


class EncodingTests(unittest.TestCase):
    def test_every_table_round_trips_and_keeps_its_census_id(self):
        words = set()
        for table_id in range(MACHINES):
            tm = table_from_index(table_id, 2)
            word = encode_table(tm)
            words.add(word)
            self.assertEqual(decode_tape(word), tm)
            self.assertEqual(int(table_ids(np.array([word]))[0]), table_id)
        self.assertEqual(words, set(range(MACHINES)))

    def test_uniform_tapes_and_layout(self):
        self.assertEqual(decode_tape(0), {q: {b: [0, 0, 1] for b in (0, 1)} for q in (1, 2)})
        self.assertEqual(decode_tape(MACHINES - 1),
                         {q: {b: [1, 1, 2] for b in (0, 1)} for q in (1, 2)})
        tm = {1: {0: [1, 0, 1], 1: [0, 1, 1]}, 2: {0: [0, 0, 2], 1: [1, 1, 2]}}
        self.assertEqual(tape_string(encode_table(tm)), "100 010 001 111")

    def test_pack_round_trip(self):
        fields = (4095, 1234, 11, 7, 1, 0)
        self.assertEqual(tuple(int(v) for v in unpack(pack(*fields))), fields)

    def test_write_classes(self):
        labels = write_class(np.arange(MACHINES))
        counts = {name: int((labels == name).sum()) for name in set(labels)}
        self.assertEqual(counts, {"always_writes_0": 256, "always_writes_1": 256,
                                  "writes_what_it_reads": 256, "other": 3328})
        self.assertEqual(labels[0], "always_writes_0")
        self.assertEqual(labels[MACHINES - 1], "always_writes_1")


class DynamicsTests(unittest.TestCase):
    def test_step_matches_scalar_reference(self):
        states = random_states(3000, 1)
        expected = [reference_step(s) for s in states]
        np.testing.assert_array_equal(step(states), expected)

    def test_writes_use_the_old_tables(self):
        # Head 1 overwrites the entry head 2 is about to use; head 2 must still use the old one.
        tm1 = {q: {b: [1, 1, 1] for b in (0, 1)} for q in (1, 2)}   # writes 1 at cell 0
        tm2 = {q: {b: [0, 1, 1] for b in (0, 1)} for q in (1, 2)}   # entry (1,*) writes 0
        t1, t2, *_ = unpack(step(np.array([pack(encode_table(tm1), encode_table(tm2), 0, 0, 0, 0)])))
        self.assertEqual(int(t2[0]) & 1, 1)                          # machine 2 now writes 1 ...
        self.assertEqual(int(t1[0]) & 1, 0)                          # ... but wrote 0 this step

    def test_uniform_programs_are_frozen(self):
        for word in (0, MACHINES - 1):
            start = np.array([pack(word, word, 3, 8, 0, 1)])
            result = analyze(start)
            self.assertEqual(int(result["period"][0]), TAPE_LENGTH)
            self.assertTrue(result["both_frozen"][0] and result["same_program"][0])
            self.assertEqual(int(result["attractors"]["rewrites1"][0]), 0)

    def test_cycles_match_brute_force(self):
        starts = random_states(400, 2)
        transient, period, on_cycle = find_cycles(starts)
        for i, start in enumerate(starts.tolist()):
            seen, state, time = {}, start, 0
            while state not in seen:
                seen[state] = time
                state = int(reference_step(state))
                time += 1
            self.assertEqual((int(transient[i]), int(period[i])), (seen[state], time - seen[state]))
            self.assertGreaterEqual(seen[int(on_cycle[i])], seen[state])


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.words1 = rng.integers(0, MACHINES, 5000)
        cls.words2 = rng.integers(0, MACHINES, 5000)
        cls.result = analyze(pack(cls.words1, cls.words2, 0, 0, 0, 0))

    def test_bookkeeping(self):
        r, a = self.result, self.result["attractors"]
        self.assertEqual(int(a["basin"].sum()), 5000)
        np.testing.assert_array_equal(np.bincount(a["index"], minlength=len(a["basin"])), a["basin"])
        np.testing.assert_allclose(a["occupancy"].sum(axis=1), [1, 1])
        self.assertAlmostEqual(float(r["occupancy"].sum()), 1)
        self.assertEqual(int(r["held"].sum()), 10000)
        self.assertTrue(np.all(r["survives"] <= r["held"]))
        self.assertTrue(np.all(r["untouched"] <= r["both_frozen"]))
        # A canonical state lies on its own cycle and is its smallest state.
        for i in np.argsort(-a["basin"])[:20]:
            state, smallest = a["state"][i:i + 1], int(a["state"][i])
            for _ in range(int(a["period"][i])):
                state = step(state)
                smallest = min(smallest, int(state[0]))
            self.assertEqual((int(state[0]), smallest), (int(a["state"][i]), int(a["state"][i])))

    def test_exchanging_the_machines_mirrors_the_outcome(self):
        mirrored = analyze(pack(self.words2, self.words1, 0, 0, 0, 0))
        np.testing.assert_array_equal(mirrored["transient"], self.result["transient"])
        np.testing.assert_array_equal(mirrored["period"], self.result["period"])
        np.testing.assert_array_equal(mirrored["same_program"], self.result["same_program"])
        np.testing.assert_array_equal(mirrored["survives"], self.result["survives"])
        np.testing.assert_allclose(mirrored["occupancy"], self.result["occupancy"])

    def test_summary_partitions_the_starts(self):
        summary = summarize(self.words1, self.words2, self.result)
        self.assertEqual(sum(summary["outcomes"].values()), 5000)
        self.assertEqual(sum(total for total, _, _ in
                             summary["outcome_by_write_one_entries"].values()), 5000)
        self.assertEqual(summary["frozen_attractors"] + summary["dynamic_attractors"],
                         summary["attractors"])
        self.assertEqual(sum(row["machines"] for row in summary["by_write_class"].values()), MACHINES)
        self.assertAlmostEqual(
            sum(row["late_occupancy"] for row in summary["by_write_class"].values()), 1)
        # The uniform programs dominate; a loose bound guards against regressions.
        self.assertGreater(summary["outcomes"]["both_all_zero"]
                           + summary["outcomes"]["both_all_one"], 4000)

    def test_command_writes_a_complete_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs"
            command = [sys.executable, "-m", "scripts.mutual_rewriting", "--pairs", "400",
                       "--start-positions", "2", "7", "--start-states", "2", "1",
                       "--output-dir", str(output)]
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"machines.csv", "largest_attractors.csv", "attractors.npz",
                              "pairs.npz", "summary.json", "README.md"})
            with (output / "machines.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["table_id"]) for row in rows], list(range(MACHINES)))
            self.assertEqual(sum(int(row["held"]) for row in rows), 800)
            pairs, attractors = np.load(output / "pairs.npz"), np.load(output / "attractors.npz")
            self.assertEqual(len(pairs["attractor"]), 400)
            self.assertEqual(int(attractors["basin"].sum()), 400)
            np.testing.assert_array_equal(attractors["period"][pairs["attractor"]], pairs["period"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["parameters"]["start_states"], [2, 1])
            again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("not empty", again.stderr)


if __name__ == "__main__":
    unittest.main()
