import unittest

import numpy as np

from turing_machines.sample_halt_times import SPACE_SIZES, decode_machine_ids, simulate_batch
from turing_machines.sample_random_tape_halt_times import (
    halt_reachable, random_tapes, scalar_trace, simulate_random_batch,
)


class RandomTapeHaltTimeTests(unittest.TestCase):
    def test_random_tape_stream_is_batch_independent(self):
        whole = random_tapes(np.random.default_rng(8), 29, 151)
        rng = np.random.default_rng(8)
        pieces = np.concatenate([random_tapes(rng, n, 151) for n in (1, 7, 21)])
        np.testing.assert_array_equal(whole, pieces)
        self.assertTrue(np.all((whole == 0) | (whole == 1)))

    def test_halt_after_107_and_cutoff_are_distinct(self):
        # q=1 scans right through ones and halts at its first zero.
        actions = np.zeros((1, 8), dtype=np.uint8)
        actions[0, 0] = 19  # write 1, move right, Halt
        actions[0, 1] = 12  # write 1, move right, stay in state 1
        tape = np.zeros((1, 303), dtype=np.uint8)
        tape[0, 151:301] = 1
        times, proof = simulate_random_batch(actions, tape.copy(), 151)
        self.assertEqual(times[0], 151)
        self.assertFalse(proof[0])
        shorter = tape[:, 1:302].copy()  # same physical input, now centered at 150
        times, proof = simulate_random_batch(actions, shorter, 150)
        self.assertEqual(times[0], -1)
        self.assertFalse(proof[0])
        blank = np.zeros_like(tape)
        times, _ = simulate_random_batch(actions, blank, 151)
        self.assertEqual(times[0], 1)

    def test_immediate_halt_on_both_initial_bits(self):
        actions = np.zeros((2, 8), dtype=np.uint8)
        actions[0, 0], actions[1, 1] = 16, 19
        tape = np.zeros((2, 21), dtype=np.uint8)
        tape[1, 10] = 1
        times, proof = simulate_random_batch(actions, tape, 10)
        np.testing.assert_array_equal(times, [1, 1])
        np.testing.assert_array_equal(proof, [False, False])
        self.assertEqual(tape[0, 10], 0)
        self.assertEqual(tape[1, 10], 1)

    def test_graph_reachability_and_certification(self):
        actions = np.zeros((2, 8), dtype=np.uint8)
        actions[0, 0:2] = 1  # state 1 -> state 2 -> state 3 -> state 4 -> H
        actions[0, 2:4] = 2
        actions[0, 4:6] = 3
        actions[0, 6:8] = 16
        actions[1, 7] = 16  # H exists, but state 1 stays in state 1
        reachable = halt_reachable(actions)
        self.assertTrue(np.all(reachable[0]))
        self.assertFalse(reachable[1, 1])
        times, proof = simulate_random_batch(actions, np.zeros((2, 21), dtype=np.uint8), 10)
        np.testing.assert_array_equal(times, [4, -1])
        np.testing.assert_array_equal(proof, [False, True])

    def test_random_trajectories_match_scalar_reference_without_pruning(self):
        rng = np.random.default_rng(31)
        ids = rng.integers(0, SPACE_SIZES['exactly-one'], size=300, dtype=np.uint64)
        tables = decode_machine_ids(ids)
        initial = random_tapes(rng, len(ids), 300)
        times, proof = simulate_random_batch(tables, initial.copy(), 300)
        without, _ = simulate_random_batch(tables, initial.copy(), 300, graph_pruning=False)
        np.testing.assert_array_equal(times, without)
        for i, table in enumerate(tables):
            scalar_time, _ = scalar_trace(table, initial[i], 300)
            self.assertEqual(times[i], scalar_time)
            if proof[i]:
                self.assertEqual(scalar_time, -1)

    def test_blank_tape_matches_original_batch_simulator(self):
        rng = np.random.default_rng(21)
        tables = decode_machine_ids(rng.integers(0, SPACE_SIZES['exactly-one'], size=1000, dtype=np.uint64))
        expected, _ = simulate_batch(tables)
        actual, _ = simulate_random_batch(tables, np.zeros((1000, 215), dtype=np.uint8), 107)
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
