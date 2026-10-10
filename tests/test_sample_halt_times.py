import unittest

import numpy as np

from turing_machines.coupled_chiral_ants import simulate as scalar_simulate
from turing_machines.sample_halt_times import (
    BOUND, SPACE_SIZES, aggregate, as_dict, decode_machine_ids, simulate_batch,
)


def encode_table(tm):
    actions = []
    for q in range(1, 5):
        for b in (0, 1):
            w, d, s = tm[q][b]
            actions.append(w * 8 + d * 4 + s - 1 if s else 16 + w * 2 + d)
    return actions


class HaltTimeTests(unittest.TestCase):
    def test_machine_id_bijection_exactly_one(self):
        rng = np.random.default_rng(11)
        ids = rng.integers(0, SPACE_SIZES["exactly-one"], 1000, dtype=np.uint64)
        ids[:2] = [0, SPACE_SIZES["exactly-one"] - 1]
        tables = decode_machine_ids(ids)
        np.testing.assert_array_equal((tables >= 16).sum(axis=1), 1)
        for number, table in zip(ids, tables):
            entry = int(np.flatnonzero(table >= 16)[0])
            rebuilt = entry + ((int(table[entry]) - 16) << 3)
            for rank, action in enumerate(np.delete(table, entry)):
                rebuilt += int(action) << (5 + 4 * rank)
            self.assertEqual(rebuilt, int(number))

    def test_independent_ids_round_trip(self):
        rng = np.random.default_rng(12)
        ids = rng.integers(0, SPACE_SIZES["independent"], 1000, dtype=np.uint64)
        tables = decode_machine_ids(ids, "independent")
        for number, table in zip(ids, tables):
            self.assertEqual(sum(int(c) * 20**i for i, c in enumerate(table)), int(number))

    def test_seed_is_batch_independent(self):
        for sampling, size in SPACE_SIZES.items():
            rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
            whole = rng1.integers(0, size, 1001, dtype=np.uint64)
            pieces = np.concatenate([rng2.integers(0, size, n, dtype=np.uint64) for n in (1, 503, 497)])
            np.testing.assert_array_equal(whole, pieces)

    def test_known_champions_including_final_transition(self):
        champion107 = {
            1: {0: [1, 1, 2], 1: [1, 0, 2]},
            2: {0: [1, 0, 1], 1: [0, 0, 3]},
            3: {0: [1, 1, 0], 1: [1, 0, 4]},
            4: {0: [1, 1, 4], 1: [0, 1, 1]},
        }
        champion96 = {
            1: {0: [1, 1, 2], 1: [0, 1, 3]},
            2: {0: [1, 0, 1], 1: [1, 1, 1]},
            3: {0: [1, 1, 0], 1: [1, 1, 4]},
            4: {0: [1, 0, 4], 1: [0, 0, 2]},
        }
        times, ones = simulate_batch(np.array([encode_table(champion107), encode_table(champion96)]))
        np.testing.assert_array_equal(times, [107, 96])
        np.testing.assert_array_equal(ones, [13, 13])

    def test_immediate_halt_and_unreachable_halt(self):
        tables = np.zeros((6, 8), dtype=np.uint8)
        tables[:4, 0] = np.arange(16, 20)
        tables[4, 7] = 16  # State 1 always moves left and remains in state 1.
        tables[5] = 4     # State 1 always moves right; no Halt entries.
        times, ones = simulate_batch(tables)
        np.testing.assert_array_equal(times, [1, 1, 1, 1, -1, -1])
        np.testing.assert_array_equal(ones, [0, 0, 1, 1, -1, -1])

    def test_random_samples_against_independent_sparse_simulator(self):
        rng = np.random.default_rng(8)
        for sampling, size in SPACE_SIZES.items():
            tables = decode_machine_ids(rng.integers(0, size, 500, dtype=np.uint64), sampling)
            times, ones = simulate_batch(tables)
            for i, actions in enumerate(tables):
                tm = as_dict(actions)
                self.assertEqual(encode_table(tm), list(actions))
                # Run beyond the bound as an extra cross-check on non-halters.
                reference = scalar_simulate(tm, 2 * BOUND)
                expected = (reference["steps"], reference["ones"]) if reference["halted"] else (-1, -1)
                self.assertEqual((int(times[i]), int(ones[i])), expected)

    def test_aggregation_keeps_infinite_mass_separate(self):
        counts, nonhalting, survival = aggregate(np.array([1, 1, 3, 107, -1, -1]))
        self.assertEqual(nonhalting, 2)
        self.assertEqual(counts.sum(), 4)
        self.assertEqual(counts[107], 1)
        self.assertEqual(survival[0], 1)
        self.assertEqual(survival[107], 2 / 6)
        self.assertTrue(np.all(np.diff(survival) <= 0))


if __name__ == "__main__":
    unittest.main()
