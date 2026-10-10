import unittest

import numpy as np

from turing_machines.coupled_chiral_ants import simulate as simulate_sparse
from turing_machines.many_turing_machines import simulate
from turing_machines.simulate_busybeaver4 import blank_reference, configuration, machine


class BusyBeaverFourTests(unittest.TestCase):
    def test_only_halt_target_changes(self):
        original, modified = machine(0), machine()
        changed = [(q, b, j) for q in original for b in (0, 1) for j in range(3)
                   if original[q][b][j] != modified[q][b][j]]
        self.assertEqual(changed, [(3, 0, 2)])
        self.assertEqual(modified[3][0], [1, 1, 3])
        self.assertTrue(all(1 <= row[b][2] <= 4 for row in modified.values() for b in (0, 1)))

    def test_original_score_and_shared_prefix(self):
        original = simulate_sparse(machine(0), 200)
        self.assertEqual((original["halted"], original["steps"], original["ones"]), (True, 107, 13))
        rows, ones = blank_reference(107)
        self.assertEqual(rows[:106], original["trace"][:106])
        last = dict(original["trace"][-1], state_after=3)
        self.assertEqual(rows[-1], last)
        self.assertEqual(ones, set(original["ones_positions"]))

    def test_blank_run_against_independent_sparse_trace(self):
        rows, ones = blank_reference(500)
        args = configuration(steps=500, plot_width=1000)
        result = simulate(args, tm_override=machine())
        np.testing.assert_array_equal(result["position_trace"][:, 0],
                                      [0] + [row["position_after"] for row in rows])
        self.assertEqual(set(np.flatnonzero(result["tape"]) + result["tape_min"]), ones)
        self.assertEqual(result["final_states"][0], rows[-1]["state_after"])
        self.assertEqual(result["spacetime"][0].sum(), 0)

    def test_periodic_ensemble_against_scalar_majority_updates(self):
        args = configuration(copies=6, steps=80, seed=7, plot_width=12,
                             boundary="periodic", tape_length=12,
                             one_density=.5, spacing=1, record_every=1)
        result = simulate(args, tm_override=machine())
        tape = (np.random.default_rng(args.seed).random(args.tape_length) < .5).astype(np.uint8)
        x = (np.arange(args.copies) - args.copies // 2) * args.spacing
        q = np.ones(args.copies, dtype=int)
        tm = machine()
        coincident = conflicting = 0
        for t in range(args.steps + 1):
            np.testing.assert_array_equal(result["spacetime"][t], tape)
            np.testing.assert_array_equal(result["position_trace"][t], x)
            np.testing.assert_array_equal(result["head_occupancy"][t],
                                          np.bincount(x % args.tape_length, minlength=args.tape_length))
            if t == args.steps:
                break
            proposals, next_x, next_q = {}, x.copy(), q.copy()
            for i in range(args.copies):
                cell = int(x[i] % args.tape_length)
                write, direction, state = tm[int(q[i])][int(tape[cell])]
                proposals.setdefault(cell, []).append(write)
                next_x[i] += 2 * direction - 1
                next_q[i] = state
            for cell, votes in proposals.items():
                coincident += len(votes) > 1
                conflicting += len(set(votes)) > 1
                balance = 2 * sum(votes) - len(votes)
                if balance:
                    tape[cell] = int(balance > 0)
            x, q = next_x, next_q
        np.testing.assert_array_equal(result["final_states"], q)
        self.assertEqual(result["coincident_cells"], coincident)
        self.assertEqual(result["conflicting_cells"], conflicting)
        self.assertGreater(coincident, 0)
        self.assertGreater(conflicting, 0)


if __name__ == "__main__":
    unittest.main()
