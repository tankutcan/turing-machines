"""Validate all two-state rule structures and finite-time drift classification."""

from collections import Counter
from itertools import product
import json
import unittest

import numpy as np

from turing_machines.analyze_two_state_structure import state_structure, velocity_statistics
from turing_machines.order_two_state_zoo import rule_observables
from tests.test_order_two_state_zoo import machine


class StateStructureTests(unittest.TestCase):
    def test_all_4096_rules_against_reachability(self):
        counts = Counter()
        exact, alternating = 0, 0
        for entries in product(list(product((0, 1), (0, 1), (1, 2))), repeat=4):
            rule = rule_observables(machine(entries))
            result = state_structure(rule)
            counts[result["structure_class"]] += 1
            exact += result["exact_counterpropagating_sectors"]
            alternating += result["strict_alternation"]
            matrix = np.array(json.loads(result["adjacency_source_columns"]))
            np.testing.assert_array_equal(matrix.sum(axis=0), [2, 2])
            reachable = (matrix > 0) | np.eye(2, dtype=bool)
            for k in range(2):
                for i in range(2):
                    for j in range(2):
                        reachable[i, j] |= reachable[i, k] and reachable[k, j]
            sccs = json.loads(result["strongly_connected_components"])
            for i, j in product(range(2), repeat=2):
                same = any(i + 1 in component and j + 1 in component for component in sccs)
                self.assertEqual(same, bool(reachable[i, j] and reachable[j, i]))
            for component in json.loads(result["sink_components"]):
                for source in component:
                    for target in {1, 2} - set(component):
                        self.assertEqual(matrix[target - 1, source - 1], 0)
            # Exact tape-independent population charges require c_next = c_current
            # separately for each of the four read-labelled transitions.
            constraints = []
            for q in (1, 2):
                for b in (0, 1):
                    row = np.zeros(2)
                    row[rule[f"S_{q}_{b}"] - 1] += 1
                    row[q - 1] -= 1
                    constraints.append(row)
            dimension = 2 - np.linalg.matrix_rank(constraints)
            self.assertEqual(result["extra_conserved_state_populations"], dimension - 1)
        self.assertEqual(counts, {"I": 256, "S1": 768, "S2": 768, "C": 2304})
        self.assertEqual(exact, 32)
        self.assertEqual(alternating, 256)

    def test_incoming_edge_uses_source_columns(self):
        rule = rule_observables(machine([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]]))
        result = state_structure(rule)
        self.assertEqual(json.loads(result["adjacency_source_columns"]), [[2, 1], [0, 1]])
        self.assertEqual(result["structure_class"], "S1")
        self.assertEqual(result["isolated_proper_components"], "[]")

    def test_opposite_moves_do_not_imply_conserved_sectors(self):
        isolated = rule_observables(machine([[0, 1, 1], [0, 1, 1], [0, 0, 2], [0, 0, 2]]))
        connected = rule_observables(machine([[0, 1, 2], [0, 1, 2], [0, 0, 1], [0, 0, 1]]))
        self.assertTrue(state_structure(isolated)["exact_counterpropagating_sectors"])
        self.assertFalse(state_structure(connected)["exact_counterpropagating_sectors"])
        self.assertTrue(state_structure(connected)["strict_alternation"])

    def test_velocity_threshold_boundaries(self):
        result = velocity_statistics(np.array([-1, -0.01, 0, 0.01, 1]), 0.01, 0.2)
        self.assertEqual((result["left_count"], result["near_zero_count"], result["right_count"]), (1, 3, 1))
        self.assertTrue(result["counterflow_flag"])
        self.assertEqual(result["velocity_fit_mean"], 0)
        self.assertFalse(velocity_statistics(np.array([-1, 0, 0, 0, 1]), 0.01, 0.21)["counterflow_flag"])

    def test_bad_velocities(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                velocity_statistics(np.array(values), 0.01, 0.05)


if __name__ == "__main__":
    unittest.main()
