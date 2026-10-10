"""Checks of rule averages, encoding and complete two-state bin degeneracies."""

from collections import Counter
from itertools import product
import json
from math import comb
import unittest

from turing_machines.order_two_state_zoo import grouped_records, rule_observables


def machine(entries):
    return {1: {0: entries[0], 1: entries[1]}, 2: {0: entries[2], 1: entries[3]}}


class RuleObservablesTests(unittest.TestCase):
    def test_extremes_and_balanced(self):
        for write, direction in product((0, 1), repeat=2):
            result = rule_observables(machine([[write, direction, 1]] * 4))
            self.assertEqual(result["mean_write_bit"], write)
            self.assertEqual(result["mean_move"], 2 * direction - 1)
        result = rule_observables(machine([[0, 0, 1], [1, 1, 2]] * 2))
        self.assertEqual(result["mean_write_bit"], 0.5)
        self.assertEqual(result["mean_move"], 0)

    def test_json_keys_and_next_states(self):
        tm = machine([[0, 1, 1], [1, 0, 2], [1, 1, 1], [1, 1, 2]])
        result = rule_observables(tm)
        self.assertEqual(result, rule_observables(json.loads(json.dumps(tm))))
        self.assertEqual(result["mean_move"], 0.5)
        self.assertEqual(result["mean_write_bit"], 0.75)
        other = rule_observables(machine([[w, d, 3 - s] for row in tm.values() for w, d, s in row.values()]))
        self.assertEqual(result["mean_move"], other["mean_move"])
        self.assertEqual(result["mean_write_bit"], other["mean_write_bit"])

    def test_all_4096_machines(self):
        counts = Counter()
        transitions = list(product((0, 1), (0, 1), (1, 2)))
        for entries in product(transitions, repeat=4):
            result = rule_observables(machine(entries))
            counts[result["write_one_entries"], result["move_right_entries"]] += 1
            self.assertEqual(result["mean_move"], sum(2 * e[1] - 1 for e in entries) / 4)
        self.assertEqual(sum(counts.values()), 4096)
        for w, d in product(range(5), repeat=2):
            self.assertEqual(counts[w, d], 16 * comb(4, w) * comb(4, d))

    def test_bad_tables_rejected(self):
        for tm in ({1: {0: [0, 0, 1], 1: [1, 1, 1]}},
                   machine([[0, -1, 1]] * 4), machine([[0, 0, 3]] * 4),
                   machine([[0, 0]] * 4)):
            with self.assertRaises(ValueError):
                rule_observables(tm)

    def test_grouping_keeps_all_seeds_in_order(self):
        records = [{"seed": seed, **rule_observables(machine([[1, 0, 1]] * 4))}
                   for seed in (19, 2, 7)]
        self.assertEqual([r["seed"] for r in grouped_records(records)[4, 0]], [2, 7, 19])


if __name__ == "__main__":
    unittest.main()
