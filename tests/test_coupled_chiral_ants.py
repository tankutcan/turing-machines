import unittest

from turing_machines.coupled_chiral_ants import evaluate, make_chiral_tm, simulate, with_halt


class ChiralAntTests(unittest.TestCase):
    def test_four_state_scores(self):
        _, _, rows, runs = evaluate("LRLR", 32)
        expected = {(1, 0): (1, 1), (1, 1): (13, 3), (2, 0): (2, 2), (2, 1): (6, 2),
                    (3, 0): (7, 3), (3, 1): (3, 1), (4, 0): (4, 2), (4, 1): (8, 2)}
        self.assertEqual(len(rows), 8)
        for key, (steps, ones) in expected.items():
            self.assertTrue(runs[key]["halted"])
            self.assertEqual((runs[key]["steps"], runs[key]["ones"]), (steps, ones))
        self.assertEqual(runs[1, 1]["ones_positions"], [-3, -1, 2])

    def test_two_state_reference(self):
        result = simulate(with_halt(make_chiral_tm("LR"), 2, 1), 20)
        self.assertEqual((result["steps"], result["ones"]), (6, 2))
        self.assertEqual(result["ones_positions"], [-1, 2])

    def test_identical_modes_and_censored_outcome(self):
        _, _, rows, _ = evaluate("LL", 20)
        self.assertEqual([row["halt_step"] for row in rows], [1, "", 2, ""])
        self.assertEqual(rows[1]["status"], "not_halted_within_budget")

    def test_parent_reduction_and_spatial_reflection(self):
        two = simulate(make_chiral_tm("LR"), 256)
        four = simulate(make_chiral_tm("LRLR"), 256)
        mirror = simulate(make_chiral_tm("RLRL"), 256)
        for a, b, c in zip(two["snapshots"], four["snapshots"], mirror["snapshots"]):
            self.assertEqual((a[0], a[1], a[3]), (b[0], b[1], b[3]))
            self.assertEqual(a[2], (b[2] - 1) % 2 + 1)
            self.assertEqual(c[1], -b[1])
            self.assertEqual(c[3], frozenset(-x for x in b[3]))

    def test_halt_only_changes_one_target(self):
        base = make_chiral_tm("LRLR")
        altered = with_halt(base, 1, 1)
        self.assertEqual(base[1][1], [0, 1, 2])
        self.assertEqual(altered[1][1], [0, 1, 0])
        for q in base:
            for b in base[q]:
                if (q, b) != (1, 1):
                    self.assertEqual(base[q][b], altered[q][b])

    def test_invalid_words(self):
        for word in ("", "LX", "lr"):
            with self.assertRaises(ValueError):
                make_chiral_tm(word)


if __name__ == "__main__":
    unittest.main()
