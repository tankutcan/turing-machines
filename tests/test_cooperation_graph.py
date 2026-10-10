"""Independent execution and archive checks for the exhaustive game graph."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from turing_machines.cooperation_graph import (
    all_actions, analyze, components, constant_strategy_responses, graph_adjacency,
    payoff_ranking, program_bits,
)


def scalar_action(reader, opponent):
    # Decode via binary text rather than the vector evaluator's bit shifts.
    code = list(map(int, f"{reader:08b}"[::-1]))
    tape = list(map(int, f"{opponent:08b}"[::-1]))
    rules = {(q, b): (code[4 * (q - 1) + 2 * b],
                       1 + code[4 * (q - 1) + 2 * b + 1])
             for q in (1, 2) for b in (0, 1)}
    state = 1
    for bit in tape:
        action, state = rules[state, bit]
    return action


class CooperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions = all_actions()

    def test_every_decision_matches_scalar_execution(self):
        expected = [[scalar_action(i, j) for j in range(256)] for i in range(256)]
        np.testing.assert_array_equal(self.actions, expected)

    def test_constant_outputs_and_copy_read_bit(self):
        # Constant writers: all next-state choices leave the output unchanged.
        for state_bits in range(16):
            base = sum(((state_bits >> k) & 1) << (2 * k + 1) for k in range(4))
            self.assertTrue(np.all(self.actions[base] == 0))
            self.assertTrue(np.all(self.actions[base | 85] == 1))
        # Tape 00100010 in read order: copies each read symbol, stays in state 1.
        np.testing.assert_array_equal(self.actions[68], np.arange(256) >> 7)
        self.assertEqual(program_bits(68), [0, 0, 1, 0, 0, 0, 1, 0])

    def test_output_complement_symmetry(self):
        # Flipping all W bits preserves the state trajectory on a fixed input.
        # Exactly half the ensemble cooperates with every fixed opponent.
        np.testing.assert_array_equal(self.actions[np.arange(256) ^ 85], 1 - self.actions)
        np.testing.assert_array_equal(self.actions.sum(axis=0), np.full(256, 128))

    def test_edges_encode_all_four_outcomes(self):
        # Off-diagonal blocks: CC, CD, DC, DD respectively.
        actions = np.zeros((8, 8), dtype=np.uint8)
        actions[0, 1] = actions[1, 0] = actions[2, 3] = actions[5, 4] = 1
        np.fill_diagonal(actions, 1)
        graph = graph_adjacency(actions)
        self.assertEqual(set(zip(*np.nonzero(graph))), {(0, 1), (1, 0), (2, 3), (5, 4)})
        self.assertTrue(np.all(actions.diagonal() == 1))

    def test_summary_bookkeeping(self):
        s, rows = analyze(self.actions)
        outcomes = s["distinct_pair_outcomes"]
        self.assertEqual(sum(outcomes.values()), 32640)
        self.assertEqual(s["directed_edges_excluding_self_loops"],
                         2 * outcomes["CC"] + outcomes["CD"] + outcomes["DC"])
        self.assertEqual(sum(row["in_degree"] for row in rows),
                         sum(row["out_degree"] for row in rows))
        self.assertEqual(sorted(sum(s["strongly_connected_components"], [])), list(range(256)))
        self.assertEqual(s["self_play_outcomes"]["CC"] + s["self_play_outcomes"]["DD"], 256)

    def test_payoffs_for_each_action_pair_ignore_self_play(self):
        for x, y, expected in ((0, 0, (1, 1)), (0, 1, (5, 0)),
                               (1, 0, (0, 5)), (1, 1, (3, 3))):
            rows = payoff_ranking(np.array([[1, x], [y, 1]]))
            self.assertEqual(tuple(row["total_payoff"] for row in rows), expected)
            self.assertEqual(tuple(row["mean_payoff"] for row in rows), expected)

    def test_ranking_matches_graph_formula_and_defector_bound(self):
        rows = payoff_ranking(self.actions)
        graph = graph_adjacency(self.actions).astype(int)
        formula = 255 + 4 * graph.sum(axis=0) - graph.sum(axis=1) - (graph & graph.T).sum(axis=1)
        np.testing.assert_array_equal([row["total_payoff"] for row in rows], formula)
        winners = [row for row in rows if row["payoff_rank"] == 1]
        self.assertEqual(len(winners), 34)
        for row in winners:
            self.assertEqual((row["games_DC"], row["games_DD"], row["games_CC"], row["games_CD"]),
                             (128, 127, 0, 0))
            self.assertEqual(row["total_payoff"], 767)
        chosen = [row["machine_id"] for row in winners]
        self.assertFalse(self.actions[np.ix_(chosen, chosen)].any())

    def test_strong_components_distinguish_direction(self):
        a = np.zeros((5, 5), dtype=np.uint8)
        a[0, 1] = a[1, 0] = a[1, 2] = a[2, 3] = a[3, 2] = 1
        self.assertEqual(components(a), [[0, 1], [2, 3], [4]])

    def test_constant_strategy_search_checks_all_targets(self):
        summary, rows = constant_strategy_responses(self.actions)
        defectors, cooperators = summary["always_defector_ids"], summary["always_cooperator_ids"]
        self.assertEqual((len(defectors), len(cooperators)), (34, 34))
        # Re-evaluate both target classes independently for every candidate.
        errors = [sum(scalar_action(i, j) for j in defectors)
                  + sum(1 - scalar_action(i, j) for j in cooperators) for i in range(256)]
        self.assertEqual([row["constant_strategy_errors"] for row in rows], errors)
        self.assertEqual(summary["exact_match_ids"], [])
        self.assertEqual(summary["best_match_ids"], [7])
        self.assertEqual(summary["minimum_errors"], 14)
        self.assertEqual((rows[7]["defects_against_always_defectors"],
                          rows[7]["cooperates_with_always_cooperators"]), (24, 30))
        literal = summary["literal_constant_output_rules"]
        self.assertEqual((len(literal["defector_ids"]), len(literal["cooperator_ids"])), (16, 16))
        self.assertEqual(literal["exact_match_ids"], [7, 39, 88, 216])
        # A simple rule-bit reader suffices for the literal classes, but not the behavioral ones.
        np.testing.assert_array_equal(self.actions[39], (np.arange(256) >> 6) & 1)

    def test_machine_seven_reads_parity_of_trailing_zeros_before_last_cell(self):
        for opponent in range(256):
            trailing_zeros = 0
            for bit in program_bits(opponent)[6::-1]:
                if bit:
                    break
                trailing_zeros += 1
            self.assertEqual(int(self.actions[7, opponent]), int(trailing_zeros % 2 == 0))

    def test_exact_payoff_distribution(self):
        summary, records = analyze(self.actions)
        distribution = summary["payoffs"]["distribution"]
        self.assertEqual(sum(row["machines"] for row in distribution), 256)
        self.assertEqual(sum(row["probability"] for row in distribution), 1)
        self.assertEqual(len(distribution), 69)
        self.assertEqual(sum(row["total_payoff"] * row["machines"] for row in distribution),
                         sum(row["total_payoff"] for row in records))
        self.assertAlmostEqual(summary["payoffs"]["population_mean"],
                               np.mean([row["mean_payoff"] for row in records]))

    def test_archive_round_trip_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-m", "scripts.cooperation_graph", "--output-dir", directory]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = Path(directory)
            archive = np.load(output / "adjacency.npz")
            np.testing.assert_array_equal(archive["actions"], self.actions)
            graph = archive["adjacency"]
            edges = np.zeros_like(graph)
            with (output / "edges.csv").open() as handle:
                for row in csv.DictReader(handle):
                    edges[int(row["source"]), int(row["target"])] = 1
            np.testing.assert_array_equal(graph, edges)
            ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
            root = ET.parse(output / "cooperation.graphml")
            self.assertEqual(len(root.findall(".//g:node", ns)), 256)
            self.assertEqual(len(root.findall(".//g:edge", ns)), int(graph.sum()))
            with (output / "games.csv").open() as handle:
                games = list(csv.DictReader(handle))
            self.assertEqual(len(games), 32896)
            for row in games:
                i, j = int(row["machine_i"]), int(row["machine_j"])
                self.assertEqual((int(row["action_i"]), int(row["action_j"])),
                                 (self.actions[i, j], self.actions[j, i]))
            metadata = json.loads((output / "summary.json").read_text())
            self.assertEqual(metadata["summary"]["directed_edges_excluding_self_loops"], int(graph.sum()))
            with (output / "payoff_ranking.csv").open() as handle:
                ranking = list(csv.DictReader(handle))
            self.assertEqual(len(ranking), 256)
            self.assertEqual([int(row["total_payoff"]) for row in ranking],
                             sorted((int(row["total_payoff"]) for row in ranking), reverse=True))
            with (output / "best_machines.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 34)
            with (output / "payoff_distribution.csv").open() as handle:
                distribution = list(csv.DictReader(handle))
            self.assertEqual(sum(int(row["machines"]) for row in distribution), 256)
            with (output / "constant_strategy_responses.csv").open() as handle:
                responses = list(csv.DictReader(handle))
            self.assertEqual((len(responses), int(responses[0]["machine_id"])), (256, 7))
            for name in ("payoff_distribution", "constant_strategy_responses"):
                for extension in ("png", "pdf"):
                    self.assertGreater((output / f"{name}.{extension}").stat().st_size, 1000)
            self.assertNotIn("__GRAPH_DATA__", (output / "index.html").read_text())
            again = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("not empty", again.stderr)


if __name__ == "__main__":
    unittest.main()
