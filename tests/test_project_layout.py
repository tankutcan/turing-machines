"""Check the flat package and the root-based experiment commands."""

import importlib
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    "many_turing_machines",
    "generate_spacetime_zoo",
    "generate_one_state_zoo",
    "coupled_chiral_ants",
    "simulate_busybeaver4",
    "sample_halt_times",
    "sample_random_tape_halt_times",
    "order_two_state_zoo",
    "analyze_two_state_structure",
    "alignment_census",
    "mutual_rewriting",
    "cooperation_graph",
)


class ProjectLayoutTests(unittest.TestCase):
    def test_launchers_use_package_implementations(self):
        for name in COMMANDS:
            with self.subTest(command=name):
                launcher = importlib.import_module(f"scripts.{name}")
                implementation = importlib.import_module(f"turing_machines.{name}")
                self.assertIs(launcher.main, implementation.main)

    def test_all_commands_offer_help_from_project_root(self):
        for name in COMMANDS:
            with self.subTest(command=name):
                result = subprocess.run(
                    [sys.executable, "-m", f"scripts.{name}", "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
