#!/usr/bin/env python3
"""Test halt insertions in a cyclic sequence of bit-flipping chiral TM modes.

One head, initial state 1, initial position 0, infinite blank binary tape.
L: 0 -> (1,left), 1 -> (0,right). R reverses these move directions.
Rules use the project's [write_bit, move_bit, next_state] convention, with
move_bit=0/1 for left/right. This standalone simulator reserves next_state=0
for Halt; many_turing_machines.py does not support that additional convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def make_chiral_tm(word: str) -> dict:
    if not word or any(c not in "LR" for c in word):
        raise ValueError("The chirality word must be a nonempty string of L and R")
    return {q: {b: [1 - b, b if chirality == "L" else 1 - b, q % len(word) + 1]
                for b in (0, 1)} for q, chirality in enumerate(word, 1)}


def with_halt(tm: dict, state: int, read: int) -> dict:
    altered = {q: {b: entry.copy() for b, entry in row.items()} for q, row in tm.items()}
    altered[state][read][2] = 0
    return altered


def simulate(tm: dict, steps: int) -> dict:
    """Sparse infinite-tape simulation; include the write/move into Halt."""
    ones, position, state = set(), 0, 1
    trace, snapshots, first_use = [], [(0, position, state, frozenset(ones))], {}
    for step in range(1, steps + 1):
        read = int(position in ones)
        write, move_bit, next_state = tm[state][read]
        move = 2 * move_bit - 1
        first_use.setdefault((state, read), step)
        if write:
            ones.add(position)
        else:
            ones.discard(position)
        trace.append({"step": step, "state_before": state, "position_before": position,
                      "read": read, "write": write, "move": move,
                      "state_after": next_state, "position_after": position + move,
                      "ones_after": len(ones)})
        position, state = position + move, next_state
        snapshots.append((step, position, state, frozenset(ones)))
        if state == 0:
            break
    return {"halted": state == 0, "steps": len(trace), "ones": len(ones),
            "ones_positions": sorted(ones), "final_position": position,
            "trace": trace, "snapshots": snapshots, "first_use": first_use}


def evaluate(word: str, steps: int) -> tuple[dict, dict, list[dict], dict]:
    base = make_chiral_tm(word)
    parent = simulate(base, steps)
    rows, variants = [], {}
    for q in base:
        for b in (0, 1):
            result = simulate(with_halt(base, q, b), steps)
            expected = parent["first_use"].get((q, b))
            assert result["halted"] == (expected is not None)
            if expected is not None:
                assert result["steps"] == expected
            variants[q, b] = result
            rows.append({"halt_state": q, "halt_read": b,
                         "status": "halted" if result["halted"] else "not_halted_within_budget",
                         "halt_step": result["steps"] if result["halted"] else "",
                         "ones_at_halt": result["ones"] if result["halted"] else "",
                         "first_use_in_parent": expected if expected is not None else ""})
    return base, parent, rows, variants


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_run(result: dict, word: str, halt_rule: tuple, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    snapshots = result["snapshots"]
    visited = [s[1] for s in snapshots]
    written = [x for s in snapshots for x in s[3]]
    low, high = min(visited + written) - 1, max(visited + written) + 1
    tape = np.array([[int(x in s[3]) for x in range(low, high + 1)] for s in snapshots])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(tape, origin="lower", interpolation="nearest", aspect="auto",
              extent=(low - 0.5, high + 0.5, -0.5, result["steps"] + 0.5),
              cmap=ListedColormap(["#171717", "#f3f3f3"]), vmin=0, vmax=1)
    ax.plot(visited, [s[0] for s in snapshots], color="#e44d3b", marker="o",
            markerfacecolor="none", markersize=9, linewidth=1.5)
    ax.set_xlabel("Tape position x")
    ax.set_ylabel("Time (completed transitions)")
    ax.set_xticks(range(low, high + 1))
    tick_indices = np.unique(np.linspace(0, len(snapshots) - 1, min(30, len(snapshots)), dtype=int))
    ax.set_yticks(tick_indices)
    states = ax.secondary_yaxis("right")
    states.set_yticks(tick_indices, labels=["Halt" if snapshots[i][2] == 0
                                         else str(snapshots[i][2]) for i in tick_indices])
    states.set_ylabel("Head state after transition")
    ax.set_title(f"Cyclic chiral modes {word}: halt at state {halt_rule[0]}, read {halt_rule[1]}\n"
                 f"{len(word)} working states; 1 head; blank infinite tape; initial q=1, x=0\n"
                 f"{result['steps']} steps; {result['ones']} final ones | light=1, dark=0, red=head",
                 fontsize=11)
    fig.savefig(output / "spacetime_best.pdf")
    fig.savefig(output / "spacetime_best.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chiralities", default="LRLR")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    try:
        base, parent, rows, variants = evaluate(args.chiralities, args.steps)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir or Path("results") / f"coupled_chiral_ants_{args.chiralities}"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "halt_placements.csv", rows)
    write_csv(output / "parent_trace.csv", parent["trace"])
    machines = {"non_halting_parent": base}
    for q, b in variants:
        machines[f"halt_q{q}_read{b}"] = with_halt(base, q, b)
        write_csv(output / f"trace_halt_q{q}_read{b}.csv", variants[q, b]["trace"])
    (output / "transition_tables.json").write_text(json.dumps(machines, indent=2) + "\n")
    halted = [(key, result) for key, result in variants.items() if result["halted"]]
    best_rule, best = max(halted, key=lambda pair: pair[1]["steps"])
    best_keys = [key for key, result in halted if result["steps"] == best["steps"]]
    summary = {
        "generator": Path(__file__).name,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "chiralities": args.chiralities, "states": len(base), "heads": 1,
        "initial_state": 1, "initial_position": 0, "initial_tape": "all zero",
        "boundary": "unbounded", "randomness": "none", "simulation_budget": args.steps,
        "rule_encoding": "[write_bit, move_bit, next_state]; move_bit 0=left, 1=right; next_state 0=Halt",
        "step_convention": "one read/write/move/state-change, including the transition into Halt",
        "all_halt_placements_resolved": len(halted) == len(variants),
        "best_observed_runtime": best["steps"], "best_runtime_halt_rules": best_keys,
        "best_runtime_final_ones": best["ones"], "best_runtime_ones_positions": best["ones_positions"],
        "maximum_observed_ones_at_halt": max(r["ones"] for _, r in halted),
        "placements": rows,
    }
    if len(base) == 4:
        summary["known_busy_beaver_bounds"] = {
            "steps": 107, "ones": 13, "source": "https://bbchallenge.org/~pascal.michel/bbc"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_run(best, args.chiralities, best_rule, output)
    lines = [f"# Cyclic chiral ants: {args.chiralities}", "",
             "One head on an unbounded all-zero tape, initially at x=0 in state 1. "
             "Every rule flips the bit. L moves left on 0 and right on 1; R does the reverse. "
             "The working state advances cyclically regardless of the read bit.", "",
             "Each experiment replaces exactly one next-state entry with Halt, retaining its flip and move. "
             "One complete transition counts as one step, including the transition into Halt. "
             "All results are deterministic; there is no random seed.", "",
             "## Parent transition rules", "", "| State | Chirality | Read 0: write, move, next | Read 1: write, move, next |",
             "| --- | --- | --- | --- |"]
    for q, row in base.items():
        entries = [f"{w}, {'right' if d else 'left'}, {s}" for w, d, s in row.values()]
        lines.append(f"| {q} | {args.chiralities[q-1]} | {entries[0]} | {entries[1]} |")
    lines += ["", "## All halt placements", "", "| State | Read | Halt step | Final ones | Status |",
              "| --- | --- | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['halt_state']} | {row['halt_read']} | {row['halt_step']} | "
                     f"{row['ones_at_halt']} | {row['status']} |")
    lines += ["", f"Best observed runtime: **{best['steps']} steps**, by halting at {best_keys}. "
              f"The plotted choice leaves **{best['ones']} ones**, at {best['ones_positions']}.", "",
              "Every placement halts, so this comparison is exhaustive for this cycle."
              if summary["all_halt_placements_resolved"] else
              "Some placements remain unresolved within the simulation budget; do not interpret the observed maximum as a proof of optimality.", ""]
    if len(base) == 4:
        lines += ["The established four-state, binary Busy Beaver maxima are **107 steps** and **13 ones**. "
                  "[Source: Pascal Michel](https://bbchallenge.org/~pascal.michel/bbc)", ""]
    if args.chiralities == "LRLR":
        lines += ["Before adding Halt, identifying states 1 and 3 as L and states 2 and 4 as R "
                  "reduces this machine exactly to the two-state LR model. Its tape and head trajectory are "
                  "identical at every time; the extra labels track the phase modulo four. A halt can distinguish "
                  "those labels, delaying its first encounter to step 13.", ""]
    lines += ["## Files and reproduction", "", "[Winning spacetime PDF](spacetime_best.pdf) · "
              "[PNG](spacetime_best.png) · [All rules](transition_tables.json) · "
              "[Halt scores](halt_placements.csv) · [Summary](summary.json)", "",
              "`parent_trace.csv` contains the non-halting parent run. `trace_halt_q*_read*.csv` "
              "contains the complete trace for each halt placement up to halting or the budget.", "",
              "From the project root:", "", "```sh",
              "env MPLCONFIGDIR=/tmp/turing-machines-matplotlib .venv/bin/python -m scripts.coupled_chiral_ants "
              f"--chiralities {args.chiralities} --steps {args.steps}", "```", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
