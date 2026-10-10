#!/usr/bin/env python3
"""Census of asymptotic state alignment in identical-table Turing-machine gases.

N identical heads share a periodic binary tape.  Updates are synchronous with
majority writes (ties keep the old bit), as in ``many_turing_machines.py``.
A gas is "aligned" when at least ``--threshold`` of its heads occupy a single
internal state at every step of the late window, in every replicate started
from random internal states.  One further run starts all heads in state 1.

Omit ``--samples`` to enumerate every labeled table (one or two states only);
otherwise that many tables are drawn with ``make_tm``.  Tables use the project
convention ``tm[state][read_bit] = [write_bit, move_bit, next_state]``.

Each run starts from uniformly random head positions, unlike the evenly spaced
zoo layout.  Results are finite-time, finite-size classifications, not proofs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .many_turing_machines import make_tm, table_arrays

MAX_ENUMERATED_TABLES = 4096
UNIFORM_TAPE = 0.01      # m <= this, or >= 1 - this, counts as a uniform tape
NEAR_HALF_TAPE = 0.1     # |m - 1/2| <= this counts as near one half
ENTRY_IN_USE = 1e-6      # late-window occupation above this marks an entry in use


def table_space_size(n_states: int) -> int:
    return (4 * n_states) ** (2 * n_states)


def table_from_index(index: int, n_states: int) -> dict[int, dict[int, list[int]]]:
    """Bijection from 0..(4n)^(2n)-1 to labeled tables.

    Base-4n digits, low digit first, in entry order (1,0),(1,1),(2,0),...
    Each digit is write*2n + move_bit*n + (next_state-1).
    """
    if not 0 <= index < table_space_size(n_states):
        raise ValueError("table index outside the labeled table space")
    tm: dict[int, dict[int, list[int]]] = {}
    for state in range(1, n_states + 1):
        tm[state] = {}
        for read_bit in (0, 1):
            index, digit = divmod(index, 4 * n_states)
            tm[state][read_bit] = [digit // (2 * n_states), (digit // n_states) % 2,
                                   digit % n_states + 1]
    return tm


def index_from_table(tm: dict[int, dict[int, list[int]]]) -> int:
    n_states = len(tm)
    index = 0
    for state in range(n_states, 0, -1):
        for read_bit in (1, 0):
            write_bit, move_bit, next_state = tm[state][read_bit]
            digit = write_bit * 2 * n_states + move_bit * n_states + next_state - 1
            index = index * 4 * n_states + digit
    return index


def static_structure(tm: dict[int, dict[int, list[int]]]) -> dict[str, Any]:
    """Tape-independent properties of the table; states are numbered from 1.

    A fixed entry (q,b) has S_q(b)=q and W_q(b)=b.  "Synchronizing" means some
    common read word sends every state to one state; "weakly mergeable" lets
    each head read its own word of the same length.
    """
    n = len(tm)
    states = range(1, n + 1)
    S = {q: (tm[q][0][2], tm[q][1][2]) for q in states}
    absorbing = [q for q in states if S[q] == (q, q)]
    fixed = [[q, b] for q in states for b in (0, 1)
             if tm[q][b][2] == q and tm[q][b][0] == b]

    reach = {q: {q, *S[q]} for q in states}
    for k in states:
        for q in states:
            if k in reach[q]:
                reach[q] |= reach[k]

    def all_pairs_merge(common_word: bool) -> bool:
        merged = {(q, q) for q in states}
        symbols = [(a, a) for a in (0, 1)] if common_word else [
            (a, b) for a in (0, 1) for b in (0, 1)]
        grew = True
        while grew:
            grew = False
            for p in states:
                for q in states:
                    if (p, q) not in merged and any(
                            (S[p][a], S[q][b]) in merged for a, b in symbols):
                        merged.add((p, q))
                        grew = True
        return len(merged) == n * n

    # Reversible single head: (q,b) -> (S,W) is a bijection and the move
    # direction is a function of the target state.
    images = {(tm[q][b][2], tm[q][b][0]) for q in states for b in (0, 1)}
    one_direction = all(
        len({tm[q][b][1] for q in states for b in (0, 1) if tm[q][b][2] == target}) <= 1
        for target in states)
    return {
        "absorbing_states": absorbing,
        "reachable_absorbing_states": [
            q for q in absorbing if all(q in reach[p] for p in states)],
        "fixed_entries": fixed,
        "symbol_blind": all(S[q][0] == S[q][1] for q in states),
        "synchronizing": all_pairs_merge(True),
        "weakly_mergeable": all_pairs_merge(False),
        "reversible": len(images) == 2 * n and one_direction,
    }


def simulate_gas(
    tm: dict[int, dict[int, list[int]]],
    *,
    tape_length: int,
    copies: int,
    steps: int,
    seed: Any,
    initial_state_mode: str = "random",
    one_density: float = 0.5,
    late_fraction: float = 0.25,
    threshold: float = 0.99,
    initial: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Evolve one gas and measure its late-time state alignment.

    ``initial`` optionally supplies (tape, positions, states) with states
    numbered from 1; otherwise they are drawn from ``seed`` in that order.
    The entry occupation is ordered (1,0),(1,1),(2,0),... and averaged over
    heads and over the late window.
    """
    write_table, move_table, state_table = table_arrays(tm)
    write_table = write_table.astype(np.int64)
    move_table = move_table.astype(np.int64)
    next_index = state_table.astype(np.int64) - 1
    n = len(tm)
    if initial is None:
        rng = np.random.default_rng(seed)
        tape = (rng.random(tape_length) < one_density).astype(np.int64)
        positions = rng.integers(0, tape_length, copies)
        if initial_state_mode == "random":
            states = rng.integers(0, n, copies)
        elif initial_state_mode == "fixed":
            states = np.zeros(copies, dtype=np.int64)
        else:
            raise ValueError(f"unknown initial state mode: {initial_state_mode}")
    else:
        tape = np.asarray(initial[0], dtype=np.int64).copy()
        positions = np.asarray(initial[1], dtype=np.int64) % tape_length
        states = np.asarray(initial[2], dtype=np.int64) - 1

    late = steps - max(1, int(round(steps * late_fraction)))
    alignment = np.empty(steps)
    dominant = np.empty(steps, dtype=np.int64)
    occupation = np.zeros(2 * n)
    flips = 0
    net_moves = 0
    for step in range(steps):
        populations = np.bincount(states, minlength=n)
        dominant[step] = populations.argmax()
        alignment[step] = populations[dominant[step]] / copies

        read_bits = tape[positions]
        proposed = write_table[states, read_bits]
        moves = move_table[states, read_bits]
        counts = np.bincount(positions, minlength=tape_length)
        ones = np.bincount(positions, weights=proposed, minlength=tape_length)
        # Majority vote per occupied cell; ties and empty cells keep the old bit.
        new_tape = np.where(2 * ones > counts, 1, np.where(2 * ones < counts, 0, tape))
        if step >= late:
            occupation += np.bincount(2 * states + read_bits, minlength=2 * n)
            flips += int(np.count_nonzero(new_tape != tape))
            net_moves += int(moves.sum())
        states = next_index[states, read_bits]
        tape = new_tape
        positions = (positions + moves) % tape_length

    window = (steps - late) * copies
    occupation /= window
    below = np.flatnonzero(alignment < threshold)
    late_populations = occupation.reshape(n, 2).sum(axis=1)
    return {
        "alignment_mean": float(alignment[late:].mean()),
        "alignment_min": float(alignment[late:].min()),
        "dominant_state": int(late_populations.argmax()) + 1,
        "dominant_states_late": int(len(np.unique(dominant[late:]))),
        # Equals ``steps`` when the final step is still below threshold.
        "alignment_time": int(below[-1] + 1) if len(below) else 0,
        "entry_occupation": [round(float(value), 6) for value in occupation],
        "entries_in_use": int(np.count_nonzero(occupation > ENTRY_IN_USE)),
        "velocity": net_moves / window,
        "flip_rate": flips / window,
        "tape_density": float(tape.mean()),
        "occupied_cells": int(len(np.unique(positions * n + states))),
        "final_tape": tape,
        "final_positions": positions,
        "final_states": states + 1,
    }


def alignment_level(run: dict[str, Any], threshold: float) -> str:
    if run["alignment_min"] >= 1.0:
        return "exact"
    return "near" if run["alignment_min"] >= threshold else "none"


def tape_class(density: float) -> str:
    if density <= UNIFORM_TAPE or density >= 1 - UNIFORM_TAPE:
        return "uniform"
    return "near_half" if abs(density - 0.5) <= NEAR_HALF_TAPE else "intermediate"


def entry_pattern(run: dict[str, Any]) -> str:
    """Which table entries remain in use, up to relabeling."""
    used = np.asarray(run["entry_occupation"]).reshape(-1, 2) > ENTRY_IN_USE
    count = int(used.sum())
    if count != 2:
        return f"{count} entries" if count != 1 else "1 entry"
    if used.any(axis=1).sum() == 1:
        return "2 entries, same state"
    return "2 entries, same symbol" if used.any(axis=0).sum() == 1 else "2 entries, diagonal"


def alignment_mechanism(structure: dict[str, Any], run: dict[str, Any]) -> str:
    """How an aligned gas is held in its dominant state."""
    if run["dominant_states_late"] > 1:
        return "cycling"
    state = run["dominant_state"]
    if state in structure["absorbing_states"]:
        return "absorbing_state"
    used = np.asarray(run["entry_occupation"]).reshape(-1, 2) > ENTRY_IN_USE
    entries = [[int(q) + 1, int(b)] for q, b in zip(*np.nonzero(used))]
    if len(entries) == 1 and entries[0] in structure["fixed_entries"]:
        return "fixed_entry"
    return "other"


RUN_FIELDS = ("alignment_mean", "alignment_min", "dominant_state", "dominant_states_late",
              "alignment_time", "entries_in_use", "entry_occupation", "velocity",
              "flip_rate", "tape_density", "occupied_cells")


def census_row(job: tuple[int, int, dict[str, Any]]) -> tuple[dict[str, Any], dict]:
    """Classify one table.  Seeds depend only on (seed, table_id, replicate)."""
    n_states, table_id, p = job
    if p["samples"] is None:
        tm = table_from_index(table_id, n_states)
    else:
        tm = make_tm(n_states, np.random.default_rng([p["seed"], 0, table_id]))
    structure = static_structure(tm)
    common = dict(tape_length=p["tape_length"], copies=p["copies"], steps=p["steps"],
                  one_density=p["one_density"], late_fraction=p["late_fraction"],
                  threshold=p["threshold"])
    runs = {f"random{r + 1}": simulate_gas(tm, seed=[p["seed"], 1, table_id, r], **common)
            for r in range(p["replicates"])}
    runs["fixed"] = simulate_gas(tm, seed=[p["seed"], 2, table_id],
                                 initial_state_mode="fixed", **common)
    first = runs["random1"]
    levels = [alignment_level(runs[f"random{r + 1}"], p["threshold"])
              for r in range(p["replicates"])]
    aligned = all(level != "none" for level in levels)
    row: dict[str, Any] = {
        "table_id": table_id, "states": n_states,
        **{key: json.dumps(value) if isinstance(value, list) else value
           for key, value in structure.items()},
        "aligned": aligned,
        "exactly_aligned": all(level == "exact" for level in levels),
        "mechanism": alignment_mechanism(structure, first) if aligned else "",
        "entry_pattern": entry_pattern(first),
        "tape_class": tape_class(first["tape_density"]),
        "fixed_start_stays_aligned": runs["fixed"]["alignment_min"] >= 1.0,
    }
    for label, run in runs.items():
        for field in RUN_FIELDS:
            value = run[field]
            row[f"{label}_{field}"] = json.dumps(value) if isinstance(value, list) else value
    return row, tm


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts behind the report; conditional entries are [aligned, total]."""
    aligned = [row for row in rows if row["aligned"]]

    def has(row: dict[str, Any], key: str) -> bool:
        return json.loads(row[key]) != []

    groups = {
        "reachable_absorbing_state": [r for r in rows if has(r, "reachable_absorbing_states")],
        "any_absorbing_state": [r for r in rows if has(r, "absorbing_states")],
        "fixed_entry_without_absorbing_state": [
            r for r in rows if has(r, "fixed_entries") and not has(r, "absorbing_states")],
        "neither": [r for r in rows
                    if not has(r, "fixed_entries") and not has(r, "absorbing_states")],
        "synchronizing": [r for r in rows if r["synchronizing"]],
        "not_synchronizing": [r for r in rows if not r["synchronizing"]],
        "not_weakly_mergeable": [r for r in rows if not r["weakly_mergeable"]],
        "symbol_blind": [r for r in rows if r["symbol_blind"]],
        "reversible": [r for r in rows if r["reversible"]],
    }
    times = sorted(row["random1_alignment_time"] for row in aligned)
    fraction = len(aligned) / len(rows)
    return {
        "tables": len(rows),
        "aligned": len(aligned),
        "aligned_fraction": fraction,
        # Binomial error for sampled tables; an exhaustive census has no sampling error.
        "aligned_fraction_standard_error": float(np.sqrt(fraction * (1 - fraction) / len(rows))),
        "exactly_aligned": sum(row["exactly_aligned"] for row in rows),
        "fixed_start_stays_aligned": sum(row["fixed_start_stays_aligned"] for row in rows),
        "aligned_and_fixed_start_stays_aligned": sum(
            row["fixed_start_stays_aligned"] for row in aligned),
        "mechanisms": dict(Counter(row["mechanism"] for row in aligned)),
        "aligned_by_structure": {name: [sum(r["aligned"] for r in group), len(group)]
                                 for name, group in groups.items()},
        "aligned_by_tape_class": {
            name: [sum(r["aligned"] for r in rows if r["tape_class"] == name),
                   sum(r["tape_class"] == name for r in rows)]
            for name in ("uniform", "intermediate", "near_half")},
        "aligned_by_entry_pattern": {
            name: [sum(r["aligned"] for r in rows if r["entry_pattern"] == name), count]
            for name, count in sorted(Counter(r["entry_pattern"] for r in rows).items())},
        "alignment_time_quantiles_10_50_90_100": [
            int(np.quantile(times, q, method="lower")) for q in (0.1, 0.5, 0.9, 1.0)
        ] if times else None,
    }


def write_report(output: Path, summary: dict[str, Any], parameters: dict[str, Any]) -> None:
    p = parameters
    source = ("all labeled tables" if p["samples"] is None
              else f"{p['samples']} tables drawn with make_tm, seed {p['seed']}")

    def table(title: str, counts: dict[str, list[int]]) -> list[str]:
        lines = [f"| {title} | Tables | Aligned |", "| --- | ---: | ---: |"]
        for name, (hit, total) in counts.items():
            share = f" ({hit / total:.1%})" if total else ""
            lines.append(f"| {name.replace('_', ' ')} | {total} | {hit}{share} |")
        return lines

    lines = [
        f"# State-alignment census: {p['states']} states", "",
        "[Per-table CSV](census.csv) · [Transition tables](transition_tables.json) · "
        "[Summary and provenance](summary.json)", "",
        f"{summary['tables']} tables ({source}). Each gas has N={p['copies']} identical heads on a "
        f"periodic tape of L={p['tape_length']}, evolved for T={p['steps']} synchronous steps with "
        f"majority writes (ties keep the old bit). Initial tape is Bernoulli({p['one_density']:g}); "
        "initial head positions are uniformly random.", "",
        f"**Aligned** means at least {p['threshold']:.0%} of heads share one state at every step of the "
        f"final {p['late_fraction']:.0%} of the run, in all {p['replicates']} replicates started from "
        "uniformly random states. This is a finite-time, finite-size classification.", "",
        f"Aligned: **{summary['aligned']} of {summary['tables']} "
        f"({summary['aligned_fraction']:.1%})**; exactly aligned (every head): "
        f"{summary['exactly_aligned']}. Starting all heads in state 1 instead, "
        f"{summary['fixed_start_stays_aligned']} tables are exactly aligned in the late window.", "",
        "Mechanisms among aligned tables: "
        + (", ".join(f"{name.replace('_', ' ')} {count}"
                     for name, count in sorted(summary["mechanisms"].items())) or "none") + ".", "",
        "## Alignment by table structure", "",
        *table("Structure", summary["aligned_by_structure"]), "",
        "An absorbing state q has S_q(0)=S_q(1)=q; it is reachable when every state can reach it in "
        "the state graph. A fixed entry (q,b) has S_q(b)=q and W_q(b)=b. Symbol-blind tables have "
        "S_q(0)=S_q(1) for every q. The groups overlap.", "",
        "## Alignment by final tape density", "",
        *table("Final tape (first replicate)", summary["aligned_by_tape_class"]), "",
        f"Uniform: density within {UNIFORM_TAPE:g} of 0 or 1. Near half: within {NEAR_HALF_TAPE:g} "
        "of 0.5. The density is a single snapshot at the final step. A tape is written once per "
        "occupied cell, so one stray head can set the density of a gas condensed into a few stacks.", "",
        "## Alignment by entries still in use", "",
        *table("Late-window entries (first replicate)", summary["aligned_by_entry_pattern"]), "",
        "## Reproduce", "", "```sh", p["reproduce"], "```", "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", type=int, default=2, help="internal states n")
    parser.add_argument("--samples", type=int,
                        help="number of random tables; omit to enumerate all tables")
    parser.add_argument("--copies", type=int, default=400, help="identical heads N")
    parser.add_argument("--tape-length", type=int, default=4000, help="ring length L")
    parser.add_argument("--steps", type=int, default=8000, help="synchronous steps T")
    parser.add_argument("--replicates", type=int, default=2,
                        help="independent runs from random initial states")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--one-density", type=float, default=0.5,
                        help="initial probability of tape bit 1")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="minimum late-window fraction of heads in one state")
    parser.add_argument("--late-fraction", type=float, default=0.25,
                        help="final fraction of the run used for late-time measurements")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.states, args.copies, args.tape_length, args.replicates, args.workers) < 1:
        parser.error("states, copies, tape-length, replicates and workers must be positive")
    if args.steps < 2 or args.seed < 0:
        parser.error("--steps must be at least 2 and --seed nonnegative")
    if not (0.5 < args.threshold <= 1 and 0 < args.late_fraction <= 1
            and 0 <= args.one_density <= 1):
        parser.error("need 0.5 < threshold <= 1, 0 < late-fraction <= 1, 0 <= one-density <= 1")
    if args.samples is None and table_space_size(args.states) > MAX_ENUMERATED_TABLES:
        parser.error(f"{table_space_size(args.states):,} tables is too many to enumerate; "
                     "give --samples")
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")
    output = args.output_dir or Path("results") / f"alignment_census_{args.states}_states"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory is not empty; choose a new --output-dir")

    parameters = {key: getattr(args, key) for key in (
        "states", "samples", "copies", "tape_length", "steps", "replicates", "seed",
        "one_density", "threshold", "late_fraction")}
    count = table_space_size(args.states) if args.samples is None else args.samples
    jobs = [(args.states, table_id, parameters) for table_id in range(count)]
    start_time = time.perf_counter()
    if args.workers == 1:
        results = [census_row(job) for job in jobs]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(census_row, jobs, chunksize=8)
    rows = [row for row, _ in results]

    with (output / "census.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "transition_tables.json").write_text(
        json.dumps({row["table_id"]: tm for row, tm in results}) + "\n")
    summary = summarize(rows)
    parameters["reproduce"] = (
        f".venv/bin/python -m scripts.{Path(__file__).stem} --states {args.states} "
        + (f"--samples {args.samples} " if args.samples is not None else "")
        + f"--copies {args.copies} --tape-length {args.tape_length} --steps {args.steps} "
        f"--replicates {args.replicates} --seed {args.seed} --one-density {args.one_density} "
        f"--threshold {args.threshold} --late-fraction {args.late_fraction} "
        f"--output-dir {output}_repeat")
    provenance = {
        "generator": Path(__file__).name,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "parameters": parameters,
        "table_source": ("exhaustive enumeration; table_id is the index decoded by table_from_index"
                         if args.samples is None else
                         "make_tm(states, default_rng([seed, 0, table_id]))"),
        "table_encoding": "tm[state][read_bit] = [write_bit, move_bit, next_state]; move_bit 0=left, 1=right",
        "seed_semantics": "random-start replicate r uses default_rng([seed, 1, table_id, r]); "
                          "the fixed-start run uses default_rng([seed, 2, table_id]); "
                          "each draws tape, then positions, then states",
        "update_semantics": "synchronous; majority write per occupied cell, ties keep the old bit",
        "entry_order": "(1,0),(1,1),(2,0),... for every entry_occupation list",
        "simulation_seconds": time.perf_counter() - start_time,
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(output, summary, parameters)
    print(json.dumps({key: summary[key] for key in (
        "tables", "aligned", "aligned_fraction", "exactly_aligned", "mechanisms")}, indent=2))
    print(f"Saved {(output / 'README.md').resolve()}")


if __name__ == "__main__":
    main()
