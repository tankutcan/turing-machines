#!/usr/bin/env python3
"""Two two-state Turing machines that rewrite each other's programs.

Each machine's transition table is stored on its own periodic 12-cell tape.
Machine 1's head reads and writes the tape that encodes machine 2, and machine
2's head reads and writes the tape that encodes machine 1.  Every step both
heads read, look up their action in the table currently on their own tape,
write, move, and change state; the two writes land simultaneously.

Tape layout: entry k = 2*(state-1) + read_bit occupies cells 3k, 3k+1, 3k+2,
holding write_bit, move_bit (0=left, 1=right) and next_state-1.  A table is
reported by the same ``table_id`` as ``alignment_census.table_from_index``.

The joint state (two tapes, two head positions, two head states) is finite and
the map is deterministic, so every start reaches a cycle.  By default all
4096^2 ordered pairs of programs are run from the chosen head start.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

TAPE_LENGTH = 12
MACHINES = 1 << TAPE_LENGTH


def encode_table(tm: dict[int, dict[int, list[int]]]) -> int:
    """Tape word of a table: bit p of the word is tape cell p."""
    word = 0
    for state in (1, 2):
        for read_bit in (0, 1):
            write_bit, move_bit, next_state = tm[state][read_bit]
            entry = write_bit | (move_bit << 1) | ((next_state - 1) << 2)
            word |= entry << (3 * (2 * (state - 1) + read_bit))
    return word


def decode_tape(word: int) -> dict[int, dict[int, list[int]]]:
    tm: dict[int, dict[int, list[int]]] = {1: {}, 2: {}}
    for state in (1, 2):
        for read_bit in (0, 1):
            entry = (word >> (3 * (2 * (state - 1) + read_bit))) & 7
            tm[state][read_bit] = [entry & 1, (entry >> 1) & 1, (entry >> 2) + 1]
    return tm


def tape_string(word: int) -> str:
    """Cells 0..11 left to right, grouped by table entry."""
    bits = [str((word >> cell) & 1) for cell in range(TAPE_LENGTH)]
    return " ".join("".join(bits[i:i + 3]) for i in range(0, TAPE_LENGTH, 3))


def table_ids(words: np.ndarray) -> np.ndarray:
    """Convert tape words to ``alignment_census`` table indices."""
    words = np.asarray(words, dtype=np.int64)
    ids = np.zeros_like(words)
    for k in range(4):
        entry = (words >> (3 * k)) & 7
        digit = ((entry & 1) << 2) | (entry & 2) | (entry >> 2)
        ids |= digit << (3 * k)
    return ids


def pack(t1, t2, x1, x2, q1, q2):
    """Joint state as one integer; head states are stored as 0 and 1."""
    return t1 | (t2 << 12) | (x1 << 24) | (x2 << 28) | (q1 << 32) | (q2 << 33)


def unpack(state):
    return (state & 4095, (state >> 12) & 4095, (state >> 24) & 15, (state >> 28) & 15,
            (state >> 32) & 1, (state >> 33) & 1)


def step(state: np.ndarray) -> np.ndarray:
    """One synchronous update of every joint state in the array."""
    t1, t2, x1, x2, q1, q2 = unpack(state)
    read1 = (t2 >> x1) & 1
    read2 = (t1 >> x2) & 1
    act1 = (t1 >> (3 * (2 * q1 + read1))) & 7
    act2 = (t2 >> (3 * (2 * q2 + read2))) & 7
    new_t2 = (t2 & ~(1 << x1)) | ((act1 & 1) << x1)
    new_t1 = (t1 & ~(1 << x2)) | ((act2 & 1) << x2)
    new_x1 = (x1 + (act1 & 2) - 1) % TAPE_LENGTH
    new_x2 = (x2 + (act2 & 2) - 1) % TAPE_LENGTH
    return pack(new_t1, new_t2, new_x1, new_x2, act1 >> 2, act2 >> 2)


def find_cycles(start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact (transient length, period, a state on the cycle) for every start.

    Periods come from Brent's algorithm run on all trajectories at once; the
    transient is then the first time at which states one period apart agree.
    """
    start = np.asarray(start, dtype=np.int64)
    period = np.zeros(len(start), dtype=np.int64)
    on_cycle = np.zeros(len(start), dtype=np.int64)
    active = np.arange(len(start))
    current, saved = start.copy(), start.copy()
    power, lam = 1, 0
    while len(active):
        current = step(current)
        lam += 1
        hit = current == saved
        if hit.any():
            period[active[hit]] = lam
            on_cycle[active[hit]] = current[hit]
            active, current, saved = active[~hit], current[~hit], saved[~hit]
        if lam == power:
            saved, power, lam = current.copy(), 2 * power, 0

    ahead = start.copy()
    active = np.arange(len(start))
    done = 0
    while len(active):
        active = active[period[active] > done]
        ahead[active] = step(ahead[active])
        done += 1
    transient = np.zeros(len(start), dtype=np.int64)
    behind = start.copy()
    active = np.flatnonzero(behind != ahead)
    while len(active):
        behind[active] = step(behind[active])
        ahead[active] = step(ahead[active])
        transient[active] += 1
        active = active[behind[active] != ahead[active]]
    return transient, period, on_cycle


def canonical_cycle_states(on_cycle: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Smallest packed state on each cycle; identifies the attractor."""
    states, inverse = np.unique(on_cycle, return_inverse=True)
    periods = np.zeros(len(states), dtype=np.int64)
    periods[inverse] = period
    smallest, current = states.copy(), states.copy()
    active = np.arange(len(states))
    done = 1
    while True:
        active = active[periods[active] > done]
        if not len(active):
            break
        current[active] = step(current[active])
        smallest[active] = np.minimum(smallest[active], current[active])
        done += 1
    return smallest[inverse]


def describe_attractors(canonical: np.ndarray, period: np.ndarray) -> dict[str, np.ndarray]:
    """One record per attractor reached, plus each start's attractor index.

    ``occupancy[slot, word]`` is the basin-weighted share of late time that
    machine ``slot`` spends as tape ``word``; each slot's row sums to one.
    A slot is frozen when its tape never changes around the cycle.
    """
    states, index, basin = np.unique(canonical, return_inverse=True, return_counts=True)
    periods = np.zeros(len(states), dtype=np.int64)
    periods[index] = period
    weight = basin / (periods * len(canonical))
    occupancy = np.zeros((2, MACHINES))
    rewrites = np.zeros((2, len(states)), dtype=np.int64)
    current = states.copy()
    active = np.arange(len(states))
    done = 0
    while len(active):
        t1, t2 = unpack(current[active])[:2]
        occupancy[0] += np.bincount(t1, weights=weight[active], minlength=MACHINES)
        occupancy[1] += np.bincount(t2, weights=weight[active], minlength=MACHINES)
        current[active] = step(current[active])
        n1, n2 = unpack(current[active])[:2]
        rewrites[0, active] += n1 != t1
        rewrites[1, active] += n2 != t2
        done += 1
        active = active[periods[active] > done]
    t1, t2, x1, x2, q1, q2 = unpack(states)
    return {"state": states, "period": periods, "basin": basin, "index": index,
            "tape1": t1, "tape2": t2, "rewrites1": rewrites[0], "rewrites2": rewrites[1],
            "occupancy": occupancy}


def analyze(start: np.ndarray) -> dict[str, Any]:
    """Cycle structure of every start and per-machine attraction statistics.

    For each tape word, over all starts and both slots:
    ``survives``  starts holding the word whose slot stays frozen at it;
    ``captured``  starts not holding the word whose slot freezes at it;
    ``occupancy`` basin-weighted share of late time spent as the word.
    """
    transient, period, on_cycle = find_cycles(start)
    attractors = describe_attractors(canonical_cycle_states(on_cycle, period), period)
    index = attractors["index"]
    initial = unpack(np.asarray(start, dtype=np.int64))[:2]
    held = np.zeros(MACHINES, dtype=np.int64)
    survives = np.zeros(MACHINES, dtype=np.int64)
    captured = np.zeros(MACHINES, dtype=np.int64)
    frozen_slots = []
    for slot in (0, 1):
        frozen = attractors[f"rewrites{slot + 1}"][index] == 0
        final = attractors[f"tape{slot + 1}"][index]
        kept = frozen & (final == initial[slot])
        held += np.bincount(initial[slot], minlength=MACHINES)
        survives += np.bincount(initial[slot][kept], minlength=MACHINES)
        captured += np.bincount(final[frozen & ~kept], minlength=MACHINES)
        frozen_slots.append(frozen)
    return {"transient": transient, "period": period, "attractors": attractors,
            "held": held, "survives": survives, "captured": captured,
            "occupancy": attractors["occupancy"].mean(axis=0),
            "both_frozen": frozen_slots[0] & frozen_slots[1],
            "untouched": (frozen_slots[0] & frozen_slots[1]
                          & (attractors["tape1"][index] == initial[0])
                          & (attractors["tape2"][index] == initial[1])),
            "same_program": (frozen_slots[0] & frozen_slots[1]
                             & (attractors["tape1"][index] == attractors["tape2"][index]))}


def write_class(words: np.ndarray) -> np.ndarray:
    """Coarse class of each table by its four write bits (cells 0, 3, 6, 9)."""
    words = np.asarray(words, dtype=np.int64)
    writes = np.stack([(words >> (3 * k)) & 1 for k in range(4)])
    labels = np.full(words.shape, "other", dtype=object)
    labels[(writes == np.array([0, 1, 0, 1])[:, None]).all(axis=0)] = "writes_what_it_reads"
    labels[writes.sum(axis=0) == 0] = "always_writes_0"
    labels[writes.sum(axis=0) == 4] = "always_writes_1"
    return labels


def summarize(words1: np.ndarray, words2: np.ndarray, result: dict[str, Any]) -> dict[str, Any]:
    attractors = result["attractors"]
    index = attractors["index"]
    count = len(index)
    frozen_attractor = (attractors["rewrites1"] == 0) & (attractors["rewrites2"] == 0)
    final1, final2 = attractors["tape1"][index], attractors["tape2"][index]
    frozen = result["both_frozen"]
    all_zero = frozen & (final1 == 0) & (final2 == 0)
    all_one = frozen & (final1 == MACHINES - 1) & (final2 == MACHINES - 1)
    untouched = result["untouched"] & ~all_zero & ~all_one
    outcomes = {
        "both_all_zero": int(all_zero.sum()),
        "both_all_one": int(all_one.sum()),
        "untouched": int(untouched.sum()),
        "other_frozen": int((frozen & ~all_zero & ~all_one & ~untouched).sum()),
        "dynamic": int((~frozen).sum()),
    }
    write_ones = sum(((words1 >> (3 * k)) & 1) + ((words2 >> (3 * k)) & 1) for k in range(4))
    by_writes = {}
    for ones in range(9):
        chosen = write_ones == ones
        if chosen.any():
            by_writes[str(ones)] = [int(chosen.sum()), int(all_zero[chosen].sum()),
                                    int(all_one[chosen].sum())]
    all_words = np.arange(MACHINES)
    classes = write_class(all_words)
    held = np.maximum(result["held"], 1)
    survival = result["survives"] / held
    by_class = {}
    for name in ("always_writes_0", "always_writes_1", "writes_what_it_reads", "other"):
        chosen = classes == name
        by_class[name] = {
            "machines": int(chosen.sum()),
            "late_occupancy": float(result["occupancy"][chosen].sum()),
            "mean_survival": float(survival[chosen].mean()),
            "never_present_late": int((result["occupancy"][chosen] == 0).sum()),
        }
    dynamic = ~frozen_attractor
    return {
        "starts": count,
        "outcomes": outcomes,
        "same_program_at_end": int(result["same_program"].sum()),
        "attractors": int(len(frozen_attractor)),
        "frozen_attractors": int(frozen_attractor.sum()),
        "dynamic_attractors": int(dynamic.sum()),
        "distinct_frozen_program_pairs": int(len(np.unique(
            attractors["tape1"][frozen_attractor] * MACHINES + attractors["tape2"][frozen_attractor]))),
        "dynamic_attractor_periods": {str(period): number for period, number in sorted(
            Counter(attractors["period"][dynamic].tolist()).items())},
        "transient_mean": float(result["transient"].mean()),
        "transient_max": int(result["transient"].max()),
        "period_max": int(result["period"].max()),
        "periods_by_starts": {str(period): number for period, number in sorted(
            Counter(result["period"].tolist()).items())},
        "machines_never_present_late": int((result["occupancy"] == 0).sum()),
        "machines_never_surviving": int((result["survives"] == 0).sum()),
        "machines_never_captured": int((result["captured"] == 0).sum()),
        "outcome_by_write_one_entries": by_writes,
        "by_write_class": by_class,
    }


def write_report(output: Path, summary: dict[str, Any], machines: list[dict[str, Any]],
                 parameters: dict[str, Any]) -> None:
    starts = summary["starts"]
    share = lambda number: f"{number / starts:.2%}"
    source = ("all 4096² ordered pairs of programs" if parameters["pairs"] is None
              else f"{starts:,} random ordered pairs (seed {parameters['seed']})")
    x1, x2 = parameters["start_positions"]
    q1, q2 = parameters["start_states"]
    names = {"both_all_zero": "both programs become all 0", "both_all_one": "both programs become all 1",
             "untouched": "both programs end as they started", "other_frozen": "other frozen pair",
             "dynamic": "programs keep rewriting each other"}
    lines = [
        "# Mutual rewriting of two-state programs", "",
        "[Per-machine CSV](machines.csv) · [Largest attractors](largest_attractors.csv) · "
        "[All attractors](attractors.npz) · [Per-start data](pairs.npz) · "
        "[Summary and provenance](summary.json)", "",
        f"{source}. Machine 1's head acts on the 12-cell periodic tape that encodes machine 2, and "
        f"machine 2's head on the tape that encodes machine 1. Heads start at cells {x1} and {x2} in "
        f"states {q1} and {q2}. Each step both heads read, act by the table currently on their own "
        "tape, write, move and change state. The joint state is finite, so every start reaches a "
        "cycle; all results below are exact for the starts that were run.", "",
        "Tape layout: entry k = 2·(state−1) + read bit occupies cells 3k, 3k+1, 3k+2, holding the "
        "write bit, the move bit (0 left, 1 right) and next state − 1. The all-0 tape is therefore "
        "the program \"write 0, move left, go to state 1\" and the all-1 tape is \"write 1, move "
        "right, go to state 2\".", "",
        "## Where pairs end up", "", "| Outcome | Starts | Share |", "| --- | ---: | ---: |",
        *[f"| {names[key]} | {number:,} | {share(number)} |"
          for key, number in summary["outcomes"].items()], "",
        f"The two programs are identical at the end for {share(summary['same_program_at_end'])} of "
        f"starts. Mean transient {summary['transient_mean']:.1f} steps (maximum "
        f"{summary['transient_max']}); longest cycle {summary['period_max']} steps. A frozen pair "
        "still has moving heads, so its cycle length is set by head motion. "
        f"{summary['attractors']:,} distinct cycles were reached: {summary['frozen_attractors']:,} "
        f"frozen ones covering {summary['distinct_frozen_program_pairs']:,} program pairs, and "
        f"{summary['dynamic_attractors']:,} in which at least one program keeps changing.", "",
        "## Outcome by the number of write-1 entries in the two starting tables", "",
        "| Write-1 entries (of 8) | Starts | Both all 0 | Both all 1 |", "| ---: | ---: | ---: | ---: |",
        *[f"| {ones} | {total:,} | {zero / total:.1%} | {one / total:.1%} |"
          for ones, (total, zero, one) in summary["outcome_by_write_one_entries"].items()], "",
        "## Machines by what they write", "",
        "| Class | Machines | Share of late-time occupancy | Mean survival | Never present late |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| {name.replace('_', ' ')} | {row['machines']} | {row['late_occupancy']:.2%} | "
          f"{row['mean_survival']:.1%} | {row['never_present_late']} |"
          for name, row in summary["by_write_class"].items()], "",
        "Late-time occupancy is the share of cycle time, weighted by basin size and averaged over "
        "both slots, that a program spends as a given table; a uniform share would be 1/4096. "
        "Survival is the fraction of starts holding a table in which that slot ends frozen at the "
        "same table. Captured counts starts that did not hold the table but freeze at it.", "",
        f"{summary['machines_never_present_late']} tables are never present at late times, "
        f"{summary['machines_never_surviving']} never survive, and "
        f"{summary['machines_never_captured']} are never the frozen endpoint of another table.", "",
        "## Most attracting tables", "",
        "| Table id | Tape | Occupancy × 4096 | Survival | Captured |", "| ---: | --- | ---: | ---: | ---: |",
        *[f"| {row['table_id']} | `{row['tape']}` | {row['enrichment']:.1f} | "
          f"{row['survival_fraction']:.1%} | {row['captured']:,} |"
          for row in sorted(machines, key=lambda row: -row["late_occupancy"])[:12]], "",
        "## Reproduce", "", "```sh", parameters["reproduce"], "```", "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int,
                        help="number of random ordered pairs; omit to run all 4096^2")
    parser.add_argument("--seed", type=int, default=0, help="seed for --pairs")
    parser.add_argument("--start-positions", type=int, nargs=2, default=(0, 0),
                        metavar=("X1", "X2"), help="initial head cells, 0..11")
    parser.add_argument("--start-states", type=int, nargs=2, default=(1, 1),
                        metavar=("Q1", "Q2"), help="initial head states, 1 or 2")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mutual_rewriting"))
    args = parser.parse_args()
    if args.pairs is not None and args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.seed < 0 or any(not 0 <= x < TAPE_LENGTH for x in args.start_positions):
        parser.error("seed must be nonnegative and start positions in 0..11")
    if any(q not in (1, 2) for q in args.start_states):
        parser.error("start states must be 1 or 2")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory is not empty; choose a new --output-dir")

    start_time = time.perf_counter()
    words_by_id = np.argsort(table_ids(np.arange(MACHINES)))
    if args.pairs is None:
        id1, id2 = np.divmod(np.arange(MACHINES * MACHINES, dtype=np.int64), MACHINES)
    else:
        rng = np.random.default_rng(args.seed)
        id1 = rng.integers(0, MACHINES, args.pairs)
        id2 = rng.integers(0, MACHINES, args.pairs)
    words1, words2 = words_by_id[id1], words_by_id[id2]
    result = analyze(pack(words1, words2, args.start_positions[0], args.start_positions[1],
                          args.start_states[0] - 1, args.start_states[1] - 1))
    attractors = result["attractors"]
    summary = summarize(words1, words2, result)

    classes = write_class(np.arange(MACHINES))
    machines = []
    for table_id in range(MACHINES):
        word = int(words_by_id[table_id])
        held = int(result["held"][word])
        machines.append({
            "table_id": table_id, "tape": tape_string(word), "write_class": classes[word],
            "late_occupancy": float(result["occupancy"][word]),
            "enrichment": float(result["occupancy"][word] * MACHINES),
            "held": held, "survives": int(result["survives"][word]),
            "survival_fraction": result["survives"][word] / held if held else 0.0,
            "captured": int(result["captured"][word]),
        })
    with (output / "machines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(machines[0]))
        writer.writeheader()
        writer.writerows(machines)

    t1, t2, x1, x2, q1, q2 = unpack(attractors["state"])
    with (output / "largest_attractors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["basin", "period", "rewrites_per_cycle_1", "rewrites_per_cycle_2",
                         "table_id_1", "tape_1", "table_id_2", "tape_2",
                         "head_1", "head_2", "state_1", "state_2"])
        for i in np.argsort(-attractors["basin"], kind="stable")[:500]:
            writer.writerow([attractors["basin"][i], attractors["period"][i],
                             attractors["rewrites1"][i], attractors["rewrites2"][i],
                             table_ids(t1[i]), tape_string(int(t1[i])),
                             table_ids(t2[i]), tape_string(int(t2[i])),
                             x1[i], x2[i], q1[i] + 1, q2[i] + 1])
    np.savez_compressed(
        output / "attractors.npz", table_id_1=table_ids(t1).astype(np.uint16),
        table_id_2=table_ids(t2).astype(np.uint16), head_1=x1.astype(np.uint8),
        head_2=x2.astype(np.uint8), state_1=(q1 + 1).astype(np.uint8),
        state_2=(q2 + 1).astype(np.uint8), period=attractors["period"].astype(np.uint16),
        basin=attractors["basin"], rewrites_per_cycle_1=attractors["rewrites1"].astype(np.uint16),
        rewrites_per_cycle_2=attractors["rewrites2"].astype(np.uint16))
    np.savez_compressed(
        output / "pairs.npz", table_id_1=id1.astype(np.uint16), table_id_2=id2.astype(np.uint16),
        transient=result["transient"].astype(np.uint16), period=result["period"].astype(np.uint16),
        attractor=attractors["index"].astype(np.uint32))

    parameters = {
        "pairs": args.pairs, "seed": args.seed,
        "start_positions": list(args.start_positions), "start_states": list(args.start_states),
        "reproduce": (f".venv/bin/python -m scripts.{Path(__file__).stem} "
                      + (f"--pairs {args.pairs} --seed {args.seed} " if args.pairs is not None else "")
                      + f"--start-positions {args.start_positions[0]} {args.start_positions[1]} "
                      f"--start-states {args.start_states[0]} {args.start_states[1]} "
                      f"--output-dir {output}_repeat"),
    }
    provenance = {
        "generator": Path(__file__).name,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "parameters": parameters,
        "tape_layout": "entry k = 2*(state-1)+read_bit in cells 3k,3k+1,3k+2 = "
                       "write_bit, move_bit (0=left, 1=right), next_state-1",
        "table_id": "alignment_census.table_from_index index of the same table",
        "update_semantics": "both heads read the old tapes and the old tables, then both writes, "
                            "moves and state changes are applied together",
        "attractor": "a cycle of the joint state (two tapes, two head cells, two head states), "
                     "identified in attractors.npz by its smallest packed state; rows of pairs.npz "
                     "index into attractors.npz",
        "simulation_seconds": time.perf_counter() - start_time,
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(output, summary, machines, parameters)
    print(json.dumps({key: summary[key] for key in (
        "starts", "outcomes", "attractors", "dynamic_attractors", "transient_max",
        "period_max")}, indent=2))
    print(f"Saved {(output / 'README.md').resolve()}")


if __name__ == "__main__":
    main()
