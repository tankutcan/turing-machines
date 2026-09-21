#!/usr/bin/env python3
"""Simulate random Turing-machine agents on a shared random tape.

By default all agents are identical.  ``--machine-ensemble independent``
instead samples a separate transition table for every agent.  The table
convention matches ``turing-machines.ipynb``::

    tm[state][read_bit] = [write_bit, move_bit, next_state]

where ``move_bit == 0`` moves left and ``move_bit == 1`` moves right.

All heads update synchronously.  They read the tape before any writes for that
step, propose writes, and then move.  When several heads write different bits
to the same cell, ``--collision-policy`` determines the resulting bit.  The
default policy takes the majority proposal and leaves the old bit unchanged on
a tie.  Thus copies interact only through their shared tape, without an
arbitrary sequential update order.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_tm(n_states: int, rng: np.random.Generator) -> dict[int, dict[int, list[int]]]:
    """Return a random binary Turing-machine transition dictionary."""
    tm: dict[int, dict[int, list[int]]] = {}
    for state in range(1, n_states + 1):
        tm[state] = {}
        for read_bit in (0, 1):
            tm[state][read_bit] = [
                int(rng.integers(0, 2)),
                int(rng.integers(0, 2)),
                int(rng.integers(1, n_states + 1)),
            ]
    return tm


def table_arrays(
    tm: dict[int, dict[int, list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the notebook-style dictionary to arrays for vectorized lookup."""
    n_states = len(tm)
    writes = np.empty((n_states, 2), dtype=np.uint8)
    moves = np.empty((n_states, 2), dtype=np.int8)
    next_states = np.empty((n_states, 2), dtype=np.int32)
    for state in range(1, n_states + 1):
        for read_bit in (0, 1):
            write_bit, move_bit, next_state = tm[state][read_bit]
            writes[state - 1, read_bit] = write_bit
            moves[state - 1, read_bit] = 2 * move_bit - 1
            next_states[state - 1, read_bit] = next_state
    return writes, moves, next_states


def resolve_writes(
    tape: np.ndarray,
    tape_min: int,
    positions: np.ndarray,
    proposed: np.ndarray,
    policy: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Apply simultaneous writes and return (coincident, conflicting) counts."""
    order = np.argsort(positions, kind="stable")
    sorted_positions = positions[order]
    sorted_writes = proposed[order]

    starts = np.r_[0, np.flatnonzero(np.diff(sorted_positions)) + 1]
    unique_positions = sorted_positions[starts]
    counts = np.diff(np.r_[starts, len(positions)])
    ones = np.add.reduceat(sorted_writes.astype(np.int32), starts)

    coincident = int(np.count_nonzero(counts > 1))
    conflicting_mask = (ones > 0) & (ones < counts)
    conflicting = int(np.count_nonzero(conflicting_mask))
    old = tape[unique_positions - tape_min]

    if policy == "majority":
        resolved = old.copy()
        resolved[2 * ones > counts] = 1
        resolved[2 * ones < counts] = 0
    elif policy == "keep":
        resolved = sorted_writes[starts].copy()
        resolved[conflicting_mask] = old[conflicting_mask]
    elif policy == "zero":
        resolved = sorted_writes[starts].copy()
        resolved[conflicting_mask] = 0
    elif policy == "one":
        resolved = sorted_writes[starts].copy()
        resolved[conflicting_mask] = 1
    elif policy == "random":
        # Choose one of the actual proposals uniformly for each occupied cell.
        offsets = (rng.random(len(starts)) * counts).astype(np.int64)
        resolved = sorted_writes[starts + offsets]
    else:  # Guard against calls that bypass argparse.
        raise ValueError(f"unknown collision policy: {policy}")

    tape[unique_positions - tape_min] = resolved
    return coincident, conflicting


def record_plot_window(
    tape: np.ndarray,
    tape_min: int,
    positions: np.ndarray,
    plot_min: int,
    plot_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Record tape bits and head occupancy in a fixed spatial window."""
    start = plot_min - tape_min
    tape_row = tape[start : start + plot_width].copy()
    heads = np.zeros(plot_width, dtype=np.uint16)
    visible = (positions >= plot_min) & (positions < plot_min + plot_width)
    np.add.at(heads, positions[visible] - plot_min, 1)
    return tape_row, heads


def random_sequential_step(
    tape: np.ndarray,
    tape_min: int,
    positions: np.ndarray,
    states: np.ndarray,
    write_table: np.ndarray,
    move_table: np.ndarray,
    state_table: np.ndarray,
    copy_indices: np.ndarray | None,
    rng: np.random.Generator,
) -> int:
    """Update every agent once in random order and return co-occupied cells.

    Updates at different starting cells commute: agents write before moving and
    cannot read their destination until the next sweep.  We therefore update
    singleton cells vectorially and explicitly randomize only the relative
    order of agents that start a sweep on the same cell.  This is exactly
    equivalent to a full random permutation, but avoids a Python loop over all
    agents.
    """
    priorities = rng.random(len(positions))
    order = np.lexsort((priorities, positions))
    sorted_positions = positions[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_positions)) + 1]
    counts = np.diff(np.r_[starts, len(positions)])
    coincident = int(np.count_nonzero(counts > 1))

    singleton_agents = order[starts[counts == 1]]
    if len(singleton_agents):
        reads = tape[positions[singleton_agents] - tape_min]
        state_indices = states[singleton_agents] - 1
        if copy_indices is None:
            writes = write_table[state_indices, reads]
            moves = move_table[state_indices, reads]
            next_states = state_table[state_indices, reads]
        else:
            writes = write_table[singleton_agents, state_indices, reads]
            moves = move_table[singleton_agents, state_indices, reads]
            next_states = state_table[singleton_agents, state_indices, reads]
        tape[positions[singleton_agents] - tape_min] = writes
        positions[singleton_agents] += moves
        states[singleton_agents] = next_states

    for start, count in zip(starts[counts > 1], counts[counts > 1]):
        cell = int(sorted_positions[start])
        tape_index = cell - tape_min
        for agent in order[start : start + count]:
            read_bit = int(tape[tape_index])
            state_index = int(states[agent] - 1)
            if copy_indices is None:
                write_bit = write_table[state_index, read_bit]
                move = move_table[state_index, read_bit]
                next_state = state_table[state_index, read_bit]
            else:
                write_bit = write_table[agent, state_index, read_bit]
                move = move_table[agent, state_index, read_bit]
                next_state = state_table[agent, state_index, read_bit]
            tape[tape_index] = write_bit
            positions[agent] += move
            states[agent] = next_state

    return coincident


def fit_velocities(position_trace: np.ndarray, burn_index: int) -> np.ndarray:
    """Fit one late-time velocity per head by ordinary least squares."""
    times = np.arange(burn_index, len(position_trace), dtype=np.float64)
    centered_times = times - times.mean()
    denominator = float(centered_times @ centered_times)
    if denominator == 0:
        raise ValueError("the post-burn-in interval must contain at least two samples")
    return centered_times @ position_trace[burn_index:] / denominator


def simulate(
    args: argparse.Namespace,
    tm_override: dict[int, dict[int, list[int]]] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    machine_ensemble = getattr(args, "machine_ensemble", "identical")
    if tm_override is not None:
        if machine_ensemble != "identical":
            raise ValueError("tm_override requires the identical machine ensemble")
        if len(tm_override) != args.states:
            raise ValueError("tm_override state count does not match args.states")
        tm = tm_override
        write_table, move_table, state_table = table_arrays(tm)
        copy_indices = None
    elif machine_ensemble == "identical":
        tm: dict[int, dict[int, list[int]]] | list[dict[int, dict[int, list[int]]]]
        tm = make_tm(args.states, rng)
        write_table, move_table, state_table = table_arrays(tm)
        copy_indices = None
    elif machine_ensemble == "independent":
        tm = [make_tm(args.states, rng) for _ in range(args.copies)]
        tables = [table_arrays(machine) for machine in tm]
        write_table = np.stack([table[0] for table in tables])
        move_table = np.stack([table[1] for table in tables])
        state_table = np.stack([table[2] for table in tables])
        copy_indices = np.arange(args.copies)
    else:
        raise ValueError(f"unknown machine ensemble: {machine_ensemble}")

    # Keep exact integer spacing. For periodic systems, these coordinates are
    # retained as unwrapped positions while tape access uses coordinates mod L.
    unwrapped_positions = (
        np.arange(args.copies, dtype=np.int64) - args.copies // 2
    ) * args.spacing
    initial_positions = unwrapped_positions.copy()
    boundary = getattr(args, "boundary", "open")
    tape_length = getattr(args, "tape_length", None)
    if boundary == "periodic":
        if tape_length is None or tape_length < 1:
            raise ValueError("periodic boundaries require a positive tape_length")
        if args.plot_width > tape_length:
            raise ValueError("plot_width cannot exceed a periodic tape's length")
        positions = np.mod(unwrapped_positions, tape_length)
        plot_min = 0
        tape_min = 0
        tape_max = tape_length - 1
    elif boundary == "open":
        positions = unwrapped_positions.copy()
        plot_min = -(args.plot_width // 2)
        plot_max = plot_min + args.plot_width - 1
        padding = 2
        tape_min = min(int(positions.min()) - args.steps - padding, plot_min)
        tape_max = max(int(positions.max()) + args.steps + padding, plot_max)
    else:
        raise ValueError(f"unknown boundary condition: {boundary}")
    initial_tape_positions = positions.copy()
    tape = (rng.random(tape_max - tape_min + 1) < args.one_density).astype(np.uint8)

    # Draw random internal states only after constructing the machine and tape.
    # This preserves both of those objects when fixed and random initial-state
    # runs use the same seed, making the two modes directly comparable.
    initial_state_mode = getattr(args, "initial_state_mode", "fixed")
    if initial_state_mode == "fixed":
        states = np.ones(args.copies, dtype=np.int32)
    elif initial_state_mode == "random":
        states = rng.integers(1, args.states + 1, size=args.copies, dtype=np.int32)
    else:
        raise ValueError(f"unknown initial state mode: {initial_state_mode}")
    initial_states = states.copy()

    position_trace = np.empty((args.steps + 1, args.copies), dtype=np.int32)
    position_trace[0] = unwrapped_positions
    recorded_times = list(range(0, args.steps + 1, args.record_every))
    if recorded_times[-1] != args.steps:
        recorded_times.append(args.steps)
    spacetime = np.empty((len(recorded_times), args.plot_width), dtype=np.uint8)
    head_occupancy = np.empty((len(recorded_times), args.plot_width), dtype=np.uint16)
    record_at = {time: row for row, time in enumerate(recorded_times)}
    spacetime[0], head_occupancy[0] = record_plot_window(
        tape, tape_min, positions, plot_min, args.plot_width
    )

    coincident_cells = 0
    conflicting_cells = 0
    update_scheme = getattr(args, "update_scheme", "synchronous")
    for step in range(1, args.steps + 1):
        if update_scheme == "synchronous":
            read_bits = tape[positions - tape_min]
            state_indices = states - 1
            if copy_indices is None:
                proposed_writes = write_table[state_indices, read_bits]
                moves = move_table[state_indices, read_bits]
                next_states = state_table[state_indices, read_bits]
            else:
                proposed_writes = write_table[copy_indices, state_indices, read_bits]
                moves = move_table[copy_indices, state_indices, read_bits]
                next_states = state_table[copy_indices, state_indices, read_bits]

            coincident, conflicting = resolve_writes(
                tape,
                tape_min,
                positions,
                proposed_writes,
                args.collision_policy,
                rng,
            )
            coincident_cells += coincident
            conflicting_cells += conflicting
            unwrapped_positions = unwrapped_positions + moves
            positions = positions + moves
            if boundary == "periodic":
                positions %= tape_length
            states = next_states
        elif update_scheme == "random-sequential":
            old_positions = positions.copy()
            coincident_cells += random_sequential_step(
                tape,
                tape_min,
                positions,
                states,
                write_table,
                move_table,
                state_table,
                copy_indices,
                rng,
            )
            unwrapped_positions += positions - old_positions
            if boundary == "periodic":
                positions %= tape_length
        else:
            raise ValueError(f"unknown update scheme: {update_scheme}")
        position_trace[step] = unwrapped_positions

        if step in record_at:
            row = record_at[step]
            spacetime[row], head_occupancy[row] = record_plot_window(
                tape, tape_min, positions, plot_min, args.plot_width
            )

    burn_index = int(round(args.steps * args.burn_in_fraction))
    velocities = fit_velocities(position_trace, burn_index)
    net_velocities = (
        position_trace[-1] - position_trace[burn_index]
    ) / (args.steps - burn_index)

    return {
        "tm": tm,
        "tape": tape,
        "tape_min": tape_min,
        "initial_positions": initial_positions,
        "initial_tape_positions": initial_tape_positions,
        "initial_states": initial_states,
        "final_states": states,
        "position_trace": position_trace,
        "recorded_times": np.asarray(recorded_times),
        "spacetime": spacetime,
        "head_occupancy": head_occupancy,
        "plot_min": plot_min,
        "boundary": boundary,
        "tape_length": len(tape) if boundary == "periodic" else None,
        "final_positions": positions,
        "velocities": velocities,
        "net_velocities": net_velocities,
        "burn_index": burn_index,
        "coincident_cells": coincident_cells,
        "conflicting_cells": conflicting_cells,
    }


def save_results(result: dict[str, Any], args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    with (output / "tm.json").open("w") as handle:
        json.dump(result["tm"], handle, indent=2)

    with (output / "velocities.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "copy_id",
                "initial_position",
                "initial_state",
                "final_position",
                "final_state",
                "velocity_fit",
                "velocity_net",
            ]
        )
        for copy_id in range(args.copies):
            writer.writerow(
                [
                    copy_id,
                    int(result["initial_positions"][copy_id]),
                    int(result["initial_states"][copy_id]),
                    int(result["position_trace"][-1, copy_id]),
                    int(result["final_states"][copy_id]),
                    float(result["velocities"][copy_id]),
                    float(result["net_velocities"][copy_id]),
                ]
            )

    np.savez_compressed(
        output / "simulation.npz",
        position_trace=result["position_trace"],
        initial_states=result["initial_states"],
        final_states=result["final_states"],
        velocities=result["velocities"],
        net_velocities=result["net_velocities"],
        recorded_times=result["recorded_times"],
        spacetime=result["spacetime"],
        head_occupancy=result["head_occupancy"],
        plot_positions=np.arange(
            result["plot_min"], result["plot_min"] + args.plot_width
        ),
        final_tape=result["tape"],
        final_tape_positions=np.arange(
            result["tape_min"], result["tape_min"] + len(result["tape"])
        ),
    )

    rounded = np.round(result["velocities"], args.peak_decimals)
    peak_counts = Counter(float(value) for value in rounded)
    peaks = [
        {"velocity": velocity, "count": count}
        for velocity, count in sorted(
            peak_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    summary = {
        "seed": args.seed,
        "states": args.states,
        "copies": args.copies,
        "machine_ensemble": getattr(args, "machine_ensemble", "identical"),
        "initial_state_mode": getattr(args, "initial_state_mode", "fixed"),
        "initial_state_counts": {
            str(state): int(count)
            for state, count in sorted(Counter(result["initial_states"]).items())
        },
        "update_scheme": getattr(args, "update_scheme", "synchronous"),
        "boundary": getattr(args, "boundary", "open"),
        "tape_length": getattr(args, "tape_length", None),
        "steps": args.steps,
        "spacing": args.spacing,
        "burn_in_step": result["burn_index"],
        "collision_policy": args.collision_policy,
        "coincident_cell_events": result["coincident_cells"],
        "conflicting_write_events": result["conflicting_cells"],
        "mean_velocity": float(np.mean(result["velocities"])),
        "mean_absolute_velocity": float(np.mean(np.abs(result["velocities"]))),
        "rms_velocity": float(np.sqrt(np.mean(result["velocities"] ** 2))),
        "velocity_standard_deviation": float(np.std(result["velocities"])),
        "rounded_velocity_peaks": peaks,
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    extent = [
        result["plot_min"] - 0.5,
        result["plot_min"] + args.plot_width - 0.5,
        -0.5 * args.record_every,
        args.steps + 0.5 * args.record_every,
    ]
    ax.imshow(
        result["spacetime"],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="gray",
        extent=extent,
    )
    visible_heads = np.ma.masked_where(
        result["head_occupancy"] == 0, result["head_occupancy"]
    )
    ax.imshow(
        visible_heads,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="autumn",
        alpha=0.75,
        extent=extent,
    )
    ax.set(
        title=(
            "Shared tape spacetime (heads in red/yellow)\n"
            f"seed={args.seed}, states={args.states}, copies={args.copies}, "
            f"initial states={getattr(args, 'initial_state_mode', 'fixed')}, "
            f"steps={args.steps}"
        ),
        xlabel="Tape position",
        ylabel="Time",
    )
    fig.savefig(output / "spacetime.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(result["velocities"], bins=args.histogram_bins, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"Late-time velocities (fit from t={result['burn_index']})",
        xlabel="Velocity (cells per step)",
        ylabel="Number of copies",
    )
    fig.savefig(output / "velocity_histogram.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    trace = result["position_trace"]
    selected = np.linspace(
        0, args.copies - 1, min(args.trajectory_count, args.copies), dtype=int
    )
    ax.plot(np.arange(args.steps + 1), trace[:, selected], linewidth=0.7, alpha=0.7)
    ax.axvline(result["burn_index"], color="black", linestyle="--", linewidth=0.8)
    ax.set(title="Head trajectories", xlabel="Time", ylabel="Tape position")
    fig.savefig(output / "trajectories.png", dpi=200)
    plt.close(fig)

    print(f"Wrote results to {output.resolve()}")
    print(
        "Velocity: "
        f"mean={summary['mean_velocity']:.5f}, "
        f"mean|v|={summary['mean_absolute_velocity']:.5f}, "
        f"rms={summary['rms_velocity']:.5f}"
    )
    print(
        f"Interactions: {result['coincident_cells']} coincident-cell events, "
        f"{result['conflicting_cells']} conflicting-write events"
    )
    print("Most common rounded velocities:")
    for peak in peaks[:10]:
        print(f"  {peak['velocity']: .{args.peak_decimals}f}: {peak['count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=20, help="states in the random TM")
    parser.add_argument("--copies", type=int, default=200, help="identical TM copies")
    parser.add_argument("--steps", type=int, default=5000, help="synchronous time steps")
    parser.add_argument("--spacing", type=int, default=5, help="initial head spacing")
    parser.add_argument("--seed", type=int, default=1, help="TM, tape, and conflict RNG seed")
    parser.add_argument(
        "--machine-ensemble",
        choices=("identical", "independent"),
        default="identical",
        help="share one TM across copies or sample one random TM per copy",
    )
    parser.add_argument(
        "--initial-state-mode",
        choices=("fixed", "random"),
        default="fixed",
        help="start every agent in state 1 or sample states uniformly and independently",
    )
    parser.add_argument(
        "--one-density", type=float, default=0.5, help="initial probability of tape bit 1"
    )
    parser.add_argument(
        "--collision-policy",
        choices=("majority", "keep", "zero", "one", "random"),
        default="majority",
        help="rule for different simultaneous writes to one cell",
    )
    parser.add_argument(
        "--update-scheme",
        choices=("synchronous", "random-sequential"),
        default="synchronous",
        help="simultaneous updates or one shuffled agent sweep per time step",
    )
    parser.add_argument(
        "--boundary",
        choices=("open", "periodic"),
        default="open",
        help="open line or finite periodic ring",
    )
    parser.add_argument(
        "--tape-length",
        type=int,
        help="number of cells; required with --boundary periodic",
    )
    parser.add_argument(
        "--burn-in-fraction",
        type=float,
        default=0.5,
        help="fraction of early trajectory excluded from velocity fits",
    )
    parser.add_argument(
        "--plot-width", type=int, default=1200, help="fixed tape width in spacetime image"
    )
    parser.add_argument(
        "--record-every", type=int, default=5, help="record every Nth spacetime row"
    )
    parser.add_argument("--histogram-bins", type=int, default=51)
    parser.add_argument(
        "--trajectory-count", type=int, default=50, help="maximum trajectories drawn"
    )
    parser.add_argument(
        "--peak-decimals", type=int, default=4, help="rounding used to count velocity peaks"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("many_tm_results")
    )
    args = parser.parse_args()

    if args.states < 1 or args.copies < 1 or args.steps < 2:
        parser.error("--states and --copies must be positive; --steps must be at least 2")
    if args.spacing < 1 or args.plot_width < 1 or args.record_every < 1:
        parser.error("--spacing, --plot-width, and --record-every must be positive")
    if not 0.0 <= args.one_density <= 1.0:
        parser.error("--one-density must lie between 0 and 1")
    if not 0.0 <= args.burn_in_fraction < 1.0:
        parser.error("--burn-in-fraction must lie in [0, 1)")
    if args.boundary == "periodic" and (args.tape_length is None or args.tape_length < 1):
        parser.error("--boundary periodic requires a positive --tape-length")
    if args.boundary == "periodic" and args.plot_width > args.tape_length:
        parser.error("--plot-width cannot exceed --tape-length on a periodic tape")
    burn_index = int(round(args.steps * args.burn_in_fraction))
    if args.steps - burn_index < 1:
        parser.error("burn-in leaves too few points for a velocity fit")
    return args


def main() -> None:
    args = parse_args()
    result = simulate(args)
    save_results(result, args)


if __name__ == "__main__":
    main()
