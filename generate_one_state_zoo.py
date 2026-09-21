#!/usr/bin/env python3
"""Enumerate all 16 one-state binary TMs and make a spacetime zoo.

The four-bit rule encoding is [w0, d0, w1, d1], with d=0 for left and
d=1 for right.  Every machine uses the same seed and therefore the same
initial tape; only its transition rule changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_spacetime_zoo import contact_thumbnail, spacetime_rgb
from many_turing_machines import simulate


def one_state_tm(machine_id: int) -> dict[int, dict[int, list[int]]]:
    """Decode machine_id using bits [w0, d0, w1, d1]."""
    if not 0 <= machine_id < 16:
        raise ValueError("one-state machine_id must lie in [0, 15]")
    w0, d0, w1, d1 = ((machine_id >> shift) & 1 for shift in (3, 2, 1, 0))
    return {1: {0: [w0, d0, 1], 1: [w1, d1, 1]}}


def rule_bits(machine_id: int) -> str:
    return f"{machine_id:04b}"


def rule_description(machine_id: int) -> str:
    tm = one_state_tm(machine_id)[1]
    w0, d0, _ = tm[0]
    w1, d1, _ = tm[1]
    return f"0→{w0}{'R' if d0 else 'L'}, 1→{w1}{'R' if d1 else 'L'}"


def simulation_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        states=1,
        copies=args.copies,
        steps=args.steps,
        spacing=args.spacing,
        plot_width=args.plot_width,
        one_density=args.one_density,
        collision_policy=args.collision_policy,
        burn_in_fraction=args.burn_in_fraction,
        record_every=args.record_every,
        machine_ensemble="identical",
        initial_state_mode="fixed",
        update_scheme=args.update_scheme,
        boundary=args.boundary,
        tape_length=args.tape_length,
    )


def panel_title(machine_id: int, args: argparse.Namespace, compact: bool) -> str:
    base = (
        f"id={machine_id:02d} bits={rule_bits(machine_id)}  "
        f"{rule_description(machine_id)}"
    )
    if compact:
        return (
            f"{base}\nseed={args.seed} N={args.copies} T={args.steps} "
            f"bc={args.boundary} L={args.tape_length}"
        )
    return (
        f"{base}\nseed={args.seed} | states=1 | copies={args.copies} | "
        f"steps={args.steps}\nupdate={args.update_scheme} | "
        f"collision={args.collision_policy} | boundary={args.boundary} | "
        f"L={args.tape_length}\nspacing={args.spacing} | "
        f"p(1)={args.one_density:g} | record every={args.record_every}"
    )


def save_plot(
    rgb: np.ndarray,
    result: dict[str, Any],
    machine_id: int,
    args: argparse.Namespace,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.imshow(
        rgb,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            result["plot_min"] - 0.5,
            result["plot_min"] + args.plot_width - 0.5,
            0,
            args.steps,
        ),
    )
    ax.set_title(panel_title(machine_id, args, compact=False), fontsize=9)
    ax.set_xlabel("Tape position")
    ax.set_ylabel("Time")
    fig.savefig(path, dpi=args.png_dpi)
    plt.close(fig)


def generate(args: argparse.Namespace) -> None:
    output = args.output_dir
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    thumbnails: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    transition_tables: dict[str, Any] = {}
    sim_args = simulation_args(args)

    with (output / "velocities.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "machine_id",
                "rule_bits_w0_d0_w1_d1",
                "copy_id",
                "initial_unwrapped_position",
                "final_unwrapped_position",
                "initial_tape_position",
                "final_tape_position",
                "velocity_fit",
                "velocity_net",
            ]
        )

        for machine_id in range(16):
            tm = one_state_tm(machine_id)
            result = simulate(sim_args, tm_override=tm)
            rgb = spacetime_rgb(result)
            image_name = f"spacetime_machine_{machine_id:02d}_{rule_bits(machine_id)}.png"
            save_plot(rgb, result, machine_id, args, plots_dir / image_name)
            thumbnails.append(contact_thumbnail(rgb, args.contact_pixels))
            transition_tables[str(machine_id)] = tm

            for copy_id in range(args.copies):
                writer.writerow(
                    [
                        machine_id,
                        rule_bits(machine_id),
                        copy_id,
                        int(result["initial_positions"][copy_id]),
                        int(result["position_trace"][-1, copy_id]),
                        int(result["initial_tape_positions"][copy_id]),
                        int(result["final_positions"][copy_id]),
                        float(result["velocities"][copy_id]),
                        float(result["net_velocities"][copy_id]),
                    ]
                )

            manifest_rows.append(
                {
                    "machine_id": machine_id,
                    "rule_bits_w0_d0_w1_d1": rule_bits(machine_id),
                    "rule": rule_description(machine_id),
                    "image": f"plots/{image_name}",
                    "seed": args.seed,
                    "states": 1,
                    "copies": args.copies,
                    "steps": args.steps,
                    "update_scheme": args.update_scheme,
                    "boundary": args.boundary,
                    "tape_length": args.tape_length,
                    "spacing": args.spacing,
                    "one_density": args.one_density,
                    "collision_policy": args.collision_policy,
                    "burn_in_fraction": args.burn_in_fraction,
                    "mean_velocity": float(np.mean(result["velocities"])),
                    "mean_absolute_velocity": float(
                        np.mean(np.abs(result["velocities"]))
                    ),
                    "rms_velocity": float(
                        np.sqrt(np.mean(result["velocities"] ** 2))
                    ),
                    "coincident_cell_events": result["coincident_cells"],
                    "conflicting_write_events": result["conflicting_cells"],
                }
            )
            print(
                f"Generated machine {machine_id:02d}/15 "
                f"({rule_bits(machine_id)}: {rule_description(machine_id)})",
                flush=True,
            )

    with (output / "transition_tables.json").open("w") as handle:
        json.dump(transition_tables, handle, indent=2)

    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    configuration = {
        "generator": Path(__file__).name,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulator": "many_turing_machines.py",
        "numpy_version": np.__version__,
        "bit_generator": "numpy.random.PCG64",
        "enumeration": {
            "machine_ids": [0, 15],
            "count": 16,
            "bit_order": ["write_on_0", "direction_on_0", "write_on_1", "direction_on_1"],
            "direction_encoding": {"0": "left", "1": "right"},
            "next_state": 1,
        },
        "control_semantics": (
            "Every machine receives the same initial tape and head positions because "
            "all runs use the same seed and an explicit transition-table override."
        ),
        "parameters": {
            "seed": args.seed,
            "states": 1,
            "copies": args.copies,
            "steps": args.steps,
            "update_scheme": args.update_scheme,
            "boundary": args.boundary,
            "tape_length": args.tape_length,
            "spacing": args.spacing,
            "one_density": args.one_density,
            "collision_policy": args.collision_policy,
            "burn_in_fraction": args.burn_in_fraction,
            "plot_width": args.plot_width,
            "record_every": args.record_every,
        },
    }
    with (output / "configuration.json").open("w") as handle:
        json.dump(configuration, handle, indent=2)

    rows = math.ceil(16 / args.grid_columns)
    fig, axes = plt.subplots(
        rows,
        args.grid_columns,
        figsize=(args.grid_columns * args.pdf_panel_size, rows * args.pdf_panel_size),
        squeeze=False,
    )
    for machine_id, ax in enumerate(axes.flat):
        if machine_id >= 16:
            ax.axis("off")
            continue
        ax.imshow(thumbnails[machine_id], origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(panel_title(machine_id, args, compact=True), fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)

    fig.suptitle("Complete one-state binary Turing-machine spacetime zoo", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.94, wspace=0.08, hspace=0.22)
    pdf_path = output / "spacetime_zoo.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=args.pdf_dpi)
    plt.close(fig)

    print(f"PDF contact sheet: {pdf_path.resolve()}")
    print(f"Individual plots: {plots_dir.resolve()}")
    print(f"Manifest: {(output / 'manifest.csv').resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="common tape seed")
    parser.add_argument("--copies", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--spacing", type=int, default=5)
    parser.add_argument("--one-density", type=float, default=0.5)
    parser.add_argument(
        "--collision-policy",
        choices=("majority", "keep", "zero", "one", "random"),
        default="majority",
    )
    parser.add_argument(
        "--update-scheme",
        choices=("synchronous", "random-sequential"),
        default="synchronous",
    )
    parser.add_argument("--boundary", choices=("open", "periodic"), default="periodic")
    parser.add_argument("--tape-length", type=int, default=10000)
    parser.add_argument("--burn-in-fraction", type=float, default=0.5)
    parser.add_argument("--plot-width", type=int, default=10000)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--contact-pixels", type=int, default=500)
    parser.add_argument("--png-dpi", type=int, default=150)
    parser.add_argument("--pdf-dpi", type=int, default=150)
    parser.add_argument("--pdf-panel-size", type=float, default=3.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("spacetime_zoo_1_state_all_16_periodic_L10000_T10000"),
    )
    args = parser.parse_args()

    positive = {
        "copies": args.copies,
        "steps": args.steps,
        "spacing": args.spacing,
        "plot_width": args.plot_width,
        "record_every": args.record_every,
        "grid_columns": args.grid_columns,
        "contact_pixels": args.contact_pixels,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if not 0 <= args.one_density <= 1:
        parser.error("--one-density must lie in [0, 1]")
    if not 0 <= args.burn_in_fraction < 1:
        parser.error("--burn-in-fraction must lie in [0, 1)")
    if args.boundary == "periodic" and args.tape_length < 1:
        parser.error("--boundary periodic requires a positive --tape-length")
    if args.boundary == "periodic" and args.plot_width > args.tape_length:
        parser.error("--plot-width cannot exceed --tape-length on a periodic tape")
    return args


if __name__ == "__main__":
    generate(parse_args())
