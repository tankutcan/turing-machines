#!/usr/bin/env python3
"""Generate a labeled zoo of shared-tape Turing-machine spacetime plots.

Each seed generates both a new random transition table and a new random tape.
The output contains individual PNGs, machine-readable metadata, all measured
velocities, all transition tables, and a one-page PDF contact sheet.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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

from many_turing_machines import simulate


def labeled_title(seed: int, args: argparse.Namespace, compact: bool = False) -> str:
    """Return a complete, reproducibility-oriented label for one panel."""
    if compact:
        tape_length = args.tape_length if args.boundary == "periodic" else "open"
        return (
            f"seed={seed}  S={args.states}  N={args.copies}  T={args.steps}  "
            f"rules={args.machine_ensemble}\n"
            f"bc={args.boundary}  L={tape_length}  upd={args.update_scheme}  "
            f"vote={args.collision_policy}\n"
            f"q0={args.initial_state_mode}  dx={args.spacing}  p1={args.one_density:g}  W={args.plot_width}  "
            f"r={args.record_every}"
        )
    collision = args.collision_policy if args.update_scheme == "synchronous" else "n/a"
    tape_length = args.tape_length if args.boundary == "periodic" else "unbounded"
    return (
        f"seed={seed} | states={args.states} | copies={args.copies} | "
        f"rules={args.machine_ensemble}\n"
        f"initial states={args.initial_state_mode} | steps={args.steps} | "
        f"spacing={args.spacing} | p(1)={args.one_density:g}\n"
        f"update={args.update_scheme} | collision={collision} | boundary={args.boundary}\n"
        f"tape length={tape_length} | plot width={args.plot_width} | record every={args.record_every}"
    )


def spacetime_rgb(result: dict[str, Any]) -> np.ndarray:
    """Render tape bits as grayscale, single heads red, and stacked heads yellow."""
    tape = result["spacetime"]
    occupancy = result["head_occupancy"]
    gray = np.where(tape[..., None] == 0, 12, 243).astype(np.uint8)
    rgb = np.repeat(gray, 3, axis=2)
    rgb[occupancy == 1] = (225, 45, 35)
    rgb[occupancy > 1] = (255, 190, 0)
    return rgb


def save_individual_plot(
    rgb: np.ndarray,
    result: dict[str, Any],
    seed: int,
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
    ax.set_title(labeled_title(seed, args), fontsize=10)
    ax.set_xlabel("Tape position")
    ax.set_ylabel("Time")
    fig.savefig(path, dpi=args.png_dpi)
    plt.close(fig)


def contact_thumbnail(rgb: np.ndarray, maximum: int) -> np.ndarray:
    """Downsample an RGB spacetime image for a memory-efficient contact sheet."""
    height, width = rgb.shape[:2]
    row_indices = np.linspace(0, height - 1, min(height, maximum), dtype=int)
    column_indices = np.linspace(0, width - 1, min(width, maximum), dtype=int)
    return rgb[row_indices][:, column_indices]


def simulation_args(seed: int, args: argparse.Namespace) -> SimpleNamespace:
    """Build the subset of arguments consumed by many_turing_machines.simulate."""
    return SimpleNamespace(
        seed=seed,
        states=args.states,
        copies=args.copies,
        steps=args.steps,
        spacing=args.spacing,
        plot_width=args.plot_width,
        one_density=args.one_density,
        collision_policy=args.collision_policy,
        burn_in_fraction=args.burn_in_fraction,
        record_every=args.record_every,
        machine_ensemble=args.machine_ensemble,
        initial_state_mode=args.initial_state_mode,
        update_scheme=args.update_scheme,
        boundary=args.boundary,
        tape_length=args.tape_length,
    )


def generate_zoo(args: argparse.Namespace) -> None:
    output = args.output_dir
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.start_seed, args.start_seed + args.count))
    thumbnails: list[np.ndarray] = []
    transition_tables: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []

    velocities_path = output / "velocities.csv"
    with velocities_path.open("w", newline="") as velocity_handle:
        velocity_writer = csv.writer(velocity_handle)
        velocity_writer.writerow(
            [
                "seed",
                "copy_id",
                "initial_unwrapped_position",
                "final_unwrapped_position",
                "initial_tape_position",
                "initial_state",
                "final_tape_position",
                "velocity_fit",
                "velocity_net",
            ]
        )

        for index, seed in enumerate(seeds, start=1):
            result = simulate(simulation_args(seed, args))
            rgb = spacetime_rgb(result)
            image_name = f"spacetime_seed_{seed:06d}.png"
            save_individual_plot(rgb, result, seed, args, plots_dir / image_name)
            thumbnails.append(contact_thumbnail(rgb, args.contact_pixels))
            transition_tables[str(seed)] = result["tm"]

            for copy_id in range(args.copies):
                velocity_writer.writerow(
                    [
                        seed,
                        copy_id,
                        int(result["initial_positions"][copy_id]),
                        int(result["position_trace"][-1, copy_id]),
                        int(result["initial_tape_positions"][copy_id]),
                        int(result["initial_states"][copy_id]),
                        int(result["final_positions"][copy_id]),
                        float(result["velocities"][copy_id]),
                        float(result["net_velocities"][copy_id]),
                    ]
                )

            manifest_rows.append(
                {
                    "seed": seed,
                    "image": f"plots/{image_name}",
                    "states": args.states,
                    "copies": args.copies,
                    "machine_ensemble": args.machine_ensemble,
                    "initial_state_mode": args.initial_state_mode,
                    "update_scheme": args.update_scheme,
                    "boundary": args.boundary,
                    "tape_length": args.tape_length,
                    "steps": args.steps,
                    "spacing": args.spacing,
                    "one_density": args.one_density,
                    "collision_policy": args.collision_policy,
                    "plot_width": args.plot_width,
                    "record_every": args.record_every,
                    "burn_in_fraction": args.burn_in_fraction,
                    "mean_velocity": float(np.mean(result["velocities"])),
                    "mean_absolute_velocity": float(np.mean(np.abs(result["velocities"]))),
                    "rms_velocity": float(np.sqrt(np.mean(result["velocities"] ** 2))),
                    "coincident_cell_events": result["coincident_cells"],
                    "conflicting_write_events": result["conflicting_cells"],
                }
            )

            if index == 1 or index % 10 == 0 or index == args.count:
                print(f"Generated {index}/{args.count} systems (latest seed={seed})", flush=True)

    if args.machine_ensemble == "independent":
        transition_table_path = output / "transition_tables.json.gz"
        with gzip.open(transition_table_path, "wt") as handle:
            json.dump(transition_tables, handle, separators=(",", ":"))
    else:
        transition_table_path = output / "transition_tables.json"
        with transition_table_path.open("w") as handle:
            json.dump(transition_tables, handle, indent=2)

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    configuration = {
        "generator": Path(__file__).name,
        "generator_sha256": script_hash,
        "numpy_version": np.__version__,
        "bit_generator": "numpy.random.PCG64",
        "transition_tables": transition_table_path.name,
        "seed_semantics": (
            "Each seed initializes its machine ensemble, random initial states when "
            "requested, and random tape."
        ),
        "update_semantics": (
            "Synchronous updates resolve simultaneous writes with collision_policy. "
            "Random-sequential updates shuffle agents every sweep and apply each "
            "read, write, state transition, and move immediately."
        ),
        "boundary_semantics": (
            "Periodic positions are reduced modulo tape_length for tape access and "
            "spacetime plots; velocity fits use separately tracked unwrapped positions."
        ),
        "parameters": {
            "start_seed": args.start_seed,
            "count": args.count,
            "states": args.states,
            "copies": args.copies,
            "machine_ensemble": args.machine_ensemble,
            "initial_state_mode": args.initial_state_mode,
            "update_scheme": args.update_scheme,
            "boundary": args.boundary,
            "tape_length": args.tape_length,
            "steps": args.steps,
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

    rows = math.ceil(args.count / args.grid_columns)
    panel_size = args.pdf_panel_size
    fig, axes = plt.subplots(
        rows,
        args.grid_columns,
        figsize=(args.grid_columns * panel_size, rows * panel_size),
        squeeze=False,
    )
    for panel_index, ax in enumerate(axes.flat):
        if panel_index >= args.count:
            ax.axis("off")
            continue
        ax.imshow(thumbnails[panel_index], origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(labeled_title(seeds[panel_index], args, compact=True), fontsize=5.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)

    fig.suptitle(
        f"Shared-tape random Turing-machine zoo ({args.count} systems)",
        fontsize=16,
        y=0.998,
    )
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.012, top=0.955, wspace=0.08, hspace=0.22)
    pdf_path = output / "spacetime_zoo.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=args.pdf_dpi)
    plt.close(fig)

    print(f"PDF contact sheet: {pdf_path.resolve()}")
    print(f"Individual plots: {plots_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--copies", type=int, default=200)
    parser.add_argument(
        "--machine-ensemble",
        choices=("identical", "independent"),
        default="identical",
        help="share one transition table or sample one independently per agent",
    )
    parser.add_argument(
        "--initial-state-mode",
        choices=("fixed", "random"),
        default="fixed",
        help="start every agent in state 1 or sample states uniformly and independently",
    )
    parser.add_argument("--steps", type=int, default=5000)
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
    parser.add_argument(
        "--boundary", choices=("open", "periodic"), default="open"
    )
    parser.add_argument("--tape-length", type=int)
    parser.add_argument("--burn-in-fraction", type=float, default=0.5)
    parser.add_argument("--plot-width", type=int, default=4000)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--grid-columns", type=int, default=10)
    parser.add_argument("--contact-pixels", type=int, default=300)
    parser.add_argument("--png-dpi", type=int, default=150)
    parser.add_argument("--pdf-dpi", type=int, default=150)
    parser.add_argument("--pdf-panel-size", type=float, default=2.8)
    parser.add_argument("--output-dir", type=Path, default=Path("spacetime_zoo"))
    args = parser.parse_args()

    positive = {
        "count": args.count,
        "states": args.states,
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
    if args.boundary == "periodic" and (args.tape_length is None or args.tape_length < 1):
        parser.error("--boundary periodic requires a positive --tape-length")
    if args.boundary == "periodic" and args.plot_width > args.tape_length:
        parser.error("--plot-width cannot exceed --tape-length on a periodic tape")
    return args


def main() -> None:
    generate_zoo(parse_args())


if __name__ == "__main__":
    main()
