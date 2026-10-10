#!/usr/bin/env python3
"""Classify saved two-state TM graphs and annotate their rule-ordered zoo.

Uses the archived transition tables, spacetime panels and per-head velocities;
no new simulation is required. All graph statements hold for either read bit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .order_two_state_zoo import (
    DEFAULT_SOURCE, archived_panels, grouped_records, load_zoo, plot_zoo, sha256,
)


def state_structure(rule: dict) -> dict:
    """Exhaustive SCC classification for two states, with source columns.

    An isolated *proper* sector is a singleton here. The whole {1,2} graph
    being closed is not evidence of two separately conserved populations.
    """
    destinations = {q: {rule[f"S_{q}_{b}"] for b in (0, 1)} for q in (1, 2)}
    a12, a21 = 2 in destinations[1], 1 in destinations[2]
    if a12 and a21:
        tag, sccs, sinks, isolated = "C", [[1, 2]], [[1, 2]], []
    elif a12:
        tag, sccs, sinks, isolated = "S2", [[1], [2]], [[2]], []
    elif a21:
        tag, sccs, sinks, isolated = "S1", [[1], [2]], [[1]], []
    else:
        tag, sccs, sinks, isolated = "I", [[1], [2]], [[1], [2]], [[1], [2]]
    constant_moves = {q: rule[f"M_{q}_0"] == rule[f"M_{q}_1"] for q in (1, 2)}
    matrix = [[sum(rule[f"S_{q}_{b}"] == j for b in (0, 1))
               for q in (1, 2)] for j in (1, 2)]
    return {
        "structure_class": tag,
        "adjacency_source_columns": json.dumps(matrix),
        "strongly_connected_components": json.dumps(sccs),
        "sink_components": json.dumps(sinks),
        "isolated_proper_components": json.dumps(isolated),
        "scc_count": len(sccs), "sink_count": len(sinks),
        "weak_component_count": 2 if tag == "I" else 1,
        "extra_conserved_state_populations": int(tag == "I"),
        "strict_alternation": destinations == {1: {2}, 2: {1}},
        "q1_constant_move": constant_moves[1], "q2_constant_move": constant_moves[2],
        "q1_rule_mean_move": (rule["M_1_0"] + rule["M_1_1"]) / 2,
        "q2_rule_mean_move": (rule["M_2_0"] + rule["M_2_1"]) / 2,
        "exact_counterpropagating_sectors": bool(
            tag == "I" and all(constant_moves.values())
            and rule["M_1_0"] == -rule["M_2_0"]
        ),
    }


def velocity_statistics(values: np.ndarray, threshold: float, fraction: float) -> dict:
    """Finite-window drift classification; not an asymptotic ballistic test."""
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected nonempty, finite head velocities")
    left, right = int(np.sum(values < -threshold)), int(np.sum(values > threshold))
    count = len(values)
    return {
        "head_count": count, "left_count": left, "right_count": right,
        "near_zero_count": count - left - right,
        "left_fraction": left / count, "right_fraction": right / count,
        "near_zero_fraction": (count - left - right) / count,
        "counterflow_flag": left / count >= fraction and right / count >= fraction,
        "velocity_fit_mean": float(np.mean(values)),
        "velocity_fit_std": float(np.std(values)),
        "velocity_fit_min": float(np.min(values)), "velocity_fit_max": float(np.max(values)),
        "velocity_fit_q05": float(np.quantile(values, 0.05)),
        "velocity_fit_median": float(np.median(values)),
        "velocity_fit_q95": float(np.quantile(values, 0.95)),
    }


def analyze(source: Path, threshold: float, fraction: float):
    configuration, records = load_zoo(source)
    heads = defaultdict(list)
    with (source / "velocities.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            heads[int(row["seed"])].append(row)
    if set(heads) != {r["seed"] for r in records}:
        raise ValueError("Velocity seeds do not match the transition tables")
    by_state, velocity_arrays = [], {}
    copies = configuration["parameters"]["copies"]
    for record in records:
        seed = record["seed"]
        rows = sorted(heads[seed], key=lambda r: int(r["copy_id"]))
        if [int(r["copy_id"]) for r in rows] != list(range(copies)):
            raise ValueError(f"Seed {seed}: missing or duplicated head records")
        initial = np.array([int(r["initial_state"]) for r in rows])
        if not np.all(np.isin(initial, [1, 2])):
            raise ValueError("Initial state outside {1,2}")
        velocities = np.array([float(r["velocity_fit"]) for r in rows])
        net = np.array([float(r["velocity_net"]) for r in rows])
        if not np.all(np.isfinite(net)) or np.any(np.abs(net) > 1 + 1e-10):
            raise ValueError("Invalid net velocities")
        record.update(state_structure(record))
        record.update(velocity_statistics(velocities, threshold, fraction))
        record["measured_mean_velocity_net"] = float(np.mean(net))
        record["velocity_threshold"] = threshold
        record["minimum_counterflow_fraction"] = fraction
        if not np.isclose(record["velocity_fit_mean"], record["measured_mean_velocity_fit"],
                          atol=1e-10, rtol=0):
            raise ValueError(f"Seed {seed}: fitted velocities disagree with manifest")
        for q in (1, 2):
            mask = initial == q
            record[f"initial_population_q{q}"] = int(mask.sum())
            if not mask.any():
                continue
            by_state.append({"seed": seed, "initial_state": q,
                             "structure_class": record["structure_class"],
                             "state_is_conserved": record["structure_class"] == "I",
                             **velocity_statistics(velocities[mask], threshold, fraction),
                             "velocity_net_mean": float(np.mean(net[mask]))})
            # For isolated fixed-direction states, this is an exact kinematic identity.
            if record["structure_class"] == "I" and record[f"q{q}_constant_move"]:
                expected = record[f"M_{q}_0"]
                if not (np.allclose(velocities[mask], expected, atol=1e-10, rtol=0)
                        and np.allclose(net[mask], expected, atol=1e-10, rtol=0)):
                    raise ValueError(f"Seed {seed}, state {q}: violates exact fixed velocity")
        velocity_arrays[seed] = velocities
    return configuration, records, by_state, velocity_arrays


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(records: list[dict], velocities: dict, fraction: float) -> dict:
    return {
        "class_counts": dict(Counter(r["structure_class"] for r in records)),
        "zero_mean_move_class_counts": dict(Counter(r["structure_class"] for r in records
                                                    if r["mean_move"] == 0)),
        "isolated_seeds": [r["seed"] for r in records if r["structure_class"] == "I"],
        "strict_alternation_seeds": [r["seed"] for r in records if r["strict_alternation"]],
        "exact_counterpropagating_seeds": [r["seed"] for r in records
                                          if r["exact_counterpropagating_sectors"]],
        "opposite_drift_seeds": [r["seed"] for r in records if r["counterflow_flag"]],
        "threshold_sensitivity": [
            {"velocity_threshold": threshold, "minimum_fraction_each_direction": fraction,
             "seeds": [seed for seed, v in velocities.items()
                       if np.mean(v < -threshold) >= fraction
                       and np.mean(v > threshold) >= fraction]}
            for threshold in (0.001, 0.01, 0.05, 0.1)
        ],
    }


def write_report(prefix: Path, records: list[dict], by_state: list[dict], summary: dict,
                 configuration: dict, threshold: float, fraction: float) -> None:
    p = configuration["parameters"]
    meanings = {"I": "two isolated singleton sectors", "S1": "sink {1}, with incoming edge(s)",
                "S2": "sink {2}, with incoming edge(s)", "C": "one strongly connected component {1,2}"}
    lines = ["# Two-state connectivity and measured drift", "",
             f"[Annotated PDF]({prefix.name}.pdf) · [PNG]({prefix.name}.png) · "
             f"[Per-machine CSV]({prefix.name}.csv) · [Initial-state velocity groups]({prefix.name}_by_initial_state.csv) · "
             f"[Provenance and summary]({prefix.name}.json)", "",
             f"Analyzed {len(records)} archived runs, with {p['copies']} identical heads per run, "
             f"n=2 states, T={p['steps']}, {p['boundary']} boundaries (L={p.get('tape_length')}), "
             f"{p['update_scheme']} updates and {p['collision_policy']} writes. No simulations were rerun.", "",
             "## Plot annotations", "",
             "The horizontal/vertical axes remain uniform rule-mean move/write bit. Colored borders and "
             "I/S1/S2/C labels describe state connectivity, not the tape colors. Tape 1 is light and 0 is dark; "
             "red/gold indicate head occupancy. Every panel retains its original seed.", "",
             f"L/R give rounded percentages of heads with fitted v < −{threshold:g} / v > {threshold:g}; "
             f"the remaining heads have |v| ≤ {threshold:g}. An asterisk means at least {100*fraction:g}% "
             "of heads in each direction. B marks exact conserved sectors moving at ±1; C~ marks strict "
             "state alternation. Plot percentages are rounded; CSV fractions are unrounded.", "",
             f"Velocities are unwrapped-position least-squares slopes over the final "
             f"{100*(1-p['burn_in_fraction']):g}% of each run, as in the original archive. "
             "Opposite signs over a finite window do not establish persistent ballistic modes.", "",
             "## Structural counts", "", "| Class | Meaning | All runs | Mean move = 0 |",
             "| --- | --- | ---: | ---: |"]
    for tag, meaning in meanings.items():
        lines.append(f"| {tag} | {meaning} | {summary['class_counts'].get(tag, 0)} | "
                     f"{summary['zero_mean_move_class_counts'].get(tag, 0)} |")
    lines += ["", "I has two exactly conserved state populations (one independent quantity beyond total head count). "
              "S1/S2 have a nondecreasing sink population, not a separately conserved population. C has no "
              "nontrivial tape-independent conserved linear combination of the two state populations. "
              "C~ swaps the state populations each step; their difference alternates sign, rather than being conserved.", "",
              "For I, heads never change state, but the sectors can still interact through their shared tape. "
              "Conserved sector populations do not generally imply independent motion. For the B subclass, "
              "each state's direction is independent of the read bit and opposite to the other state's direction: "
              "these head trajectories are exactly ballistic and independent of tape feedback.", "",
              "## Isolated-sector runs", "",
              "| Seed | Mean move | Mean write | Initial N1/N2 | Mean v in q1/q2 | Exact opposite ±1 sectors |",
              "| --- | ---: | ---: | --- | --- | --- |"]
    state_lookup = {(r["seed"], r["initial_state"]): r for r in by_state}
    for r in records:
        if r["structure_class"] != "I":
            continue
        means = [state_lookup.get((r["seed"], q), {}).get("velocity_fit_mean") for q in (1, 2)]
        mean_text = "/".join("empty" if v is None else f"{v:+.6f}" for v in means)
        lines.append(f"| {r['seed']} | {r['mean_move']:g} | {r['mean_write_bit']:g} | "
                     f"{r['initial_population_q1']}/{r['initial_population_q2']} | {mean_text} | "
                     f"{'yes' if r['exact_counterpropagating_sectors'] else 'no'} |")
    lines += ["", "The per-initial-state CSV groups heads by their initial state, not their late-time state. "
              "Only I guarantees that this label remains the same throughout the run.", "",
              "## Sensitivity of the opposite-drift flag", "",
              f"Each direction must contain at least {100*fraction:g}% of the heads.", "",
              "| Speed threshold | Flagged seeds |", "| ---: | --- |"]
    for entry in summary["threshold_sensitivity"]:
        lines.append(f"| {entry['velocity_threshold']:g} | {', '.join(map(str, entry['seeds'])) or 'none'} |")
    lines += ["", "This distinguishes exact graph/kinematic statements from threshold-dependent finite-time "
              "measurements. No assumption is made that all graph-allowed paths can be realized by a tape.", "",
              "## Reproduce", "", "From the project root:", "", "```sh",
              "env MPLCONFIGDIR=/tmp/turing-machines-matplotlib .venv/bin/python -m scripts.analyze_two_state_structure",
              "env MPLCONFIGDIR=/tmp/turing-machines-matplotlib .venv/bin/python -m unittest -v tests.test_order_two_state_zoo tests.test_two_state_structure",
              "```", "", "Use --velocity-threshold and --counterflow-fraction to change the finite-time flag. "
              "Use --source-dir or --output-prefix for other saved two-state zoos. "
              "Relative source paths are interpreted from the project root.", ""]
    prefix.with_suffix(".md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("results/two_state_rule_averages/zoo_tm2_state_structure"))
    parser.add_argument("--velocity-threshold", type=float, default=0.01)
    parser.add_argument("--counterflow-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.velocity_threshold < 1:
        parser.error("--velocity-threshold must be in [0,1)")
    if not 0 < args.counterflow_fraction <= 0.5:
        parser.error("--counterflow-fraction must be in (0,0.5]")
    config, records, by_state, velocities = analyze(
        args.source_dir, args.velocity_threshold, args.counterflow_fraction)
    panels = archived_panels(args.source_dir, [r["seed"] for r in records])
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".csv"), sorted(records, key=lambda r:
              (-r["mean_write_bit"], r["mean_move"], r["seed"])))
    write_csv(prefix.with_name(prefix.name + "_by_initial_state.csv"), by_state)
    plot_zoo(records, panels, config["parameters"], prefix, annotate_structure=True,
             velocity_threshold=args.velocity_threshold, counterflow_fraction=args.counterflow_fraction)
    summary = make_summary(records, velocities, args.counterflow_fraction)
    groups = grouped_records(records)
    provenance = {
        "generator": Path(__file__).name, "generator_sha256": sha256(Path(__file__)),
        "renderer": "order_two_state_zoo.py", "renderer_sha256": sha256(Path(__file__).with_name("order_two_state_zoo.py")),
        "source_directory": str(args.source_dir), "source_configuration": config,
        "source_sha256": {name: sha256(args.source_dir / name) for name in
                          ("configuration.json", "manifest.csv", "transition_tables.json", "velocities.csv", "spacetime_zoo.pdf")},
        "velocity_threshold": args.velocity_threshold, "counterflow_fraction": args.counterflow_fraction,
        "image_provenance": "Archived contact-sheet rasters, with verified seed order; time increases upward; no simulation reruns",
        "classification": {"I": "two isolated singleton SCCs", "S1": "proper sink {1}",
                           "S2": "proper sink {2}", "C": "one SCC {1,2}"},
        "extra_conserved_state_populations": "Dimension of tape-independent conserved linear state-population quantities minus one for total head count",
        "summary": summary,
        "bins": [{"mean_write_bit": w / 4, "mean_move": d / 2 - 1,
                  "seeds": [r["seed"] for r in groups[w, d]]}
                 for w in range(4, -1, -1) for d in range(5)],
    }
    prefix.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(prefix, records, by_state, summary, config,
                 args.velocity_threshold, args.counterflow_fraction)
    print(json.dumps(summary, indent=2))
    print(f"Annotated zoo and analysis: {prefix.resolve()}.*")


if __name__ == "__main__":
    main()
