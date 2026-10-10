"""Arrange a saved identical-TM_2(2) zoo by uniform transition-table averages.

No simulations are rerun. Poppler's pdfimages and pdftotext extract the archived
contact-sheet panels and verify their seed order. Outputs: PDF, PNG, CSV, JSON.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_SOURCE = Path(
    "results/spacetime_zoo_2_states_1000_identical_agents_"
    "random_initial_states_periodic_L10000_T10000"
)

STRUCTURE_COLORS = {"I": "#8e44ad", "S1": "#d89000", "S2": "#00876c", "C": "#2878b5"}


def rule_observables(tm: dict) -> dict[str, int | float]:
    """Give every (q,b) entry weight 1/4, including dynamically unused entries.

    The stored direction d is a BIT: M = 2*d - 1, not d itself.
    Accept both integer keys (simulator) and string keys (JSON archive).
    """
    if {str(q) for q in tm} != {"1", "2"}:
        raise ValueError("Expected exactly two states, numbered 1 and 2")
    entries = []
    result = {}
    for q in (1, 2):
        row = tm[q] if q in tm else tm[str(q)]
        if {str(b) for b in row} != {"0", "1"}:
            raise ValueError("Each state must have exactly the read bits 0 and 1")
        for b in (0, 1):
            entry = row[b] if b in row else row[str(b)]
            if len(entry) != 3:
                raise ValueError("Expected [write_bit, move_bit, next_state]")
            w, d, s = entry
            if w not in (0, 1) or d not in (0, 1) or s not in (1, 2):
                raise ValueError(f"Invalid two-state transition: {entry}")
            entries.append((int(w), int(d)))
            result.update({f"W_{q}_{b}": int(w), f"M_{q}_{b}": 2 * int(d) - 1,
                           f"S_{q}_{b}": int(s)})
    ones = sum(w for w, _ in entries)
    rights = sum(d for _, d in entries)
    return {"mean_write_bit": ones / 4, "mean_move": (2 * rights - 4) / 4,
            "write_one_entries": ones, "move_right_entries": rights, **result}


def load_zoo(source: Path) -> tuple[dict, list[dict]]:
    configuration = json.loads((source / "configuration.json").read_text())
    parameters = configuration["parameters"]
    if (configuration["generator"] != "generate_spacetime_zoo.py"
            or parameters["states"] != 2
            or parameters["machine_ensemble"] != "identical"):
        raise ValueError("Input must be an identical two-state generate_spacetime_zoo archive")
    tables = json.loads((source / "transition_tables.json").read_text())
    with (source / "manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    seeds = [int(row["seed"]) for row in manifest]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Manifest must have distinct seeds and at least one run")
    if len(seeds) != parameters["count"] or set(map(str, seeds)) != set(tables):
        raise ValueError("Configuration, manifest and transition-table seeds disagree")
    records = []
    for row in manifest:
        for key, value in parameters.items():
            if key in row and str(value) != row[key]:
                raise ValueError(f"Seed {row['seed']}: inconsistent parameter {key}")
        seed = int(row["seed"])
        records.append({"seed": seed, **rule_observables(tables[str(seed)]),
                        "measured_mean_velocity_fit": float(row["mean_velocity"]),
                        "source_image": str(source / row["image"])})
    return configuration, records


def archived_panels(source: Path, seeds: list[int]) -> dict[int, np.ndarray]:
    """Extract panels in the original generator's row-major artist order.

    This generator draws one image per seed, with no other image artists.
    Validate its printed seed sequence and image count before pairing them.
    Extracted PNGs are already top-to-bottom: t=T at top, t=0 at bottom.
    """
    for command in ("pdfimages", "pdftotext"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Missing {command}: install Poppler to read archived PDFs")
    pdf = source / "spacetime_zoo.pdf"
    labels = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                            check=True, capture_output=True, text=True).stdout
    printed_seeds = [int(s) for s in re.findall(r"seed\s*=\s*(\d+)", labels)]
    if printed_seeds != seeds:
        raise ValueError("PDF seed order differs from the manifest; refusing to relabel images")
    with tempfile.TemporaryDirectory(prefix="tm-rule-panels-") as temp:
        prefix = Path(temp) / "panel"
        subprocess.run(["pdfimages", "-png", str(pdf), str(prefix)], check=True)
        images = sorted(Path(temp).glob("panel-*.png"),
                        key=lambda p: int(p.stem.split("-")[-1]))
        if len(images) != len(seeds):
            raise ValueError("Expected exactly one PDF image per seed")
        return {seed: plt.imread(path) for seed, path in zip(seeds, images)}


def grouped_records(records: list[dict]) -> dict[tuple[int, int], list[dict]]:
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["seed"]):
        groups[record["write_one_entries"], record["move_right_entries"]].append(record)
    return groups


def plot_zoo(records: list[dict], panels: dict[int, np.ndarray], parameters: dict,
             prefix: Path, *, annotate_structure: bool = False,
             velocity_threshold: float = 0.01, counterflow_fraction: float = 0.05) -> None:
    groups = grouped_records(records)
    fig = plt.figure(figsize=(25, 28 if annotate_structure else 26), facecolor="white")
    left, right = 0.087, 0.987
    bottom, top = (0.110, 0.858) if annotate_structure else (0.090, 0.895)
    grid = fig.add_gridspec(5, 5, left=left, right=right, bottom=bottom, top=top,
                           wspace=0.035, hspace=0.045)
    for row, ones in enumerate(range(4, -1, -1)):
        for rights in range(5):
            cell = fig.add_subplot(grid[row, rights])
            cell.set_facecolor("#f3f5f7")
            cell.set_xticks([])
            cell.set_yticks([])
            for spine in cell.spines.values():
                spine.set_color("#bcc5ce")
            runs = groups[ones, rights]
            cell.text(0.025, 0.975, f"{len(runs)} run{'s' if len(runs) != 1 else ''}",
                      transform=cell.transAxes, va="top", fontsize=11, color="#465568")
            if not runs:
                cell.text(0.5, 0.5, "Not sampled", ha="center", va="center",
                          transform=cell.transAxes, color="#8290a0", fontsize=15)
            else:
                if annotate_structure:
                    bounds = cell.get_position()
                    cell_width = bounds.width * fig.get_figwidth()
                    cell_height = bounds.height * fig.get_figheight()
                    title_inches = 0.25
                    # Maximize the square image size after reserving two title
                    # lines, rather than squeezing a dense bin into four columns.
                    columns = max(range(1, min(6, len(runs)) + 1), key=lambda c: min(
                        cell_width * 0.94 * 0.94 / c,
                        cell_height * 0.86 * 0.975 / math.ceil(len(runs) / c) - title_inches,
                    ))
                else:
                    columns = min(4, math.ceil(math.sqrt(len(runs))))
                rows = math.ceil(len(runs) / columns)
                # Full x/t range in each inset, with the same orientation as the archive.
                for k, run in enumerate(runs):
                    i, j = divmod(k, columns)
                    dx, dy = 0.94 / columns, 0.86 / rows
                    image_height = (dy * 0.975 - title_inches / cell_height
                                    if annotate_structure else dy * 0.83)
                    ax = cell.inset_axes([0.03 + j * dx, 0.035 + (rows - 1 - i) * dy,
                                          dx * 0.94, image_height])
                    ax.imshow(panels[run["seed"]], origin="upper", aspect="equal",
                              extent=(0, parameters["plot_width"], 0, parameters["steps"]),
                              interpolation="none")
                    ax.set_anchor("N")
                    if annotate_structure:
                        tag = run["structure_class"]
                        if run["strict_alternation"]:
                            tag += "~"
                        if run["counterflow_flag"]:
                            tag += "*"
                        if run["exact_counterpropagating_sectors"]:
                            tag += "B"
                        ax.set_title(
                            f"seed {run['seed']} {tag}\n"
                            f"L{100 * run['left_fraction']:.0f} R{100 * run['right_fraction']:.0f}",
                            fontsize=6.5, pad=2, linespacing=1.1,
                        )
                    else:
                        ax.set_title(f"seed {run['seed']}", fontsize=8, pad=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.8 if annotate_structure else 0.35)
                        if annotate_structure:
                            spine.set_color(STRUCTURE_COLORS[run["structure_class"]])
            if rights == 0:
                cell.set_ylabel(f"{ones / 4:g}", fontsize=20, rotation=0,
                                labelpad=22, va="center")
            if row == 4:
                cell.set_xlabel(f"{rights / 2 - 1:g}", fontsize=20, labelpad=12)

    title = ("Two-state zoo: state connectivity and measured head drift" if annotate_structure
             else "Two-state Turing-machine zoo: ordered by rule averages")
    fig.suptitle(title,
                 fontsize=26, y=0.979)
    seeds = sorted(r["seed"] for r in records)
    seed_text = (f"seeds {seeds[0]}–{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1))
                 else "seeds listed in panels and companion CSV")
    p = parameters
    geometry = (f"periodic L={p['tape_length']}" if p["boundary"] == "periodic"
                else f"unbounded tape; displayed width={p['plot_width']}")
    collision = (f"{p['collision_policy']} writes (ties keep old bit)"
                 if p["update_scheme"] == "synchronous" and p["collision_policy"] == "majority"
                 else f"write policy={p['collision_policy']}"
                 if p["update_scheme"] == "synchronous" else "immediate sequential writes")
    fig.text(0.5, 0.952,
             f"{len(records)} runs ({seed_text}) | n=2 states | N={p['copies']} identical heads/run | "
             f"T={p['steps']} | {geometry}\n"
             f"{p['update_scheme']} updates; {collision} | initial states: {p['initial_state_mode']} | "
             f"initial tape: Bernoulli({p['one_density']:g})\n"
             f"initial spacing={p['spacing']}; x(0,i)={p['spacing']}[i−1−floor(N/2)] | "
             f"snapshots every {p['record_every']} steps; archived overview resolution",
             ha="center", va="center", fontsize=12, linespacing=1.6)
    if annotate_structure:
        meanings = {"I": "two isolated states", "S1": "sink {1}",
                    "S2": "sink {2}", "C": "one SCC {1,2}"}
        handles = [Line2D([0], [0], color=STRUCTURE_COLORS[tag], linewidth=3,
                          label=f"{tag}: {meaning} ({sum(r['structure_class'] == tag for r in records)} runs)")
                   for tag, meaning in meanings.items()]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.921),
                   ncol=4, frameon=False, fontsize=12)
        fig.text(0.54, 0.888,
                 f"L/R: rounded % of heads with fitted v < −{velocity_threshold:g} / v > {velocity_threshold:g}; "
                 "the remainder have near-zero fitted drift. C~: states strictly alternate.\n"
                 f"*: at least {100 * counterflow_fraction:g}% in each direction. "
                 "B: exact conserved sectors moving at v=+1 and −1. Other opposite-sign drifts need not be ballistic.",
                 ha="center", va="center", fontsize=11, linespacing=1.6)
    fig.text((left + right) / 2, 0.063 if annotate_structure else 0.043,
             r"Mean move: $\overline{M}=\frac{1}{4}\sum_{q=1}^{2}\sum_{b=0}^{1}M_q(b)$",
             fontsize=22, ha="center")
    fig.text(0.021, (top + bottom) / 2,
             r"Mean written bit: $\overline{W}=\frac{1}{4}\sum_{q=1}^{2}\sum_{b=0}^{1}W_q(b)$",
             fontsize=22, va="center", rotation=90)
    fig.text(0.5, 0.025 if annotate_structure else 0.019,
             "Within panels: space →, time ↑ (same x/t scale ratio) | dark/light: tape 0/1 | red: one head; gold: multiple heads\n"
             "Both coordinates weight the four rules equally; mean move is not the measured drift. "
             "Seeds increase within each cell; empty cells are unsampled, not impossible.",
             ha="center", va="center", fontsize=12, linespacing=1.6)
    fig.savefig(prefix.with_suffix(".pdf"), metadata={
        "Title": title,
        "Subject": "Horizontal: mean move; vertical: mean written bit; no simulation reruns",
    })
    fig.savefig(prefix.with_suffix(".png"), dpi=110)
    plt.close(fig)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("results/two_state_rule_averages/zoo_tm2_rule_averages"))
    args = parser.parse_args()
    source, prefix = args.source_dir, args.output_prefix
    configuration, records = load_zoo(source)
    panels = archived_panels(source, [r["seed"] for r in records])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Same order as the figure: W descending, M ascending, then seed ascending.
    ordered = sorted(records, key=lambda r: (-r["mean_write_bit"], r["mean_move"], r["seed"]))
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    plot_zoo(records, panels, configuration["parameters"], prefix)
    groups = grouped_records(records)
    provenance: dict[str, Any] = {
        "generator": Path(__file__).name,
        "generator_sha256": sha256(Path(__file__)),
        "source_directory": str(source),
        "source_sha256": {name: sha256(source / name) for name in
                          ("spacetime_zoo.pdf", "manifest.csv", "configuration.json", "transition_tables.json")},
        "source_configuration": configuration,
        "definitions": {
            "mean_write_bit": "sum W_q(b) / 4 over q=1,2 and b=0,1",
            "mean_move": "sum M_q(b) / 4; stored move bit d is converted to M=2*d-1",
            "measured_mean_velocity_fit": "Original manifest mean_velocity: mean over heads of late-time OLS slopes of unwrapped position; burn_in_fraction is in source_configuration",
        },
        "layout": "M increases left to right; W increases bottom to top; increasing seed within each cell; all panels have equal physical x/t aspect ratio",
        "image_provenance": "Original PDF raster panels extracted by pdfimages in generator drawing order; printed seed order verified with pdftotext; no reruns, flips, or new physical sampling",
        "pdfimages_version": subprocess.run(["pdfimages", "-v"], capture_output=True, text=True).stderr.splitlines()[0],
        "matplotlib_version": matplotlib.__version__,
        "panel_count": len(records),
        "bins": [{"mean_write_bit": w / 4, "mean_move": d / 2 - 1,
                  "count": len(groups[w, d]), "seeds": [r["seed"] for r in groups[w, d]]}
                 for w in range(4, -1, -1) for d in range(5)],
    }
    prefix.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved PDF, PNG, CSV and provenance JSON: {prefix.resolve()}.*")
    print(f"{len(records)} runs, {len([g for g in groups.values() if g])}/25 occupied bins")


if __name__ == "__main__":
    main()
