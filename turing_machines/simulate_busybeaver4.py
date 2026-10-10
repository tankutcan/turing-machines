#!/usr/bin/env python3
"""Plot the 107-step BB(4) table with its sole Halt target replaced by state 3.

Uses the existing synchronous shared-tape simulator. The single-head run has
an unbounded blank tape; the identical-agent ensemble has a random periodic
tape. Rules are [write_bit, move_bit, next_state], with 0/1 moves left/right.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .generate_spacetime_zoo import spacetime_rgb
from .many_turing_machines import simulate


def machine(halt_target: int = 3) -> dict:
    return {
        1: {0: [1, 1, 2], 1: [1, 0, 2]},
        2: {0: [1, 0, 1], 1: [0, 0, 3]},
        3: {0: [1, 1, halt_target], 1: [1, 0, 4]},
        4: {0: [1, 1, 4], 1: [0, 1, 1]},
    }


def blank_reference(steps: int) -> tuple[list[dict], set[int]]:
    """Independent sparse single-head trace, also locating its visited window."""
    tm, ones, q, x = machine(), set(), 1, 0
    rows = []
    for t in range(1, steps + 1):
        b = int(x in ones)
        w, d, s = tm[q][b]
        m = 2 * d - 1
        if w:
            ones.add(x)
        else:
            ones.discard(x)
        rows.append(dict(step=t, state_before=q, position_before=x, read=b,
                         write=w, move=m, state_after=s, position_after=x + m,
                         ones_after=len(ones)))
        q, x = s, x + m
    return rows, ones


def configuration(*, copies=1, steps=10000, seed=0, plot_width=100,
                  boundary="open", tape_length=None, one_density=0.0,
                  spacing=5, record_every=1) -> SimpleNamespace:
    return SimpleNamespace(states=4, copies=copies, steps=steps, seed=seed,
                           spacing=spacing, plot_width=plot_width,
                           boundary=boundary, tape_length=tape_length,
                           one_density=one_density, record_every=record_every,
                           machine_ensemble="identical", initial_state_mode="fixed",
                           update_scheme="synchronous", collision_policy="majority",
                           burn_in_fraction=0.5)


def save_data(output: Path, name: str, result: dict, config: SimpleNamespace) -> dict:
    np.savez_compressed(
        output / f"{name}.npz",
        position_trace=result["position_trace"],
        initial_positions=result["initial_positions"],
        initial_tape_positions=result["initial_tape_positions"],
        initial_states=result["initial_states"], final_states=result["final_states"],
        recorded_times=result["recorded_times"], spacetime=result["spacetime"],
        head_occupancy=result["head_occupancy"],
        plot_positions=np.arange(result["plot_min"], result["plot_min"] + result["spacetime"].shape[1]),
        final_tape=result["tape"], tape_min=result["tape_min"],
        velocities=result["velocities"], net_velocities=result["net_velocities"],
    )
    return dict(
        **vars(config), plot_min=result["plot_min"],
        saved_plot_width=result["spacetime"].shape[1],
        initial_tape_ones_in_saved_window=int(result["spacetime"][0].sum()),
        final_tape_ones=int(result["tape"].sum()),
        final_state_counts={str(q): int(np.count_nonzero(result["final_states"] == q))
                            for q in range(1, 5)},
        mean_late_velocity=float(result["velocities"].mean()),
        min_late_velocity=float(result["velocities"].min()),
        max_late_velocity=float(result["velocities"].max()),
        coincident_cell_events=int(result["coincident_cells"]),
        conflicting_write_events=int(result["conflicting_cells"]),
    )


def draw_panel(ax, result, tmax, xmin, xmax, *, head_line=False):
    times = result["recorded_times"]
    count = int(np.searchsorted(times, tmax, side="right"))
    start, stop = xmin - result["plot_min"], xmax - result["plot_min"] + 1
    subset = {key: result[key][:count, start:stop]
              for key in ("spacetime", "head_occupancy")}
    dt = int(times[1] - times[0])
    ax.imshow(spacetime_rgb(subset), origin="lower", interpolation="nearest", aspect="auto",
              extent=(xmin - .5, xmax + .5, times[0] - dt / 2, times[count - 1] + dt / 2))
    if head_line:
        x = result["position_trace"][:tmax + 1, 0]
        ax.plot(x, np.arange(len(x)), color="#df3225", linewidth=.6)
    ax.set(xlabel="Tape position x", ylabel="Time (completed transitions)",
           xlim=(xmin - .5, xmax + .5), ylim=(-.5, tmax + .5))


def make_plot(result: dict, config: SimpleNamespace, *, single: bool):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    low = result["plot_min"]
    high = low + result["spacetime"].shape[1] - 1
    draw_panel(axes[0], result, config.steps, low, high, head_line=single)
    axes[0].set_title("Full run (entire visited region)" if single else "Full periodic tape", fontsize=12)
    zoom_t = min(config.steps, 250 if single else 1000)
    if single:
        early_x = result["position_trace"][:zoom_t + 1, 0]
        zlow, zhigh = max(low, int(early_x.min()) - 3), min(high, int(early_x.max()) + 3)
    else:
        zlow, zhigh = 0, min(high, 499)
    draw_panel(axes[1], result, zoom_t, zlow, zhigh, head_line=single)
    axes[1].set_title(f"Early-time detail: t = 0–{zoom_t}", fontsize=12)
    if single and zoom_t >= 107:
        axes[1].axhline(107, color="#256bad", linewidth=.9, linestyle="--")
        axes[1].text(.98, 107 / zoom_t + .015, "Original machine halts at t=107",
                     transform=axes[1].transAxes, ha="right", fontsize=8,
                     color="#154675", bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    kind = "Single head on a blank tape" if single else "Identical heads on a shared random tape"
    tape_info = "blank infinite tape; no random choices" if single else (
        f"Bernoulli p(1)=0.5; seed={config.seed}; periodic L={config.tape_length}; spacing={config.spacing}")
    fig.suptitle(
        f"Four-state Busy Beaver table, modified H → 3 | {kind}\n"
        f"n=4; N={config.copies}; T={config.steps}; all initial q=1 | {tape_info}\n"
        f"Synchronous updates; majority writes, ties keep old bit; snapshot interval={config.record_every}\n"
        "0 = dark; 1 = light; one head = red; multiple heads = gold; time runs upward",
        fontsize=11,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--copies", type=int, default=1000)
    parser.add_argument("--tape-length", type=int, default=10000)
    parser.add_argument("--spacing", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-record-every", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/busybeaver4_halt_to_3"))
    args = parser.parse_args()
    if min(args.copies, args.tape_length, args.spacing, args.ensemble_record_every) < 1 or args.steps < 2:
        parser.error("counts, length, spacing, and recording interval must be positive; steps >= 2")
    if args.seed < 0 or args.steps % args.ensemble_record_every or args.ensemble_record_every > args.steps:
        parser.error("seed must be nonnegative and ensemble recording interval must divide steps")
    if (args.copies - 1) * args.spacing >= args.tape_length:
        parser.error("initial head span must be less than tape length")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory is not empty; choose a new --output-dir")

    print("Simulating single head on blank unbounded tape...", flush=True)
    rows, final_ones = blank_reference(args.steps)
    xs = [0] + [row["position_after"] for row in rows]
    low, high = min(xs) - 10, max(xs) + 10
    width = 2 * max(abs(low), abs(high)) + 3
    single_config = configuration(steps=args.steps, seed=args.seed, plot_width=width)
    single = simulate(single_config, tm_override=machine())
    assert np.array_equal(single["position_trace"][:, 0], xs)
    assert set((np.flatnonzero(single["tape"]) + single["tape_min"]).tolist()) == final_ones
    assert single["final_states"][0] == rows[-1]["state_after"]
    offset = low - single["plot_min"]
    for key in ("spacetime", "head_occupancy"):
        single[key] = single[key][:, offset:offset + high - low + 1].copy()
    single["plot_min"] = low
    with (output / "single_blank_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    single_summary = save_data(output, "single_blank", single, single_config)
    single_summary.update(final_position=xs[-1], visited_min=min(xs), visited_max=max(xs),
                          modified_rule_uses=sum(row["state_before"] == 3 and row["read"] == 0 for row in rows))

    print("Simulating identical-head ensemble on random periodic tape...", flush=True)
    ensemble_config = configuration(
        copies=args.copies, steps=args.steps, seed=args.seed, spacing=args.spacing,
        plot_width=args.tape_length, boundary="periodic", tape_length=args.tape_length,
        one_density=.5, record_every=args.ensemble_record_every,
    )
    ensemble = simulate(ensemble_config, tm_override=machine())
    assert np.all(ensemble["head_occupancy"].sum(axis=1) == args.copies)
    ensemble_summary = save_data(output, "ensemble_random", ensemble, ensemble_config)
    with (output / "ensemble_velocities.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["copy_id", "initial_tape_position", "initial_state", "final_unwrapped_position",
                         "final_state", "velocity_fit_last_half", "velocity_net_last_half"])
        for i in range(args.copies):
            writer.writerow([i, ensemble["initial_tape_positions"][i], ensemble["initial_states"][i],
                             ensemble["position_trace"][-1, i], ensemble["final_states"][i],
                             ensemble["velocities"][i], ensemble["net_velocities"][i]])

    print("Rendering spacetime figures...", flush=True)
    with PdfPages(output / "spacetime_comparison.pdf") as pdf:
        for name, result, config, is_single in (
            ("single_blank", single, single_config, True),
            ("ensemble_random", ensemble, ensemble_config, False),
        ):
            fig = make_plot(result, config, single=is_single)
            fig.savefig(output / f"{name}.png", dpi=180)
            fig.savefig(output / f"{name}.pdf")
            pdf.savefig(fig)
            plt.close(fig)

    metadata = {
        "model": "107-step, 13-one BB(4) champion with H replaced by state 3",
        "source": "https://bbchallenge.org/~pascal.michel/ha",
        "modified_entry": {"state": 3, "read": 0, "write": 1, "move": 1, "next_state": 3},
        "rule_encoding": "[write_bit, move_bit, next_state]; move_bit 0=left, 1=right",
        "transition_table": machine(), "original_transition_table_halt_is_0": machine(0),
        "numpy_version": np.__version__, "rng": "numpy.random.default_rng / PCG64",
        "seed_semantics": "fixed rules; seed controls only initial random tape; all heads start in q=1",
        "initial_positions": "x_i=(i-floor(N/2))*spacing, reduced modulo L on the ring",
        "collision_semantics": "read old tape; vote on writes; tie preserves old bit; then all heads move/change state",
        "snapshots": "row at t stores the tape and heads AFTER t complete transitions; row 0 is initial",
        "single_blank": single_summary, "ensemble_random": ensemble_summary,
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in (Path(__file__).name, "many_turing_machines.py", "generate_spacetime_zoo.py")},
        "reproduce": (f"env MPLCONFIGDIR=/tmp/turing-machines-matplotlib .venv/bin/python -m scripts.{Path(__file__).stem} "
                      f"--steps {args.steps} --copies {args.copies} --tape-length {args.tape_length} "
                      f"--spacing {args.spacing} --seed {args.seed} --ensemble-record-every {args.ensemble_record_every} "
                      "--output-dir results/busybeaver4_halt_to_3_repeat"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Four-state Busy Beaver: H replaced by 3</title>
<style>body{{max-width:1300px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#243044}}
img{{width:100%;height:auto}}a{{color:#175d74}}h2{{margin-top:40px}}code{{background:#edf1f4;padding:3px}}
table{{border-collapse:collapse}}td,th{{padding:8px 18px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Four-state Busy Beaver: H → 3</h1>
<p>The 107-step, 13-one champion is made non-halting by changing only
<code>(q=3, read 0) → (write 1, move right, next 3)</code>.
Both experiments run for {args.steps:,} steps. Every head starts in state 1.</p>
<table><tr><th>State</th><th>Read 0: (write, move, next)</th><th>Read 1: (write, move, next)</th></tr>
<tr><td>1</td><td>(1, +1, 2)</td><td>(1, −1, 2)</td></tr>
<tr><td>2</td><td>(1, −1, 1)</td><td>(0, −1, 3)</td></tr>
<tr><td>3</td><td>(1, +1, 3)</td><td>(1, −1, 4)</td></tr>
<tr><td>4</td><td>(1, +1, 4)</td><td>(0, +1, 1)</td></tr></table>
<p>Time runs upward. Dark = 0, light = 1, red = one head, gold = multiple heads.
Each image includes an early-time detail; full-resolution snapshots are saved separately.</p>
<h2>One head on a blank, unbounded tape</h2>
<p>Initial x=0; no random choices. At the end: x={xs[-1]}, {len(final_ones)} ones,
state {rows[-1]['state_after']}. The blue line marks when the original machine would halt;
the modified machine continues. Snapshots every step.</p>
<a href="single_blank.pdf"><img src="single_blank.png" alt="Single-head spacetime: full trajectory and first 250 steps"></a>
<h2>{args.copies:,} identical heads on a shared random ring</h2>
<p>Seed {args.seed}; independent Bernoulli(1/2) initial bits; periodic length {args.tape_length:,}.
Initial positions xᵢ=(i−floor(N/2))×{args.spacing} modulo L. Heads are not independently simulated:
they interact through the shared tape. Updates are synchronous; majority writes, ties preserve the old bit.
Snapshots every {args.ensemble_record_every} steps. There were {ensemble['coincident_cells']:,}
co-occupied cell-time events, including {ensemble['conflicting_cells']:,} conflicting-write events.</p>
<a href="ensemble_random.pdf"><img src="ensemble_random.png" alt="Shared random-tape ensemble: full periodic tape and early-time spatial detail"></a>
<h2>Files and reproducibility</h2>
<p><a href="spacetime_comparison.pdf">Both figures as a two-page PDF</a> ·
<a href="metadata.json">Parameters, exact tables, code hashes, and reproduction command</a> ·
<a href="single_blank_trace.csv">Every single-head transition</a> ·
<a href="ensemble_velocities.csv">Ensemble velocities</a> ·
<a href="single_blank.npz">Single-head snapshot data</a> ·
<a href="ensemble_random.npz">Ensemble snapshot data</a></p>
<p>NPZ files include every unwrapped head position, sampled tape and occupancy rows, times,
initial/final states, and final tape. The single-head CSV records its state at every step.
The rule change removes halting, so these are not Busy Beaver competition scores.</p>
</html>
"""
    (output / "index.html").write_text(report)
    print(json.dumps({"single_blank": single_summary, "ensemble_random": ensemble_summary}, indent=2))
    print(f"Saved report: {(output / 'index.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
