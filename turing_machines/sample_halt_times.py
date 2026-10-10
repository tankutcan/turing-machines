#!/usr/bin/env python3
"""Sample the blank-tape halting-time distribution of four-state binary TMs.

The proven bound S(4,2)=107 classifies every sample, including non-halters.
Default measure: uniform labeled tables with exactly one halting entry.
Each sample has one head and its own all-zero, unbounded tape; no interaction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coupled_chiral_ants import simulate as scalar_simulate

BOUND = 107
SOURCE = "https://bbchallenge.org/~pascal.michel/bbc"
SPACE_SIZES = {"exactly-one": 8 * 4 * 16**7, "independent": 20**8}
# Codes 0..15: write*8 + direction_bit*4 + (next_state-1).
# Codes 16..19: 16 + write*2 + direction_bit; next_state=H=0.
WRITE = np.array([c // 8 for c in range(16)] + [0, 0, 1, 1], dtype=np.uint8)
MOVE = np.array([2 * ((c // 4) % 2) - 1 for c in range(16)] + [-1, 1, -1, 1], dtype=np.int16)
NEXT = np.array([1 + c % 4 for c in range(16)] + [0] * 4, dtype=np.uint8)


def decode_machine_ids(ids: np.ndarray, sampling: str = "exactly-one") -> np.ndarray:
    """Bijection from integer IDs to labeled tables; columns are 1/0,1/1,...,4/1."""
    ids = np.asarray(ids, dtype=np.uint64)
    if ids.ndim != 1 or np.any(ids >= SPACE_SIZES[sampling]):
        raise ValueError("machine IDs must be a 1-D array inside the selected table space")
    tables = np.empty((len(ids), 8), dtype=np.uint8)
    if sampling == "exactly-one":
        halt_entry = ids & np.uint64(7)
        halt_action = np.uint64(16) + ((ids >> np.uint64(3)) & np.uint64(3))
        for entry in range(8):
            # Seven non-H entries occur in increasing state/read order.
            rank = entry - (entry > halt_entry).astype(np.int64)
            shift = (5 + 4 * rank).astype(np.uint64)
            action = (ids >> shift) & np.uint64(15)
            tables[:, entry] = np.where(halt_entry == entry, halt_action, action)
    elif sampling == "independent":
        rest = ids.copy()
        for entry in range(8):
            tables[:, entry] = rest % np.uint64(20)
            rest //= np.uint64(20)
    else:
        raise ValueError(f"unknown sampling measure: {sampling}")
    return tables


def as_dict(actions: np.ndarray) -> dict:
    return {q: {b: [int(WRITE[c]), int((MOVE[c] + 1) // 2), int(NEXT[c])]
                for b in (0, 1) for c in [actions[2 * (q - 1) + b]]}
            for q in range(1, 5)}


def simulate_batch(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return halting times and ones at halt; -1 denotes proven non-halting.

Within 107 steps the head cannot leave [-107,107], so the finite storage
below exactly represents the unbounded blank tape, with no wraparound.
"""
    tables = np.asarray(tables)
    if tables.ndim != 2 or tables.shape[1] != 8 or np.any((tables < 0) | (tables > 19)):
        raise ValueError("expected an (N,8) array of action codes in 0..19")
    count = len(tables)
    tape = np.zeros((count, 2 * BOUND + 1), dtype=np.uint8)
    x = np.full(count, BOUND, dtype=np.int16)
    state = np.ones(count, dtype=np.uint8)
    halt_time = np.full(count, -1, dtype=np.int16)
    active = np.arange(count)
    for step in range(1, BOUND + 1):
        if not len(active):
            break
        positions = x[active]
        read = tape[active, positions]
        action = tables[active, 2 * (state[active] - 1) + read]
        tape[active, positions] = WRITE[action]
        x[active] += MOVE[action]
        state[active] = NEXT[action]
        halted = state[active] == 0
        halt_time[active[halted]] = step
        active = active[~halted]
    ones = tape.sum(axis=1).astype(np.int16)
    ones[halt_time == -1] = -1
    return halt_time, ones


def aggregate(times: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    finite = times > 0
    counts = np.bincount(times[finite], minlength=BOUND + 1)
    nonhalting = int((~finite).sum())
    survival = (len(times) - np.cumsum(counts)) / len(times)
    assert int(counts.sum()) + nonhalting == len(times)
    return counts, nonhalting, survival


def render(counts, nonhalting, summary, output):
    finite = int(counts.sum())
    total = finite + nonhalting
    t = np.arange(1, BOUND + 1)
    # Zero is used only for plotting the empty conditional distribution.
    pmf = counts[1:] / max(finite, 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.4, .85]})
    axes[0].bar(t[:20], pmf[:20], color="#287d8e", width=.8)
    axes[0].set(title="Halting times: first 20 steps", xlabel="Halt time t (transitions)",
                ylabel="P(T = t | machine halts)", xlim=(.3, 20.7))
    axes[0].set_xticks([1, 5, 10, 15, 20])
    positive = counts[1:] > 0
    if finite:
        axes[1].bar(t[positive], pmf[positive], color="#287d8e", width=.9)
        axes[1].set_yscale("log")
        axes[1].set_ylim(.5 / finite, 1)
    axes[1].axvline(BOUND, color="#b14632", linestyle="--", linewidth=1.2)
    axes[1].set(title="Full finite-time distribution (log scale)",
                xlabel="Halt time t (transitions)", ylabel="P(T = t | machine halts)",
                xlim=(0, 112))
    axes[1].set_xticks([1, 20, 40, 60, 80, 107])
    axes[1].text(.98, .97, f"Proven maximum: {BOUND}\nSample maximum: {summary['max_observed_halt_time']}",
                ha="right", va="top", transform=axes[1].transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    fractions = np.array([finite, nonhalting]) / total
    bars = axes[2].bar(["Halts", "Never halts"], fractions, color=["#287d8e", "#a36547"], width=.65)
    for bar, count, fraction in zip(bars, [finite, nonhalting], fractions):
        axes[2].text(bar.get_x() + bar.get_width() / 2, fraction + .035,
                     f"{fraction:.2%}\n({count:,})", ha="center", fontsize=10)
    axes[2].set(title="All sampled machines", ylabel="Fraction of all samples", ylim=(0, 1.13))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    sampling_label = "uniform tables with exactly one H entry" if summary["sampling"] == "exactly-one" else "uniform tables with independent next targets in {1,2,3,4,H}"
    fig.suptitle(
        f"Four-state, two-symbol Turing machines | {total:,} independent draws | seed={summary['seed']}\n"
        f"{sampling_label}; labeled states; sampling with replacement\n"
        "One head per machine; all-zero unbounded tape; initial q=1, x=0; halting transition counts as a step",
        fontsize=11,
    )
    fig.savefig(output / "halt_time_distribution.png", dpi=180)
    fig.savefig(output / "halt_time_distribution.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--sampling", choices=tuple(SPACE_SIZES), default="exactly-one")
    parser.add_argument("--output-dir", type=Path, default=Path("results/halt_times_4state_2symbol"))
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error("samples and batch-size must be positive; seed must be nonnegative")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory is not empty; choose a new --output-dir")
    start_time = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    ids = np.empty(args.samples, dtype=np.uint64)
    times = np.empty(args.samples, dtype=np.int16)
    ones = np.empty(args.samples, dtype=np.int16)
    next_progress = 0
    for start in range(0, args.samples, args.batch_size):
        stop = min(args.samples, start + args.batch_size)
        ids[start:stop] = rng.integers(0, SPACE_SIZES[args.sampling], size=stop - start, dtype=np.uint64)
        times[start:stop], ones[start:stop] = simulate_batch(decode_machine_ids(ids[start:stop], args.sampling))
        if stop >= next_progress or stop == args.samples:
            print(f"Classified {stop:,}/{args.samples:,} machines ({time.perf_counter()-start_time:.1f}s)", flush=True)
            next_progress = stop + max(args.samples // 10, 1)

    counts, nonhalting, survival = aggregate(times)
    finite = args.samples - nonhalting
    finite_times = times[times > 0]
    summary = dict(
        states=4, symbols=2, samples=args.samples, seed=args.seed, sampling=args.sampling,
        table_space_size=SPACE_SIZES[args.sampling], sampling_with_replacement=True,
        batch_size=args.batch_size, halted=finite, nonhalting=nonhalting,
        halting_fraction=finite / args.samples, nonhalting_fraction=nonhalting / args.samples,
        halting_fraction_standard_error=float(np.sqrt(finite * nonhalting / args.samples**3)),
        mean_halt_time_given_halts=float(finite_times.mean()) if finite else None,
        median_halt_time_given_halts=float(np.median(finite_times)) if finite else None,
        max_observed_halt_time=int(finite_times.max()) if finite else None,
        max_observed_ones_at_halt=int(ones.max()) if finite else None,
        counts_at_proven_maximum=int(counts[BOUND]),
        proven_bound=BOUND, bound_source=SOURCE,
        initial_state=1, initial_position=0, initial_tape="all zeros", boundary="unbounded",
        heads_per_machine=1, interactions="none; each sample has its own tape",
        step_convention="complete read/write/move/state-change, including transition into H",
        classification="not halted after 107 completed transitions => never halts, by S(4,2)=107",
        labeled_states="1,2,3,4; H=0 is additional; unused/unreachable states are allowed",
        rng="numpy.random.default_rng / PCG64", numpy_version=np.__version__,
        generator=Path(__file__).name,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        machine_id_encoding=("bits 0..2: halting entry; bits 3..4: halt action write*2+direction; "
                             "then seven 4-bit non-H actions in increasing entry order")
                            if args.sampling == "exactly-one" else "eight base-20 action digits, low digit first",
        action_encoding="0..15: write*8+direction*4+next_state-1; 16..19: 16+write*2+direction, next=H",
        entry_order="(1,0),(1,1),(2,0),(2,1),(3,0),(3,1),(4,0),(4,1)",
        direction_encoding="0=left, 1=right",
        dataset="samples.npz: row index=draw index; machine_ids, halt_times, ones_at_halt; -1 denotes never halts",
        normalization="finite PMF shown conditional on halting; unconditional probabilities and nonhalting mass also saved",
        simulation_seconds=time.perf_counter() - start_time,
        reproduce=(f"env MPLCONFIGDIR=/tmp/turing-machines-matplotlib .venv/bin/python -m scripts.{Path(__file__).stem} "
                   f"--samples {args.samples} --seed {args.seed} --sampling {args.sampling} "
                   f"--batch-size {args.batch_size} --output-dir results/halt_times_4state_2symbol_repeat"),
    )
    np.savez_compressed(output / "samples.npz", machine_ids=ids, halt_times=times, ones_at_halt=ones)
    with (output / "distribution.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["halt_time", "count", "probability_all_samples", "probability_given_halts"])
        for t in range(1, BOUND + 1):
            writer.writerow([t, counts[t], counts[t] / args.samples, counts[t] / finite if finite else ""])
        writer.writerow(["infinity", nonhalting, nonhalting / args.samples, ""])
    with (output / "survival.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "probability_T_greater_than_t", "probability_T_greater_than_t_given_halts"])
        cumulative = np.cumsum(counts)
        for t in range(BOUND + 1):
            writer.writerow([t, survival[t], (finite - cumulative[t]) / finite if finite else ""])

    # Independently replay the longest sampled halter, using the sparse simulator.
    if finite:
        draw = int(np.argmax(times))
        best_tm = as_dict(decode_machine_ids(ids[draw:draw+1], args.sampling)[0])
        replay = scalar_simulate(best_tm, BOUND)
        assert replay["halted"] and (replay["steps"], replay["ones"]) == (int(times[draw]), int(ones[draw]))
        (output / "longest_sample.json").write_text(json.dumps(dict(
            draw_index=draw, machine_id=int(ids[draw]), transition_table=best_tm,
            halt_time=int(times[draw]), ones_at_halt=int(ones[draw]),
            rule_encoding="[write_bit, move_bit, next_state]; next_state=0 is Halt",
        ), indent=2) + "\n")
        with (output / "longest_sample_trace.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(replay["trace"][0]))
            writer.writeheader()
            writer.writerows(replay["trace"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(counts, nonhalting, summary, output)
    measure = ("Uniform over labeled transition tables with exactly one halt entry. Equivalently, "
               "sample an ordinary non-halting table and uniformly choose one of its eight next-state "
               "entries to replace by H, retaining its write and move. The other seven targets are "
               "uniform in states 1–4; every write and move is an independent fair choice.") if args.sampling == "exactly-one" else (
               "Uniform over labeled tables with independent write bits, move directions, and next-state "
               "targets in {1,2,3,4,H}. Zero, one, or multiple halting entries are allowed.")
    longest_link = (' · <a href="longest_sample.json">Longest sampled halter</a>'
                    ' · <a href="longest_sample_trace.csv">Its complete trace</a>') if finite else ""
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Halting times: four-state binary Turing machines</title>
<style>body{{max-width:1300px;margin:40px auto;padding:0 24px;font:17px/1.65 system-ui;color:#243044}}
img{{width:100%;height:auto}}a{{color:#175d74}}h2{{margin-top:36px}}table{{border-collapse:collapse}}
td,th{{padding:8px 22px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#edf1f4;padding:3px}}
.note{{padding:16px 22px;background:#f0f5f7;border-left:4px solid #287d8e}}</style></head><body>
<h1>Halting times of four-state, two-symbol machines</h1>
<p>{args.samples:,} independent draws, seed {args.seed}. Every machine starts in state 1 at x=0,
with one head on its own infinite all-zero tape. There are no interactions or periodic boundaries.</p>
<p class="note">The proven blank-tape bound is <strong>S(4,2)=107</strong>. We execute up to and
including transition 107. A machine still running after that is classified as <strong>never halting</strong>,
not merely censored. This conclusion does not apply to random input tapes or interacting gases.
<a href="{SOURCE}">Source: Pascal Michel's Busy Beaver records</a>.</p>
<a href="halt_time_distribution.pdf"><img src="halt_time_distribution.png" alt="Conditional halting-time distributions and separate halting versus non-halting fractions"></a>
<h2>Sample statistics</h2>
<table><tr><th>Quantity</th><th>Observed value</th></tr>
<tr><td>Halting</td><td>{finite:,} ({finite / args.samples:.4%})</td></tr>
<tr><td>Never halting</td><td>{nonhalting:,} ({nonhalting / args.samples:.4%})</td></tr>
<tr><td>Mean finite halt time</td><td>{summary['mean_halt_time_given_halts']}</td></tr>
<tr><td>Median finite halt time</td><td>{summary['median_halt_time_given_halts']}</td></tr>
<tr><td>Longest observed halt time</td><td>{summary['max_observed_halt_time']}</td></tr>
<tr><td>Samples halting exactly at 107</td><td>{summary['counts_at_proven_maximum']}</td></tr></table>
<p>The first two plots are conditional on eventual halting. The right plot keeps the probability
mass at infinite halt time separate. Zero-count finite bins are retained in the CSV; on a logarithmic
axis they cannot be drawn. An empty bin in this finite sample does not prove that a halt time is impossible.</p>
<h2>Which random-machine measure?</h2><p>{measure}</p>
<p>There are {SPACE_SIZES[args.sampling]:,} labeled tables under this convention. Draws are with replacement:
duplicate tables are retained. State relabelings and left/right reflections are not identified;
unreachable states are not excluded. Thus the sample is not uniform over symmetry classes or restricted
to machines that visit all four working states. H is an additional state, not one of the four.</p>
<p>Probabilities depend on the sampling convention. The script also supports
<code>--sampling independent</code>, where each next-state target is independently uniform over
the four working states and H, allowing zero or several halt entries.</p>
<h2>Files and reproduction</h2>
<p><a href="halt_time_distribution.pdf">Figure PDF</a> · <a href="distribution.csv">Distribution CSV</a> ·
<a href="survival.csv">Survival probabilities</a> · <a href="samples.npz">All sampled rules and outcomes</a> ·
<a href="summary.json">Parameters, encoding, code hash, and reproduction command</a>{longest_link}</p>
<p>The compressed dataset stores machine IDs plus exact halt times and ones at halt; −1 denotes
non-halting, not a time. Decode rules using <code>decode_machine_ids</code> and <code>as_dict</code> in
<a href="{html.escape(os.path.relpath(Path(__file__).resolve(), output.resolve()), quote=True)}">sample_halt_times.py</a>. The seeded sample is independent of batch size.</p>
</body></html>
"""
    (output / "index.html").write_text(report)
    print(json.dumps({k: summary[k] for k in ("samples", "halted", "nonhalting", "halting_fraction",
          "mean_halt_time_given_halts", "max_observed_halt_time", "counts_at_proven_maximum")}, indent=2))
    print(f"Saved {(output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
