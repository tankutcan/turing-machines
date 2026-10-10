#!/usr/bin/env python3
"""Repeat a saved four-state halt-time sample with independent Bernoulli(1/2) tapes.

There is no blank-tape Busy Beaver bound on these input tapes. Times of -1 mean
not halted within the observation window, never an automatic non-halting proof.
Unreachable-H state graphs can safely be excluded from further simulation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sample_halt_times import BOUND, WRITE, MOVE, NEXT, as_dict, decode_machine_ids


def tape_width(cutoff: int) -> int:
    # A multiple of eight avoids discarded partial random words at batch edges.
    return ((2 * cutoff + 1 + 7) // 8) * 8


def random_tapes(rng: np.random.Generator, count: int, cutoff: int) -> np.ndarray:
    return rng.integers(0, 2, size=(count, tape_width(cutoff)), dtype=np.uint8)


def halt_reachable(tables: np.ndarray) -> np.ndarray:
    """Reachability of H from each state, ignoring tape constraints (necessary only)."""
    reachable = np.zeros((len(tables), 5), dtype=bool)
    reachable[:, 0] = True
    targets = NEXT[tables].reshape(-1, 4, 2)
    rows = np.arange(len(tables))[:, None, None]
    for _ in range(4):
        reachable[:, 1:] |= reachable[rows, targets].any(axis=2)
    return reachable


def simulate_random_batch(tables: np.ndarray, tape: np.ndarray, cutoff: int,
                          *, graph_pruning: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Mutate tape, returning times (-1 if no observed halt) and graph certificates.

The head starts at column cutoff (physical x=0). The array contains its entire
light cone, so these are unbounded-tape trajectories, not periodic ones.
Graph exclusion is exact: if neither bit can lead from the current state to H,
this trajectory cannot halt on any continuation of its tape.
"""
    tables = np.asarray(tables)
    if cutoff < 1 or tables.ndim != 2 or tables.shape[1] != 8:
        raise ValueError("positive cutoff and an (N,8) rule table are required")
    if np.any((tables < 0) | (tables > 19)):
        raise ValueError("action codes must be in 0..19")
    if tape.ndim != 2 or tape.shape[0] != len(tables) or tape.shape[1] < 2 * cutoff + 1:
        raise ValueError("tape must contain the complete [-cutoff,cutoff] light cone")
    size = len(tables)
    x = np.full(size, cutoff, dtype=np.int32)
    q = np.ones(size, dtype=np.uint8)
    times = np.full(size, -1, dtype=np.int32)
    reachable = halt_reachable(tables) if graph_pruning else np.ones((size, 5), dtype=bool)
    certified = ~reachable[:, 1]
    active = np.flatnonzero(~certified)
    for step in range(1, cutoff + 1):
        if not len(active):
            break
        positions = x[active]
        read = tape[active, positions]
        action = tables[active, 2 * (q[active] - 1) + read]
        tape[active, positions] = WRITE[action]
        x[active] += MOVE[action]
        q[active] = NEXT[action]
        halted = q[active] == 0
        times[active[halted]] = step
        excluded = ~reachable[active, q[active]]
        certified[active[excluded]] = True
        active = active[~halted & ~excluded]
    return times, certified


def scalar_trace(actions, initial_tape, cutoff):
    """Independent dictionary-tape replay for a selected sampled machine."""
    tape = {i - cutoff: int(b) for i, b in enumerate(initial_tape)}
    tm = as_dict(actions)
    x, q, rows = 0, 1, []
    for step in range(1, cutoff + 1):
        read = tape[x]
        write, direction, next_state = tm[q][read]
        move = 2 * direction - 1
        tape[x] = write
        rows.append(dict(step=step, state_before=q, position_before=x, read=read,
                         write=write, move=move, state_after=next_state, position_after=x + move))
        x, q = x + move, next_state
        if q == 0:
            return step, rows
    return -1, rows


def make_plot(counts, baseline, summary, output):
    observed = int(counts.sum())
    total, cutoff = summary["samples"], summary["cutoff"]
    t = np.arange(1, cutoff + 1)
    pmf = counts[1:] / max(observed, 1)
    baseline_counts = np.bincount(baseline[baseline > 0], minlength=BOUND + 1)
    baseline_pmf = baseline_counts[1:] / max(int(baseline_counts.sum()), 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.4, .95]})
    early = min(20, cutoff)
    axes[0].bar(t[:early], pmf[:early], width=.8, color="#287d8e", label="Random tape")
    axes[0].plot(np.arange(1, early + 1), baseline_pmf[:early], color="#a36547", marker=".",
                 linewidth=1, label="Blank tape (same TMs)")
    axes[0].set(title="First 20 steps", xlabel="Halt time t (transitions)",
                ylabel="P(T = t | T ≤ cutoff)", xlim=(.3, early + .7))
    axes[0].legend(frameon=False, fontsize=8)
    positive = counts[1:] > 0
    if observed:
        axes[1].vlines(t[positive], .5 / observed, pmf[positive], color="#287d8e", alpha=.65, linewidth=.9)
        axes[1].scatter(t[positive], pmf[positive], color="#287d8e", s=9)
        axes[1].set(xscale="log", yscale="log", ylim=(.5 / observed, 1))
    axes[1].axvline(BOUND, color="#a36547", linestyle="--", linewidth=1.1)
    axes[1].axvline(cutoff, color="#555555", linestyle=":", linewidth=1.1)
    axes[1].set(title="Full observed range (log–log)", xlabel="Halt time t (transitions)",
                ylabel="P(T = t | T ≤ cutoff)", xlim=(.8, cutoff * 1.15))
    axes[1].text(.98, .97, f"107: blank-tape bound only\nObservation cutoff: {cutoff:,}\nLongest observed: {summary['max_observed_halt_time']}",
                 ha="right", va="top", transform=axes[1].transAxes, fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    values = [observed, total - observed]
    bars = axes[2].bar(["Halted by\ncutoff", "Not halted\nby cutoff"], np.array(values) / total,
                       color=["#287d8e", "#a36547"], width=.65)
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value / total + .035,
                     f"{value / total:.2%}\n({value:,})", ha="center", fontsize=10)
    axes[2].set(title="All sampled TM/tape pairs", ylabel="Fraction of all samples", ylim=(0, 1.13))
    axes[2].text(.5, .98, "Not halted ≠ never halts", ha="center", va="top",
                 transform=axes[2].transAxes, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Four states, two symbols | {total:,} TM/tape pairs | exactly one H entry per TM\n"
        f"Same rule tables as blank-tape sample (machine seed={summary['machine_seed']}); independent Bernoulli(1/2) tapes (seed={summary['tape_seed']})\n"
        f"One head per unbounded tape; q₀=1, x₀=0; cutoff={cutoff:,} complete transitions; no interactions",
        fontsize=10.5,
    )
    fig.savefig(output / "halt_time_distribution.png", dpi=180)
    fig.savefig(output / "halt_time_distribution.pdf")
    plt.close(fig)

    # Preserve the original semilog presentation as well as the full log–log range.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    plot_end = min(cutoff, max(120, int(summary["max_observed_halt_time"] or 1)))
    for ax, end in zip(axes, [min(120, cutoff), plot_end]):
        visible = positive & (t <= end)
        ax.scatter(t[visible], pmf[visible], s=12, color="#287d8e", label="Random tape: conditional on halt by cutoff")
        bt = np.arange(1, len(baseline_pmf) + 1)
        mask = (baseline_pmf > 0) & (bt <= end)
        ax.scatter(bt[mask], baseline_pmf[mask], s=10, marker="x", color="#a36547", label="Blank tape: conditional on eventual halt")
        if observed:
            ax.set(yscale="log", ylim=(.5 / observed, 1))
        ax.axvline(BOUND, linestyle="--", linewidth=.8, color="#777777")
        ax.set(xlabel="Halt time t", ylabel="Conditional probability per integer step", xlim=(0, end * 1.02),
               title=f"Semilog comparison: t ≤ {end}")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"Same {total:,} four-state TMs, different initial tapes | machine seed={summary['machine_seed']}, tape seed={summary['tape_seed']}\n"
                 f"Independent fair bits; q₀=1, x₀=0; random-tape cutoff={cutoff:,}; 107 is not a random-tape bound", fontsize=10)
    fig.savefig(output / "semilog_comparison.png", dpi=180)
    fig.savefig(output / "semilog_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("results/halt_times_4state_2symbol"))
    parser.add_argument("--cutoff", type=int, default=10000)
    parser.add_argument("--tape-seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=Path("results/halt_times_4state_2symbol_random_tape"))
    args = parser.parse_args()
    if args.cutoff < 20 or args.batch_size < 1 or args.tape_seed < 0:
        parser.error("cutoff must be at least 20; batch-size positive; tape-seed nonnegative")
    config = json.loads((args.source_dir / "summary.json").read_text())
    if (config["states"], config["symbols"], config["sampling"]) != (4, 2, "exactly-one"):
        parser.error("this comparison requires the four-state, binary, exactly-one-H baseline")
    with np.load(args.source_dir / "samples.npz") as source:
        ids, baseline = source["machine_ids"], source["halt_times"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory is not empty; choose a new --output-dir")
    rng = np.random.default_rng(args.tape_seed)
    times = np.empty(len(ids), dtype=np.int32)
    certified = np.empty(len(ids), dtype=bool)
    best_time, best_index, best_initial = 0, None, None
    started = time.perf_counter()
    next_progress = 0
    for start in range(0, len(ids), args.batch_size):
        stop = min(len(ids), start + args.batch_size)
        tables = decode_machine_ids(ids[start:stop], config["sampling"])
        tape = random_tapes(rng, stop - start, args.cutoff)
        batch_times, proof = simulate_random_batch(tables, tape, args.cutoff)
        times[start:stop], certified[start:stop] = batch_times, proof
        if batch_times.max() > best_time:
            local = int(np.argmax(batch_times))
            best_index, best_time = start + local, int(batch_times[local])
        if stop >= next_progress or stop == len(ids):
            print(f"Simulated {stop:,}/{len(ids):,} pairs ({time.perf_counter()-started:.1f}s)", flush=True)
            next_progress = stop + max(len(ids) // 20, 1)

    # Recover the selected initial tape by replaying only the random-number stream.
    if best_index is not None:
        replay_rng = np.random.default_rng(args.tape_seed)
        for start in range(0, best_index + 1, args.batch_size):
            stop = min(len(ids), start + args.batch_size)
            initial = random_tapes(replay_rng, stop - start, args.cutoff)
            if start <= best_index < stop:
                best_initial = initial[best_index - start].copy()
        best_actions = decode_machine_ids(ids[best_index:best_index+1])[0]
        verified_time, trace = scalar_trace(best_actions, best_initial, args.cutoff)
        assert verified_time == best_time
        np.savez_compressed(output / "longest_sample_initial_tape.npz", tape=best_initial,
                            physical_positions=np.arange(len(best_initial)) - args.cutoff)
        (output / "longest_sample.json").write_text(json.dumps(dict(
            draw_index=best_index, machine_id=int(ids[best_index]), halt_time=best_time,
            blank_tape_halt_time=int(baseline[best_index]),
            transition_table=as_dict(best_actions),
            rule_encoding="[write_bit, move_bit, next_state]; move 0=left,1=right; next 0=H",
        ), indent=2) + "\n")
        with (output / "longest_sample_trace.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
    counts = np.bincount(times[times > 0], minlength=args.cutoff + 1)
    observed = int(counts.sum())
    remaining = len(ids) - observed
    np.savez_compressed(output / "samples.npz", machine_ids=ids, halt_times=times,
                        graph_certified_nonhalting=certified, blank_tape_halt_times=baseline)
    summary = dict(
        states=4, symbols=2, samples=len(ids), sampling="exactly-one",
        machine_seed=config["seed"], tape_seed=args.tape_seed, cutoff=args.cutoff,
        one_density=.5, initial_state=1, initial_position=0, boundary="unbounded",
        heads_per_tape=1, interactions="none", tape_assignment="independent tape for every draw, including duplicate TM tables",
        halted_by_cutoff=observed, not_halted_by_cutoff=remaining,
        graph_certified_nonhalting=int(certified.sum()),
        unresolved_at_cutoff=int(np.count_nonzero((times < 0) & ~certified)),
        halted_fraction_by_cutoff=observed / len(ids),
        mean_halt_time_observed=float(times[times > 0].mean()) if observed else None,
        max_observed_halt_time=best_time if observed else None,
        halts_after_107=int(np.count_nonzero(times > BOUND)),
        halts_after_1000=int(np.count_nonzero(times > 1000)),
        blank_tape_halting_fraction=float(np.mean(baseline > 0)),
        normalization="PMF conditional on T <= cutoff; remaining mass is T > cutoff, NOT T=infinity",
        classification="-1 in halt_times means not observed to halt; graph flag separately certifies no path to H",
        tape_storage=f"columns 0..{tape_width(args.cutoff)-1} correspond to x=-{args.cutoff} upward; no boundaries reached",
        tape_rng="numpy.random.default_rng / PCG64; sequential uint8 fair bits, rows in sample order",
        numpy_version=np.__version__, batch_size=args.batch_size,
        reproducibility="seed + fixed cutoff + sample order reproduce tapes; padding makes stream batch-independent",
        source_directory=str(args.source_dir), source_summary=config,
        code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in (Path(__file__).name, "sample_halt_times.py")},
        runtime_seconds=time.perf_counter() - started,
        reproduce=(f"env MPLCONFIGDIR=/tmp/turing-machines-matplotlib .venv/bin/python -m scripts.{Path(__file__).stem} "
                   f"--source-dir {args.source_dir} --cutoff {args.cutoff} --tape-seed {args.tape_seed} "
                   f"--batch-size {args.batch_size} --output-dir results/halt_times_4state_2symbol_random_tape_repeat"),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "distribution.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["halt_time", "count", "probability_all_samples", "probability_given_halt_by_cutoff"])
        for t in range(1, args.cutoff + 1):
            writer.writerow([t, counts[t], counts[t] / len(ids), counts[t] / observed if observed else ""])
        writer.writerow([f">{args.cutoff}", remaining, remaining / len(ids), ""])
    with (output / "survival.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "probability_T_greater_than_time"])
        for t, cumulative in enumerate(np.cumsum(counts)):
            writer.writerow([t, (len(ids) - cumulative) / len(ids)])
    make_plot(counts, baseline, summary, output)
    longest_link = '<a href="longest_sample.json">Longest observed halter</a> · <a href="longest_sample_trace.csv">Its trace</a> · <a href="longest_sample_initial_tape.npz">Its initial tape</a>' if observed else ""
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Halting times from random tapes</title>
<style>body{{max-width:1300px;margin:40px auto;padding:0 24px;font:17px/1.65 system-ui;color:#243044}}
img{{width:100%;height:auto}}a{{color:#175d74}}table{{border-collapse:collapse}}td,th{{padding:8px 22px;border-bottom:1px solid #ddd;text-align:left}}
.note{{padding:16px 22px;background:#f0f5f7;border-left:4px solid #287d8e}}</style></head><body>
<h1>Four-state halting times: random initial tapes</h1>
<p>The same {len(ids):,} sampled rule tables as the blank-tape experiment, with exactly one halt entry
per table. Every draw now has its own independent Bernoulli(1/2) tape. One head, initial q=1 and x=0,
unbounded tape, no interactions. Machine seed={config['seed']}; tape seed={args.tape_seed}.</p>
<p class="note">The observation window is {args.cutoff:,} transitions, including the transition into H.
The 107-step Busy Beaver bound applies to <em>blank</em> tapes, not these random inputs.
Runs still going at the cutoff are not automatically classified as never halting.
<a href="https://bbchallenge.org/~pascal.michel/bbc">Busy Beaver definition and bound</a>.</p>
<a href="halt_time_distribution.pdf"><img src="halt_time_distribution.png" alt="Random-tape halting-time distribution and observed versus not-observed halting fractions"></a>
<h2>Results</h2><table><tr><th>Quantity</th><th>Observed value</th></tr>
<tr><td>Halted by cutoff</td><td>{observed:,} ({observed / len(ids):.4%})</td></tr>
<tr><td>Not halted by cutoff</td><td>{remaining:,} ({remaining / len(ids):.4%})</td></tr>
<tr><td>Of those, graph-certified never halting</td><td>{int(certified.sum()):,}</td></tr>
<tr><td>Of those, unresolved at cutoff</td><td>{summary['unresolved_at_cutoff']:,}</td></tr>
<tr><td>Halts after step 107</td><td>{summary['halts_after_107']:,}</td></tr>
<tr><td>Longest observed halt time</td><td>{summary['max_observed_halt_time']}</td></tr>
<tr><td>Mean among observed halters</td><td>{summary['mean_halt_time_observed']}</td></tr></table>
<p>The histograms condition on halting <em>by this cutoff</em>, not eventual halting. The remaining
fraction is P(T &gt; cutoff), not an estimate of a proven mass at infinity. Graph certification is separate:
if H is unreachable from the current internal state even when allowing both read bits, that trajectory
cannot halt. No other non-halting claims or period-detection assumptions are used.</p>
<h2>Semilog comparison with the blank-tape sample</h2>
<a href="semilog_comparison.pdf"><img src="semilog_comparison.png" alt="Semilog comparison of random and blank initial tape halting distributions"></a>
<p>The rule tables are paired exactly. The blank curve is conditional on eventual halting (which was
decided using the blank-tape bound); the random-tape curve is conditional on halting within the finite window.
Zero-count bins cannot appear on logarithmic axes and are not proof that a time is impossible.</p>
<h2>Data and reproducibility</h2><p><a href="distribution.csv">Distribution CSV</a> ·
<a href="survival.csv">Survival probabilities</a> · <a href="samples.npz">All sampled rules and outcomes</a> ·
<a href="summary.json">Parameters, seeds, code hashes, and reproduction command</a> · {longest_link}</p>
<p>In samples.npz, halt_times=-1 means no observed halt. Graph certificates are stored separately.
The full random initial tape has infinitely many ones almost surely, so a global final count of ones is
not reported. Initial tape bits are generated once and persist until overwritten; they are not redrawn
when revisited. Only the causally accessible finite interval is allocated, without periodic wrapping.</p>
</body></html>"""
    (output / "index.html").write_text(report)
    print(json.dumps({k: summary[k] for k in ("samples", "cutoff", "halted_by_cutoff", "not_halted_by_cutoff",
          "graph_certified_nonhalting", "unresolved_at_cutoff", "max_observed_halt_time", "halts_after_107")}, indent=2))
    print(f"Saved {(output / 'index.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
