"""Exhaustive cooperation graph of two-state, one-pass binary transducers.

Program cells 2k,2k+1 store W(q,b), S(q,b)-1, with k=2*(q-1)+b.
Machine IDs are the resulting eight-bit words, cell 0 least significant.
Motion is fixed rightward and omitted from the code. Both machines start in
state 1, scan an immutable opponent code once, and use the eighth emitted bit
as the action. A[i,j] = 1 iff i cooperates with j. Main graphs omit self-loops;
raw actions and self-play retain all diagonal decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

MACHINES = 256
TAPE_LENGTH = 8


def program_bits(machine_id: int) -> list[int]:
    if not 0 <= machine_id < MACHINES:
        raise ValueError("machine_id must be in 0..255")
    return [(machine_id >> k) & 1 for k in range(TAPE_LENGTH)]


def all_actions() -> np.ndarray:
    """Compute every reader/opponent decision, including self-play."""
    reader = np.arange(MACHINES, dtype=np.uint16)[:, None]
    opponent = np.arange(MACHINES, dtype=np.uint16)[None, :]
    state = np.zeros((MACHINES, MACHINES), dtype=np.uint16)
    for cell in range(TAPE_LENGTH):
        bit = (opponent >> cell) & 1
        entry = (reader >> (2 * (2 * state + bit))) & 3
        action = entry & 1
        state = entry >> 1
    return action.astype(np.uint8)


def graph_adjacency(actions: np.ndarray) -> np.ndarray:
    adjacency = actions.copy()
    np.fill_diagonal(adjacency, 0)
    return adjacency


def payoff_ranking(actions: np.ndarray) -> list[dict]:
    """Score each machine against all other machines, equally weighted.

    Payoffs from the focal player's perspective: DD=1, DC=5, CD=0, CC=3.
    Self-play is excluded, even if its action is stored in the input matrix.
    Rank is one plus the number of strictly higher-scoring machines.
    """
    n = len(actions)
    if n < 2 or actions.shape != (n, n) or not np.isin(actions, [0, 1]).all():
        raise ValueError("actions must be a square binary matrix of size at least two")
    off_diagonal = ~np.eye(n, dtype=bool)
    counts = {}
    for label, own, other in (("DD", 0, 0), ("DC", 0, 1), ("CD", 1, 0), ("CC", 1, 1)):
        counts[label] = ((actions == own) & (actions.T == other) & off_diagonal).sum(axis=1)
    total = counts["DD"] + 5 * counts["DC"] + 3 * counts["CC"]
    return [{"machine_id": i, "payoff_rank": int(1 + (total > total[i]).sum()),
             "opponents": n - 1, **{f"games_{key}": int(value[i]) for key, value in counts.items()},
             "total_payoff": int(total[i]), "mean_payoff": float(total[i] / (n - 1))}
            for i in range(n)]


def constant_strategy_responses(actions: np.ndarray) -> tuple[dict, list[dict]]:
    """Test the requested D-on-always-D, C-on-always-C response pattern.

    Behavioral classes use all inputs, including self-play; a class member's
    row is constant. For this ensemble the 256 codes exhaust all eight-bit
    inputs. Also report the smaller classes with four identical W entries.
    """
    defectors = np.flatnonzero((actions == 0).all(axis=1))
    cooperators = np.flatnonzero((actions == 1).all(axis=1))
    if not len(defectors) or not len(cooperators):
        raise ValueError("both constant-behavior classes must be present")
    d_errors = actions[:, defectors].sum(axis=1).astype(int)
    c_errors = (1 - actions[:, cooperators]).sum(axis=1).astype(int)
    errors = d_errors + c_errors
    records = [{"machine_id": i, "cooperates_with_always_defectors": int(d_errors[i]),
                "defects_against_always_defectors": int(len(defectors) - d_errors[i]),
                "cooperates_with_always_cooperators": int(len(cooperators) - c_errors[i]),
                "defects_against_always_cooperators": int(c_errors[i]),
                "constant_strategy_errors": int(errors[i]),
                "constant_strategy_accuracy": float(1 - errors[i] / (len(defectors) + len(cooperators)))}
               for i in range(len(actions))]
    writes = np.array([program_bits(i)[::2] for i in range(MACHINES)])
    literal_d = np.flatnonzero((writes == 0).all(axis=1))
    literal_c = np.flatnonzero((writes == 1).all(axis=1))
    literal_errors = actions[:, literal_d].sum(axis=1) + (1 - actions[:, literal_c]).sum(axis=1)
    best = np.flatnonzero(errors == errors.min()).tolist()
    summary = {
        "definition": "constant final output on every opponent code, including self-play",
        "always_defector_ids": defectors.tolist(), "always_cooperator_ids": cooperators.tolist(),
        "exact_match_ids": np.flatnonzero(errors == 0).tolist(),
        "best_match_ids": best, "minimum_errors": int(errors.min()),
        "targets": int(len(defectors) + len(cooperators)),
        "best_matches": [{**records[i],
                          "misclassified_defector_ids": defectors[actions[i, defectors] == 1].tolist(),
                          "misclassified_cooperator_ids": cooperators[actions[i, cooperators] == 0].tolist()}
                         for i in best],
        "literal_constant_output_rules": {
            "definition": "W(q,b) is identical in all four rule entries",
            "defector_ids": literal_d.tolist(), "cooperator_ids": literal_c.tolist(),
            "exact_match_ids": np.flatnonzero(literal_errors == 0).tolist()},
    }
    return summary, records


def components(adjacency: np.ndarray) -> list[list[int]]:
    """Strongly connected components from exact Boolean reachability."""
    reach = adjacency.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(len(reach)):
        reach |= reach[:, k, None] & reach[None, k, :]
    remaining = set(range(len(reach)))
    groups = []
    while remaining:
        first = min(remaining)
        group = np.flatnonzero(reach[first] & reach[:, first]).tolist()
        groups.append(group)
        remaining.difference_update(group)
    return sorted(groups, key=lambda group: (-len(group), group[0]))


def analyze(actions: np.ndarray) -> tuple[dict, list[dict]]:
    a = graph_adjacency(actions)
    i, j = np.triu_indices(MACHINES, 1)
    outcomes = 2 * actions[i, j] + actions[j, i]
    counts = dict(zip(("DD", "DC", "CD", "CC"),
                      np.bincount(outcomes, minlength=4).tolist()))
    indegree, outdegree = a.sum(axis=0), a.sum(axis=1)
    groups = components(a)
    scc = {node: index for index, group in enumerate(groups) for node in group}
    records = []
    for node in range(MACHINES):
        tape = program_bits(node)
        records.append({
            "machine_id": node,
            "program": " ".join("".join(map(str, tape[k:k + 2])) for k in range(0, 8, 2)),
            "in_degree": int(indegree[node]), "out_degree": int(outdegree[node]),
            "mutual_neighbors": int((a[node] & a[:, node]).sum()),
            "self_cooperates": int(actions[node, node]), "scc": scc[node],
            "rules": [[tape[k], tape[k + 1] + 1] for k in range(0, 8, 2)],
        })
    ranking = payoff_ranking(actions)
    response_summary, response_records = constant_strategy_responses(actions)
    for record, payoff, response in zip(records, ranking, response_records):
        record.update(payoff)
        record.update(response)
    winners = [record["machine_id"] for record in ranking if record["payoff_rank"] == 1]
    totals = np.array([record["total_payoff"] for record in ranking])
    scores, frequencies = np.unique(totals, return_counts=True)
    # Each orientation of a triangle is counted once, up to cyclic rotation.
    integer_a = a.astype(np.int64)
    triangles = int(((integer_a @ integer_a) * integer_a.T).sum() // 3)
    summary = {
        "nodes": MACHINES, "distinct_pair_games": len(i),
        "self_play_games": MACHINES, "total_games_including_self_play": len(i) + MACHINES,
        "directed_edges_excluding_self_loops": int(a.sum()),
        "edge_density": float(a.sum() / (MACHINES * (MACHINES - 1))),
        "distinct_pair_outcomes": counts,
        "self_play_outcomes": {"CC": int(np.trace(actions)),
                               "DD": MACHINES - int(np.trace(actions))},
        "reciprocated_edge_fraction": float((a & a.T).sum() / a.sum()),
        "strongly_connected_components": groups,
        "weak_component_sizes": [len(group) for group in components(a | a.T)],
        "zero_out_degree_nodes": np.flatnonzero(outdegree == 0).tolist(),
        "zero_in_degree_nodes": np.flatnonzero(indegree == 0).tolist(),
        "directed_three_cycles": triangles,
        "distinct_action_profiles_including_self_play": len(np.unique(actions, axis=0)),
        "in_degree_range": [int(indegree.min()), int(indegree.max())],
        "cooperators_per_opponent_including_self": np.unique(actions.sum(axis=0)).tolist(),
        "constant_strategy_responses": response_summary,
        "payoffs": {
            "matrix": {"DD": 1, "DC": 5, "CD": 0, "CC": 3},
            "convention": "first action is focal machine; uniform over 255 other machines; self-play excluded",
            "winner_ids": winners, "winner_count": len(winners),
            "maximum_total_payoff": records[winners[0]]["total_payoff"],
            "maximum_mean_payoff": records[winners[0]]["mean_payoff"],
            "rank_convention": "competition rank: one plus number of strictly higher scores",
            "distribution_convention": "equal weight per labeled machine; each score averages 255 other opponents",
            "distribution": [{"total_payoff": int(score), "mean_payoff": float(score / 255),
                              "machines": int(count), "probability": float(count / MACHINES)}
                             for score, count in zip(scores, frequencies)],
            "population_mean": float(totals.mean() / 255),
            "population_median": float(np.median(totals) / 255),
            "population_standard_deviation": float(totals.std() / 255),
            "minimum_mean_payoff": float(totals.min() / 255),
        },
    }
    return summary, records


def save_graphml(path: Path, adjacency: np.ndarray, records: list[dict]) -> None:
    namespace = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", namespace)
    tag = lambda name: f"{{{namespace}}}{name}"
    root = ET.Element(tag("graphml"))
    keys = {"program": "string", "in_degree": "int", "out_degree": "int",
            "mutual_neighbors": "int", "self_cooperates": "int", "scc": "int",
            "payoff_rank": "int", "total_payoff": "int", "mean_payoff": "double"}
    for name, kind in keys.items():
        ET.SubElement(root, tag("key"), {"id": name, "for": "node",
                      "attr.name": name, "attr.type": kind})
    graph = ET.SubElement(root, tag("graph"), {"id": "cooperation", "edgedefault": "directed"})
    for record in records:
        node = ET.SubElement(graph, tag("node"), {"id": str(record["machine_id"])})
        for name in keys:
            ET.SubElement(node, tag("data"), {"key": name}).text = str(record[name])
    for i, j in zip(*np.nonzero(adjacency)):
        ET.SubElement(graph, tag("edge"), {"source": str(i), "target": str(j)})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def save_figure(output: Path, adjacency: np.ndarray, records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    outdegree = np.array([row["out_degree"] for row in records])
    indegree = np.array([row["in_degree"] for row in records])
    order = np.lexsort((np.arange(MACHINES), indegree, outdegree))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fig.suptitle("Cooperation among all 256 two-state machines", fontsize=17)
    axes[0].imshow(adjacency[np.ix_(order, order)], interpolation="nearest",
                   cmap=ListedColormap(["#f3f5f8", "#176b83"]), vmin=0, vmax=1)
    axes[0].set(title="Directed adjacency · row cooperates with column",
                xlabel="Target (ordered by out-degree, then in-degree)", ylabel="Source (same order)")
    degree_pairs, count = np.unique(np.stack([outdegree, indegree], axis=1),
                                    axis=0, return_counts=True)
    axes[1].scatter(degree_pairs[:, 0], degree_pairs[:, 1], s=14 * count,
                    alpha=0.6, color="#176b83", edgecolors="white", linewidths=0.4)
    axes[1].set(title="Degree structure · marker area counts machines",
                xlabel="Out-degree: opponents this machine cooperates with",
                ylabel="In-degree: opponents cooperating with this machine",
                xlim=(-9, 264), ylim=(126, 129), yticks=[127, 128])
    axes[1].grid(alpha=0.15)
    fig.text(0.5, -0.025, "One left-to-right scan of the opponent's 8-bit program; final output is the action. Self-loops excluded.",
             ha="center", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"cooperation_graph.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_payoff_and_response_figures(output: Path, summary: dict, records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distribution = summary["payoffs"]["distribution"]
    x = np.array([row["mean_payoff"] for row in distribution])
    count = np.array([row["machines"] for row in distribution])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    fig.suptitle("Payoffs across all 256 two-state machines", fontsize=16)
    axes[0].vlines(x, 0, count, color="#176b83", linewidth=1.6)
    axes[0].scatter(x, count, s=15, color="#176b83", zorder=3)
    axes[0].set(title="Exact distribution · no binning", ylabel="Number of machines",
                xlabel="Mean payoff against 255 other machines", ylim=(0, 53))
    for value, label in ((381 / 255, "34 always-cooperators"), (767 / 255, "34 always-defectors")):
        axes[0].annotate(label, (value, 34), xytext=(value, 45), ha="center", fontsize=9,
                         arrowprops={"arrowstyle": "-", "color": "#778b99"})
    axes[1].step(np.r_[x[0] - .05, x], np.r_[0, count.cumsum() / MACHINES],
                 where="post", color="#176b83", linewidth=2)
    axes[1].set(title="Cumulative distribution", xlabel="Mean payoff against 255 other machines",
                ylabel="Fraction of machines at or below this payoff", ylim=(0, 1.04))
    for ax in axes:
        ax.set_xlim(1.37, 3.14)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, -.03, "CC = 3, DD = 1, DC = 5, CD = 0 · Self-play excluded · Equal weight per labeled machine",
             ha="center", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"payoff_distribution.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    responses = summary["constant_strategy_responses"]
    n_d, n_c = len(responses["always_defector_ids"]), len(responses["always_cooperator_ids"])
    pairs = np.array([[row["cooperates_with_always_defectors"],
                       row["cooperates_with_always_cooperators"]] for row in records])
    unique, multiplicity = np.unique(pairs, axis=0, return_counts=True)
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    ax.scatter(unique[:, 0], unique[:, 1], s=20 + 12 * multiplicity, color="#176b83",
               alpha=.65, edgecolors="white", linewidths=.7, label="Two-state machines (area reflects count)")
    ax.scatter([0], [n_c], marker="*", s=190, color="#172b3a", label="Requested perfect response")
    for i in responses["best_match_ids"]:
        dx, cy = pairs[i]
        ax.scatter([dx], [cy], s=100, color="#d47e28", zorder=4)
        ax.annotate(f"Machine {i}: {n_d - dx}/{n_d} D correct, {cy}/{n_c} C correct",
                    (dx, cy), xytext=(dx + 2, cy - 4), fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "#d47e28"})
    ax.set(title="Can a machine distinguish constant strategies?",
           xlabel=f"Cooperates with always-defectors (out of {n_d}) · fewer is better",
           ylabel=f"Cooperates with always-cooperators (out of {n_c}) · more is better",
           xlim=(-2, n_d + 2), ylim=(-2, n_c + 3), aspect="equal")
    ax.legend(loc="lower center", fontsize=9, framealpha=.95)
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"constant_strategy_responses.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_results(output: Path) -> dict:
    actions = all_actions()
    adjacency = graph_adjacency(actions)
    summary, records = analyze(actions)
    np.savez_compressed(output / "adjacency.npz", adjacency=adjacency, actions=actions,
                        machine_ids=np.arange(MACHINES), self_actions=actions.diagonal())
    with (output / "machines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in records[0] if key != "rules"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    ordered = sorted(records, key=lambda record: (-record["total_payoff"], record["machine_id"]))
    for filename, selected in (("payoff_ranking.csv", ordered),
                               ("constant_strategy_responses.csv", sorted(records, key=lambda row: (
                                   row["constant_strategy_errors"], -row["total_payoff"], row["machine_id"]))),
                               ("best_machines.csv", [row for row in ordered if row["payoff_rank"] == 1])):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[key for key in records[0] if key != "rules"],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    with (output / "payoff_distribution.csv").open("w", newline="") as handle:
        distribution = summary["payoffs"]["distribution"]
        writer = csv.DictWriter(handle, fieldnames=list(distribution[0]))
        writer.writeheader()
        writer.writerows(distribution)
    with (output / "edges.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        writer.writerows(zip(*np.nonzero(adjacency)))
    with (output / "games.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["machine_i", "machine_j", "action_i", "action_j", "outcome"])
        for i in range(MACHINES):
            for j in range(i, MACHINES):
                x, y = int(actions[i, j]), int(actions[j, i])
                writer.writerow([i, j, x, y, "DC"[x] + "DC"[y]])
    metadata = {
        "model": {"states": 2, "symbols": 2, "motion": "fixed rightward",
                  "initial_state": 1, "tape_length": 8,
                  "encoding": "entry k=2*(q-1)+b: cells 2k,2k+1 are W(q,b),S(q,b)-1",
                  "machine_id": "sum(tape[cell] * 2**cell); distinct from full-TM census IDs",
                  "decision": "eighth emitted W bit; immutable opponent code; fresh state each match",
                  "edge": "source cooperates with target, regardless of target action",
                  "self_play": "included in actions and games.csv; omitted from graph adjacency and edges",
                  "outcome_order": "i,j with i<j; C=1, D=0"},
        "summary": summary,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "numpy_version": np.__version__,
                       "command": ".venv/bin/python -m scripts.cooperation_graph --output-dir results/cooperation_graph_n2_repeat"},
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_graphml(output / "cooperation.graphml", adjacency, records)
    save_figure(output, adjacency, records)
    save_payoff_and_response_figures(output, summary, records)
    payload = {"nodes": records, "adjacency": adjacency.tolist(), "summary": summary}
    template = Path(__file__).with_suffix(".html").read_text()
    (output / "index.html").write_text(template.replace("__GRAPH_DATA__", json.dumps(payload)))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cooperation_graph_n2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output directory is not empty; choose a new --output-dir")
    summary = write_results(args.output_dir)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "strongly_connected_components"}, indent=2))
    print(f"Saved {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
