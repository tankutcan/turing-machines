# Figures used in tm-stat-mech.tex

The spacetime-zoo appendix embeds the 13 PDFs in this folder. They are byte-for-byte copies of the archived contact sheets, containing 1,216 panels in total. No simulations or image resampling were performed when adding them to the notes.

Captions and reading conventions are maintained in `tm-stat-mech.tex`. `zoo-manifest.json` records each figure's source, checksum, complete original configuration, and normalized parameters. Older configurations omit fields that were then fixed simulator conventions: identical machines, fixed initial state 1, synchronous updating, and an unbounded tape. Those legacy defaults are recorded explicitly rather than presented as original metadata.

| Figure | Original results directory |
| --- | --- |
| [zoo_tm1_all16.pdf](zoo_tm1_all16.pdf) | `results/spacetime_zoo_1_state_all_16_periodic_L10000_T10000/` |
| [zoo_tm2_random_states_periodic.pdf](zoo_tm2_random_states_periodic.pdf) | `results/spacetime_zoo_2_states_1000_identical_agents_random_initial_states_periodic_L10000_T10000/` |
| [zoo_tm3_fixed_states.pdf](zoo_tm3_fixed_states.pdf) | `results/spacetime_zoo_3_states_1000_agents/` |
| [zoo_tm3_seeds100-199.pdf](zoo_tm3_seeds100-199.pdf) | `results/spacetime_zoo_3_states_1000_agents_seeds_100_199/` |
| [zoo_tm3_blank_tape.pdf](zoo_tm3_blank_tape.pdf) | `results/spacetime_zoo_3_states_1000_agents_blank_tape/` |
| [zoo_tm3_async.pdf](zoo_tm3_async.pdf) | `results/spacetime_zoo_3_states_1000_identical_agents_async/` |
| [zoo_tm6_fixed_states.pdf](zoo_tm6_fixed_states.pdf) | `results/spacetime_zoo_6_states_1000_agents/` |
| [zoo_tm6_random_states.pdf](zoo_tm6_random_states.pdf) | `results/spacetime_zoo_6_states_1000_identical_agents_random_initial_states/` |
| [zoo_tm6_random_states_periodic.pdf](zoo_tm6_random_states_periodic.pdf) | `results/spacetime_zoo_6_states_1000_identical_agents_random_initial_states_periodic_L10000_T10000/` |
| [zoo_tm20_N200.pdf](zoo_tm20_N200.pdf) | `results/spacetime_zoo/` |
| [zoo_tm20_N1000.pdf](zoo_tm20_N1000.pdf) | `results/spacetime_zoo_1000_agents/` |
| [zoo_tm3_independent.pdf](zoo_tm3_independent.pdf) | `results/spacetime_zoo_3_states_1000_independent_agents/` |
| [zoo_tm3_independent_periodic.pdf](zoo_tm3_independent_periodic.pdf) | `results/spacetime_zoo_3_states_1000_independent_agents_periodic_L5000_T10000/` |

Each source directory retains its `manifest.csv`, individual plots, velocity data, and exact transition tables (`transition_tables.json`, or `.json.gz` for independent-machine ensembles). The original configuration records the generator hash and NumPy version. Matching seeds across different geometries or sampling configurations need not produce the same initial tape.

## Scientific conventions

- The notation is TM_n(2): n internal states and two tape symbols. The functions M, W, S specify movement, writing, and next state.
- Original panel headings use `S` for n and `W` for display width; they are not the functions S_q and W_q.
- Space increases rightward; time increases upward. Dark/light pixels show tape symbols 0/1. Red is one head; yellow is two or more heads at a displayed site.
- Snapshots were recorded every five steps (sweeps for asynchronous runs), and the archived contact sheets further downsample them. Narrow trajectories can be absent from the overview sampling.
- `open` denotes an unbounded tape viewed through a fixed window, not an absorbing boundary. The zoos contain no reservoirs.
- Synchronous writes use majority vote, preserving the old symbol on ties. Asynchronous updates apply each agent once per random-order sweep without voting.
- The one-state grid enumerates IDs 0–15 in bit order [W_1(0), (M_1(0)+1)/2, W_1(1), (M_1(1)+1)/2]. Every one-state rule has S_1(b)=1 and uses the same initial tape, seed 0.

## Rebuild the document

Run from the project root:

```sh
pdflatex -interaction=nonstopmode -halt-on-error tm-stat-mech.tex
bibtex tm-stat-mech
pdflatex -interaction=nonstopmode -halt-on-error tm-stat-mech.tex
pdflatex -interaction=nonstopmode -halt-on-error tm-stat-mech.tex
```

