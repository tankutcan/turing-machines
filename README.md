# turing-machines

Statistical mechanics of Turing machines: shared-tape gases, spacetime zoos,
state-space structure, and Busy Beaver experiments.

## Layout

```text
turing-machines/
├── turing_machines/       # simulation and analysis implementations
├── scripts/               # short command-line launchers
├── tests/                 # automated checks
├── results/               # generated figures, data, and run metadata
├── figure/                # figures included in the LaTeX write-up
├── turing-machines.ipynb   # exploratory notebook
├── tm-stat-mech.tex        # write-up (with .bib and compiled .pdf)
├── busybeavers.md          # reference notes (also available as HTML)
└── requirements.txt
```

This is a flat package layout: `turing_machines/` lives directly in the project
root, with no `src/` directory or package-installation step. The underscore in
the package name allows Python imports. Implementations retain their original
module names; `scripts/` just calls their `main()` functions.

## First-time setup

After cloning the repository, open a terminal in the project root (the folder
containing `requirements.txt`). With Python 3 installed, run the following
commands on macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

This creates a local Python environment in `.venv/` and installs the dependencies
listed in `requirements.txt`. The environment is machine-specific and is not
included in the GitHub repository; each checkout needs its own setup. You only
need to do this once per checkout, not before every experiment.

## Running experiments

After completing setup, run the commands below from the project root. They use
the Python environment you just created; no separate activation is required.

For example, this generates a small, two-state zoo with a periodic tape:

```sh
.venv/bin/python -m scripts.generate_spacetime_zoo \
  --states 2 --count 4 --copies 20 --steps 200 --spacing 5 \
  --boundary periodic --tape-length 128 --plot-width 128 \
  --record-every 2 --grid-columns 2 --pdf-panel-size 5 \
  --output-dir results/example_two_state_zoo
```

Use `--help` to see a command's options and defaults:

```sh
.venv/bin/python -m scripts.many_turing_machines --help
```

Available commands, all run as `.venv/bin/python -m scripts.<name>`:

- `many_turing_machines`: simulate a gas and measure velocities.
- `generate_spacetime_zoo`: sample machines and make a spacetime contact sheet.
- `generate_one_state_zoo`: enumerate the 16 one-state machines.
- `order_two_state_zoo`: arrange a saved zoo by average write and move rules.
- `analyze_two_state_structure`: annotate a saved zoo with state-space structure.
- `alignment_census`: classify tables by whether their gas aligns into one state.
- `coupled_chiral_ants`: study the coupled-ant machines.
- `mutual_rewriting`: two two-state machines rewriting each other's program tapes.
- `cooperation_graph`: all pairwise games among 256 two-state, right-moving
  machines; edges point from each cooperator to its opponent. Run
  `.venv/bin/python -m scripts.cooperation_graph` and open
  `results/cooperation_graph_n2/index.html` for the interactive graph. The archive
  includes GraphML, edge and game CSVs, NumPy adjacency, and PNG/PDF figures.
  `payoff_ranking.csv` ranks all machines by average payoff against the 255
  others using CC=3, DD=1, DC=5, CD=0; `best_machines.csv` contains all tied winners.
  `payoff_distribution.{csv,png,pdf}` gives the exact distribution across machines.
  `constant_strategy_responses.{csv,png,pdf}` tests how each machine responds to
  the always-defecting and always-cooperating behavioral classes; the summary
  separately checks tables whose four output rules are all identical.
  Each machine reads the opponent's eight-bit `(W, S-1)` program once from state
  1 and uses the last output bit as its action. Self-play is saved separately
  from the graph, which has no self-loops.
- `simulate_busybeaver4`: simulate the four-state winner with its halt redirected.
- `sample_halt_times`: sample four-state machines on blank tapes.
- `sample_random_tape_halt_times`: reuse sampled machines on random tapes.

The two saved-zoo analysis commands also require Poppler's `pdfimages` and
`pdftotext` on `PATH` (on macOS: `brew install poppler`).

Paths supplied to commands are relative to the working directory unless
absolute. Existing defaults have been preserved; use `--output-dir` or
`--output-prefix`, as applicable, to choose a new destination inside `results/`.
Some commands refuse nonempty destinations; others can overwrite outputs, so
choose a fresh destination for a new experiment.

If Matplotlib cannot write to its default cache directory, prefix a command
with `env MPLCONFIGDIR=/tmp/turing-machines-matplotlib`.

## Using the code in Python

From the project root or the root-level notebook:

```python
from turing_machines.many_turing_machines import make_tm, simulate
from turing_machines.coupled_chiral_ants import make_chiral_tm
```

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Older commands and saved results

Replace an old invocation such as `python generate_spacetime_zoo.py ...` with
`python -m scripts.generate_spacetime_zoo ...`. Use module invocation (`-m`),
not `python scripts/generate_spacetime_zoo.py`, so the project package can be
imported without installation. Imports similarly gain the `turing_machines.`
prefix; source files now live in `turing_machines/`.

Existing results, figures, notebooks, and LaTeX files have not been moved or
regenerated. Archived reports retain their original commands and code hashes
as historical provenance; apply the command substitution above when rerunning
them. In new metadata, source-file basenames and their hashes refer to the
implementations in `turing_machines/`, not the launchers in `scripts/`.
