# Busy beavers and coupled chiral machines

A reference collection in the notation of [the notes](tm-stat-mech.tex).
The champion tables below follow Pascal Michel's published collection; the
chiral examples are our own comparisons. All five scores were checked by
direct simulation. Last checked: 2026-09-21.

## Conventions

We consider one head on an infinite binary tape, initially all zeros, with
$x_0=0$ and $q_0=1$. There are $n$ working states and an additional halt state
$H$, which does **not** count toward $n$. These are single-machine experiments,
not many-agent gases or finite periodic-tape simulations.

For the bit $b=V_t(x_t)$ read **before** writing, the update is

$$
V_{t+1}(x_t)=W_{q_t}(b),\qquad
x_{t+1}=x_t+M_{q_t}(b),\qquad
q_{t+1}=S_{q_t}(b).
$$

All other tape sites are unchanged. Here $M=-1$ means left and $M=+1$ means
right. One complete read/write/move/state-change counts as **one step**,
including the final transition into $H$. The initial configuration is time zero.

For a halting machine, let $T$ denote its number of steps and $\sigma$ its
number of ones remaining at halt, not the cumulative number of writes.
The maxima over halting machines are conventionally called $S(n,2)$ and
$\Sigma(n,2)$, respectively. The scalar $S(n,2)$ is distinct from our
next-state function $S_q(b)$.

In the code, rules have the form
`tm[q][b] = [write_bit, move_bit, next_state]`, where
$M=2\,\texttt{move_bit}-1$. The standalone
[chiral-machine simulator](turing_machines/coupled_chiral_ants.py) uses `next_state=0` for
$H$; this halt encoding is not supported by the many-agent simulator.

## Collection at a glance

The established binary-tape maxima are $S(2,2)=6$, $\Sigma(2,2)=4$,
$S(4,2)=107$, and $\Sigma(4,2)=13$.
[Source: Pascal Michel, Busy Beaver Competitions](https://bbchallenge.org/~pascal.michel/bbc).

| Example | Working states | Steps $T$ | Final ones $\sigma$ | Status |
| --- | ---: | ---: | ---: | --- |
| Classical two-state champion | 2 | 6 | 4 | Maximizes both scores |
| Coupled chiral pair $LR$, with a halt | 2 | 6 | 2 | Matches the time maximum, not the ones maximum |
| Four-state, 107-step champion | 4 | 107 | 13 | Maximizes both scores |
| Four-state, 96-step champion | 4 | 96 | 13 | Maximizes ones, not time |
| Cyclic $LRLR$, best single halt insertion | 4 | 13 | 3 | Comparison model, not a champion |

This is a selection of explicit representatives, not an enumeration of all
equivalent machines. Reflecting every move preserves both scores on a blank
infinite tape, as does consistently relabeling states and the initial state.

## Classical two-state champion

| State $q$ | $W_q(b)$ | $M_q(b)$ | $S_q(0)$ | $S_q(1)$ |
| --- | --- | --- | --- | --- |
| 1 | $1$ | $1-2b$ | 2 | 2 |
| 2 | $1$ | $2b-1$ | 1 | $H$ |

This machine halts after **6 steps with 4 ones**. Both states always write 1:
they have opposite bit-dependent steering, but are not bit-flipping ants.
[Source: Michel's two-state table](https://bbchallenge.org/~pascal.michel/ha).

## Coupled chiral pair: a six-step bit-flipping example

Our chirality labels mean

$$
L:\quad W(b)=1-b,\quad M(b)=2b-1,
\qquad
R:\quad W(b)=1-b,\quad M(b)=1-2b.
$$

Thus $L$ moves left on 0 and right on 1, whereas $R$ does the reverse.
These are internal modes of **one head**, not two separate agents.

Start with the non-halting cycle $1L\to2R\to1L$, then replace only
$S_2(1)$ by $H$:

| State $q$ | $W_q(b)$ | $M_q(b)$ | $S_q(0)$ | $S_q(1)$ |
| --- | --- | --- | --- | --- |
| 1 ($L$) | $1-b$ | $2b-1$ | 2 | 2 |
| 2 ($R$) | $1-b$ | $1-2b$ | 1 | $H$ |

Direct simulation gives **6 steps and 2 final ones**, at positions $-1$ and
$2$. It matches the two-state time maximum but is not a productivity winner.
In particular, its final transition writes 0; we retain that write when
inserting the halt, rather than replacing the entire rule with a conventional
`write 1, move right, halt` instruction.

## Four-state champion: 107 steps, 13 ones

This is the champion attaining both four-state maxima. Numbered states
$1,2,3,4$ correspond to the conventional $A,B,C,D$.
[Source: Michel's four-state table](https://bbchallenge.org/~pascal.michel/ha).

| State $q$ | $W_q(b)$ | $M_q(b)$ | $S_q(0)$ | $S_q(1)$ |
| --- | --- | --- | --- | --- |
| 1 | $1$ | $1-2b$ | 2 | 2 |
| 2 | $1-b$ | $-1$ | 1 | 3 |
| 3 | $1$ | $1-2b$ | $H$ | 4 |
| 4 | $1-b$ | $+1$ | 4 | 1 |

Interpreted as local modes:

- States 1 and 3 always write 1 and have $R$-type steering.
- State 2 flips the bit and always moves left.
- State 4 flips the bit and always moves right.
- The next-state routing depends on the read bit; it is not a fixed cycle.

Consequently, **none of its states is exactly an $L$ or $R$ bit-flipping ant**
as defined above. It combines the steering and writing ingredients differently.
The halting event is the first read of 0 in state 3, on step 107.

## Another four-state productivity champion: 96 steps, 13 ones

This distinct machine also attains the maximum final number of ones, but
does not attain the maximum runtime.
[Source: Michel's four-state table](https://bbchallenge.org/~pascal.michel/ha).

| State $q$ | $W_q(b)$ | $M_q(b)$ | $S_q(0)$ | $S_q(1)$ |
| --- | --- | --- | --- | --- |
| 1 | $1-b$ | $+1$ | 2 | 3 |
| 2 | $1$ | $2b-1$ | 1 | 1 |
| 3 | $1$ | $+1$ | $H$ | 4 |
| 4 | $1-b$ | $-1$ | 4 | 2 |

Here states 1 and 4 flip while moving in fixed opposite directions; state 2
always writes 1 with $L$-type steering; state 3 always writes 1 and moves right.
The halting event is again the first read of 0 in state 3, now on step 96.

For both displayed four-state champions, all working states are mutually
reachable in the directed transition graph, excluding the halting vertex.
Neither decomposes into isolated one-state species. This observation alone
does not explain the long runtime: the evolving tape constrains which graph
paths are actually followed.

## Comparison: the four-mode cycle $1L\to2R\to3L\to4R\to1L$

The non-halting parent has

$$
W_q(b)=1-b,\qquad
M_q(b)=
\begin{cases}
2b-1,&q=1,3,\\
1-2b,&q=2,4,
\end{cases}
\qquad
S_q(b)=1+(q\bmod4).
$$

Replacing one of its eight next-state entries by $H$, without changing the
write or move, gives the following exhaustive results:

| Halt at state $q$ | Read bit $b$ | Steps | Final ones |
| --- | --- | ---: | ---: |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 13 | 3 |
| 2 | 0 | 2 | 2 |
| 2 | 1 | 6 | 2 |
| 3 | 0 | 7 | 3 |
| 3 | 1 | 3 | 1 |
| 4 | 0 | 4 | 2 |
| 4 | 1 | 8 | 2 |

The longest run halts at $(q,b)=(1,1)$ after **13 steps**, leaving ones at
$-3,-1,2$. All eight placements halt, so these are resolved scores, not
time-budget lower bounds. This cycle does not attain either four-state record.

Before adding a halt, identifying states 1 and 3 with $L$, and 2 and 4 with
$R$, reproduces the two-state $LR$ head-and-tape trajectory exactly. The
extra labels record the phase modulo four. A halt can distinguish those
labels, but the parent has no new tape dynamics under this projection.

[Full results and reproduction command](results/coupled_chiral_ants_LRLR/README.md)
and [spacetime plot](results/coupled_chiral_ants_LRLR/spacetime_best.pdf).
The plot uses light for 1, dark for 0, and red for the head.

## Uniform rule averages

For comparison with our zoo, define the averages over all $2n$ rule entries,
including the halting entry:

$$
\overline W=\frac{1}{2n}\sum_{q=1}^n\sum_{b=0}^1 W_q(b),
\qquad
\overline M=\frac{1}{2n}\sum_{q=1}^n\sum_{b=0}^1 M_q(b).
$$

| Example | $\overline W$ | $\overline M$ |
| --- | ---: | ---: |
| Classical two-state champion | $1$ | $0$ |
| Coupled chiral pair $LR$ | $1/2$ | $0$ |
| Four-state 107-step champion | $3/4$ | $0$ |
| Four-state 96-step champion | $3/4$ | $1/4$ |
| Four-mode cycle $LRLR$, any halt insertion | $1/2$ | $0$ |

These are properties of the rule tables, not time averages. In particular,
$\overline M$ need not equal the measured head velocity, since the trajectory
does not generally sample state/read pairs uniformly.

## Using non-halting parents to search for candidates

For a parent machine without a halt, define the first execution time of a rule:

$$
\tau_{q,b}=\min\{t+1:q_t=q,\ V_t(x_t)=b\},
$$

with $\tau_{q,b}=\infty$ if it is never used. Replacing only $S_q(b)$ by
$H$ makes the altered machine halt at exactly $\tau_{q,b}$, if finite.
The two trajectories agree through the write and move of that final step.
Thus the best runtime among single halt insertions is the largest finite
first-use time, not a state-graph cycle length.

Productivity must be evaluated separately: it is the number of ones after
that particular write. An exact recurrence of the full configuration
$(q,x,V)$ before an unvisited halt rule is reached instead certifies that
the corresponding halt insertion will never be encountered. A simulation
timeout alone does not certify non-halting.

## Rechecking the scores

From the project root, the following uses signed moves in `rules`, converts
them to the simulator's move-bit encoding, and checks the five examples:

```sh
.venv/bin/python - <<'PY'
from turing_machines.coupled_chiral_ants import evaluate, simulate, make_chiral_tm, with_halt

# Each row contains read-0 and read-1 triples (write, signed move, next).
# Working states are 1-based; next=0 denotes Halt.
rules = {
    "two_state": [
        [(1, +1, 2), (1, -1, 2)],
        [(1, -1, 1), (1, +1, 0)],
    ],
    "four_state_107": [
        [(1, +1, 2), (1, -1, 2)],
        [(1, -1, 1), (0, -1, 3)],
        [(1, +1, 0), (1, -1, 4)],
        [(1, +1, 4), (0, +1, 1)],
    ],
    "four_state_96": [
        [(1, +1, 2), (0, +1, 3)],
        [(1, -1, 1), (1, +1, 1)],
        [(1, +1, 0), (1, +1, 4)],
        [(1, -1, 4), (0, -1, 2)],
    ],
}
expected = {"two_state": (6, 4), "four_state_107": (107, 13),
            "four_state_96": (96, 13)}
for name, rows in rules.items():
    tm = {q: {b: [w, (m + 1) // 2, s]
              for b, (w, m, s) in enumerate(row)}
          for q, row in enumerate(rows, 1)}
    result = simulate(tm, 200)
    score = result["steps"], result["ones"]
    assert result["halted"] and score == expected[name]
    print(name, score)

result = simulate(with_halt(make_chiral_tm("LR"), 2, 1), 200)
assert result["halted"] and (result["steps"], result["ones"]) == (6, 2)
print("chiral_LR", (result["steps"], result["ones"]))

_, _, _, variants = evaluate("LRLR", 200)
assert all(result["halted"] for result in variants.values())
best = max(variants.values(), key=lambda result: result["steps"])
assert (best["steps"], best["ones"]) == (13, 3)
assert max(result["ones"] for result in variants.values()) == 3
print("chiral_LRLR_best", (best["steps"], best["ones"]))
PY
```

These finite simulations verify the displayed machines' scores, not the
global Busy Beaver upper bounds; those rely on the established results
cited above.
