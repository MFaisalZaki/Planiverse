# Water distribution (contamination containment)

A contaminant enters a drinking-water network at one junction. Every node downstream of it
drinks a share, and which nodes those are depends on where the water is flowing — which is
the solution of a nonlinear system over the whole network. The operator's move is to close
pipes. Closing a pipe reroutes the flow, so it changes the answer everywhere at once.

The goal is to contain the contamination *without cutting off the customers*, and that
tension is the problem: every closed pipe contains a little more and costs a little more
service.

- **Class:** `WaterNetworkEnv`
- **Import:** `from planiverse.problems.real_world_problems.water_distribution.environment import WaterNetworkEnv`
- **Source:** [`environment.py`](../../planiverse/problems/real_world_problems/water_distribution/environment.py)
- **Dependencies:** `wntr` — and nothing else. The benchmark networks ship inside it, so
  unlike the Game Boy environments there is nothing to supply.

## Why this is not a PDDL domain

Not "PDDL would be verbose here". Two properties, both measured on the shipped networks:

**Effects are global.** On `Net3`, closing a single pipe changes the pressure at **94 of the
97 nodes**. The effect of `close pipe 123` is not a set of facts to add and delete; it is
"re-solve the hydraulics and the transport, and see what happens".

**Effects are not monotone.** On `Net1` with the source at node 12, closing pipe `110` makes
the contamination **worse** — it pushes flow down a path that reaches more customers. A
delete-list cannot express that, and neither can a planner that assumes closing more pipes
contains more.

There is a third, softer one: which pipes matter is not readable off the topology. On `Net3`
the source has four pipes on it; two carry essentially all the contamination and two do
nothing at all. Nothing structural distinguishes them — only the solve does.

## Quickstart

```python
from planiverse.problems.real_world_problems.water_distribution.environment import WaterNetworkEnv

env = WaterNetworkEnv()
env.fix_index(0)                      # Net1, contaminant at junction 23
state, info = env.reset()

print(state)
# closed: nothing
# contaminated delivered: 13.6%
# service: 100.0%

for action, successor in env.successors(state):
    print(action, successor.contaminated, successor.service)
# close_pipe_11 0.0669 1.0
# close_pipe_111 0.1358 1.0
# ...

env.close()                           # removes the simulator's scratch directory
```

## Scenarios

`fix_index(i)` picks a network and the junction the contaminant enters at. Nine scenarios,
ordered by how deep a solution is:

| Index | Network | Source | Contaminated at t=0 | Solved at depth |
|---|---|---|---|---|
| 0 | Net1 | 23 | 13.6% | 2 |
| 1 | Net3 | 123 | 61.9% | 2 |
| 2 | Net3 | 199 | 41.4% | 2 |
| 3 | Net3 | 121 | 56.3% | 2 |
| 4 | Net1 | 21 | 30.0% | 3 |
| 5 | Net3 | 119 | 56.2% | 3 |
| 6 | Net1 | 12 | 40.8% | 4 |
| 7 | Net1 | 22 | 28.7% | 4 |
| 8 | Net1 | 11 | 80.1% | **7** |

**`solved_at` is a measurement, not an estimate** — the shallowest depth at which a solution
was actually found. It is recorded because an instance whose goal nobody has reached is a
poor benchmark entry: a planner cannot tell "unreachable" from "not found yet". Every
scenario here has been solved.

`Net2` was tried and **dropped** for exactly that reason. Its sources sit on trunk mains
that cannot be rerouted around, and a width-16 beam search to depth 8 with every one of its
40 pipes available plateaued at 36% contamination and stopped improving. Rather than ship an
instance that is probably unsolvable, it is not in the list.

The sources are not arbitrary either. `rank_sources(network_file)` runs every junction of a
network as the source and reports how much of the delivered water ends up contaminated; the
scenarios were chosen off that ranking, and the function is kept so the choice can be
re-derived rather than taken on trust.

### Index 8 is the interesting one

`Net1` with the source at node 11 contaminates 80% of everything delivered. It is
**solvable** — proved exhaustively over the full powerset of its 12 pipes, with exactly
three feasible closures at depth 7, one of which reaches zero contamination at **100%**
service. And a width-16 beam search to depth 8 does not find it.

It is also where the obvious heuristic fails. Ranking states by contamination alone marches
straight into "close everything": contamination zero, service zero, goal failed. Getting
this instance needs a heuristic that respects the service constraint on the way, not just at
the end.

## Determinism

The design rests on this, so it is tested rather than assumed: the same set of closed pipes
simulates **bit-identically** every time (max absolute pressure difference `0.0` across
repeated runs). Three consequences:

- **The closed set is a sufficient statistic for the state.** Two states with the same pipes
  shut *are* the same state, so `__eq__` and `__hash__` are on the closed set alone and
  search can close over them.
- **Results are memoised** on the closed set, which is only sound because of the above.
- **`simulate` re-runs from scratch** and is therefore an independent check on `successors`
  rather than a restatement of it.

`depth` is deliberately *not* part of state identity — with a step counter in there no
successor could ever equal its parent and the self-loop filter would be dead code, the trap
the [urban planning](urban-planning.md#known-quirks) environment fell into.

## State representation

| Field | Meaning |
|---|---|
| `closed` | frozenset of closed pipe names — the whole state |
| `contaminated` | share of all delivered water that came from the source |
| `service` | share of expected demand actually delivered |
| `pressure_deficit` | mean shortfall below the required pressure |
| `depth` | plan length; not part of identity |

### Literals

```
closed(PIPE)
contaminated(N)      # bucketed into twentieths
service(N)
```

The two metrics are bucketed because a planner keyed on raw floats would treat every state
as novel. The closed set is what identifies the state; the buckets are for width-based
methods that measure novelty over atoms.

### Hydraulics are pressure-driven

`demand_model = "PDD"`, not the demand-driven default. It matters: under the demand-driven
model a node takes its full demand no matter how little pressure is left, so closing pipes
would look free and there would be no trade-off to plan against.

One wrinkle this surfaced: a junction may have a **negative** demand, which is an injection
into the network rather than a customer drawing from it. `Net2` has one, and counted as
written it drags the network's total expected demand to −0.02 and makes the service ratio
meaningless — it came out at −1791%. Both sides of the ratio are clipped at zero.

## Actions

`WaterNetworkAction(pipe)` — close one pipe. That is the entire action set, and the entire
operator interface. Cost is 1 each.

The candidate pipes default to those within `radius=2` hops of the source. The full pipe
list is a branching factor of 117 on `Net3`, nearly all of it nowhere near the incident.
`WaterNetworkEnv(radius=None)` offers every pipe, which is the honest setting and a slow
one — and it is what the `Net2` investigation above used, so the restriction is not what
made those instances unsolvable.

## Goal and terminal

- **Goal** — contaminated ≤ 2% **and** service ≥ 80%. Both halves are needed: closing every
  pipe at the source contains perfectly and is not a solution.
- **Terminal** — service < 50%. Sound because service is **monotone** in the closed set: a
  network cannot deliver more water with more pipes shut, so a collapsed state can never
  recover. Note that contamination is *not* monotone, which is why only one of the two can
  be used this way.

Both are absorbing: `successors` returns `[]`, so a plan cannot wander past the end.

## Planning

The shape to know before pointing a planner at this: branching factor 6–22, solution depths
2–7, and about 0.03–0.05 s per expansion (each one is a full hydraulic and transport solve).
Caching on the closed set does most of the work — the same configuration is reached by many
orderings.

The heuristic to write is not "minimise contamination". That fails on index 8, as above. It
has to trade contamination against service, and the useful signal is that service is
monotone while contamination is not: you can bound service from the closed set alone, but
containment you have to simulate for.

## Housekeeping

`EpanetSimulator.run_sim()` writes `temp.inp`, `temp.bin` and `temp.rpt` into the *current
working directory* unless told otherwise, and expansion runs it hundreds of times. This
environment routes all of it into a temp directory, and `close()` removes it. There is a
test that asserts nothing is left behind in the working directory.

## Attribution

Built on [WNTR](https://github.com/USEPA/WNTR), the US EPA's Python interface to EPANET
(BSD). The `Net1`, `Net2` and `Net3` benchmark networks ship with it.

## Files

| Path | What |
|---|---|
| [`environment.py`](../../planiverse/problems/real_world_problems/water_distribution/environment.py) | `WaterNetworkEnv`, `WaterNetworkState`, `WaterNetworkAction`, `rank_sources` |
| [`tests/test_water_distribution.py`](../../tests/test_water_distribution.py) | Tests, including the determinism guarantees |
