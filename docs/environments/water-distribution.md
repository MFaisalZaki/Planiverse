# Water distribution

This environment models containing contamination in a drinking-water network. A contaminant enters
at one junction, and every node downstream of it drinks a share. Which nodes those are depends on
where the water is flowing, which is the solution of a nonlinear system over the whole network.
The operator's move is to close pipes, and closing a pipe reroutes the flow, so it changes the
answer everywhere at once.

The goal is to contain the contamination without cutting off the customers, and that tension is
the problem: every closed pipe contains a little more and costs a little more service.

- **Class:** `WaterNetworkEnv`
- **Import:** `from planiverse.environments.water_network.environment import WaterNetworkEnv`
- **Source:** [`environment.py`](../../planiverse/environments/water_network/environment.py)
- **Instances:** 9 scenarios, indices `0`–`8`
- **Dependencies:** `wntr`. The benchmark networks ship inside it, so there is nothing to supply.

This is not a PDDL domain, and not merely because PDDL would be verbose here. Two properties, both
measured on the shipped networks, rule it out.

First, effects are global. On `Net3`, closing a single pipe changes the pressure at 94 of the 97
nodes, so the effect of `close pipe 123` is not a set of facts to add and delete but an
instruction to re-solve the hydraulics and the transport and see what happens.

Second, effects are not monotone. On `Net1` with the source at node 12, closing pipe `110` makes
the contamination *worse*, because it pushes flow down a path that reaches more customers. A
delete-list cannot express that, and neither can a planner that assumes closing more pipes
contains more.

A third property is softer: which pipes matter is not readable off the topology. On `Net3` the
source has four pipes on it, two of which carry essentially all the contamination and two of which
do nothing at all, and nothing structural distinguishes them.

## Quickstart

```python
from planiverse.environments.water_network.environment import WaterNetworkEnv

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

env.close()                           # removes the simulator's scratch directory
```

## Scenarios

`fix_index(i)` picks a network together with the junction the contaminant enters at. There are
nine scenarios, ordered by how deep a solution is:

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
| 8 | Net1 | 11 | 80.1% | 7 |

`solved_at` is a measurement rather than an estimate: it is the shallowest depth at which a
solution was actually found. We record it because an instance whose goal nobody has reached is a
poor benchmark entry, since a planner cannot tell "unreachable" from "not found yet". Every
scenario here has been solved.

We tried `Net2` and dropped it for exactly that reason. Its sources sit on trunk mains that cannot
be rerouted around, and a width-16 beam search to depth 8 with every one of its 40 pipes available
plateaued at 36% contamination and stopped improving. Rather than ship an instance that is
probably unsolvable, we left it out.

The sources are not arbitrary either. `rank_sources(network_file)` runs every junction of a
network as the source and reports how much of the delivered water ends up contaminated, and we
chose the scenarios off that ranking. The function is kept so the choice can be re-derived rather
than taken on trust.

Index 8 is the interesting one. `Net1` with the source at node 11 contaminates 80% of everything
delivered, and it is solvable: we proved that exhaustively over the full powerset of its 12 pipes,
finding exactly three feasible closures at depth 7, one of which reaches zero contamination at
100% service. A width-16 beam search to depth 8 does not find it. It is also where the obvious
heuristic fails, since ranking states by contamination alone marches straight into "close
everything", giving contamination zero, service zero, and the goal failed.

## Determinism

The design rests on determinism, so we tested it rather than assuming it: the same set of closed
pipes simulates bit-identically every time, at a maximum absolute pressure difference of `0.0`
across repeated runs. Three consequences follow.

- **The closed set is a sufficient statistic for the state.** Two states with the same pipes shut
  are the same state, so `__eq__` and `__hash__` are on the closed set alone and search can close
  over them.
- **Results are memoised** on the closed set.
- **`simulate` re-runs from scratch**, so it is an independent check on `successors`.

`depth` is deliberately not part of state identity. With a step counter in there no successor
could ever equal its parent, and the self-loop filter would be dead code.

## State

| Field | Meaning |
|---|---|
| `closed` | frozenset of closed pipe names; the whole state |
| `contaminated` | share of all delivered water that came from the source |
| `service` | share of expected demand actually delivered |
| `pressure_deficit` | mean shortfall below the required pressure |
| `depth` | plan length; not part of identity |

The literals are:

```
closed(PIPE)
contaminated(N)      # bucketed into twentieths
service(N)
```

The two metrics are bucketed because a planner keyed on raw floats would treat every state as
novel. The closed set is what identifies the state, and the buckets are there for width-based
methods that measure novelty over atoms.

Hydraulics are pressure-driven, `demand_model = "PDD"`, rather than the demand-driven default, and
that choice matters: under the demand-driven model a node takes its full demand no matter how
little pressure is left, so closing pipes would look free and there would be no trade-off to plan
against. Note that a junction may have a *negative* demand, which is an injection into the network
rather than a customer drawing from it. Counted as written it drags the network's total expected
demand below zero and makes the service ratio meaningless, so both sides of the ratio are clipped
at zero.

## Actions

`WaterNetworkAction(pipe)` closes one pipe. That is the entire action set, and the entire operator
interface, with a cost of 1 each.

The candidate pipes default to those within `radius=2` hops of the source, because the full pipe
list is a branching factor of 117 on `Net3`, nearly all of it nowhere near the incident.
`WaterNetworkEnv(radius=None)` offers every pipe, which is the honest setting and a slow one. Note
that the `Net2` investigation above used the unrestricted setting, so the restriction is not what
made those instances unsolvable.

## Goal and terminal

- **Goal** (`is_goal`): contaminated ≤ 2% and service ≥ 80%. Both halves are needed: closing every
  pipe at the source contains perfectly and is not a solution.
- **Terminal** (`is_terminal`): service < 50%. Sound because service is monotone in the closed
  set: a network cannot deliver more water with more pipes shut, so a collapsed state can never
  recover. Contamination is not monotone, which is why only one of the two can be used this way.

Both are absorbing (i.e., no action leads out of them), so `successors` returns `[]`.

## Shape of the search

The shape to know before pointing a planner at this is a branching factor between 6 and 22,
solution depths between 2 and 7, and about 0.03 to 0.05 s per expansion, each one a full hydraulic
and transport solve. Caching on the closed set does most of the work, because the same
configuration is reached by many orderings.

The heuristic to write is not "minimise contamination", which fails on index 8 as described above.
It has to trade contamination against service, and the useful signal is that service is monotone
while contamination is not: service can be bounded from the closed set alone, whereas containment
has to be simulated for.

## Rendering

`str(state)` describes the network in a few lines. `render_trace` typesets it into a contact
sheet, PDF, GIF or directory of PNGs:

```python
from planiverse.planners.width import IteratedBFWS
from planiverse.benchmark import measures

env = WaterNetworkEnv()
env.fix_index(0)
env.reset()

result = IteratedBFWS(max_width=1000, progress=measures.water_network).solve(env)
trace = env.simulate(result.plan)

env.render_trace(trace, "water_network.gif")                                 # animated
env.render_trace(trace, "water_network.png", actions=result.plan, env=env)   # contact sheet
```

`wntr` imports `pkg_resources`, which setuptools removed in version 81, so pin `setuptools<81` in
your environment. See [docs/rendering.md](../rendering.md) for the other output formats.

## Housekeeping

`EpanetSimulator.run_sim()` writes `temp.inp`, `temp.bin` and `temp.rpt` into the current working
directory unless told otherwise, and expansion runs it hundreds of times. This environment routes
all of it into a temp directory, and `close()` removes it.

## Attribution

Built on [WNTR](https://github.com/USEPA/WNTR), the US EPA's Python interface to EPANET (BSD). The
`Net1`, `Net2` and `Net3` benchmark networks ship with it.

## Files

| Path | What |
|---|---|
| [`environment.py`](../../planiverse/environments/water_network/environment.py) | `WaterNetworkEnv`, `WaterNetworkState`, `WaterNetworkAction`, `rank_sources` |
| [`tests/test_water_distribution.py`](../../tests/test_water_distribution.py) | Tests |
