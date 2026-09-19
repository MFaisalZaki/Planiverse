# Network attack

This environment models penetration testing as a planning problem. The attacker starts outside a
segmented network and must compromise every sensitive host by scanning, exploiting services and
escalating privileges. We wrap [NASim](https://github.com/MFaisalZaki/NetworkAttackSimulator) and
make it deterministic, so a planner can reason about it rather than sampling it.

- **Class:** `EnvNASim`
- **Import:** `from planiverse.environments.network_attack.network_attack import EnvNASim`
- **Source:** [`network_attack.py`](../../planiverse/environments/network_attack/network_attack.py)
- **Instances:** 18 NASim benchmark scenarios, indices `0` to `17`
- **Generator:** `generate_instance(seed, hosts=5, services=3, **options)`; see [Generating networks](#generating-networks)
- **Dependencies:** `nasim`, the `MFaisalZaki/NetworkAttackSimulator` fork pinned in
  `pyproject.toml`. Upstream `nasim` will not work: `generative_step` and the patched internals
  (`_perform_wiretapping`, `has_required_remote_permission`) come from the fork.

Related upstream work is [PenGym](https://github.com/cyb3rlab/PenGym).

## A solved instance

![BFWS's plan for network_attack instance 0](../renders/network_attack.png)

BFWS's plan for instance `0` (`tiny`): 6 actions, shown as a contact sheet. Each frame is
the state's own text, which lists the hosts discovered so far and says of each whether it is
reachable, compromised or rooted, so the frames show the attack advancing host by host, and
the captions carry the actions: exploit, subnet scan, exploit, privilege escalation, exploit,
privilege escalation, and the goal.

```python
from planiverse.environments.network_attack.network_attack import EnvNASim
from planiverse.benchmark import measures
from planiverse.planners.width import IteratedBFWS

env = EnvNASim()
env.set_index(0)
env.reset()

result = IteratedBFWS(max_width=1000, progress=measures.network_attack).solve(env)
trace = env.simulate(result.plan)
env.render_trace(trace, "network_attack.png", actions=result.plan, env=env)
```

See [docs/rendering.md](../rendering.md) for the other output formats.

## Quickstart

```python
from planiverse.environments.network_attack.network_attack import EnvNASim

env = EnvNASim()
env.set_index(0)             # 'tiny'
state, info = env.reset()

for action, successor in env.successors(state):
    print(action, env.is_goal(successor))

trace = env.simulate([a for a, _ in env.successors(state)][:3])
```

You can name a scenario directly instead of calling `set_index`, or load your own scenario file:

```python
env = EnvNASim(scenario_name="small-honeypot")
state, _ = env.reset()

env = EnvNASim(scenario_yaml="/path/to/scenario.yaml")   # loaded via nasim.load
state, _ = env.reset()
```

## Scenarios

`set_index(i)` maps to NASim's benchmark scenarios, loaded via `nasim.make_benchmark(name)` with
`seed=0`:

| Index | Scenario | | Index | Scenario |
|---|---|---|---|---|
| 0 | `tiny` | | 9 | `tiny-gen` |
| 1 | `tiny-hard` | | 10 | `tiny-gen-rgoal` |
| 2 | `tiny-small` | | 11 | `small-gen` |
| 3 | `small` | | 12 | `small-gen-rgoal` |
| 4 | `small-honeypot` | | 13 | `medium-gen` |
| 5 | `small-linear` | | 14 | `large-gen` |
| 6 | `medium` | | 15 | `huge-gen` |
| 7 | `medium-single-site` | | 16 | `pocp-1-gen` |
| 8 | `medium-multi-site` | | 17 | `pocp-2-gen` |

The `-gen` scenarios are procedurally generated, and the rest are hand-authored. `reset()`
raises unless an instance has been selected with `set_index`, `set_instance` or
`generate_instance`, or named in the constructor.

## Generating networks

`generate_instance` hands the drawing to NASim's own scenario generator, selects the result,
and returns it as a dict that `set_instance` accepts back:

```python
env = EnvNASim()
network = env.generate_instance(seed=7, hosts=8, services=4, num_os=3)
# {'hosts': 8, 'services': 4, 'seed': 7, 'num_os': 3}
state, info = env.reset()
```

`hosts` and `services` fix the size; anything else in `options` goes straight to
`nasim.scenarios.generator.ScenarioGenerator.generate` (`num_os`, `num_processes`,
`num_exploits`, `num_privescs`, `r_sensitive`, `r_user`, `uniform`, `alpha_H`, `alpha_V`,
`lambda_V`, `base_host_value`, `host_discovery_value`, `step_limit`, and the rest). The
seed goes into the dict with them, so the same dict builds the same network every time,
including on replay through `simulate`. With `solvable` (the default) the draw is searched
breadth-first for up to `search_limit` (2000) expansions and kept only if a plan was found:
`witness` holds it and the instance records `solved_at`. NASim's generated networks keep
their sensitive hosts reachable, so the check is on this environment's own transition
function rather than on the draw; a network too large to decide within the budget is redrawn
from the next seed, up to `attempts` (20) times.

An instance is one of three dicts: `{"scenario": name}` (what `set_index` selects),
`{"yaml": path}` (a scenario file of your own), or the generated shape above.

## Determinism

NASim is a stochastic reinforcement-learning environment, in which exploits succeed with
probability `action.prob`. That is fatal for deterministic search, so this module replaces
`Network.perform_action` at import time with a copy that keeps every precondition check and
removes the random failure roll:

```python
if action.is_exploit() and host_compromised:
    # host already compromised so exploits don't fail due to randomness
    pass
# elif np.random.rand() > action.prob:
#     return next_state, ActionResult(False, 0.0, undefined_error=True)
```

Preconditions still apply, and their failures are deterministic outcomes rather than dice rolls:
unreachable or undiscovered targets return a connection error, remote actions without permission
return a permission error, exploits against a service the firewall blocks return a connection
error, and privilege escalation on an uncompromised host returns a connection error.

The same patch adds `__hash__` to every action class (`Exploit`, `PrivilegeEscalation`,
`ServiceScan`, `OSScan`, `SubnetScan`, `ProcessScan`, `NoOp`), hashing on `str(self)` so actions
can live in sets.

Both patches are applied by `setattr` on import of this module and are global to the process, so
importing this environment changes NASim's behaviour for everything else in the same interpreter.
That is worth knowing if you also use NASim directly.

## State

`NASimState` subclasses NASim's `State`, keeping its `tensor` and `host_num_map`, and adds
literals:

| Literal | Meaning |
|---|---|
| `at(x,y,val)` | Cell `(x, y)` of the state tensor holds `val`; one per cell |
| `compromised_host_N` | Sensitive host `N` is owned at `AccessLevel.ROOT` |

The `at(...)` literals are a direct, lossless transcription of the NASim tensor, with one literal
per cell, covering every host's discovery, reachability and compromise flags, OS, services and
processes. Nothing is abstracted away, which makes the literal set large but exact, so a planner's visited set is precise. The `compromised_host_N` literals are the goal-relevant summary layered on
top.

## Actions

The action set comes from NASim's own `env.action_space.actions`, which enumerates every ground
action for the scenario.

| Action | Effect |
|---|---|
| `ServiceScan` | Reveal services on a host |
| `OSScan` | Reveal the host OS |
| `SubnetScan` | Discover hosts in a subnet; requires a compromised foothold |
| `ProcessScan` | Reveal running processes |
| `Exploit` | Compromise a host through a service; grants access, subject to firewall rules |
| `PrivilegeEscalation` | Raise access on a compromised host, given the right OS or process |
| `NoOp` | Do nothing |

`successors` applies each action through `env.generative_step(state, action)` and drops any action
that leaves the state unchanged, so failed preconditions produce no successor. That is what keeps
the branching factor manageable, since only actions that actually accomplish something are
offered.

## Goal and terminal

- **Goal** (`is_goal`): `env.network.all_sensitive_hosts_compromised(state)`, every sensitive host
  owned at root.
- **Terminal** (`is_terminal`): always `False`. The source comment flags this as a known gap: dead
  ends exist in this environment but are not detected, so a planner must bound its own search.

## Rendering

`str(state)` lists the hosts discovered so far, one line each, with whether the host is
reachable, compromised or rooted and whether it is sensitive. NASim's own `State.__str__`
prints the address space and nothing that changes, so `NASimState` overrides it; the full
state, every cell of the tensor, is in `state.literals`. See
[docs/rendering.md](../rendering.md) for the other output formats.

## Notes and limits

- **`reset()` rebuilds the environment every call**, re-running `make_benchmark`. It is not cheap;
  do not call it inside a loop.
- **`successors` needs `reset()` first.** It reads `self.actionslist`, which is `None` until
  reset.
- **`is_terminal` is always `False`**; see [Goal and terminal](#goal-and-terminal).

## Files

| Path | What |
|---|---|
| [`network_attack.py`](../../planiverse/environments/network_attack/network_attack.py) | The `perform_action` patch, `NASimState`, `EnvNASim` |
| [`tests/test_network_attack.py`](../../tests/test_network_attack.py) | Tests |
