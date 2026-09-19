# Epidemic

An outbreak on Covasim. A town of five to ten thousand people, a few dozen of them infected on
day one, and a decision a week for ten to fourteen weeks: leave the town open, ask for
distancing, or lock it down. The hospitals have a number of beds, the town has a budget of
disruption it will bear, and the goal is the horizon reached with the deaths under the target,
the beds never all taken, and the budget respected. Who meets whom, who falls ill, who needs a
bed and who dies are the model's, and the effect of a week's level on the weeks after is only
known by running it, which keeps the environment out of PDDL.

[Covasim](https://github.com/institutefordiseasemodeling/covasim) (MIT; Kerr et al. 2021,
https://doi.org/10.1371/journal.pcbi.1009149) is the Institute for Disease Modeling's
agent-based model of COVID-19: a synthetic population with household, school, work and community
contact layers, an infection with a viral-load course, and a severity ladder from symptoms to
the hospital, intensive care and death. It is a dependency installed from PyPI; nothing of it
is included here.

- **Import:** `from planiverse.environments.epidemic.environment import EpidemicEnv`
- **Source:** [`planiverse/environments/epidemic/environment.py`](../../planiverse/environments/epidemic/environment.py)
- **Instances:** 100 outbreaks, indices `0` to `99`, all drawn by the generator at recorded seeds
- **Generator:** `generate_instance(seed, population=None, weeks=None, ...)`; see [Generating outbreaks](#generating-outbreaks)
- **Dependency:** `covasim`

## How a trajectory problem becomes a goal

An epidemic is a trajectory problem: what matters is what happens every week, not a final
configuration. The environment turns it into an initial-to-goal problem with the reduction the
grid and flood environments use, and [NOTES.md](../../NOTES.md) sets it out in full:

| The trajectory asks for | The environment does |
|---|---|
| the hospital load never over capacity | `is_terminal`: the first week that breaks it is a dead end |
| disruption within a budget | points accumulate in the state; a week the budget cannot pay for is not offered |
| few deaths by the end | deaths accumulate in the state; the goal is the horizon with deaths under the target |
| the horizon itself | the week is in the state |

The target is what the best of a few scripted schedules achieves, so every instance is
solvable by construction and the planner is asked to do at least as well.

## Quickstart

```python
from planiverse.environments.epidemic.environment import EpidemicEnv, EpidemicAction

env = EpidemicEnv()
env.set_index(0)
state, info = env.reset()          # info: outbreak, weeks, capacity, budget, target, generated
print(state)                       # week 0 of 12: 0 infectious, 0 in hospital (peak 0 of ...), ...

state, saved = env.step("lockdown")
for action, child in env.successors(state):
    print(action, child.load, child.deaths)
```

## The rules

1. The town is Covasim's default population at the instance's size with its seed infections,
   run from its seed, so the same schedule always gives the same epidemic.
2. A week is one decision: `open`, `distancing` or `lockdown`, which scale the model's
   transmission rate by 1.0, 0.6 or 0.3 and cost 0, 1 or 3 points of disruption. A level the
   remaining budget cannot pay for is not offered.
3. The hospital load is the people severely or critically ill on any day. A week in which it
   exceeds the beds is a dead end.
4. The goal is the last week reached with the deaths under the target, the beds never exceeded
   and the points within the budget. The horizon reached with more deaths is a dead end.

## State

`EpidemicState` holds the schedule so far and the readings at the end of it: the day, the
infectious, the hospital load now and at its peak, the deaths, the infections and the points
spent. Identity is the schedule: the model replays exactly from its seed, and a copy of a
running simulation does not carry the random stream with it, so every expansion replays the
epidemic from day zero. Twelve weeks of ten thousand people take a third of a second.

`literals` names each week's level, the week, the deaths and points exactly, and the loads
and infectious in bands:

```
chosen(0, open)
chosen(1, lockdown)
week(2)
deaths(1)
points(3)
in_hospital(20)
peak(20)
infectious(150)
```

## Actions

`open`, `distancing` and `lockdown`, each costing 1; `get_actions(state)` leaves out what the
budget cannot pay for.

## Outbreaks

The hundred outbreaks are the generator's own draws, embedded in the module as the plain data
`set_instance` takes (`seed`, `population`, `infected`, `weeks`, `capacity`, `budget`, `target`)
with the seed each came from beside it, and the schedule each was accepted on is in
`tests/data/epidemic_solutions.json`. Beds are one to two and a half per cent of the
population, budgets six to twenty-one points, horizons ten to fourteen weeks.

## Generating outbreaks

`generate_instance` draws an outbreak, selects it, and returns it as the dict `set_instance`
takes back:

```python
env = EpidemicEnv()
outbreak = env.generate_instance(seed=7, weeks=12)
print(env.witness, env.witness_expansions)     # the schedule it was accepted on, and the schedules measured
```

| Option | Default | What it does |
|---|---|---|
| `population` | 5,000, 8,000 or 10,000 | the town |
| `weeks` | 10, 12 or 14 | the horizon |
| `attempts` | 40 | draws before giving up with `GenerationError` |

A draw is measured by four scripted schedules: the town left open, distancing throughout (when
the budget allows), the budget spent on lockdown first, and a lockdown after two open weeks.
The target is the fewest deaths any schedule that keeps under the beds and within the budget
reaches; a draw is thrown back when none does, or when leaving the town open already meets it.
The method is generate-and-test with the target set off a reference policy (Togelius,
Yannakakis, Stanley and Browne, 2011, https://doi.org/10.1109/TCIAIG.2011.2148116).

## Files

| File | Contents |
|---|---|
| [`environment.py`](../../planiverse/environments/epidemic/environment.py) | `EpidemicAction`, `EpidemicState`, `replay`, `EpidemicEnv`, `OUTBREAKS` |
| [`tests/test_operational_four.py`](../../tests/test_operational_four.py) | Tests |
| [`tests/data/epidemic_solutions.json`](../../tests/data/epidemic_solutions.json) | The schedule each outbreak was accepted on |
