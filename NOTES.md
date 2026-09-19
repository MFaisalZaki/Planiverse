# Notes

Working notes on the last round of changes, for checking. Each section says what was done, what
it rests on, and what to look at. The environment pages under `docs/environments/` carry the
detail; this file carries the reasoning and the loose ends.

## 1. How a trajectory problem becomes a planning problem

The operational simulators (an epidemic, a traffic grid, an airspace, a reservoir system, and
before them the water network, the grid, the crop, the flood city and the Micropolis city) do
not come with a goal state. What they ask for is a property of the whole trajectory: a hospital
load never over capacity, separation never lost, a river never short, and a cost accumulated
over the horizon kept small. A search planner asks for an initial state, a successor function
and a goal test. The reduction that bridges the two has three parts, and every operational
environment in the tree uses all three.

**A constraint that must hold throughout becomes a dead end.** `is_terminal` is true of the
first state that breaks it. A planner never expands a dead end, so any plan it returns respects
the constraint at every step by construction, and the constraint costs it nothing but the
states it need not look at. The grid's blackout, the epidemic's beds, the airspace's separation
standard, the reservoir's river and city floors and the traffic grid's horizon are all of this
kind.

**An accumulated quantity becomes part of the state, and the goal bounds it.** Deaths so far,
vehicle-seconds so far, the farm's shortfall so far and the sum of exit times are carried in
the state like any other reading. The goal is then "the horizon reached with the accumulator
under the target", which is an ordinary final-state test. This is bounded-cost planning
(Stern, Puzis and Felner, "Potential search: a bounded-cost search algorithm", ICAPS 2011):
the optimisation is turned into a decision problem by fixing the bound, and the margin under
the bound is left as the quality measure. The bound has to come from somewhere, and it comes
from reference policies: the generator runs a few scripted policies (fixed cycles, steady
releases, a rule-based controller, the town left open), takes the best feasible one as the
target, and keeps that policy's plan as the witness. Every instance is therefore solvable by
construction, and the planner is asked to do at least as well as the best rule of thumb.

**The horizon is time in the state.** "By week twelve" is then a goal condition like any other,
and "the horizon passed without the goal" is a dead end, since nothing can improve afterwards.

In PDDL's terms these are the trajectory constraints of PDDL3 (Gerevini and Long, "Plan
constraints and preferences in PDDL3", 2005: `always`, `sometime`, `within`, `at-most-once`),
and the reduction is the compilation of temporally extended goals into final-state goals by
augmenting the state (Baier and McIlraith, AAAI 2006; Torres and Baier, IJCAI 2015). The
difference is that we do it against a simulator rather than a compiled model: the state
carries the accumulators and the violation flags, the simulator supplies the dynamics, and the
planner sees only `successors`, `is_goal` and `is_terminal`. That is the point of the repository
(Francès, Ramírez, Lipovetzky and Geffner, "Purely declarative action descriptions are
overrated: classical planning with simulators", IJCAI 2017).

Two things the reduction does not hide:

- The plan is open-loop for one seeded scenario. Covasim, SUMO and the crop model are
  stochastic; we pin their seeds, so the plan is a plan for that morning, that outbreak, that
  season. Whether it holds on other seeds is a separate measurement, and a useful one to add
  to the benchmark (validate each plan on ten unseen seeds and report the fraction that still
  reach the goal).
- The horizon has to be short in decisions, so the actions are per-period macro decisions: a
  level for the week, a switch every thirty seconds, an instruction a minute, a release for
  the month. That is what keeps the branching and depth inside what width-based search can
  do; it is the same choice Micropolis makes with a zone a year.

The online planners in the tree (Rollout IW and π-IW) see the same environments as a reward
stream and plan a lookahead per decision, which is the other natural framing of a trajectory
problem; both run in the benchmark without any change.

## 2. The four environments, and how each maps

| Environment | Decision, period | Dead ends (`is_terminal`) | Accumulators bounded at the goal | Final-state conditions | Target from |
|---|---|---|---|---|---|
| `epidemic` (Covasim) | open, distancing or lockdown; a week | hospital load over the beds; budget exceeded; horizon with deaths over target | deaths ≤ target; disruption points ≤ budget | the last week reached | four scripted schedules |
| `traffic` (SUMO) | switch a junction, row, column, all, or hold; 30 s | horizon with vehicles left; everyone through over the target | vehicle-seconds ≤ target | every vehicle arrived | fixed cycles of 1, 2, 3 decisions, and holding, with a tenth in hand |
| `airspace` (BlueSky) | turn left or right, direct-to, hold; a minute | loss of separation; horizon with aircraft in; everyone out over the target | sum of exit times ≤ target | every aircraft out | a rule-based right- or left-turn controller, with a tenth in hand |
| `reservoir` (pywr) | release 2 to 10 and the farm on full or half; a month | river below its minimum; city below 90 %; year ending short | farm shortfall ≤ target | both reservoirs above their reserves | steady, summer-rationed and seasonal release policies; a draw a steady release solves is thrown back |

State identity: the epidemic, the traffic grid and the airspace are paths (the simulators
replay exactly from their seeds but their running state cannot be copied: Covasim's random
stream lives outside the object, SUMO's saved state re-reads the route file, BlueSky's state is
module globals), so every expansion replays from the start, at 0.1 to 0.5 s per successor. The
reservoir is a value (month, volumes, shortfalls), since pywr is memoryless beyond its storages.

Per-expansion costs measured here: epidemic 0.1 to 0.5 s, traffic 0.4 to 0.7 s, airspace 1.4 s
at the start rising to about 4 s at depth seven, reservoir a few milliseconds. The airspace is
the slow one, and it is why its aircraft are three or four and its horizon twelve to fifteen
minutes.

## 3. Things to check

- **The shot family.** Iterated BFWS over the bundled instances: artillery, all hundred solved
  in 1.1 s, median six expansions, plans of two to five shots; slingshot (every fourth level),
  all solved, median two expansions, nineteen of twenty-five in three or fewer; billiards
  (every tenth table), all solved in at most four expansions. Artillery has too little
  coupling to be worth keeping. Slingshot and billiards have coupling their instances do not
  demand; a regeneration with longer minimum plans and a shot budget no larger than the
  targets, then the same survey, would say whether they stay. Nothing has been removed.
- **Reservoir is the most PDDL-like of the four**: a mass balance with priorities is a small
  linear system, and a numeric planner could model it. It is here for what it is, a real
  water-industry simulator with a year's worth of coupled decisions; keep it or not on that.
- **Two candidates were left out.** CityLearn downloads its datasets from GitHub at run time,
  which the benchmark's rule against run-time downloads excludes. SimFire pins Python 3.9 and
  does not install on 3.11.
- **BlueSky's install.** Its published dependency list names `zmq`, a placeholder that no longer
  builds, so `pip install bluesky-simulator` fails; `scripts/install_bluesky.sh` installs the
  real dependencies and the simulator without its list. It is not in `pyproject.toml` for that
  reason. Its two data dependencies are OpenAP (LGPL-3.0) and BlueSky's navigation data
  (GPL-3.0), both compatible with the repository's licence and neither included.
- **Covasim moved numba.** Installing Covasim upgraded numba and llvmlite past what pooltool
  pins; pinning `llvmlite<0.46` and `numba<0.62` satisfies both. `pyproject.toml` does not pin
  them, and a fresh install may need the same pins if pip picks the newer numba.
- **Robustness across seeds** is not measured yet: see section 1.
- **Tycoon games.** OpenTTD is the candidate (GPL-2, free assets, headless, deterministic,
  scriptable over its admin port and Game Scripts). The first step is a feasibility probe: build
  it headless, generate a small map from a seed, drive one company action, and time a simulated
  year. OpenRCT2 and CorsixTH need the original games' files and are out on the ROM rule;
  FreeRCT is too early.

## 4. Licences, in one place

| Component | Licence | Included? |
|---|---|---|
| factory-sim | MIT | no; built from source at a pinned commit |
| MicropolisCore | GPL-3.0-or-later with EA's section 7 terms; "Micropolis" name licence | no; built from source |
| pymunk, Chipmunk2D | MIT | no; PyPI |
| pooltool | Apache-2.0 | no; PyPI |
| Covasim | MIT | no; PyPI |
| SUMO | EPL-2.0 or GPL-2.0-or-later | no; PyPI (`eclipse-sumo`, `libsumo`) |
| BlueSky; OpenAP; BlueSky navigation data | MIT; LGPL-3.0; GPL-3.0 | no; PyPI by script |
| pywr | GPL-3.0-or-later | no; PyPI |

All compatible with the repository's GPL-3.0-or-later, and none conveyed by it.
