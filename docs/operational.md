# Operational environments as planning problems

The operational simulators (an epidemic, a traffic grid, an airspace, a reservoir system, a water
network, a power grid, a crop, a flooding city, a Micropolis city and a factory) do not come with
a goal state. What they ask for is a property of the whole trajectory: a hospital load never
over capacity, separation never lost, a river never short, and a cost accumulated over the
horizon kept small. A search planner asks for an initial state, a successor function and a goal
test. The reduction that bridges the two has three parts, and every operational environment
uses all three.

**A constraint that must hold throughout becomes a dead end.** `is_terminal` is true of the
first state that breaks it. A planner never expands a dead end, so any plan it returns respects
the constraint at every step by construction. The grid's blackout, the epidemic's beds, the
airspace's separation standard, the reservoir's river and city floors and the traffic grid's
horizon are all of this kind.

**An accumulated quantity becomes part of the state, and the goal bounds it.** Deaths so far,
vehicle-seconds so far, the farm's shortfall so far and the sum of exit times are carried in
the state like any other reading. The goal is then "the horizon reached with the accumulator
under the target", which is an ordinary final-state test. This is bounded-cost planning
(Stern, Puzis and Felner, "Potential search: a bounded-cost search algorithm", ICAPS 2011):
the optimisation is turned into a decision problem by fixing the bound, and the margin under
the bound is the quality measure. The bound comes from reference policies: the generator runs
a few scripted policies (fixed cycles, steady releases, a rule-based controller, the town left
open), takes the best feasible one as the target, and keeps that policy's plan as the witness.
Every instance is therefore solvable by construction, and the planner is asked to do at least
as well as the best rule of thumb.

**The horizon is time in the state.** "By week twelve" is then a goal condition like any other,
and "the horizon passed without the goal" is a dead end, since nothing can improve afterwards.

In PDDL's terms these are the trajectory constraints of PDDL3 (Gerevini and Long, "Plan
constraints and preferences in PDDL3", 2005: `always`, `sometime`, `within`, `at-most-once`),
and the reduction is the compilation of temporally extended goals into final-state goals by
augmenting the state (Baier and McIlraith, AAAI 2006; Torres and Baier, IJCAI 2015). Here it
is done against a simulator rather than a compiled model: the state carries the accumulators
and the violation flags, the simulator supplies the dynamics, and the planner sees only
`successors`, `is_goal` and `is_terminal` (Francès, Ramírez, Lipovetzky and Geffner, "Purely
declarative action descriptions are overrated: classical planning with simulators", IJCAI
2017).

Two properties follow from the reduction:

- **A plan is open-loop for one seeded scenario.** Covasim, SUMO and the crop model are
  stochastic; each environment pins its seed, so a plan is a plan for that morning, that
  outbreak, that season.
- **The decisions are per-period macro decisions**: a level for the week, a switch every thirty
  seconds, an instruction a minute, a release for the month, a zone a year. They keep the
  number of decisions in a horizon small, and with it the depth a planner has to search.

Each environment's page under [`environments/`](environments/) gives its own dead ends,
accumulators, goal and the reference policies its targets come from.
