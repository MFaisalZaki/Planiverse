# Candidate planners for simulators: a literature survey

What else the library could implement to plan against a black-box simulator, drawn from the
planning, search, games and control literature, restricted to methods that **need no
training phase**: nothing is fitted on one set of runs and assessed on another, and no
learned model, policy, value function or feature extractor is required before the planner
can be pointed at an instance. That rules out MCTS and its descendants (see
[Excluded](#excluded-and-why)) and anything reward-driven: this is a planning library.

**Status.** The candidates below have since been implemented; see
[more-planners.md](more-planners.md) for the classes, what each one needs, the choices made
where a paper could not be read in full, and the smoke results. Five are not in the
library: 2BFS (1), prioritised IW (2), the rollout algorithm (23) and Fractal Monte Carlo
(25) are defined by an accumulated reward, which a planning suite does not have, and
time-bounded A\* (18) needs reversible actions.

Every method below needs only what the [environment contract](../../README.md#the-environment-interface)
already gives: `successors`, `literals`, `is_goal`, `is_terminal`, and for some `simulate` or
`step`. Where a method classically uses a goal count or a reward, the library's `progress`
callback stands in, as it does for SIW and BFWS today. States here are ordinary objects that
can be expanded again at any time, so "restore the simulator to this state", which several
methods below rely on, is free.

## What is already here

| Family | Classes | Reference |
|---|---|---|
| Width-based | `IW`, `SIW`, `BFWS`, `DualBFWS` | Lipovetzky & Geffner 2012, 2017 |
| Sampling | `FSXPlanner` | Wissner-Gross & Freer 2013 |

Rollout IW, π-IW and an MCTS planner were in the library when this survey was written and
have since been removed: the first two choose actions by a discounted return and the second
learns a policy as it plans, and MCTS wants a reward; none of that is planning.
| Heuristic search | `TreeSearchPlanner` (best-first with a heuristic and a cost) | Baumgarten's A\* Mario agent |

## Candidates

Grouped by family. Each entry says what the method is, what it needs from an environment, and
what would have to change for a simulator. A **priority** column at the end ranks them.

### A. Width-based methods not yet in the library

1. **2BFS** — Lipovetzky, Ramírez & Geffner, *Classical Planning with Simulators: Results on the
   Atari Video Games*, IJCAI 2015. Two open lists: one ordered by `<novelty, reward>`, the other
   by `<reward, novelty>`, expanded alternately. The paper's best planner on Atari after IW(1),
   and the one that beat UCT. Needs `successors`, `literals`, `progress` (as the reward).
   A short generalisation of `BFWS` to several queues.

2. **Prioritised IW, p-IW(1)** — Shleyfman, Tuisov & Domshlak, *Blind Search for Atari-Like
   Online Planning Revisited*, IJCAI 2016. IW's novelty filter is relaxed with a value test: a
   state whose atoms have all been seen is still kept when it reaches one of them with a higher
   accumulated reward than any earlier state that had that atom, and the earlier state is
   reopened. Best of IW(1), 2BFS and UCT on 37 of 53 Atari games. Needs the same as IW plus
   `progress`. The paper's second idea, treating the root as a bandit over actions, is optional.

3. **BFWS(R): novelty partitioned by a relaxed-plan atom set** — Lipovetzky & Geffner, *Best-First
   Width Search: Exploration and Exploitation in Classical Planning*, AAAI 2017, and Francès,
   Ramírez, Lipovetzky & Geffner, *Purely Declarative Action Descriptions are Overrated:
   Classical Planning with Simulators*, IJCAI 2017. The state of the art among width-based
   planners. `#r` counts how many atoms of a set `R` a path has reached, and novelty is measured
   within `<#g, #r>` partitions. In the simulator paper `R` is not read off a model: it is the
   set of atoms on the paths an IW(1) or IW(2) pre-search found to the goal (or to each goal atom),
   so nothing beyond `successors` and `literals` is needed. Here `R` would be the atoms along
   IW's paths to its best-`progress` states. The IJCAI 2017 paper's whole point is that this
   matches PDDL planners while ignoring action structure, which is exactly Planiverse's setting.

4. **Novelty as a heuristic (quantified novelty)** — Katz, Lipovetzky, Moshkovich & Tuisov,
   *Adapting Novelty to Classical Planning as Heuristic Search*, ICAPS 2017. Novelty is defined
   relative to a heuristic: an atom of `s` is novel if no earlier state with a heuristic value
   at most `h(s)` contained it, and the *number* of such atoms becomes a heuristic to search
   on, alone or combined with `h`. Needs `progress` as `h`. A different ordering for
   `BFWS` rather than a new search.

5. **Count-based novelty with a trimmed open list** — Rosa & Lipovetzky, *Count-Based Novelty
   Exploration in Classical Planning*, ECAI 2024. Replaces the binary seen/unseen record with
   the frequency of each tuple in the search tree, so exploration does not switch off once
   every atom has been seen once, and keeps the open list at a constant size by discarding
   nodes with the worst novelty. Directly addresses the "novelty runs out" problem that
   partitioning only postpones. Needs `literals` only.

6. **Approximate novelty search** — Singh, Lipovetzky, Ramírez & Segovia-Aguas, *Approximate
   Novelty Search*, ICAPS 2021. Novelty tables in Bloom filters and tuple sampling instead of
   exhaustive enumeration, plus an adaptive policy that skips expanding some open nodes. This
   is the cure for the `O(n^k)` cost that makes `NoveltyTable` refuse widths above 2, and it is
   memory-bounded, which matters under the benchmark's address-space cap.

7. **Boundary Extension Features (BEE)** — Teichteil-Königsbuch, Ramírez & Lipovetzky, *Boundary
   Extension Features for Width-Based Planning with Simulators on Continuous-State Domains*,
   IJCAI 2020. Instead of a hand-chosen bucketing of continuous variables, the features novelty
   is measured over are defined online as the observed range of each variable grows.
   k-BFWS-BEE beat PPO on the gym classic-control suite without learning. Relevant to the
   water, power and crop environments, whose `literals` bucket continuous quantities by hand.

8. **Hierarchical IW, HIW(1,1)** — Junyent, Gómez & Jonsson, *Hierarchical Width-Based Planning
   and Learning*, ICAPS 2021. Two IW levels: a high-level IW over abstract features that are
   discovered incrementally from the low-level search's own pruning decisions. The paper also
   adds a learned policy; the planning half does not need it. Feature discovery happens inside
   the run, not in a separate phase, so it stays within scope, but it is the closest item on
   this list to the line.

9. **Dominated action-sequence pruning** (add-on) — Jinnai & Fukunaga, *Learning to Prune
   Dominated Action Sequences in Online Black-Box Planning*, AAAI 2017 (extended 2022). Detects
   pairs of short action sequences that always reach the same state and never expands the
   dominated one, cutting simulator calls in IW and breadth-first search on Atari. The
   statistics are gathered in the same run it prunes, with no separate phase. It plugs into
   every planner here that calls `successors`.

10. **Focused-effect macro-actions** (add-on) — Allen, Katz, Klinger, Konidaris, Riemer & Tesauro,
    *Efficient Black-Box Planning Using Macro-Actions with Focused Effects*, IJCAI 2021. A search
    over the simulator for short action sequences that change few state variables, then planning
    over those macros with a goal-count heuristic, which becomes far more accurate. The macro
    discovery is a pre-search on the instance, not a fit on data. Needs `literals` to measure an
    effect's footprint.

### B. Heuristic search with black-box heuristics and knowledge-free exploration

The library has one best-first search. The satisficing-planning literature has a decade of
work on what to do when the heuristic is weak, which `progress` usually is; almost all of it
needs nothing from the model.

11. **Greedy best-first, weighted A\*, anytime restarting weighted A\*** — Richter, Thayer & Ruml,
    *The Joy of Forgetting: Faster Anytime Search via Restarting*, ICAPS 2010. Generalises
    `TreeSearchPlanner` into a configurable search with a `Budget`, and gives the benchmark an
    anytime baseline. Needs `progress` as the heuristic.

12. **Enforced Hill-Climbing (EHC)** — Hoffmann & Nebel, *The FF Planning System*, JAIR 2001.
    Breadth-first search to the first state that strictly improves the heuristic, commit, repeat.
    SIW with `progress` in place of novelty; with `is_terminal` it gets the same dead-end refusal
    `SIW` has. Also worth having as a novelty-pruned variant.

13. **Type-based exploration and ε-greedy node selection** — Xie, Müller, Holte & Imai,
    *Type-Based Exploration with Multiple Search Queues for Satisficing Planning*, AAAI 2014;
    Valenzano, Schaeffer, Sturtevant & Xie, *A Comparison of Knowledge-Based GBFS Enhancements
    and Knowledge-Free Exploration*, ICAPS 2014. A second queue buckets open nodes by `(h, g)`
    and picks a bucket then a node uniformly at random; ε-greedy simply expands a random open
    node with probability ε. Both are complete, both are a few lines on top of a best-first
    search, and both were designed for exactly the case of a heuristic that misleads.

14. **Diverse Best-First Search (DBFS)** — Imai & Kishimoto, *A Novel Technique for Avoiding
    Plateaus of Greedy Best-First Search in Satisficing Planning*, AAAI 2011. Selects the next
    node to expand probabilistically from the distribution of `g` and `h` values in the open
    list, then runs a short local greedy search from it. A complete, stochastic best-first.

15. **GBFS with local exploration (GBFS-LE)** — Xie, Müller & Holte, *Adding Local Exploration to
    Greedy Best-First Search in Satisficing Planning*, AAAI 2014, and *Understanding and Improving
    Local Exploration for GBFS*, ICAPS 2015. When the global search stalls on a plateau, start a
    local greedy search or a batch of random walks from the best node, and return with anything
    that improves. Needs `progress`; random walks want `step` or a single child from
    `successors`.

16. **Monte-Carlo Random Walks (Arvand)** — Nakhost & Müller, *Monte-Carlo Exploration for
    Deterministic Planning*, IJCAI 2009; *Towards a Second Generation Random Walk Planner*,
    IJCAI 2013; Xie, Nakhost & Müller, *Planning via Random Walk-Driven Local Search*, ICAPS
    2012 (MRW-LTS). From the current state, run many bounded random walks, evaluate only their
    endpoints, jump to the best, and restart when no walk improves for a while. Evaluations
    happen only at endpoints, which suits an expensive `is_goal`. Random walks rather than a
    tree, no bandits, no training; the walk-length adaptation and the action-bias statistics
    (MDA/MHA) are computed inside the run. Against a simulator whose `successors` returns every
    child, a walk step costs a full expansion unless the environment offers `step`.

17. **Beam search and its complete forms** — beam search; BULB, Furcy & Koenig, *Limited
    Discrepancy Beam Search*, IJCAI 2005; beam-stack search, Zhou & Hansen, ICAPS 2005;
    limited discrepancy search, Harvey & Ginsberg, IJCAI 1995; iterative broadening, Ginsberg &
    Harvey, AIJ 1992. Memory-bounded searches that keep a fixed number of the best nodes per
    depth and backtrack over discrepancies when the beam fails. Fits the benchmark's memory cap
    and an ordering by `progress` or novelty; the last two need no heuristic at all.

18. **Real-time heuristic search** — LRTA\*, Korf, *Real-Time Heuristic Search*, AIJ 1990; RTAA\*,
    Koenig & Likhachev, AAMAS 2006; Time-Bounded A\*, Björnsson, Bulitko & Sturtevant, IJCAI
    2009. Bounded lookahead, commit one action, repeat, with a per-step compute bound; LRTA\* and
    RTAA\* also raise the stored heuristic of states they leave. The "learning" is an in-run
    table update, the same kind SIW does when it starts a fresh novelty table per leg, and
    there is no phase before planning.

19. **Feature Space Search (FESS)** — Shoham & Schaeffer, *The FESS Algorithm: A Feature Based
    Approach to Single-Agent Search*, IEEE CoG 2020. Search is organised in a small feature space
    (each feature a `progress`-like measure), moving from the projected initial state toward the
    projected goal, and each feature-space step is realised by domain-space moves suggested by
    "advisors". First program to solve all 90 XSokoban levels. Natural for the box-pushing
    environments (Lolo, Amazing Tater, Puzznic) where several progress measures exist and one
    alone is a trap.

20. **Multi-heuristic alternation** — Helmert, *The Fast Downward Planning System*, JAIR 2006;
    Röger & Helmert, *The More, the Merrier*, ICAPS 2010. One open list per measure
    (`progress`, novelty, `option_count` from FSX), expanded round-robin. A framework rather than
    a planner, but it is how the library's goal-free `option_count` measure would be used
    alongside a goal-directed one.

### C. Sampling, population and swarm planners

Not tree searches, and none trains anything: each optimises an action sequence or a
population of them against the simulator and commits.

21. **Rolling Horizon Evolutionary Algorithm (RHEA)** — Perez, Samothrakis, Lucas & Rohlfshagen,
    *Rolling Horizon Evolution versus Tree Search for Navigation in Single-Player Real-Time
    Games*, GECCO 2013; Gaina, Lucas & Perez-Liebana, *Rolling Horizon Evolution Enhancements in
    General Video Game Playing*, CIG 2017; Gaina, Devlin, Lucas & Perez-Liebana, *Rolling Horizon
    Evolutionary Algorithms for General Video Game Playing*, IEEE ToG 2021. Evolve a population
    of fixed-horizon action sequences with the forward model, execute the first action of the
    best, shift the population and repeat. The standard non-MCTS agent in GVGAI, competitive with
    MCTS across its games. Needs `simulate` and a fitness (`progress`, goal, dead end).

22. **Random shooting and the Cross-Entropy Method (CEM)** — Rubinstein, *The Cross-Entropy
    Method for Combinatorial and Continuous Optimization*, 1999; iCEM, Pinneri et al.,
    *Sample-Efficient Cross-Entropy Method for Real-Time Planning*, CoRL 2020. Sample action
    sequences from a per-step categorical distribution, keep the elite fraction, refit,
    repeat, then execute in receding horizon. The standard model-predictive-control planner;
    training-free when the model is the simulator itself. iCEM adds memory across steps and
    correlated noise. Needs `simulate`.

23. **Rollout with a base policy** — Bertsekas, Tsitsiklis & Wu, *Rollout Algorithms for
    Combinatorial Optimization*, J. Heuristics 1997; Bertsekas, *Rollout, Policy Iteration, and
    Distributed Reinforcement Learning*, 2020. One-step lookahead where each child is valued by
    running a fixed base policy (greedy on `progress`, or random) to the horizon; provably no
    worse than the base policy. With a deterministic base policy there is no sampling at all.
    Cheap, and a strong baseline for the operational environments.

24. **Nested Monte-Carlo Search (NMCS)** — Cazenave, *Nested Monte-Carlo Search*, IJCAI 2009.
    A level-1 search plays out each child randomly and follows the best; level *k* uses level
    *k−1* to choose each move, keeping the best sequence found. No tree, no bandit, no training.
    Its successor NRPA (Rosin, IJCAI 2011) adapts a playout policy by gradient steps inside the
    run; it has no separate training phase but is the same kind of in-run learning π-IW does,
    so it is listed as borderline rather than as a candidate.

25. **Fractal Monte Carlo (FMC)** — Hernández Cerezo & Duran Ballester, *Fractal AI: A Fragile
    Theory of Intelligence*, arXiv 2018; *Solving Atari Games Using Fractals and Entropy*, arXiv
    2018. A swarm of walkers steps randomly; each is then compared with a random peer and clones
    onto it with a probability that rises with the peer's reward and its distance in state space,
    so the swarm spreads over reachable, rewarding states. The root action with most walkers is
    committed. Reported to need under a thousand samples per action on Atari. Despite the name
    it is a swarm, not a tree search, and it is the closest relative of the library's FSX
    walkers, sharing most of their code.

26. **Go-Explore, exploration phase** — Ecoffet, Huizinga, Lehman, Stanley & Clune, *Go-Explore:
    a New Approach for Hard-Exploration Problems*, arXiv 2019; *First Return, Then Explore*,
    Nature 2021. An archive of cells (a coarse projection of the state; `literals` or a
    projection of them), each holding the best trajectory that reached it. Repeatedly: pick a
    cell with weight favouring rarely chosen ones, restore the simulator to it, explore
    randomly for a few steps, add or improve cells. It assumes a deterministic, restorable
    simulator, which every environment here is, and in that setting the trajectory to a goal
    cell is the plan. The second, robustification phase trains a policy by imitation and is out
    of scope; it is only needed when the deployed environment is stochastic. Solved Montezuma's
    Revenge and Pitfall from this phase alone. Structurally it is count-based novelty with
    restarts, and a strong fit for the exploration-hard game environments.

27. **Novelty search and quality-diversity** — Lehman & Stanley, *Abandoning Objectives:
    Evolution Through the Search for Novelty Alone*, Evol. Comput. 2011; Mouret & Clune,
    *Illuminating Search Spaces by Mapping Elites*, arXiv 2015. Evolutionary search over action
    sequences whose selection pressure is a behaviour descriptor (final `literals`, or the
    progress vector) rather than a fitness, MAP-Elites keeping the best sequence per descriptor
    cell. Go-Explore is the planning-shaped descendant; listed separately because a population
    can be evaluated in parallel across simulator processes.

28. **Kinodynamic sampling-based planners** — EST, Hsu, Latombe & Motwani, ICRA 1997;
    kinodynamic RRT, LaValle & Kuffner, IJRR 2001; KPIECE, Şucan & Kavraki, *Kinodynamic
    Motion Planning by Interior-Exterior Cell Exploration*, WAFR 2008; SST, Li, Littlefield &
    Bekris, *Asymptotically Optimal Sampling-Based Kinodynamic Planning*, IJRR 2016. Robotics'
    answer to planning with only a forward propagator. KPIECE in particular needs no distance
    metric and no state sampling: it projects states into grid cells, prefers expanding from
    cells on the exterior of what has been covered, and applies random controls for random
    durations. Replace "control" with "action sequence" and "projection" with `literals` and it
    is a novelty-driven tree grower; SST adds a sparse, near-optimal tree with pruning. Most
    relevant to the water, power and crop environments with continuous state.

29. **Local search over action sequences** — simulated annealing, tabu and iterated local search
    on a fixed-length plan, or large-neighbourhood re-planning of a window of it with any
    search above. Generic, needs `simulate`, and useful as an improver of plans the other
    planners return rather than as a first-line planner.

### D. Blind baselines

30. **Breadth-first search with duplicate detection**, **iterative-deepening DFS** and
    **uniform-cost search**. IJCAI 2015 used breadth-first as its baseline; the benchmark has
    no blind, complete reference at all. Trivial, and they make the width-based numbers
    interpretable. Iterative deepening re-expands the shallow tree on every iteration, which is
    costly against a simulator and should be said in its docs.

## Excluded, and why

- **MCTS/UCT and descendants**: THTS (Keller & Helmert 2013), GreedyUCT-Normal (Wissow & Asai
  2023), Extreme-Value MCTS and Bilevel MCTS (Asai & Wissow 2024, 2025), open-loop tree search
  OLETS/OLMCTS (Perez et al. 2015), NRPA-as-MCTS hybrids, AlphaZero-style search. Excluded by
  the brief, and the library's own MCTS planner went with them.
- **Width-based methods with a trained component**: π-IW and π-IW+ (Junyent et al. 2019),
  learned symbolic features for IW (Dittadi, Drachmann & Bolander, AAAI 2021), width-based
  lookaheads with learnt base policies and heuristics (O'Toole, Lipovetzky & Ramírez, 2021),
  width-based planning with active learning (Dittadi, Drachmann & Bolander, ICAPS 2022). Each
  needs a network trained before or across episodes.
- **Model-learning pipelines**: learning STRIPS or numeric action models from simulator traces
  and then running a classical planner (LOCM, SAM, and successors). Training by definition.
- **World-model planners** (PETS, PlaNet, Dreamer, TD-MPC): CEM or gradient planning inside a
  learned model. The planning half is item 22; the model is the trained part.
- **LLM-based planners and LLM-guided search**: trained models.
- **Regression, plan-space, SAT and symbolic (BDD) planners**: they need an action model to
  regress or encode, which a simulator does not give. Bidirectional search is out for the
  same reason.

## Heuristics that need no model

Most of section B and several entries elsewhere want a `progress` measure. What the
literature offers without a model, beyond the per-environment callbacks the library already
uses:

- **Novelty** (in the library) and **count-based novelty** (item 5).
- **`option_count`**, the library's FSX measure of how many futures a state keeps open, and its
  information-theoretic cousin **empowerment** (Klyubin, Polani & Nehaniv, CEC 2005; Salge,
  Glackin & Polani, 2014), the channel capacity from *n*-step action sequences to resulting
  states. Both goal-free; both usable as tie-breakers or as one queue in item 20.
- **Relaxed-plan atoms `R` from an IW pre-search** (item 3), the black-box replacement for FF's
  relaxed plan.
- **Random-walk endpoint statistics** (item 16): the best `progress` reachable by short random
  walks from a state is a cheap, noisy distance estimate.
- **Distance in literal space** to a known goal literal set (Hamming or Jaccard), for the
  environments whose goal is a set of atoms rather than an opaque predicate.

## Suggested order of implementation

Ranked by expected gain on the benchmark against the effort, given what the code already has.

| Priority | Item | Why first |
|---|---|---|
| 1 | 26 Go-Explore (exploration phase) | The hard game environments are exploration problems; restore is free here; small code |
| 2 | 1 2BFS and 2 p-IW(1) | The two IJCAI 2015/2016 simulator planners the library does not have; both small changes to existing classes |
| 3 | 3 BFWS(R) | The state of the art among width-based planners, and its simulator form was designed for this contract |
| 4 | 5 Count-based novelty, 6 Approximate novelty | Fix the two structural limits of `NoveltyTable`: novelty running out, and width > 2 being unaffordable |
| 5 | 21 RHEA, 22 CEM | The strongest non-tree family from GVGAI and control; need only `simulate` |
| 6 | 12 EHC, 13 Type-GBFS and ε-greedy, 14 DBFS, 15 GBFS-LE | Cheap exploration repairs for a weak `progress` heuristic |
| 7 | 16 Arvand random walks, 23 rollout with a base policy, 24 NMCS | Sampling planners that are not MCTS |
| 8 | 17 Beam search and BULB, 30 blind baselines | Memory-bounded and complete references for the benchmark |
| 9 | 19 FESS, 25 FMC | Specialised but strong: multi-feature puzzles, and a swarm sharing FSX's code |
| 10 | 7 BEE features, 28 KPIECE/SST | For the continuous-state operational environments |
| 11 | 9, 10 pruning and macro add-ons, 8 HIW, 18 real-time search, 4 QN, 20 alternation, 27, 29 | Add-ons and variants once the above exist |
