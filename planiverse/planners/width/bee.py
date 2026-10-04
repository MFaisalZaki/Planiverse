"""Boundary Extension Features: atoms for continuous variables, defined as the search goes.

Teichteil-Königsbuch, Ramírez and Lipovetzky, *Boundary Extension Features for Width-Based
Planning with Simulators on Continuous-State Domains*, IJCAI 2020. Novelty needs atoms, and
a continuous variable has none: bucket it by hand and the bucket width decides the
problem's width for you; keep it raw and every state is novel. BEE defines the features
**online**: for each variable the search remembers the range of values it has seen, and a
state is marked novel when it pushes a boundary of that range. The features are the
boundary extensions themselves, so nothing about the variable's scale has to be known in
advance, and k-BFWS over them matched deep reinforcement learning on the gym classic-control
tasks with no learning at all.

`BoundaryExtensionFeatures` is a callable `state -> frozenset` to hand to any width planner
that takes `atoms=` (`IWSearch`, `BFWS`, `DualBFWS`). It takes a `variables(state)
-> {name: value}` callback, since the environment contract exposes numbers only through
`literals`, and each environment knows which of its quantities are continuous.

Two readings of "feature" are offered, and the pure one is the default:

* `bins=0`: the atoms are the boundary extensions only. Each time a variable's minimum or
  maximum is pushed, a fresh atom `bee(name, low|high, n)` is created and the state gets it;
  a state inside every known range gets the atom `bee(name, inside)`, seen before, so it is
  not novel on that variable's account. This is the paper's mechanism as its abstract
  states it.
* `bins>0`: additionally, the current range of each variable is cut into `bins` equal
  intervals and the state gets `bee(name, bin, i)`. The bins move as the range grows, which
  is what makes interior progress visible once the boundaries have settled.

`include_literals=True` unions the environment's own atoms in, so discrete parts of the
state keep their exact novelty.
"""


class BoundaryExtensionFeatures:
    """Online boundary features over `variables(state)`."""

    def __init__(self, variables, bins=0, include_literals=True):
        if bins < 0:
            raise ValueError(f"bins must be non-negative, got {bins}")
        self.variables = variables
        self.bins = bins
        self.include_literals = include_literals
        self.ranges = {}            # name -> [low, high, extensions]

    def __call__(self, state):
        atoms = set(state.literals) if self.include_literals else set()
        for name, value in self.variables(state).items():
            bounds = self.ranges.get(name)
            if bounds is None:
                self.ranges[name] = [value, value, 0]
                atoms.add(f"bee({name}, first)")
                continue
            low, high, extensions = bounds
            if value < low:
                bounds[0], bounds[2] = value, extensions + 1
                atoms.add(f"bee({name}, low, {extensions + 1})")
            elif value > high:
                bounds[1], bounds[2] = value, extensions + 1
                atoms.add(f"bee({name}, high, {extensions + 1})")
            else:
                atoms.add(f"bee({name}, inside)")
            if self.bins and bounds[1] > bounds[0]:
                index = min(self.bins - 1,
                            int(self.bins * (value - bounds[0]) / (bounds[1] - bounds[0])))
                atoms.add(f"bee({name}, bin, {index})")
        return frozenset(atoms)

    def __repr__(self):
        return f"<BoundaryExtensionFeatures({len(self.ranges)} variables, bins={self.bins})>"
