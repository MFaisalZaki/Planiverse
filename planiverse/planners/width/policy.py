"""π-IW: Rollout IW whose rollouts follow a policy it learns from its own lookaheads.

Junyent, Jonsson and Gómez, *Deep Policies for Width-Based Planning in Pixel Domains*
(ICAPS 2019).

Rollout IW picks the child a rollout follows uniformly at random, which is what makes its
lookahead a search rather than a guess. π-IW keeps everything else and replaces that one
choice with a sample from a policy network π(a|s), and it trains the network on the planner
itself: after every decision, the returns the lookahead backed up to the root's children are
turned into a target distribution, softmax(R/τ), and the network is pushed toward it by
cross-entropy. Nothing about the domain is told to it; the planner is its own teacher, and
the policy it learns is a compressed memory of where the lookaheads found return before,
which then steers new rollouts there first. The paper's other contribution is that the
network's last hidden layer, binarised, works as the atoms novelty is measured over, so the
planner needs no hand-made features at all; `features="learned"` does that here.

What is kept and what is changed against the paper:

* **The function approximator is a two-layer network in NumPy**, not a convolutional one
  over pixels. States here are sets of atoms, so the input is a hashed bag of them (a fixed
  number of binary inputs, each atom mapped to one by a stable hash). That keeps the library
  free of a deep-learning dependency, and it is enough for the mechanism the paper
  describes; it is not a claim to match its Atari numbers.
* **Returns come from `progress`**, the improvement in the same measure SIW and BFWS take,
  because there is no score. The root's returns are scaled to `[0, 1]` before the softmax,
  so one temperature serves every environment; the paper uses the game's raw returns.
* **Training is online, one gradient step per decision, from a replay of recent targets**,
  and it continues across episodes: an episode that ends without a goal is not wasted, the
  next one starts with what it taught. `RolloutIW`'s `max_episodes` is therefore `None`
  here, and the search runs until it finds a plan or the budget ends.
* **Actions are indexed as they appear.** Environments build actions per state and have no
  fixed vocabulary, so the output layer grows a column whenever a new action is seen, with
  zero weights, which means a new action starts at the same probability as its peers.
"""
import zlib
from collections import deque

import numpy as np

from planiverse.planners.width.rollout import RolloutIW


class PolicyNetwork:
    """A one-hidden-layer softmax policy over a growing action vocabulary, trained by Adam.

    `probabilities(x)` and `hidden(x)` are the two things the planner asks for; `update`
    is the one gradient step. Everything is float32 NumPy, and the output layer grows in
    place when `grow` is told about more actions.
    """

    def __init__(self, inputs=2048, hidden=64, learning_rate=1e-3, seed=None):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": (rng.standard_normal((inputs, hidden)) / np.sqrt(inputs)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": np.zeros((hidden, 0), dtype=np.float32),
            "b2": np.zeros(0, dtype=np.float32),
        }
        self.moments = {name: (np.zeros_like(p), np.zeros_like(p))
                        for name, p in self.params.items()}
        self.learning_rate = learning_rate
        self.steps = 0
        self.updates = 0

    @property
    def outputs(self):
        return self.params["W2"].shape[1]

    def grow(self, outputs):
        """Add zero-weight columns for actions seen for the first time."""
        extra = outputs - self.outputs
        if extra <= 0:
            return
        for name, axis in (("W2", 1), ("b2", 0)):
            pad = [(0, 0)] * self.params[name].ndim
            pad[axis] = (0, extra)
            self.params[name] = np.pad(self.params[name], pad)
            self.moments[name] = tuple(np.pad(m, pad) for m in self.moments[name])

    def hidden(self, x):
        return np.maximum(x @ self.params["W1"] + self.params["b1"], 0.0)

    def probabilities(self, x, hidden=None):
        h = self.hidden(x) if hidden is None else hidden
        logits = h @ self.params["W2"] + self.params["b2"]
        if not self.outputs:
            return np.zeros(logits.shape, dtype=np.float32)
        logits = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=-1, keepdims=True)

    def update(self, X, T, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """One Adam step on the cross-entropy between softmax(logits) and the targets `T`."""
        P = self.params
        H = self.hidden(X)
        probs = self.probabilities(X, hidden=H)
        n = X.shape[0]
        d_logits = (probs - T) / n
        grads = {
            "W2": H.T @ d_logits,
            "b2": d_logits.sum(axis=0),
        }
        d_hidden = (d_logits @ P["W2"].T) * (H > 0)
        grads["W1"] = X.T @ d_hidden
        grads["b1"] = d_hidden.sum(axis=0)
        self.steps += 1
        for name, grad in grads.items():
            m, v = self.moments[name]
            m[...] = beta1 * m + (1 - beta1) * grad
            v[...] = beta2 * v + (1 - beta2) * grad * grad
            m_hat = m / (1 - beta1 ** self.steps)
            v_hat = v / (1 - beta2 ** self.steps)
            P[name] -= (self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)).astype(
                np.float32)
        self.updates += 1
        return float(-(T * np.log(probs + 1e-12)).sum() / n)


class PiIW(RolloutIW):
    """Rollout IW with a learned policy, trained online from the lookaheads.

    ```python
    from planiverse.planners.width import PiIW, Budget

    env.set_index(0)
    result = PiIW(
        expansions_per_step=100,
        progress=lambda s: s.blocks_left,
        seed=0,
    ).solve(env, Budget(max_expansions=50_000, max_seconds=60))
    result.statistics.episodes        # how many times it started over, learning each time
    ```

    `features` picks what novelty is measured over: `"literals"` (the environment's atoms,
    the default), `"learned"` (the network's hidden layer, binarised, as the paper does in
    pixel domains) or `"both"`.
    """

    def __init__(self, width=1, expansions_per_step=100, progress=None, discount=0.99,
                 max_depth=None, max_steps=200, max_episodes=None, reuse_tree=True,
                 avoid_dead_ends=True, strict=True, seed=None, features="literals",
                 inputs=2048, hidden=64, temperature=0.5, learning_rate=1e-3, batch_size=32,
                 replay_size=10_000, updates_per_step=1, floor=1e-3, train=True):
        """
        - `temperature`: τ in softmax(R/τ) over the root's returns, scaled to `[0, 1]`
          first. Lower is more decisive.
        - `inputs`, `hidden`: the network's shape. Atoms are hashed into `inputs` bits.
        - `floor`: the least weight any candidate action gets in a rollout, so a policy
          that has become sure of itself cannot stop the rollouts exploring altogether.
        - `train`: off, and the network stays as constructed, which is uniform: the
          planner is then Rollout IW with the overhead of a network it never uses.
        """
        if features not in ("literals", "learned", "both"):
            raise ValueError(f"features must be 'literals', 'learned' or 'both', got {features!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        super().__init__(width=width, expansions_per_step=expansions_per_step,
                         progress=progress, discount=discount, max_depth=max_depth,
                         max_steps=max_steps, max_episodes=max_episodes,
                         reuse_tree=reuse_tree, avoid_dead_ends=avoid_dead_ends,
                         strict=strict, policy=None, seed=seed)
        self.features = features
        self.inputs = inputs
        self.temperature = temperature
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.floor = floor
        self.train = train
        self.network = PolicyNetwork(inputs, hidden, learning_rate, seed)
        self.replay = deque(maxlen=replay_size)
        self.actions = {}               # str(action) -> output index
        self.rng = np.random.default_rng(seed)
        self.losses = []

    # ------------------------------------------------------------------- features

    def encode(self, state):
        """A state as a hashed bag of its atoms: one binary input per hash bucket."""
        x = np.zeros(self.inputs, dtype=np.float32)
        for atom in state.literals:
            x[zlib.crc32(str(atom).encode()) % self.inputs] = 1.0
        return x

    def index(self, action):
        key = str(action)
        if key not in self.actions:
            self.actions[key] = len(self.actions)
            self.network.grow(len(self.actions))
        return self.actions[key]

    def __evaluate__(self, node):
        """The network's view of a node, computed once per lookahead and kept on it.

        Recomputed if the vocabulary has grown since, so every candidate has a column.
        """
        if node.cache is None or len(node.cache[2]) != self.network.outputs:
            x = self.encode(node.state)
            h = self.network.hidden(x)
            node.cache = (x, h, self.network.probabilities(x, hidden=h))
        return node.cache

    def __atoms__(self, node):
        if self.features == "literals":
            return node.literals
        _, h, _ = self.__evaluate__(node)
        learned = frozenset(int(i) for i in np.flatnonzero(h > 0))
        return learned if self.features == "learned" else learned | frozenset(node.literals)

    # ---------------------------------------------------------------------- rollouts

    def __pick__(self, node, candidates):
        if len(candidates) == 1:
            return candidates[0]
        indices = [self.index(child.action) for child in candidates]
        _, _, probs = self.__evaluate__(node)
        weights = [max(float(probs[i]), self.floor) for i in indices]
        return self.random.choices(candidates, weights=weights)[0]

    # ---------------------------------------------------------------------- learning

    def __committed__(self, root, child):
        """Turn the root's backed-up returns into a target and take a gradient step."""
        if not self.train or not root.children:
            return
        returns = np.array([c.reward + self.discount * c.value for c in root.children],
                           dtype=np.float32)
        span = returns.max() - returns.min()
        scaled = (returns - returns.min()) / span if span > 0 else np.zeros_like(returns)
        logits = scaled / self.temperature
        soft = np.exp(logits - logits.max())
        soft /= soft.sum()
        indices = [self.index(c.action) for c in root.children]
        target = np.zeros(self.network.outputs, dtype=np.float32)
        for i, p in zip(indices, soft):
            target[i] += p
        self.replay.append((self.encode(root.state), target))
        for _ in range(self.updates_per_step):
            batch = self.rng.choice(len(self.replay), size=min(self.batch_size,
                                                                len(self.replay)),
                                    replace=False)
            X = np.stack([self.replay[i][0] for i in batch])
            T = np.stack([np.pad(self.replay[i][1],
                                 (0, self.network.outputs - len(self.replay[i][1])))
                          for i in batch])
            self.losses.append(self.network.update(X, T))
        # The tree beneath the executed child was scored by the network before this step;
        # the reroot clears those caches so the next lookahead sees the updated policy.
