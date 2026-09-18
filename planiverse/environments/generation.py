"""What the instance generators share: a seeded draw, a bounded search, and a retry loop.

Every environment ships a fixed set of instances, selected by `set_index`. Each also has a
generator, `generate_instance(seed, **options)`, that draws a fresh instance from a seed and
selects it the same way. The generators differ in what they draw (a board, a network, a
season) but agree on three things, which live here so they are written once:

1. **The draw is a function of the seed.** `generate_instance(seed=7)` on two machines gives
   the same instance, because everything random comes from one `random.Random(seed)` and the
   checks below are deterministic.
2. **A draw is checked before it is handed out.** A random board is usually unsolvable, and
   an unsolvable instance is not an instance: a planner cannot tell "no plan" from "not yet".
   This is how the bundled instances were made in the first place: Flipull's stages were
   drawn at random and explored exhaustively, Super Mario Land's levels were each searched
   with BFWS before shipping and ranked by what that cost, and every simulator scenario
   carries the depth it was actually solved at. So the generators search each draw and keep
   only the ones a plan was found for, within a stated budget, leaving that plan on the
   environment as `witness` and what the search spent as `witness_expansions`. The budget is
   a knob (`search_limit`) and so is the check itself (`solvable=False` hands out the raw
   draw), because the check biases the generator towards instances a small search can
   solve, and a caller who wants harder ones than that has to say so.
3. **The instance is plain data.** Strings, tuples, lists and dicts, the same shape as the
   bundled instances, so it can be written to a file and given back to `set_instance` later.

The grid helpers at the end are the drawing half of the same story: the four board games
each put a ring of walls round a rectangle, scatter some scenery over it and drop objects
onto whatever floor is left, and that is written once here too.
"""
import random
from collections import deque, namedtuple


class GenerationError(RuntimeError):
    """No acceptable instance came out of the allowed number of draws."""


def rng(seed):
    """The one source of randomness a generator may use.

    `None` draws a seed from the operating system, so an unseeded call still returns an
    instance that `set_instance` reproduces; it is simply one nobody asked for by number.
    """
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 32)
    return random.Random(seed), seed


SearchOutcome = namedtuple("SearchOutcome", ["plan", "exhausted", "expansions"])


def bounded_search(env, limit, progress=None, key=None):
    """Search `env` from its initial state for a plan, expanding at most `limit` states.

    Breadth-first by default, so the plan returned is a shortest one and `len(plan)` is a
    fair measure of how deep an instance is. With `progress` (a `state -> number` guide,
    lower is better, the same shape as the benchmark's measures) it expands the most
    promising state first instead, which finds a plan sooner on the environments where
    breadth-first would drown, at the cost of the plan no longer being shortest.

    Returns `SearchOutcome(plan, exhausted, expansions)`. `plan` is a list of actions or
    `None`; `exhausted` says whether the whole reachable space was seen, which is what turns
    a `None` from "not within the budget" into "there is no plan".

    States are told apart by `literals`, which is what the planners key on too, unless `key`
    says otherwise: an environment whose literals carry history (Puzznic's record which
    blocks were cleared, and in what order) hands in the position instead, so the search
    closes on positions rather than on routes to them.
    """
    import heapq

    key = key or (lambda state: state.literals)
    start, _ = env.reset()
    if env.is_goal(start):
        return SearchOutcome([], True, 0)
    seen = {key(start)}
    expansions = 0
    if progress is None:
        frontier = deque([(start, [])])
        pop, push = frontier.popleft, frontier.append
    else:
        counter = 0
        frontier = [(progress(start), 0, start, [])]
        pop = lambda: heapq.heappop(frontier)[2:]  # noqa: E731

        def push(item):
            nonlocal counter
            counter += 1
            heapq.heappush(frontier, (progress(item[0]), counter, *item))
    while frontier:
        if expansions >= limit:
            return SearchOutcome(None, False, expansions)
        state, plan = pop()
        expansions += 1
        for action, successor in env.successors(state):
            identity = key(successor)
            if identity in seen:
                continue
            if env.is_goal(successor):
                return SearchOutcome(plan + [action], False, expansions)
            seen.add(identity)
            if env.is_terminal(successor):
                continue
            push((successor, plan + [action]))
    return SearchOutcome(None, True, expansions)


def width_search(env, limit, progress, width=2, seconds=None):
    """Search `env` with BFWS, the planner the bundled game levels were accepted with.

    The same shape as `bounded_search`, so a generator can use either: `SearchOutcome`, with
    `expansions` what BFWS spent, which is the number Super Mario Land's shipped levels are
    ranked by. Breadth-first drowns in a platformer's state space and a greedy search says
    nothing comparable about difficulty; BFWS is what the levels were checked with, so it is
    what generated ones are checked with.
    """
    from planiverse.planners.width import BFWSSearch, Budget

    result = BFWSSearch(width=width, progress=progress).solve(
        env, Budget(max_expansions=limit, max_seconds=seconds))
    return SearchOutcome(result.plan if result.solved else None,
                         result.status == "exhausted", result.statistics.expansions)


def draw_until(draw, accept, attempts, what="an instance"):
    """Call `draw(attempt)` up to `attempts` times and return the first result `accept` likes.

    Raises `GenerationError` naming what was asked for when none is, rather than returning a
    draw that failed the check: an unchecked instance handed out quietly is the one failure
    a generator must not have.
    """
    for attempt in range(attempts):
        candidate = draw(attempt)
        if candidate is not None and accept(candidate):
            return candidate
    raise GenerationError(
        f"no acceptable {what} in {attempts} draws; loosen the options, raise `attempts` or "
        f"`search_limit`, or pass solvable=False for an unchecked draw")


def solvable_draw(env, draw, attempts, search_limit, min_plan_length=1, progress=None,
                  key=None, search=None, min_expansions=0, what="an instance"):
    """Draw instances into `env` until one has a plan of at least `min_plan_length` actions.

    `draw(attempt)` returns an instance or `None`; each is loaded with `env.set_instance`
    and searched, with `bounded_search` unless `search` (a callable of the same shape, such
    as `width_search`) says otherwise. The winning instance is left selected, the plan found
    is kept on the environment as `env.witness`, and what the search spent as
    `env.witness_expansions`, so a caller can see the depth an instance was accepted at and
    what it cost; `min_expansions` rejects draws that were cheaper than that, which is a
    difficulty floor in the currency the search reports.
    """
    found = {}

    def accept(instance):
        env.set_instance(instance)
        if search is not None:
            outcome = search(env, search_limit)
        else:
            outcome = bounded_search(env, search_limit, progress, key)
        if outcome.plan is None or len(outcome.plan) < min_plan_length:
            return False
        if outcome.expansions < min_expansions:
            return False
        found["plan"], found["expansions"] = outcome.plan, outcome.expansions
        return True

    instance = draw_until(draw, accept, attempts, what)
    env.set_instance(instance)          # selecting clears any witness, so it goes on after
    env.witness, env.witness_expansions = found["plan"], found["expansions"]
    return instance


# ------------------------------------------------------------------------ drawing boards

def walled_grid(width, height, wall, floor):
    """A `height` by `width` rectangle of `floor` inside a ring of `wall`, as lists."""
    grid = [[wall] * (width + 2)]
    grid += [[wall] + [floor] * width + [wall] for _ in range(height)]
    grid.append([wall] * (width + 2))
    return grid


def interior(width, height):
    """The cells inside the ring `walled_grid` draws, row-major."""
    return [(row, col) for row in range(1, height + 1) for col in range(1, width + 1)]


def scatter(grid, random_, cells, glyph, fraction):
    """Turn a `fraction` of `cells` into `glyph`, chosen at random, and return them."""
    chosen = random_.sample(cells, int(round(fraction * len(cells))))
    for row, col in chosen:
        grid[row][col] = glyph
    return chosen


def place(grid, random_, cells, shape, free):
    """Put `shape` down somewhere every one of its squares lands on a `free` cell.

    `shape` is `(row, column, glyph)` squares relative to an anchor; the anchor is tried at
    each of `cells` in random order, and a square may only land inside the grid on a cell
    holding `free`. Returns whether it fitted; the caller draws again when it did not.
    """
    anchors = list(cells)
    random_.shuffle(anchors)
    height, width = len(grid), len(grid[0])
    for row, col in anchors:
        squares = [(row + dr, col + dc, glyph) for dr, dc, glyph in shape]
        if all(0 <= r < height and 0 <= c < width and grid[r][c] == free
               for r, c, _ in squares):
            for r, c, glyph in squares:
                grid[r][c] = glyph
            return True
    return False
