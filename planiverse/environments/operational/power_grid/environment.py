"""Restoring an electricity grid to a secure state after a line trips.

A transmission line goes out. The power it was carrying does not stop: it redistributes
over every remaining line according to Kirchhoff's laws, and some of those lines now carry
more than they are rated for. Left alone, an overloaded line trips too, which redistributes
the flow again, which is how a regional blackout happens. The operator's move is to change
the *topology*: at a substation, the equipment is split across two busbars, and reassigning
which side each line, generator and load sits on reroutes the power without switching
anything off.

That is a discrete action set (a few hundred distinct reconfigurations) sitting on top of
a transition function that is a nonlinear solve. The flow on every line is the solution of
the AC power-flow equations, found by Newton-Raphson at each step. There is no way to write
"the effect of moving this line to busbar 2 is that line 17 now carries 1.08 of its rating";
the only way to know is to solve the network.

Two properties make this hard for a declarative model, both measured on the shipped case:

* **The effect of a local action is global and numerical.** One reassignment at substation 5
  changes the loading of every line in the grid.
* **Doing nothing is not safe.** The demand time series keeps moving, so an overload that is
  survivable now may cascade three steps later. On chronic 1 the worst loading after tripping
  line 6 climbs from 1.08 to 1.98 while the operator does nothing.

    env = PowerGridEnv()
    env.set_index(0)
    state, info = env.reset()
    for action, successor in env.successors(state):
        print(action, successor.max_rho)

`generate_instance(seed)` draws a contingency of its own: a time series, a step into it, and
a line to trip there, kept only if the trip leaves the grid overloaded and, left alone,
blacked out within a few steps, which is the same N-1 test the bundled scenarios were chosen
by, and then only if a reconfiguration secures it, so that like every bundled scenario it
carries the depth it was actually solved at.

Built on Grid2Op, the framework RTE (the French transmission operator) uses for the L2RPN
competitions: https://github.com/Grid2Op/grid2op
"""
from collections import namedtuple

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng
from planiverse.rendering.text import Canvas

#: The bundled case. `test=True` uses the time series shipped inside grid2op, so nothing is
#: downloaded and every run sees the same data.
GRID_NAME = "l2rpn_case14_sandbox"

#: A line is at `rho = 1.0` when it carries exactly its thermal rating. Above that it is
#: overloaded and the grid is insecure even though nothing has failed yet.
SECURE_RHO = 1.0

#: How many steps a plan may run before the episode is abandoned. Long enough to fix an
#: overload and see it stay fixed; short enough that search terminates.
HORIZON = 10

#: `seed` is the draw a generated scenario came from, and None for the chosen ones.
Scenario = namedtuple("Scenario", ["chronic", "line", "rho_after_trip",
                                   "blackout_in", "solved_at", "offset", "seed"],
                      defaults=(0, None))

#: How far into a time series a generated contingency may start, and how soon the untended
#: grid has to black out for the draw to count as doomed rather than self-healing.
MAX_OFFSET = 200
BLACKOUT_WITHIN = 6

#: Expansions the generator's search may spend proving a drawn contingency solvable. One:
#: every bundled scenario is solved by a single reconfiguration, and one expansion tries
#: every relevant one, at a power-flow solve each.
SEARCH_LIMIT = 1

#: Draws the generator may make before giving up. About one random (series, step, line)
#: draw in eleven survives both tests, so eighty draws fail to find one about once in two
#: thousand.
ATTEMPTS = 80

#: Which line to trip, on which time series.
#:
#: Chosen by N-1 analysis: every line of every chronic was tripped in turn, and these are the
#: ones that leave the grid **standing but doomed**: it survives the trip and then blacks out
#: within `blackout_in` steps if the operator does nothing. That filter matters more than it
#: sounds. Most overloads on this case *clear themselves* as demand moves: tripping line 1 on
#: chronic 0 gives a loading of 1.019 that is back under the limit two steps later on its own.
#: An instance the null plan solves is not an instance, so those are not here.
#:
#: `rho_after_trip` and `blackout_in` are measurements, recorded so that a scenario which
#: stops reproducing fails loudly instead of quietly becoming easy. `solved_at` is the
#: shallowest depth a solution was actually found at.
SCENARIOS = (
    Scenario(0, 11, 1.013, 2, 1),
    Scenario(0, 8, 1.188, 2, 1),
    Scenario(0, 9, 1.738, 2, 1),
    Scenario(1, 16, 1.044, 4, 1),
    Scenario(1, 6, 1.078, 4, 1),
    Scenario(1, 15, 1.224, 2, 1),
    Scenario(1, 3, 1.266, 4, 1),
    Scenario(1, 17, 1.914, 2, 1),
    Scenario(2, 9, 1.664, 2, 1),
    # Drawn by `generate_instance` at the seed each line records: the same N-1 test,
    # at a step into the series rather than at its start.
    Scenario(1, 8, 1.027, 2, 1, offset=50, seed=9000),
    Scenario(2, 19, 1.059, 4, 1, offset=131, seed=9001),
    Scenario(0, 9, 1.647, 2, 1, offset=151, seed=9002),
    Scenario(2, 8, 1.198, 2, 1, offset=147, seed=9003),
    Scenario(0, 9, 1.61, 2, 1, offset=125, seed=9004),
    Scenario(2, 9, 1.638, 2, 1, offset=132, seed=9005),
    Scenario(2, 8, 1.223, 2, 1, offset=153, seed=9006),
    Scenario(1, 6, 1.034, 4, 1, offset=10, seed=9007),
    Scenario(1, 8, 1.227, 2, 1, offset=149, seed=9008),
    Scenario(1, 8, 1.038, 2, 1, offset=85, seed=9009),
    Scenario(1, 6, 1.031, 4, 1, offset=19, seed=9010),
    Scenario(2, 9, 1.347, 2, 1, offset=96, seed=9011),
    Scenario(0, 8, 1.028, 2, 1, offset=36, seed=9012),
    Scenario(1, 9, 1.668, 2, 1, offset=157, seed=9013),
    Scenario(1, 9, 1.618, 2, 1, offset=114, seed=9014),
    Scenario(0, 9, 1.617, 2, 1, offset=142, seed=9015),
    Scenario(2, 8, 1.104, 2, 1, offset=174, seed=9016),
    Scenario(1, 8, 1.172, 2, 1, offset=5, seed=9017),
    Scenario(2, 9, 1.389, 2, 1, offset=29, seed=9018),
    Scenario(2, 9, 1.296, 2, 1, offset=85, seed=9019),
    Scenario(0, 8, 1.196, 2, 1, offset=156, seed=9020),
    Scenario(0, 8, 1.186, 2, 1, offset=148, seed=9021),
    Scenario(2, 9, 1.59, 2, 1, offset=164, seed=9022),
    Scenario(1, 8, 1.077, 2, 1, offset=179, seed=9023),
    Scenario(2, 9, 1.33, 2, 1, offset=91, seed=9024),
    Scenario(2, 9, 1.26, 2, 1, offset=72, seed=9025),
    Scenario(0, 11, 1.007, 2, 1, offset=2, seed=9026),
    Scenario(1, 9, 1.668, 2, 1, offset=121, seed=9027),
    Scenario(1, 19, 1.052, 2, 1, offset=94, seed=9028),
    Scenario(2, 9, 1.708, 2, 1, offset=155, seed=9029),
    Scenario(1, 8, 1.027, 2, 1, offset=63, seed=9030),
    Scenario(1, 8, 1.089, 2, 1, offset=105, seed=9031),
    Scenario(2, 8, 1.161, 2, 1, offset=132, seed=9032),
    Scenario(1, 8, 1.052, 2, 1, offset=83, seed=9033),
    Scenario(0, 8, 1.038, 2, 1, offset=101, seed=9034),
    Scenario(0, 8, 1.092, 2, 1, offset=170, seed=9035),
    Scenario(0, 9, 1.564, 2, 1, offset=161, seed=9036),
    Scenario(2, 9, 1.298, 2, 1, offset=83, seed=9037),
    Scenario(2, 9, 1.416, 2, 1, offset=186, seed=9038),
    Scenario(2, 8, 1.12, 2, 1, offset=5, seed=9039),
    Scenario(0, 8, 1.012, 2, 1, offset=96, seed=9040),
    Scenario(2, 19, 1.064, 4, 1, offset=157, seed=9041),
    Scenario(0, 9, 1.61, 2, 1, offset=135, seed=9042),
    Scenario(1, 8, 1.212, 2, 1, offset=158, seed=9043),
    Scenario(2, 9, 1.438, 2, 1, offset=189, seed=9044),
    Scenario(2, 8, 1.026, 2, 1, offset=199, seed=9045),
    Scenario(2, 8, 1.213, 2, 1, offset=152, seed=9046),
    Scenario(0, 8, 1.095, 2, 1, offset=15, seed=9047),
    Scenario(2, 9, 1.347, 2, 1, offset=96, seed=9048),
    Scenario(2, 9, 1.341, 2, 1, offset=34, seed=9500),
    Scenario(0, 8, 1.185, 2, 1, offset=149, seed=9501),
    Scenario(0, 9, 1.644, 2, 1, offset=127, seed=9502),
    Scenario(1, 19, 1.072, 2, 1, offset=91, seed=9503),
    Scenario(1, 15, 1.036, 2, 1, offset=90, seed=9504),
    Scenario(0, 8, 1.092, 2, 1, offset=170, seed=9505),
    Scenario(1, 8, 1.242, 2, 1, offset=150, seed=9506),
    Scenario(2, 19, 1.059, 4, 1, offset=131, seed=9507),
    Scenario(0, 8, 1.105, 2, 1, offset=22, seed=9508),
    Scenario(2, 9, 1.418, 2, 1, offset=28, seed=9509),
    Scenario(0, 9, 1.621, 2, 1, offset=124, seed=9510),
    Scenario(1, 8, 1.068, 2, 1, offset=39, seed=9511),
    Scenario(0, 8, 1.051, 2, 1, offset=177, seed=9512),
    Scenario(2, 9, 1.264, 2, 1, offset=67, seed=9513),
    Scenario(1, 6, 1.033, 4, 1, offset=13, seed=9514),
    Scenario(1, 9, 1.678, 2, 1, offset=142, seed=9515),
    Scenario(2, 9, 1.387, 2, 1, offset=101, seed=9516),
    Scenario(0, 9, 1.641, 2, 1, offset=20, seed=9517),
    Scenario(2, 9, 1.386, 2, 1, offset=30, seed=9518),
    Scenario(2, 9, 1.473, 2, 1, offset=178, seed=9519),
    Scenario(1, 8, 1.172, 2, 1, offset=0, seed=9520),
    Scenario(1, 8, 1.077, 2, 1, offset=180, seed=9521),
    Scenario(1, 8, 1.077, 2, 1, offset=190, seed=9522),
    Scenario(1, 8, 1.114, 2, 1, offset=173, seed=9523),
    Scenario(1, 8, 1.186, 2, 1, offset=138, seed=9524),
    Scenario(1, 9, 1.666, 2, 1, offset=134, seed=9525),
    Scenario(0, 8, 1.17, 2, 1, offset=158, seed=9526),
    Scenario(2, 19, 1.07, 4, 1, offset=156, seed=9527),
    Scenario(1, 8, 1.16, 2, 1, offset=118, seed=9528),
    Scenario(1, 8, 1.077, 2, 1, offset=183, seed=9529),
    Scenario(1, 9, 1.635, 2, 1, offset=118, seed=9530),
    Scenario(2, 8, 1.062, 2, 1, offset=183, seed=9531),
    Scenario(2, 19, 1.046, 4, 1, offset=124, seed=9532),
    Scenario(0, 9, 1.579, 2, 1, offset=30, seed=9533),
    Scenario(2, 9, 1.464, 2, 1, offset=109, seed=9534),
    Scenario(2, 9, 1.477, 2, 1, offset=180, seed=9535),
    Scenario(2, 9, 1.671, 2, 1, offset=158, seed=9536),
    Scenario(0, 8, 1.158, 2, 1, offset=4, seed=9537),
    Scenario(1, 9, 1.673, 2, 1, offset=120, seed=9538),
    Scenario(2, 9, 1.423, 2, 1, offset=22, seed=9539),
    Scenario(2, 19, 1.041, 4, 1, offset=142, seed=9540),
    Scenario(0, 9, 1.637, 2, 1, offset=140, seed=9541),
)


#: What the picture of a state needs that the state does not hold: the substations, each
#: line's two ends, and the line the contingency tripped. One per instance, shared by all its
#: states and outside their identity.
GridMap = namedtuple("GridMap", ["n_sub", "line_or", "line_ex", "tripped"])

#: A substation's box in the picture, columns by rows.
BOX_W, BOX_H = 5, 3

#: The one-line diagram of the 14-bus case: where each substation's box sits (its top-left
#: corner, in characters) and the path each line takes between them, as corner points from
#: one box's edge to the other's. Drawn after the case's own layout in grid2op, straightened
#: so that every line runs along rows and columns. `CASE14_LABELS` places the loading of a
#: line whose longest run has no room for it, ending at the column and row given.
CASE14_SUBS = {0: (0, 8), 1: (14, 0), 2: (50, 0), 3: (50, 8), 4: (14, 8), 5: (14, 18),
               6: (58, 18), 7: (72, 18), 8: (44, 18), 9: (36, 24), 10: (24, 24),
               11: (0, 30), 12: (14, 30), 13: (44, 30)}
CASE14_ROUTES = {
    0: ((2, 7), (2, 1), (13, 1)), 1: ((5, 9), (13, 9)), 2: ((19, 1), (49, 1)),
    3: ((17, 3), (17, 5), (51, 5), (51, 7)), 4: ((15, 3), (15, 7)), 5: ((53, 3), (53, 7)),
    6: ((19, 9), (49, 9)), 7: ((19, 19), (26, 19), (26, 23)),
    8: ((13, 19), (2, 19), (2, 29)), 9: ((16, 21), (16, 29)),
    10: ((45, 21), (45, 25), (41, 25)), 11: ((47, 21), (47, 29)), 12: ((29, 25), (35, 25)),
    13: ((5, 31), (13, 31)), 14: ((19, 31), (43, 31)),
    15: ((53, 11), (53, 14), (60, 14), (60, 17)), 16: ((51, 11), (51, 14), (46, 14), (46, 17)),
    17: ((16, 11), (16, 17)), 18: ((63, 19), (71, 19)), 19: ((49, 19), (57, 19)),
}
CASE14_LABELS = {4: (13, 5), 10: (43, 22), 16: (44, 16)}

#: Line styles: the straight runs and the corners, by the directions a corner joins.
_STYLES = {
    "light": ("─", "│", {"dr": "┌", "dl": "┐", "ur": "└", "ul": "┘"}),
    "heavy": ("━", "┃", {"dr": "┏", "dl": "┓", "ur": "┗", "ul": "┛"}),
    "dashed": ("╌", "╎", {"dr": "┌", "dl": "┐", "ur": "└", "ul": "┘"}),
}


def _layout(grid):
    """Box positions and line routes: the drawn case14 diagram when the grid is that case,
    otherwise substations on a square grid and each line bent once between them."""
    if grid.n_sub == len(CASE14_SUBS) and len(grid.line_or) == len(CASE14_ROUTES) and all(
            {grid.line_or[line], grid.line_ex[line]} == _ends(CASE14_SUBS, route)
            for line, route in CASE14_ROUTES.items()):
        return CASE14_SUBS, CASE14_ROUTES, CASE14_LABELS
    columns = max(1, int(grid.n_sub ** 0.5 + 0.999))
    subs = {sub: ((sub % columns) * 14, (sub // columns) * 8) for sub in range(grid.n_sub)}
    routes = {}
    for line, (a, b) in enumerate(zip(grid.line_or, grid.line_ex)):
        (ax, ay), (bx, by) = subs[a], subs[b]
        if ay == by:
            left, right = sorted(((ax, a), (bx, b)))
            routes[line] = ((left[0] + BOX_W, ay + 1), (right[0] - 1, ay + 1))
        else:
            (top, top_y), (low, low_y) = sorted(((ax, ay), (bx, by)), key=lambda p: p[1])
            x = top + 1 + line % (BOX_W - 2)
            routes[line] = ((x, top_y + BOX_H), (x, low_y + 1),
                            (low - 1 if low > x else low + BOX_W, low_y + 1))
    return subs, routes, {}


def _ends(subs, route):
    """The two substations a route starts and ends beside."""
    def beside(point):
        x, y = point
        for sub, (bx, by) in subs.items():
            if bx - 1 <= x <= bx + BOX_W and by - 1 <= y <= by + BOX_H:
                return sub
        return None
    return {beside(route[0]), beside(route[-1])}


def _percent(rho):
    """A loading as a whole percentage, rounded down so that only an overload reads 100%."""
    return f"{int(rho * 100)}%"


def _direction(a, b):
    (ax, ay), (bx, by) = a, b
    if ax == bx:
        return "d" if by > ay else "u"
    return "r" if bx > ax else "l"


def _route(canvas, points, style):
    across, down, corners = _STYLES[style]
    for a, b in zip(points, points[1:]):
        if a[1] == b[1]:
            canvas.hline(a[0], b[0], a[1], char=across)
        else:
            canvas.vline(a[0], a[1], b[1], char=down)
    opposite = {"u": "d", "d": "u", "l": "r", "r": "l"}
    for before, at, after in zip(points, points[1:], points[2:]):
        arms = {opposite[_direction(before, at)], _direction(at, after)}
        key = ("u" if "u" in arms else "d") + ("l" if "l" in arms else "r")
        canvas.put(at[0], at[1], corners[key])


def _one_line_diagram(state, grid):
    subs, routes, label_at = _layout(grid)
    width = max(x for x, _ in subs.values()) + BOX_W + 4
    height = max(y for _, y in subs.values()) + BOX_H
    canvas = Canvas(width, height)
    connected = state.connected or (True,) * len(grid.line_or)
    labels = []
    for line, points in routes.items():
        rho = state.rhos[line] if line < len(state.rhos) else 0.0
        out = state.blackout or not connected[line]
        style = "dashed" if out else ("heavy" if rho > SECURE_RHO else "light")
        _route(canvas, points, style)
        text = "✕" if line == grid.tripped else ("off" if out else _percent(rho))
        if line in label_at:
            x, y = label_at[line]
            labels.append(((x - len(text) + 1, y), text))
            continue
        a, b = max(zip(points, points[1:]),
                   key=lambda seg: abs(seg[1][0] - seg[0][0]) + abs(seg[1][1] - seg[0][1]))
        if a[1] == b[1]:
            low, high = sorted((a[0], b[0]))
            start = max(low + 1, (low + high + 1) // 2 - (len(text) + 2) // 2)
            labels.append(((start, a[1]), f" {text} "))
        else:
            labels.append(((a[0] + 2, (a[1] + b[1]) // 2), text))
    for (x, y), text in labels:
        canvas.write(x, y, text)
    for sub, (x, y) in subs.items():
        if sub in state.split:
            canvas.fill(x, y, x + BOX_W - 1, y + BOX_H - 1, " ")
            canvas.write(x, y, "╔" + "═" * (BOX_W - 2) + "╗")
            canvas.write(x, y + BOX_H - 1, "╚" + "═" * (BOX_W - 2) + "╝")
            for row in range(y + 1, y + BOX_H - 1):
                canvas.put(x, row, "║")
                canvas.put(x + BOX_W - 1, row, "║")
        else:
            canvas.fill(x, y, x + BOX_W - 1, y + BOX_H - 1, " ")
            canvas.rect(x, y, x + BOX_W - 1, y + BOX_H - 1)
        canvas.write(x + 1, y + 1, f"{sub:>2}")
    return str(canvas)


class PowerGridAction:
    """One substation reconfiguration, or doing nothing.

    Actions are identified by their index in grid2op's `IdToAct` converter, which enumerates
    the discrete topology space in a fixed order. The index is what a plan carries, so a
    plan is a list of integers and replays identically.
    """

    def __init__(self, action_id, label=None):
        self.action_id = action_id
        self.label = label or (f"topology_{action_id}" if action_id else "do_nothing")

    def cost(self):
        return 0 if self.action_id == 0 else 1

    def __eq__(self, other):
        return isinstance(other, PowerGridAction) and self.action_id == other.action_id

    def __hash__(self):
        return hash(self.action_id)

    def __lt__(self, other):
        return self.action_id < other.action_id

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)


class PowerGridState:
    """A grid position: which actions have been taken, and what the solve says about it.

    The action path is the state. Grid2op is deterministic once the time series is pinned
    with `set_id` and the environment is seeded (verified by running the same sequence
    twice and comparing every line's loading), so replaying a path always lands in the same
    place, and two paths that agree are the same position.
    """

    def readings(self):
        """The state's numbers by name, which its OpenSpiel game gives as the named pieces
        of its observation."""
        return {"worst line loading": float(self.max_rho), "limit": 1.0,
                "overloaded": sum(1 for rho in self.rhos if rho >= 1.0)}

    def __init__(self, path, max_rho, rhos, blackout, step, survived, grid=None,
                 connected=None, split=()):
        self.path = tuple(path)
        # What the picture draws, none of it part of the state's identity: the grid's
        # topology (a `GridMap`, the same object for every state of an instance), which
        # lines are in service, and which substations run on two busbars.
        self.grid = grid
        self.connected = tuple(connected) if connected is not None else None
        self.split = frozenset(split)
        self.max_rho = max_rho
        self.rhos = tuple(rhos)
        self.blackout = blackout
        self.step = step
        self.survived = survived
        self.depth = len(self.path)

        literals = [f"acted({index},{action_id})"
                    for index, action_id in enumerate(self.path) if action_id]
        literals.append(f"step({step})")
        if blackout:
            # A blacked-out grid has no line loadings to report (`max_rho` is infinite
            # precisely so it sorts last), so the numeric atoms are simply absent.
            literals.append("blackout")
        else:
            # Bucketed: a planner keyed on the raw loading would find every state novel.
            literals.append(f"max-loading({int(max_rho * 10)})")
            literals += [f"overloaded(line-{line})"
                         for line, rho in enumerate(self.rhos) if rho > SECURE_RHO]
        if max_rho < SECURE_RHO and not blackout:
            literals.append("secure")
        self.literals = frozenset(literals)

    def is_secure(self):
        return not self.blackout and self.max_rho < SECURE_RHO

    def overloaded_lines(self):
        return tuple(line for line, rho in enumerate(self.rhos) if rho > SECURE_RHO)

    def __eq__(self, other):
        return isinstance(other, PowerGridState) and self.path == other.path

    def __hash__(self):
        return hash(self.path)

    def __lt__(self, other):
        return self.depth < other.depth

    def __repr__(self):
        loading = "blackout" if self.blackout else f"{self.max_rho:.3f}"
        return f"<PowerGridState(depth={self.depth}, max_rho={loading})>"

    def __str__(self):
        """A one-line diagram of the grid.

        Each substation is a box with its id, double-walled when it runs on two busbars,
        and each line is labelled with its loading: drawn light within its rating, heavy
        when overloaded, dashed when out of service, the line that tripped marked `✕`.
        """
        if self.grid is None:
            return f"step {self.step}" + (" · BLACKOUT" if self.blackout else "")
        return _one_line_diagram(self, self.grid)


class PowerGridEnv(Environment):
    """Fix an insecure grid with substation topology, before it cascades.

    Needs `grid2op`. The case and its time series ship inside the package, so there is
    nothing to download.
    """

    def __init__(self, horizon=HORIZON, grid_name=GRID_NAME, restrict_to_overloads=True):
        super().__init__("power_grid")
        self.horizon = horizon
        self.grid_name = grid_name
        # Every topology action is legal, but expanding all 209 of them costs half a minute
        # a node. Restricting to the substations that touch an overloaded line is what an
        # operator would consider, and is what makes search tractable (see `successors`).
        self.restrict_to_overloads = restrict_to_overloads

        self.scenario_index = None
        #: The contingency `reset` builds, as `{"chronic": ..., "line": ..., "offset": ...}`:
        #: a bundled one after `set_index`, or whatever `set_instance` was given.
        self.instance = None
        #: The plan `generate_instance` accepted the current contingency on, when it checked
        #: one, and what the search spent finding it.
        self.witness = None
        self.witness_expansions = None
        self.state = None

        self._env = None
        self._converter = None
        self._cache = {}
        self._cursor = None          # (path, live env) so expanding a parent replays once
        self._grid_map = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        """Choose which line trips, on which time series."""
        if not 0 <= index < len(SCENARIOS):
            raise IndexError(
                f"Invalid index: {index}. There are {len(SCENARIOS)} scenarios, so the "
                f"index must be 0-{len(SCENARIOS) - 1}.")
        scenario = SCENARIOS[index]
        self.set_instance({"chronic": scenario.chronic, "line": scenario.line,
                           "offset": scenario.offset,
                           "rho_after_trip": scenario.rho_after_trip,
                           "blackout_in": scenario.blackout_in,
                           "solved_at": scenario.solved_at})
        self.scenario_index = index

    def set_instance(self, instance):
        """Select a contingency: `{"chronic": i, "line": l, "offset": t}`.

        `chronic` is one of the case's time series, `line` the line that trips, and `offset`
        how many steps into the series it does so (0 for the start).
        """
        self.instance = {"chronic": int(instance["chronic"]), "line": int(instance["line"]),
                         "offset": int(instance.get("offset", 0)), **instance}
        self.scenario_index = None
        self.witness = self.witness_expansions = None
        self._cache = {}
        self.__release_cursor__()

    def generate_instance(self, seed=None, chronic=None, line=None, max_offset=MAX_OFFSET,
                          min_rho=SECURE_RHO, blackout_within=BLACKOUT_WITHIN, solvable=True,
                          search_limit=SEARCH_LIMIT, attempts=ATTEMPTS):
        """Draw a fresh contingency, select it, and return it as a dict.

        A time series (`chronic`, or one of the case's at random), a step `offset` into it
        drawn from `0` to `max_offset`, and a line (`line`, or one at random) to trip there.
        A draw is kept only if the trip leaves some line above `min_rho` of its rating and,
        with the operator doing nothing, the grid blacks out within `blackout_within` steps:
        the "standing but doomed" test the bundled scenarios were chosen by, since a grid
        that heals itself is not an instance. With `solvable` it is then searched for up to
        `search_limit` expansions (one, by default: every bundled scenario is solved by a
        single reconfiguration, and one expansion tries every relevant one) and kept only
        if a plan was found, which is left in `witness` and recorded in the instance as
        `solved_at`. Each candidate costs a few simulated steps to measure and a power-flow
        solve per relevant reconfiguration to check, so `attempts` bounds the draws.
        """
        random_, _ = rng(seed)
        # One probe environment for every draw: building one costs seconds, resetting it to
        # another series and step does not.
        probe = self.__make__()
        chronics = range(len(probe.chronics_handler.available_chronics()))
        measured = {}

        def draw(attempt):
            return {"chronic": chronic if chronic is not None else random_.choice(chronics),
                    "line": line if line is not None else random_.randrange(probe.n_line),
                    "offset": random_.randint(0, max_offset)}

        def accept(candidate):
            self.set_instance(candidate)
            try:
                probe.set_id(candidate["chronic"])
                probe.reset()
                if candidate["offset"]:
                    probe.fast_forward_chronics(candidate["offset"])
                probe.step(probe.action_space({"set_line_status": [(candidate["line"], -1)]}))
            except Exception:
                return False              # the series is shorter than the offset, or the like
            tripped = self.__measure__(probe, ())
            if tripped.blackout or tripped.max_rho <= min_rho:
                return False
            for step in range(1, blackout_within + 1):
                probe.step(probe.action_space({}))
                if probe.done:
                    break
            else:
                return False
            found = {"rho_after_trip": round(tripped.max_rho, 3), "blackout_in": step}
            if solvable:
                outcome = bounded_search(self, search_limit)
                if outcome.plan is None:
                    return False
                found.update(solved_at=len(outcome.plan), plan=outcome.plan,
                             expansions=outcome.expansions)
            measured.update(found)
            return True

        try:
            candidate = draw_until(draw, accept, attempts, "contingency")
        finally:
            probe.close()
        instance = {**candidate, **{key: measured[key] for key in measured
                                    if key not in ("plan", "expansions")}}
        self.set_instance(instance)
        if solvable:
            self.witness, self.witness_expansions = measured["plan"], measured["expansions"]
        return instance

    # ------------------------------------------------------------------ the grid

    def __make__(self):
        """A fresh grid2op environment with the stochastic parts pinned.

        `seed` and `set_id` are what make this a planning problem rather than a gamble: the
        time series is fixed, so a plan replays to the same place every time.
        """
        import grid2op
        from grid2op.Parameters import Parameters

        parameters = Parameters()
        # Leave overloads *on*: a line that stays above its rating is supposed to trip, and
        # the cascade it starts is the thing the operator is planning against.
        parameters.NO_OVERFLOW_DISCONNECTION = False
        env = grid2op.make(self.grid_name, test=True, param=parameters)
        env.seed(0)
        return env

    def __converter__(self):
        """Grid2op's enumeration of the discrete topology space, in a fixed order."""
        from grid2op.Converter import IdToAct

        if self._converter is None:
            if self._env is None:
                self._env = self.__make__()
            converter = IdToAct(self._env.action_space)
            converter.init_converter(set_line_status=False, change_line_status=False,
                                     change_bus_vect=False, set_topo_vect=True,
                                     redispatch=False)
            self._converter = converter
        return self._converter

    def __release_cursor__(self):
        if self._cursor is not None:
            self._cursor[1].close()
            self._cursor = None

    def __env_at__(self, path):
        """A live environment positioned at `path`, replayed from the start.

        Replay rather than snapshot-per-state: an environment copy is megabytes and a search
        holds thousands of states. Replaying is cheap because the result of every path is
        memoised, so a path is only ever walked once, and expanding a parent replays it once
        and then copies for each child.
        """
        env = self.__make__()
        env.set_id(self.instance["chronic"])
        env.reset()
        if self.instance["offset"]:
            env.fast_forward_chronics(self.instance["offset"])
        # The contingency: the line that trips is what creates the problem.
        env.step(env.action_space({"set_line_status": [(self.instance["line"], -1)]}))
        converter = self.__converter__()
        for action_id in path:
            env.step(converter.convert_act(action_id))
        return env

    def __measure__(self, env, path):
        observation = env.current_obs
        blackout = observation is None or bool(env.done)
        grid = self.__grid_map__(env)
        if blackout:
            return PowerGridState(path, float("inf"), (), True, len(path) + 1, False,
                                  grid=grid)
        rhos = [float(value) for value in observation.rho]
        max_rho = max(rhos) if rhos else 0.0
        # Counted from the trip, not from the start of the time series: a contingency that
        # begins `offset` steps in has the same horizon as one that begins at the start.
        step = (int(observation.current_step) - self.instance["offset"]
                if hasattr(observation, "current_step") else len(path) + 1)
        connected = tuple(bool(status) for status in observation.line_status)
        split = tuple(sub for sub in range(int(env.n_sub))
                      if 2 in set(int(bus) for bus in observation.sub_topology(sub)))
        return PowerGridState(path, max_rho, rhos, False, step, True, grid=grid,
                              connected=connected, split=split)

    def __grid_map__(self, env):
        """The topology a state's picture is drawn on, one per instance."""
        if self._grid_map is None or self._grid_map.tripped != self.instance["line"]:
            self._grid_map = GridMap(int(env.n_sub),
                                     tuple(int(sub) for sub in env.line_or_to_subid),
                                     tuple(int(sub) for sub in env.line_ex_to_subid),
                                     self.instance["line"])
        return self._grid_map

    def __state__(self, path):
        """The state a path leads to. Memoised: deterministic, so it cannot differ."""
        key = tuple(path)
        if key in self._cache:
            return self._cache[key]
        env = self.__env_at__(key)
        try:
            state = self.__measure__(env, key)
        finally:
            env.close()
        self._cache[key] = state
        return state

    # ------------------------------------------------------------------ interface

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.__release_cursor__()
        self._cache = {}
        self.state = self.__state__(())
        return self.state, {"grid": self.grid_name,
                            "chronic": self.instance["chronic"],
                            "offset": self.instance["offset"],
                            "tripped_line": self.instance["line"],
                            "scenario": self.scenario_index,
                            "generated": self.scenario_index is None,
                            "max_rho": self.state.max_rho,
                            "overloaded": list(self.state.overloaded_lines()),
                            "actions": self.__converter__().n}

    def is_goal(self, state):
        """Every line back within its rating, and the grid still standing."""
        return state.is_secure()

    def is_terminal(self, state):
        """Blacked out, or out of time.

        A blackout is absorbing in the simulator too (grid2op ends the episode), so there is
        genuinely nothing further to plan from.
        """
        return state.blackout or state.step >= self.horizon

    def __relevant_actions__(self, state):
        """Reconfigurations at substations touching an overloaded line, plus doing nothing.

        Every topology action is legal from every state, and offering all 209 costs about
        thirty seconds a node. An operator looks at the substations either end of the line
        that is overloading, and so does this; `restrict_to_overloads=False` turns it off and
        offers the lot.
        """
        import numpy as np

        converter = self.__converter__()
        if not self.restrict_to_overloads:
            return list(range(converter.n))

        env = self._env if self._env is not None else self.__make__()
        if self._env is None:
            self._env = env
        substations = set()
        for line in state.overloaded_lines():
            substations.add(int(env.line_or_to_subid[line]))
            substations.add(int(env.line_ex_to_subid[line]))
        if not substations:
            return [0]

        relevant = [0]                      # do-nothing is always available
        for action_id in range(1, converter.n):
            _, touched = converter.convert_act(action_id).get_topological_impact()
            if substations & set(int(s) for s in np.where(touched)[0]):
                relevant.append(action_id)
        return relevant

    def successors(self, state):
        """Every relevant reconfiguration, minus the ones that change nothing."""
        if self.is_goal(state) or self.is_terminal(state):
            return []

        converter = self.__converter__()
        # Replay to the parent once, then branch from a copy per child. Replaying per child
        # would repeat the whole prefix every time.
        parent = self.__env_at__(state.path)
        successors = []
        try:
            for action_id in self.__relevant_actions__(state):
                path = state.path + (action_id,)
                if path in self._cache:
                    successor = self._cache[path]
                else:
                    child = parent.copy()
                    try:
                        child.step(converter.convert_act(action_id))
                        successor = self.__measure__(child, path)
                    finally:
                        child.close()
                    self._cache[path] = successor
                if successor == state:
                    continue
                successors.append((PowerGridAction(action_id), successor))
        finally:
            parent.close()
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        action_id = action.action_id if isinstance(action, PowerGridAction) else int(action)
        return self.__state__(state.path + (action_id,))

    def get_actions(self):
        """Every topology action the case has, whatever state it is in."""
        return [PowerGridAction(action_id) for action_id in range(self.__converter__().n)]

    def close(self):
        self.__release_cursor__()
        if self._env is not None:
            self._env.close()
            self._env = None
        self._converter = None
