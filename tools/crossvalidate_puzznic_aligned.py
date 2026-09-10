"""Puzznic twin fidelity with the cursor aligned.

The twin starts its cursor on the nearest empty cell and the cartridge on a block. Before
replaying a plan in the other form, walk that form's cursor (moves only, no holds) to the cell
the plan's own form started from, then replay. Both forms print the same board layout, so a
cursor cell read off str(state) is the same cell in both.
"""
import json, os, sys, time
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.crossvalidate as cv
from planiverse.environments import make

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sandbox", "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else "puzznic-aligned.json"

def cursor_of(state):
    for r, line in enumerate(str(state).split("\n")):
        for c, ch in enumerate(line):
            if ch in "c¢":
                return (r, c)
    return None

def bfs(env, start, target, moves):
    queue = deque([(start, [])]); seen = {str(start)}
    while queue:
        state, path = queue.popleft()
        if cursor_of(state) == target:
            return path
        for action, nxt in env.successors(state):
            key = str(action)
            if key not in moves or str(nxt) in seen:
                continue
            seen.add(str(nxt)); queue.append((nxt, path + [moves[key]]))
    return None

py, gb = make("puzznic"), make("puzznic_gb")
gb.set_index(0); gb.reset()
vocabulary = list(gb.actions)                      # 'left,15' ...
twin_moves = {d: d for d in ("left", "right", "up", "down")}
gb_moves = {a.replace("+", "_with_").replace(",", "_for_"): a
            for a in vocabulary if a.split(",")[0] in ("left", "right", "up", "down")}

report = {"gb_to_sim": [], "sim_to_gb": []}
for key, source, target_env, translate, moves in (
        ("gb_to_sim", "puzznic_gb", py, lambda p: cv.gb_to_sim("puzznic", p), twin_moves),
        ("sim_to_gb", "puzznic", gb, lambda p: cv.sim_to_gb("puzznic", p, vocabulary), gb_moves)):
    for index, plan in sorted(cv.load_plans(RESULTS, source, "bfws").items()):
        row = {"index": index}
        py.set_index(index); s0, _ = py.reset()
        gb.set_index(index); g0, _ = gb.reset()
        origin = cursor_of(g0) if key == "gb_to_sim" else cursor_of(s0)
        start = s0 if key == "gb_to_sim" else g0
        translated, why = translate(plan)
        if translated is None:
            row.update(status="untranslatable", detail=why)
        else:
            t0 = time.time()
            prefix = bfs(target_env, start, origin, moves)
            if prefix is None:
                row.update(status="unreachable", detail=f"no cursor path to {origin}")
            else:
                try:
                    trace = target_env.simulate(prefix + translated)
                    row.update(status="reached" if target_env.is_goal(trace[-1]) else "failed",
                               prefix=len(prefix), seconds=round(time.time() - t0, 1))
                except Exception as error:
                    row.update(status="error", detail=f"{type(error).__name__}: {error}")
        report[key].append(row)
        print(key, index, row["status"], flush=True)
        json.dump(report, open(OUT, "w"), indent=1)
for key in report:
    rows = report[key]
    print(key, sum(r["status"] == "reached" for r in rows), "/", len(rows),
          {s: sum(r["status"] == s for r in rows) for s in ("failed", "unreachable", "untranslatable", "error")})
