"""Replay each twin's plans on the other twin, and say what the differences are.

Four of the games ship twice: a pure-Python implementation (`puzznic`, `flipull`, `lolo`,
`amazing_tater`) and the cartridge itself driven through PyBoy (`..._gb`). The cartridge is
the authority, so a plan found on it that the twin cannot replay is a defect in the twin,
and a plan found in the twin that the cartridge refuses says the twin is modelling something
it does not have. Both directions are worth running, and both are just a replay: take the
plans the benchmark already produced, spell the actions in the other twin's vocabulary, and
ask whether the last state is a goal.

    PLANIVERSE_LOLO_ROM=... python tools/crossvalidate.py --games lolo -v

`--diagnose` adds, for each plan that does not replay, the step where the two positions
first disagree and the action that produced it, which is what turns "44 rooms failed" into
"they failed on the enemies this twin freezes".

Flipull is reported but not scored: its stages are generated rather than decoded, because
the cartridge draws each stage's arrangement from an RNG seeded by boot timing and stores
only a block total and a CLEAR target per stage. There is no shared board for a plan to
cross, so the contract is checked instead.
"""
import argparse
import glob
import json
import os
import time
import traceback

from planiverse.environments import make

#: Where the benchmark left its plans. `sandbox/` is gitignored, so this is a default and
#: not a promise; `--results` overrides it.
DEFAULT_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sandbox", "results")

#: Each twin's action names against the buttons the cartridge presses for them. `GBAction`
#: spells a combination with "+", and the plan files spell the whole action `buttons_for_ticks`.
BUTTONS = {
    "puzznic": {"left": "left", "right": "right", "up": "up", "down": "down",
                "left-hold": "a+left", "right-hold": "a+right"},
    "flipull": {"up": "up", "down": "down", "throw": None},   # the throw button is probed
    "lolo": {"left": "left", "right": "right", "up": "up", "down": "down", "shoot": "a"},
    "amazing_tater": {"up": "up", "right": "right", "down": "down", "left": "left",
                      "switch": "switch"},
}

PAIRS = (("puzznic", "puzznic_gb"), ("flipull", "flipull_gb"),
         ("lolo", "lolo_gb"), ("amazing_tater", "amazing_tater_gb"))

#: Games whose two halves hold the same boards. Flipull's do not, so a plan cannot cross.
COMPARABLE = frozenset({"puzznic", "lolo", "amazing_tater"})


def unparse(pretty):
    """`"a_with_left_for_15"` back to the `"a+left,15"` a `GBAction` is built from."""
    return pretty.replace("_with_", "+").replace("_for_", ",")


def buttons_of(action):
    return unparse(action).split(",")[0]


def gb_to_sim(game, plan):
    """A cartridge plan as twin action names, or `(None, why)` if a step has no twin."""
    reverse = {buttons: name for name, buttons in BUTTONS[game].items() if buttons}
    translated = []
    for action in plan:
        pressed = buttons_of(action)
        if pressed in reverse:
            translated.append(reverse[pressed])
        elif game == "flipull" and pressed in ("a", "b"):
            translated.append("throw")          # whichever button the probe settled on
        else:
            return None, f"no twin action presses {pressed!r}"
    return translated, None


def sim_to_gb(game, plan, vocabulary):
    """A twin plan as cartridge action strings, taking each hold from the live environment.

    The hold comes from the environment's own vocabulary rather than the module's fallback,
    because Flipull and Puzznic both measure theirs off the cartridge at `reset`.
    """
    by_buttons = {buttons_of(a): unparse(a) for a in vocabulary}
    spare = [a for a in vocabulary if buttons_of(a) not in ("up", "down")]
    translated = []
    for name in plan:
        buttons = BUTTONS[game].get(name)
        if buttons is None and game == "flipull" and name == "throw":
            if len(spare) != 1:
                return None, f"cannot tell which button throws: {vocabulary}"
            translated.append(unparse(spare[0]))
            continue
        if buttons is None or buttons not in by_buttons:
            return None, f"no cartridge action for {name!r}"
        translated.append(by_buttons[buttons])
    return translated, None


def load_plans(results, environment, planner):
    """`{index: plan}` for every instance this planner solved."""
    plans = {}
    for path in glob.glob(os.path.join(results, planner, f"{environment}__*.json")):
        record = json.load(open(path))
        if record.get("search_status") == "solved" and record.get("plan") is not None:
            plans[record["index"]] = record["plan"]
    return plans


# ------------------------------------------------------------------------- diagnosis
# Comparing two positions means comparing the facts both twins model, which is per game:
# the cartridge and the twin do not agree on coordinates (the cartridge counts from the
# frame it draws, the twin from its level string), so each key is offset onto the other.

def positions(game, sim_state, gb_state):
    """What each side of the pair reads out of one step, in whatever terms it keeps them."""
    if game == "lolo":
        onboard = lambda cells: frozenset((int(r), int(c)) for r, c in cells
                                          if 0 <= r < 8 and 0 <= c < 8)
        return ((tuple(int(x) for x in sim_state.lolo), onboard(sim_state.alive | sim_state.eggs)),
                (tuple(int(x) for x in gb_state.lolo), onboard(gb_state.enemies)))
    if game == "amazing_tater":
        return (({k: v for k, v in sim_state.taters}, sim_state.active),
                ({k: (v.row, v.col) for k, v in gb_state.taters.items()}, gb_state.active))
    return (str(sim_state), str(gb_state))


def first_divergence(game, sim_env, gb_env, sim_plan, gb_plan):
    """The step at which the two twins stop agreeing, and the action that produced it.

    The two sides do not share a coordinate system -- the cartridge counts from the frame it
    draws, the twin from its level string -- so what is compared is not the readings but
    whether each side's reading *changed*. A step where one twin moved and the other did not
    is the step they parted, whichever way round it is; `moved` says which.
    """
    try:
        sim_trace = sim_env.simulate(sim_plan)
        gb_trace = gb_env.simulate(gb_plan)
    except Exception:
        return None
    readings = [positions(game, sim_state, gb_state)
                for sim_state, gb_state in zip(sim_trace, gb_trace)]
    for step in range(1, len(readings)):
        twin_moved = readings[step][0] != readings[step - 1][0]
        cartridge_moved = readings[step][1] != readings[step - 1][1]
        if twin_moved != cartridge_moved:
            return {"step": step, "action": str(sim_plan[step - 1]),
                    "moved": "twin only" if twin_moved else "cartridge only"}
    return None


# ------------------------------------------------------------------------- the crossing

def replay(env, index, plan):
    env.set_index(index)
    env.reset()
    started = time.time()
    trace = env.simulate(plan)
    return env.is_goal(trace[-1]), time.time() - started


def cross(pair, planner, results=DEFAULT_RESULTS, limit=None, verbose=False,
          diagnose=False):
    sim_name, gb_name = pair
    game = sim_name
    sim_env, gb_env = make(sim_name), make(gb_name)
    gb_env.set_index(0)
    gb_env.reset()
    vocabulary = list(gb_env.actions)
    report = {"game": game, "planner": planner, "comparable": game in COMPARABLE,
              "cartridge_vocabulary": vocabulary, "gb_to_sim": [], "sim_to_gb": []}

    directions = (
        ("gb_to_sim", gb_name, sim_env, lambda plan: gb_to_sim(game, plan)),
        ("sim_to_gb", sim_name, gb_env, lambda plan: sim_to_gb(game, plan, vocabulary)),
    )
    for key, source, target, translate in directions:
        for index, plan in sorted(load_plans(results, source, planner).items())[:limit]:
            row = {"index": index, "length": len(plan)}
            translated, why = translate(plan)
            if translated is None:
                row.update(status="untranslatable", detail=why)
            else:
                try:
                    reached, seconds = replay(target, index, translated)
                    row.update(status="reached" if reached else "failed",
                               seconds=round(seconds, 3))
                except Exception as error:
                    row.update(status="error", detail=f"{type(error).__name__}: {error}")
                    if verbose:
                        traceback.print_exc()
                if diagnose and row["status"] == "failed" and game in COMPARABLE:
                    sim_plan = translated if key == "gb_to_sim" else plan
                    gb_plan = plan if key == "gb_to_sim" else translated
                    sim_env.set_index(index); sim_env.reset()
                    gb_env.set_index(index); gb_env.reset()
                    row["diverges"] = first_divergence(
                        game, sim_env, gb_env,
                        sim_plan, [unparse(a) for a in gb_plan])
            report[key].append(row)
            if verbose:
                print(f"  {key.replace('_', ' ')} {game}@{index}: {row['status']}", flush=True)

    if game == "flipull":
        report["contract"] = flipull_contract(sim_env, gb_env)
    gb_env.close()
    return report


def flipull_contract(sim_env, gb_env):
    """Flipull's boards cannot be crossed, so check what the twin does promise: every
    stage's block total and CLEAR target."""
    rows = []
    for index in range(32):
        sim_env.set_index(index); _, a = sim_env.reset()
        gb_env.set_index(index); _, b = gb_env.reset()
        rows.append({"index": index, "twin": [a["blocks"], a["clear_target"]],
                     "cartridge": [b["blocks"], b["clear_target"]],
                     "match": (a["blocks"], a["clear_target"]) == (b["blocks"], b["clear_target"])})
    return rows


def summarise(report):
    lines = []
    for key, arrow in (("gb_to_sim", "cartridge plans replayed in the twin"),
                       ("sim_to_gb", "twin plans replayed on the cartridge")):
        rows = report[key]
        reached = sum(1 for row in rows if row["status"] == "reached")
        lines.append(f"{report['game']:16s} {arrow:38s} {reached:3d}/{len(rows):3d}")
        failed = [row["index"] for row in rows if row["status"] == "failed"]
        broken = [row["index"] for row in rows if row["status"] in ("error", "untranslatable")]
        if failed:
            lines.append(f"{'':16s}   failed: {failed}")
        if broken:
            lines.append(f"{'':16s}   not run: {broken}")
    if not report["comparable"]:
        lines.append(f"{'':16s}   (boards are generated, not shared: the counts above are "
                     f"expected to be zero)")
    if report.get("contract") is not None:
        matched = sum(1 for row in report["contract"] if row["match"])
        lines.append(f"{'':16s}   stage contract (blocks, CLEAR target) matches: "
                     f"{matched}/{len(report['contract'])}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--planner", default="bfws", help="which results directory to read")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="the benchmark's results directory (default: sandbox/results)")
    parser.add_argument("--games", nargs="*", default=None,
                        help="a subset of: " + ", ".join(pair[0] for pair in PAIRS))
    parser.add_argument("--limit", type=int, default=None, help="first N plans per direction")
    parser.add_argument("--diagnose", action="store_true",
                        help="for each failure, where the two positions first parted")
    parser.add_argument("--out", default=None, help="write the full report here as JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    pairs = [pair for pair in PAIRS if args.games is None or pair[0] in args.games]
    reports = []
    for pair in pairs:
        print(f"== {pair[0]}", flush=True)
        report = cross(pair, args.planner, args.results, args.limit, args.verbose,
                       args.diagnose)
        reports.append(report)
        print(summarise(report), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(reports, handle, indent=2)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
