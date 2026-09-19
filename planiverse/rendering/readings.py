"""What the environments whose state is a handful of readings chart, and how.

A board renders as its own text; a city, a factory or a crop describes itself in a line of
numbers, and a line of numbers per state is read better as a chart of those numbers over
the steps of the plan. Each entry here says which readings a state carries (`values`, a
function of the state) and how to lay them out (`panels`, each a titled axis holding the
series that share a unit, and optionally the reading drawn across it as a target). The
registry is keyed by the state's class, so `render_trace` finds it without being told the
environment, and an environment that is not here renders as text.
"""
from collections import namedtuple

#: One axis: its title (which names the unit), the readings drawn on it as series, and the
#: reading drawn across it as a dashed target line, if any.
Panel = namedtuple("Panel", "title series target")
#: A state's readings (`values(state) -> {name: number}`) and the panels they go on.
Readings = namedtuple("Readings", "values panels")


def micropolis(state):
    return {"population": state.population, "residents": state.residents,
            "commerce": state.commerce, "industry": state.industry,
            "funds": state.funds, "score": state.score, "target": state.target}


def factory(state):
    furnaces = [m for m in state.machines if m.kind == "furnace"]
    drills = [m for m in state.machines if m.kind == "drill"]
    return {"brought back": state.plates, "made": state.produced,
            "in furnaces": sum(m.plates for m in furnaces), "target": state.target,
            "coal": state.inventory.get("coal", 0), "ore": state.inventory.get("ore", 0),
            "drills": sum(1 for m in drills if m.status == "working"),
            "furnaces": sum(1 for m in furnaces if m.status == "working"),
            "seconds": state.tick / 60}


def tower_defence(state):
    kinds = [kind for _, kind in state.towers]
    return {"lives": state.lives, "gold": state.gold, "arrow": kinds.count("arrow"),
            "cannon": kinds.count("cannon"), "wave": state.wave}


def crop_management(state):
    return {"biomass": float(state.biomass), "yield": float(state.yield_kg),
            "water used": float(state.water_used)}


def power_grid(state):
    return {"worst line loading": float(state.max_rho), "limit": 1.0,
            "overloaded": sum(1 for rho in state.rhos if rho >= 1.0)}


def water_network(state):
    return {"contaminated": 100 * float(state.contaminated),
            "service": 100 * float(state.service), "closed": len(state.closed)}


def flood_transport(state):
    million = 1e6
    return {"total": float(state.cost) / million, "damage": float(state.damage) / million,
            "delays": float(state.delay) / million, "measures": float(state.spent) / million,
            "target": float(state.target) / million, "storm": float(state.rain or 0),
            "in place": len(state.protected)}


def epidemic(state):
    return {"infectious": state.infectious, "in hospital": state.load, "beds": state.capacity,
            "deaths": state.deaths, "allowed": state.target, "points spent": state.points,
            "budget": state.budget}


def traffic(state):
    return {"on the road": state.running, "arrived": state.arrived, "to come": state.pending,
            "vehicle-seconds": state.travel, "target": state.target}


def airspace(state):
    return {"in the sector": len(state.flying), "closest, nm": min(state.closest, 30.0),
            "standard": 5.0, "miles to go": sum(a[5] for a in state.flying),
            "exit times, s": state.exit_sum, "target": state.target}


def reservoir(state):
    city, farm, river, turbine = state.delivered
    return {"upper": state.upper, "lower": state.lower, "released": turbine, "city": city,
            "farm": farm, "river": river, "farm shortfall": state.farm_short, "allowed": state.target}


READINGS = {
    ("planiverse.environments.epidemic.environment", "EpidemicState"): Readings(
        epidemic, (Panel("people", ("infectious",), None),
                   Panel("in hospital", ("in hospital",), "beds"),
                   Panel("deaths", ("deaths",), "allowed"),
                   Panel("disruption points", ("points spent",), "budget"))),
    ("planiverse.environments.traffic.environment", "TrafficState"): Readings(
        traffic, (Panel("vehicles", ("on the road", "arrived", "to come"), None),
                  Panel("travel, vehicle-seconds", ("vehicle-seconds",), "target"))),
    ("planiverse.environments.airspace.environment", "AirspaceState"): Readings(
        airspace, (Panel("aircraft in the sector", ("in the sector",), None),
                   Panel("closest pair, nautical miles", ("closest, nm",), "standard"),
                   Panel("miles still to fly", ("miles to go",), None),
                   Panel("exit times, seconds", ("exit times, s",), "target"))),
    ("planiverse.environments.reservoir.environment", "ReservoirState"): Readings(
        reservoir, (Panel("volume", ("upper", "lower"), None),
                    Panel("flows a month", ("released", "city", "farm", "river"), None),
                    Panel("farm shortfall", ("farm shortfall",), "allowed"))),
    ("planiverse.environments.micropolis.environment", "MicropolisState"): Readings(
        micropolis, (Panel("people", ("population", "residents", "commerce", "industry"), "target"),
                     Panel("funds", ("funds",), None),
                     Panel("score", ("score",), None))),
    ("planiverse.environments.factory.environment", "FactoryState"): Readings(
        factory, (Panel("plates", ("brought back", "made", "in furnaces"), "target"),
                  Panel("held", ("coal", "ore"), None),
                  Panel("machines working", ("drills", "furnaces"), None),
                  Panel("game time, seconds", ("seconds",), None))),
    ("planiverse.environments.tower_defence.environment", "TowerState"): Readings(
        tower_defence, (Panel("lives", ("lives",), None),
                        Panel("gold", ("gold",), None),
                        Panel("towers built", ("arrow", "cannon"), None),
                        Panel("waves fought", ("wave",), None))),
    ("planiverse.environments.crop_management.environment", "CropState"): Readings(
        crop_management, (Panel("biomass and yield, kg/ha", ("biomass", "yield"), None),
                          Panel("water used, cm", ("water used",), None))),
    ("planiverse.environments.power_grid.environment", "PowerGridState"): Readings(
        power_grid, (Panel("worst line loading, share of capacity", ("worst line loading",), "limit"),
                     Panel("lines overloaded", ("overloaded",), None))),
    ("planiverse.environments.water_network.environment", "WaterNetworkState"): Readings(
        water_network, (Panel("contaminated water delivered, %", ("contaminated",), None),
                        Panel("service, %", ("service",), None),
                        Panel("pipes closed", ("closed",), None))),
    ("planiverse.environments.flood_transport.environment", "FloodState"): Readings(
        flood_transport, (Panel("cost, M DKK", ("total", "damage", "delays", "measures"), "target"),
                          Panel("worst storm so far, mm", ("storm",), None),
                          Panel("measures in place", ("in place",), None))),
}


def readings_of(state):
    """The readings registered for a state's class, or None: then it renders as text."""
    cls = type(state)
    return READINGS.get((cls.__module__, cls.__name__))
