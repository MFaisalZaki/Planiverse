"""Export a MAAT city as a Planiverse flood instance.

Runs inside a checkout of https://github.com/MLSM-at-DTU/floods_transport_rl, with its
dependencies installed (climada, osmnx, geopandas), which this repository does not need and
does not declare:

    python /path/to/planiverse/tools/export_maat_city.py --zones IndreBy --out indreby.json

The environment it writes is loaded with

    env = FloodTransportEnv()
    env.set_instance(json.load(open("indreby.json")))

MAAT's zones, their trips and their roads become the instance's zones; MAAT's zone graph
becomes its edges, with the share of each edge that lies in either zone; and the depth its
design storm leaves in a zone is the mean water depth over the zone's road segments, which is
what MAAT itself aggregates to a zone. The storms, the horizon and the period are added
here and can be edited in the file.

Not run as part of this repository's tests, since it needs MAAT's data.
"""
import argparse
import json
import sys

from planiverse.environments.flood_transport.environment import DESIGN_STORM, sample_rain
from planiverse.environments.generation import rng


def export(world, years, period, rain, seed):
    zones = []
    exposures = world.exposures["network"].gdf
    for _, zone in world.taz.iterrows():
        roads = exposures[exposures["region_id"] == zone["zoneid"]]
        km = {road_type: 0.0 for road_type in ("motorway", "trunk", "primary", "secondary",
                                                "tertiary", "other")}
        signalled, trunk_like = 0.0, 0.0
        for _, road in roads.iterrows():
            highway = road["highway"] if isinstance(road["highway"], list) else [road["highway"]]
            kind = next((name for name in ("motorway", "trunk", "primary", "secondary", "tertiary")
                         if any(str(tag).startswith(name) for tag in highway)), "other")
            km[kind] += float(road["length"]) / 1000
            if kind in ("motorway", "trunk"):
                trunk_like += float(road["length"])
                if road.get("traffic_signals") == "yes":
                    signalled += float(road["length"])
        depth = float(roads["water_depth"].mean()) if len(roads) and "water_depth" in roads else 0.0
        zones.append({
            "id": str(zone["zoneid"]),
            "x": float(zone["centroid"][0]) / 1000, "y": float(zone["centroid"][1]) / 1000,
            "supply": int(zone["supply"]) if zone["supply"] == zone["supply"] else 0,
            "demand": int(zone["demand"]) if zone["demand"] == zone["demand"] else 0,
            "roads": {kind: round(value, 3) for kind, value in km.items()},
            "signalled": round(signalled / trunk_like, 3) if trunk_like else 0.5,
            "design_depth": round(max(0.0, depth), 3),
        })
    edges = []
    for a, b, data in world.transport_network.edges(data=True):
        edges.append([str(a), str(b), round(float(data["distance"]), 4),
                      round(float(data[a]), 4), round(float(data[b]), 4)])
    random_, seed = rng(seed)
    storms = ([DESIGN_STORM] * years if rain == "design"
              else [sample_rain(random_, year) for year in range(years)])
    return {"zones": zones, "edges": edges, "rain": storms, "years": years, "period": period,
            "measures": ["elevate1"], "slack": 0.02, "seed": seed, "source": "MAAT"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zones", default="IndreBy", help="MAAT's city_zones")
    parser.add_argument("--years", type=int, default=77)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--rain", choices=("design", "klimaatlas"), default="design")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import maat.world as world_module

    world = world_module.BasicEnvironment(city_zones=args.zones, assets_types=["network"])
    world.reset()
    world.step(0)             # one storm, so the segments carry their water depths
    instance = export(world, args.years, args.period, args.rain, args.seed)
    with open(args.out, "w") as handle:
        json.dump(instance, handle, indent=1)
    print(f"wrote {args.out}: {len(instance['zones'])} zones, {len(instance['edges'])} edges")


if __name__ == "__main__":
    sys.exit(main())
