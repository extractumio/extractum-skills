#!/usr/bin/env python3
"""Fetch an OSM vector basemap + pollution-relevant OSM features for a bbox.

Usage:
  python fetch_basemap.py --bbox S W N E [--core S W N E] --out WORKDIR/base.json

--bbox  full analysis extent (roads, land use, water, coastline, tanks...)
--core  smaller extent where building footprints are included (keeps size down;
        default = 3x3 km around bbox centre)
Output base.json: {layers:{land,water,park,green,industrial,port,commercial,beach,
cemetery,building,pier,runway,rail,metro,river,road:{motorway..minor}}, tanks:[...],
named_ind:[{name,kind,c}], osm_pois:[...]}  coordinates as [lat,lon] rounded to 1e-5.
Requires: shapely (pip install shapely).
"""
import argparse, json, sys, time, urllib.parse, urllib.request
from shapely.geometry import LineString, Polygon, box, Point
from shapely.ops import linemerge, polygonize, unary_union

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = "eco-exposure-map/1.0 (personal environmental research)"

def overpass(q):
    for url in OVERPASS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:
            print("overpass failed", url, e, file=sys.stderr); time.sleep(5)
    sys.exit("Overpass unavailable")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("S", "W", "N", "E"))
    ap.add_argument("--core", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, W, N, E = a.bbox
    if a.core: cs, cw, cn, ce = a.core
    else:
        cy, cx = (S + N) / 2, (W + E) / 2; cs, cn, cw, ce = cy - .015, cy + .015, cx - .02, cx + .02
    bb = f"({S},{W},{N},{E})"; pad = f"({S-.1},{W-.1},{N+.1},{E+.1})"
    q = f"""[out:json][timeout:240];(
 way["highway"~"^(motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|residential|unclassified|living_street|pedestrian)$"]{bb};
 way["railway"~"^(rail|light_rail|subway|tram)$"]{bb};
 way["natural"="coastline"]{pad};
 way["waterway"~"^(river|stream|canal)$"]{bb};
 way["natural"="water"]{bb}; relation["natural"="water"]{bb};
 way["leisure"~"^(park|garden|nature_reserve)$"]{bb};
 way["landuse"~"^(industrial|port|forest|grass|meadow|farmland|railway|commercial|retail|cemetery|landfill|quarry|brownfield)$"]{bb};
 relation["landuse"~"^(industrial|port|landfill)$"]{bb};
 way["natural"~"^(wood|beach|sand|scrub)$"]{bb};
 way["aeroway"~"^(runway|aerodrome)$"]{bb};
 way["man_made"~"^(breakwater|pier|groyne)$"]{bb};
 way["man_made"~"^(storage_tank|silo|wastewater_plant|chimney|works)$"]{bb};
 node["man_made"~"^(storage_tank|silo|chimney|works)$"]{bb};
 way["power"~"^(plant|substation|line)$"]{bb};
 way["amenity"~"^(fuel|waste_transfer_station|recycling)$"]{bb};
 way["building"]({cs},{cw},{cn},{ce});
);out geom qt;"""
    d = overpass(q)
    BB = box(W, S, E, N)
    def enc(g, tol):
        g = g.simplify(tol, preserve_topology=False); out = []
        for x in getattr(g, "geoms", [g]):
            if x.is_empty: continue
            cs_ = x.coords if x.geom_type == "LineString" else (x.exterior.coords if x.geom_type == "Polygon" else [])
            if cs_: out.append([[round(c[1], 5), round(c[0], 5)] for c in cs_])
        return out
    L = {k: [] for k in ["water","park","green","industrial","port","commercial","beach","cemetery","landfill","building","pier","runway","rail","metro","river"]}
    roads = {k: [] for k in ["motorway","trunk","primary","secondary","tertiary","minor"]}
    tanks, coast, named, pois = [], [], [], []
    for e in d["elements"]:
        t = e.get("tags", {})
        if e["type"] == "node":
            pois.append({"lat": e["lat"], "lon": e["lon"], "kind": t.get("man_made"), "name": t.get("name"), "content": t.get("content") or t.get("substance"), "operator": t.get("operator")}); continue
        if e["type"] == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"] if p]) for m in e.get("members", []) if m.get("role") == "outer" and "geometry" in m]
            if not lines: continue
            u = unary_union(lines); u = linemerge(u) if u.geom_type == "MultiLineString" else u
            tgt = "water" if t.get("natural") == "water" else ("landfill" if t.get("landuse") == "landfill" else ("port" if t.get("landuse") == "port" else "industrial"))
            for p in polygonize(u):
                L[tgt] += enc(p.intersection(BB), 4e-5)
                if t.get("name") and tgt != "water": named.append({"name": t["name"], "kind": t.get("landuse"), "c": [round(p.centroid.y, 5), round(p.centroid.x, 5)]})
            continue
        c = [(p["lon"], p["lat"]) for p in e.get("geometry", []) if p]
        if len(c) < 2: continue
        if t.get("natural") == "coastline": coast.append(LineString(c)); continue
        closed = c[0] == c[-1] and len(c) >= 4
        hw = t.get("highway")
        if hw:
            k = hw.replace("_link", ""); k = k if k in roads else "minor"
            roads[k] += enc(LineString(c), 2e-5); continue
        if t.get("railway"):
            L["metro" if t["railway"] in ("light_rail", "subway", "tram") else "rail"] += enc(LineString(c), 2e-5); continue
        if t.get("waterway"): L["river"] += enc(LineString(c), 3e-5); continue
        if t.get("man_made") in ("breakwater", "pier", "groyne"): L["pier"] += enc(Polygon(c) if closed else LineString(c), 2e-5); continue
        if t.get("man_made") in ("storage_tank", "silo") and closed:
            p = Polygon(c); tanks.append([round(p.centroid.y, 5), round(p.centroid.x, 5), round((p.area ** .5) * 111000 * .56, 1), t.get("content") or t.get("substance") or ("silo" if t.get("man_made") == "silo" else None)]); continue
        if t.get("aeroway") == "runway": L["runway"] += enc(LineString(c), 2e-5); continue
        if t.get("man_made") in ("wastewater_plant", "works", "chimney") or t.get("power") in ("plant", "substation") or t.get("amenity"):
            P = Polygon(c) if closed else LineString(c)
            pois.append({"lat": round(P.centroid.y, 5), "lon": round(P.centroid.x, 5), "kind": t.get("man_made") or t.get("power") or t.get("amenity"), "name": t.get("name"), "operator": t.get("operator"), "voltage": t.get("voltage")})
            if t.get("power") != "line": continue
            continue
        if not closed: continue
        P = Polygon(c)
        if not P.is_valid: P = P.buffer(0)
        if t.get("building"):
            if P.area > 2e-9: L["building"] += enc(P, 1.5e-5)
            continue
        lu, nat, le = t.get("landuse"), t.get("natural"), t.get("leisure")
        k = ("water" if nat == "water" else "park" if le in ("park", "garden", "nature_reserve") else
             "green" if lu in ("forest", "grass", "meadow", "farmland") or nat in ("wood", "scrub") or t.get("aeroway") == "aerodrome" else
             "industrial" if lu in ("industrial", "railway", "quarry", "brownfield") else "port" if lu == "port" else
             "landfill" if lu == "landfill" else "commercial" if lu in ("commercial", "retail") else
             "beach" if nat in ("beach", "sand") else "cemetery" if lu == "cemetery" else None)
        if not k: continue
        if k in ("park", "green") and P.area * 1.2e10 < 600: continue
        L[k] += enc(P.intersection(BB), 3e-5)
        if k in ("industrial", "port", "landfill") and t.get("name"): named.append({"name": t["name"], "kind": lu, "c": [round(P.centroid.y, 5), round(P.centroid.x, 5)]})
    # land polygon: OSM coastline has land on its LEFT side
    if coast:
        BB2 = box(W - .1, S - .1, E + .1, N + .1)
        u = unary_union(coast); m = (linemerge(u) if u.geom_type == "MultiLineString" else u).intersection(BB2)
        polys = list(polygonize(unary_union([m, BB2.boundary])))
        probes = []
        for ln in getattr(m, "geoms", [m]):
            cs_ = list(ln.coords)
            for i in range(0, len(cs_) - 1, max(1, len(cs_) // 20)):
                (x1, y1), (x2, y2) = cs_[i], cs_[i + 1]; dx, dy = x2 - x1, y2 - y1; n = (dx * dx + dy * dy) ** .5 or 1
                probes.append(Point((x1 + x2) / 2 - dy / n * 2e-5, (y1 + y2) / 2 + dx / n * 2e-5))
        land = [p for p in polys if sum(p.contains(q) for q in probes) > 0]
        land_g = unary_union(land).intersection(box(W - .02, S - .02, E + .02, N + .02)) if land else box(W, S, E, N)
    else:
        land_g = box(W - .02, S - .02, E + .02, N + .02)
    L["land"] = enc(land_g, 3e-5); L["road"] = roads
    json.dump({"bbox": [S, W, N, E], "layers": L, "tanks": tanks, "named_ind": named, "osm_pois": pois}, open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out, {k: len(v) for k, v in L.items() if isinstance(v, list)}, "tanks", len(tanks), "named", len(named), "pois", len(pois), "coastline", bool(coast))

if __name__ == "__main__":
    main()
