#!/usr/bin/env python3
"""Bundle config + model outputs + research JSON into ONE self-contained HTML page.

Usage: python build_page.py WORKDIR/config.json
Writes WORKDIR/<slug>.html (artifact body: no <html>/<head>, for claude.ai Artifact publish)
   and WORKDIR/<slug>_standalone.html (full document; open locally or host anywhere).

Optional per-project overrides in config:
  heat:   {key:{n,unit,stops|bands,ticks,desc,sea}}  merged over DEFAULT_HEAT (drop a key by setting it to null)
  cats / icats: category dictionaries {key:{n,c}}
  refs:   reference lines per pollutant {NO2:[{v,l,c}],...}
  strings: UI string overrides (see template/app.js STR)  -> translate the UI here
"""
import json, os, sys, base64, io, re
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pollutants import POLLUTANTS, BY_KEY, GROUPS
TPL = os.path.join(HERE, "..", "template")
cfg_path = sys.argv[1]; W = os.path.dirname(os.path.abspath(cfg_path)); C = json.load(open(cfg_path)); F = C.get("files", {})
# research agents may deliver named roads / port as separate files
if F.get("roads_named") and not C.get("traffic", {}).get("named"):
    C.setdefault("traffic", {})["named"] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(cfg_path)), F["roads_named"])))
if F.get("port") and not C.get("port"):
    C["port"] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(cfg_path)), F["port"])))
def load(k, default=None):
    p = F.get(k); p = os.path.join(W, p) if p else None
    return json.load(open(p)) if p and os.path.exists(p) else default

IDX = [[0, '#3E9C6B', 0], [15, '#7DB36A', .35], [30, '#D6CF5E', .55], [45, '#EFA13C', .68], [60, '#D9532F', .78], [80, '#7A1F4F', .85]]
NO2S = [[8, '#F7EBB0', 0], [12, '#F4DB7A', .35], [16, '#F2B24F', .55], [20, '#E7822F', .65], [25, '#D2502B', .72], [30, '#A82C3A', .8], [40, '#5E1A4A', .85]]
END_BANDS = lambda b0: [[b0 + 5 * i, c] for i, c in enumerate(['#B9D66A', '#E8D84A', '#F2A33A', '#E86A2B', '#D2302E', '#9B1D4E', '#40408F'])]
DEFAULT_HEAT = {
 "index_m": {"n": "Load index: NO₂ + noise", "unit": "0–100", "stops": IDX, "ticks": [0, 20, 40, 60, 80], "desc": "50% NO₂ (model, 10→40 µg/m³) + 50% Lden (47→75 dB). Only measured/calibrated factors. Relative comparison, not a legal standard."},
 "index": {"n": "Extended index (+ odour, dust, risk)", "unit": "0–100", "stops": IDX, "ticks": [0, 20, 40, 60, 80], "desc": "Adds expert-judgement layers (odour, dust, major-hazard proximity). Use for comparison only."},
 "no2": {"n": "NO₂ annual mean (model)", "unit": "µg/m³", "stops": NO2S, "ticks": [10, 20, 30, 40], "sea": True, "desc": "Background + roads (AADT) + port + stacks, calibrated on monitoring stations (leave-one-out validated). WHO 2021: 10; EU 2030 limit: 20; EU current limit: 40."},
 "no2hi": {"n": "NO₂ — high shipping scenario", "unit": "µg/m³", "stops": NO2S, "ticks": [10, 20, 30, 40], "sea": True, "desc": "Port contribution scaled to the literature uplift near ports (+ model RMSE). Upper bound."},
 "lden": {"n": "Noise Lden", "unit": "dB(A)", "bands": END_BANDS(45), "desc": "Day-evening-night level. WHO road-traffic guideline 53 dB; typical legal limits 55–65 dB."},
 "ln": {"n": "Night noise Lnight", "unit": "dB(A)", "bands": END_BANDS(40), "desc": "Night level (23–07). WHO road guideline 45 dB, aircraft 40 dB."},
 "odour": {"n": "Odour (estimate)", "unit": "rel.", "stops": [[.05, '#C9B458', 0], [.2, '#B39A36', .4], [.4, '#8E7424', .6], [.7, '#5E4A16', .75]], "ticks": [0, .25, .5, .75, 1], "desc": "Wind-skewed kernels around odour sources (WWTPs, food processing, landfills, rivers). Qualitative."},
 "dust": {"n": "Dust / PM (estimate)", "unit": "rel.", "stops": [[.05, '#D8C3A0', 0], [.2, '#C19A62', .45], [.4, '#9E6E35', .62], [.7, '#6B4420', .78]], "ticks": [0, .25, .5, .75, 1], "desc": "Bulk terminals, scrap, demolition, concrete, mills, foundries. Qualitative."},
 "risk": {"n": "Industrial major-hazard proximity", "unit": "rel.", "stops": [[.05, '#E9A7A0', 0], [.3, '#DB6A5E', .35], [.6, '#C4392F', .5], [.9, '#8A1C2A', .62]], "ticks": [0, .25, .5, .75, 1], "desc": "Decay over typical consequence distances (tank fire / spill). Indicative; official emergency-plan zones differ."},
}
DEFAULT_CATS = {"fuel": {"n": "Oil, fuel, gas", "c": "#D1495B"}, "chem": {"n": "Chemicals, industrial gases", "c": "#7B4FB0"}, "port": {"n": "Port & shipping", "c": "#1F7A99"},
 "waste": {"n": "Waste, sewage, rivers", "c": "#5E8C31"}, "food": {"n": "Food processing", "c": "#C98B1A"}, "industry": {"n": "Industry", "c": "#9A5B34"}, "transport": {"n": "Transport, logistics", "c": "#56616A"}}
DEFAULT_ICATS = {"air": {"n": "Air", "c": "#C05A2E"}, "odour": {"n": "Odour", "c": "#8A7A1E"}, "water": {"n": "Water", "c": "#1F7A99"}, "noise": {"n": "Noise", "c": "#7B4FB0"},
 "risk": {"n": "Accidents, risk", "c": "#C4392F"}, "soil": {"n": "Soil", "c": "#7A5A3A"}, "health": {"n": "Health", "c": "#2F8F5B"}}
RC = ["#2F8F5B", "#D08B1C", "#C4392F", "#7B4FB0"]
DEFAULT_REFS = {p["key"]: [{"v": r["v"], "l": r["l"], "c": RC[min(i, 3)]} for i, r in enumerate(p["refs"])] for p in POLLUTANTS}

def norm_unit(u, default):
    u = str(u or "")
    if u in ("", "nan", "None"): return default
    return {"ug.m-3": "µg/m³", "ng.m-3": "ng/m³", "mg.m-3": "mg/m³", "ug/m3": "µg/m³", "ng/m3": "ng/m³", "mg/m3": "mg/m³", "count.cm-3": "1/cm³"}.get(u, u)
def cat_of(s):
    c = (s.get("category", "") + " " + s.get("name", "")).lower()
    for k, rx in [("fuel", r"refiner|fuel|oil|lpg|lng|tank|petrol|gas station|jet"), ("chem", r"chemic|paint|gases|hazardous|pharma|pesticid"), ("port", r"^port|dredg|shipyard|cruise|harbour|harbor"),
                  ("waste", r"wastewater|sewage|outfall|landfill|incinerat|compost|waste|river|stream|wwtp"), ("transport", r"road|motorway|highway|rail|airport|logistic|depot"),
                  ("food", r"cannery|brewery|sugar|flour|meat|fish|dairy|food")]:
        if re.search(rx, c): return k
    return "industry"

grids = load("grids"); base = load("base"); met = load("meteo", {}) or {}
nx, ny = grids["grid"]["nx"], grids["grid"]["ny"]
for k, g in grids["grids"].items():
    a = np.frombuffer(base64.b64decode(g["d"]), np.uint8).reshape(ny, nx); b = io.BytesIO()
    Image.fromarray(a, "L").save(b, "PNG", optimize=True); g["d"] = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
def enc(r):
    out = []; pa = pb = 0
    for a, b in r:
        ia, ib = round(a * 1e5), round(b * 1e5); out += [ia - pa, ib - pb]; pa, pb = ia, ib
    return out
B = {}
for k, v in base["layers"].items():
    B[k] = {c: [enc(x) for x in lines] for c, lines in v.items()} if k == "road" else [enc(x) for x in v if len(x) >= 2]
B["tanks"] = base.get("tanks", [])
src = load("sources", []) or []
for s in src:
    s.setdefault("cat", cat_of(s)); s.setdefault("impacts", s.pop("pollutants_impacts", []) if "pollutants_impacts" in s else [])
    if "seveso_tier" in s and "hazard_tier" not in s: s["hazard_tier"] = s.pop("seveso_tier")
    s["line"] = bool(re.match(r"^road", s.get("category", "")))
def gj_lines(key, tol=5e-5):
    """GeoJSON (lines/polygons, any nesting) -> [{t:'l'|'s', v, n, g, ll}] simplified (shapely if available)."""
    g = load(key); out = []
    if not g: return out
    try:
        from shapely.geometry import shape
    except ImportError:
        shape = None
    def walk(geom):
        t = geom["type"]
        if t == "GeometryCollection":
            for x in geom["geometries"]: yield from walk(x)
        elif t.startswith("Multi"):
            for c in geom["coordinates"]: yield from walk({"type": t[5:], "coordinates": c})
        else: yield geom
    for f in g["features"]:
        p = f.get("properties", {})
        for geom in walk(f["geometry"]):
            if geom["type"] not in ("LineString", "Polygon"): continue
            if shape:
                sh = shape(geom).simplify(tol)
                if sh.is_empty or (geom["type"] == "Polygon" and sh.area < 2e-8): continue
                coords = list(sh.exterior.coords if geom["type"] == "Polygon" else sh.coords)
            else:
                coords = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"]
            out.append({"t": "s" if geom["type"] == "Polygon" else "l", "v": p.get("voltage_V") or p.get("voltage"), "n": p.get("name"), "g": p.get("gridcode", p.get("class", 4)), "ll": [[round(c[1], 5), round(c[0], 5)] for c in coords]})
    return out
heat = {k: v for k, v in DEFAULT_HEAT.items() if k in grids["grids"]}
SEQ = ['#F4E3B5', '#E9B96A', '#D9803A', '#C04A30', '#8E2438', '#521A45']
EMI = [[.05, '#D9CBE8', 0], [.3, '#B79AD6', .35], [1, '#8C62BD', .55], [3, '#5E3A99', .7], [10, '#35205E', .8]]
LT = {"calibrated_n": "{label} (calibrated model)",
      "calibrated_desc": "Concentration field calibrated on {n} stations, leave-one-out RMSE ≈ {rmse} {unit}. {refs}",
      "emissions_n": "{label} — emissions-based potential",
      "emissions_desc": "NOT a measured concentration. Absolute scale: 1 = one facility releasing exactly the EU reporting threshold of this pollutant, 1 km away (wind-rose weighted, generic stack kernel). Comparable between places and pollutants; not calibrated against monitoring.",
      "metals_label": "Heavy metals (sum, threshold-weighted)", "metals_extra": " Sum over As, Cd, Cr, Cu/Zn, Hg, Ni, Pb, each divided by its reporting threshold. Hg is mostly gaseous and travels far: local kernel understates its reach."}
LT.update(C.get("layer_texts") or {})
PL = (C.get("strings") or {}).get("pollutants", {})            # {key: translated label}
label_of = lambda k: PL.get(k, BY_KEY[k]["label"])
for k, g in grids["grids"].items():
    if k.startswith("c_"):
        P_ = BY_KEY[k[2:]]; lo, hi = g["lo"], g["hi"]; st = [[lo + (hi - lo) * i / 5, c, .15 + .13 * i] for i, c in enumerate(SEQ)]; st[0][2] = 0
        fm = grids.get("field_meta", {}).get(k, {})
        heat[k] = {"n": LT["calibrated_n"].format(label=label_of(k[2:])), "unit": P_["unit"], "stops": st, "ticks": [round(lo + (hi - lo) * i / 4, 2) for i in range(5)], "sea": True,
                   "desc": LT["calibrated_desc"].format(n=len(fm.get("calib", [])), rmse=fm.get("loo_rmse"), unit=P_["unit"], refs="; ".join(r["l"] for r in P_["refs"]))}
    elif k.startswith("e_"):
        lab = LT["metals_label"] if k == "e_metals" else label_of(k[2:])
        heat[k] = {"n": LT["emissions_n"].format(label=lab), "unit": LT.get("emissions_unit", "× threshold @1 km"), "stops": EMI, "ticks": [0, 1, 3, 10],
                   "desc": LT["emissions_desc"] + (LT["metals_extra"] if k == "e_metals" else "")}
for k, v in (C.get("heat") or {}).items():
    if v is None: heat.pop(k, None)
    else: heat[k] = {**heat.get(k, {}), **v}
cfg_ui = {k: C.get(k) for k in ["title", "subtitle", "lang", "home", "areas", "radius_m", "center", "zoom", "strings", "findings", "summaryRows", "summaryIntro", "summaryNote",
          "method_html", "sources_html", "water_html", "windNote", "airIntro", "defaultHeat", "defaultStation", "years", "camsYears", "lastYearPartial", "overlays", "hazardRings", "probe", "probeIndex", "decimal", "attribution"] if C.get(k) is not None}
cfg_ui.setdefault("hazardRings", C.get("hazard", {"upper": [250, 750], "lower": [150, 400]}))
cfg_ui.setdefault("defaultHeat", "index_m" if "index_m" in heat else next(iter(heat)))
cfg_ui.setdefault("summaryRows", [
  {"label": "NO₂ µg/m³ (range)", "key": "no2", "fmt": "range", "warn": 15, "bad": 20},
  {"label": "Noise Lden mean, dB", "key": "lden_mean", "warn": 53, "bad": 60},
  {"label": "Area with Lden ≥ 65, %", "key": "lden65", "digits": 0, "warn": 3, "bad": 10},
  {"label": "Area with Ln ≥ 55, %", "key": "ln55", "digits": 0, "warn": 3, "bad": 10},
  {"label": "Index NO₂ + noise", "key": "index_m", "digits": 0, "warn": 22, "bad": 35, "bold": True},
  {"label": "Odour 0–1 (estimate)", "key": "odour", "digits": 2, "warn": .15, "bad": .35},
  {"label": "Dust 0–1 (estimate)", "key": "dust", "digits": 2, "warn": .12, "bad": .3},
  {"label": "Major-hazard proximity 0–1", "key": "risk", "digits": 2, "warn": .15, "bad": .5},
  {"label": "Extended index", "key": "index", "digits": 0, "warn": 22, "bad": 35}])
data = {"config": cfg_ui, "base": B, "grid": grids["grid"], "grids": grids["grids"], "heat": heat, "stats": grids["stats"], "calib": grids["calib"], "loo": grids["loo"], "rmse": grids["rmse"],
        "src": src, "cats": C.get("cats") or DEFAULT_CATS, "icats": C.get("icats") or DEFAULT_ICATS, "inc": load("incidents", []) or [], "aq": load("stations", []) or [],
        "refs": C.get("refs") or DEFAULT_REFS, "cams": met.get("cams"), "wind": met.get("wind"), "bath": load("bath", []) or [], "polys": load("polys", []) or [],
        "roads": [{"name": r["name"], "aadt": r.get("aadt"), "ll": r["ll"]} for r in C.get("traffic", {}).get("named", []) if r.get("ll")],
        "pollutants": [{**{k_: p[k_] for k_ in ("key", "group", "unit", "notes")}, "label": label_of(p["key"])} for p in POLLUTANTS], "groups": {**GROUPS, **((C.get("strings") or {}).get("groups") or {})},
        "coverage": [{**c, "label": label_of(c["key"]), "notes": ((C.get("strings") or {}).get("pollutant_notes") or {}).get(c["key"], c["notes"])} for c in grids.get("coverage", [])], "prtrCodes": {p["key"]: p["prtr"] for p in POLLUTANTS}, "field_meta": grids.get("field_meta", {}),
        "emitters": [{"id": f["id"], "name": f["name"], "lat": f["lat"], "lon": f["lon"], "latest_year": f["latest_year"], "activity": f.get("activity"),
                      "air": f["air"]} for f in (load("prtr", {}) or {}).get("facilities", []) if f.get("air")],
        "power": gj_lines("power"), "coast": [x for x in gj_lines("coast") if (x.get("g") or 0) >= 4], "lines": load("lines", []) or []}
for st_ in data["aq"]:
    st_["units"] = {k_: norm_unit((st_.get("units") or {}).get(k_), BY_KEY.get(k_, {}).get("unit", "")) for k_ in st_.get("pollutants", [])}
for i in data["inc"]:
    i.setdefault("y", int(str(i.get("d", "2000"))[:4]) if str(i.get("d", ""))[:4].isdigit() else 2000)
    if i.get("c") not in data["icats"]: i["c"] = next(iter(data["icats"]))
js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
css = open(os.path.join(TPL, "leaflet.css")).read() + "\n" + open(os.path.join(TPL, "app.css")).read()
shell = open(os.path.join(TPL, "shell.html")).read()
body = (shell.replace("{{TITLE}}", C["title"]).replace("{{DESC}}", C.get("subtitle", "")).replace("{{CSS}}", css)
        .replace("{{DATA}}", "window.ECO=" + js + ";").replace("{{APP}}", open(os.path.join(TPL, "app.js")).read()))
# privacy guard: refuse to write a page that contains any forbidden string (home street, house number, own name ...)
bad = [w for w in (C.get("privacy") or {}).get("forbid", []) if w and w.lower() in body.lower()]
if bad:
    for w in bad:
        i = body.lower().index(w.lower()); print(f"PRIVACY: '{w}' found: …{body[max(0, i-80):i+60]}…", file=sys.stderr)
    sys.exit("build aborted by privacy.forbid — scrub the input files (sources.json descriptions/keys, incidents, config text) and rebuild")
slug = C.get("slug") or re.sub(r"[^a-z0-9]+", "_", C["title"].lower()).strip("_")[:40] or "ecomap"
open(os.path.join(W, slug + ".html"), "w").write(body)
open(os.path.join(W, slug + "_standalone.html"), "w").write(f'<!doctype html><html lang="{C.get("lang","en")}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">' + body.replace('<div id="app">', '</head><body><div id="app">', 1) + "</body></html>")
print("wrote", slug + ".html", round(len(body) / 1e6, 2), "MB; layers:", list(heat), "; sources", len(src), "incidents", len(data["inc"]), "stations", len(data["aq"]))
