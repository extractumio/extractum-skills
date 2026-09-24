#!/usr/bin/env python3
"""All official air-quality measurements (every pollutant) near a place — EU/EEA member & cooperating countries.

Usage:
  $PY fetch_eea_aq.py --bbox S W N E --home LAT LON --out WORK/stations.json [--margin-km 10] [--years 2023 2024 2025 2026] [--history]

What it does
  1. Downloads EEA station/sampling-point metadata (cached in ~/.cache/ecomap/) and keeps every sampling point
     within bbox + margin, for EVERY pollutant (gases, PM, metals in PM10, BaP, BTEX, NH3, H2S, BC ...).
  2. Asks the EEA download API for Parquet URLs (dataset 2 = verified E1a 2013+, 1 = up-to-date E2a, 3 = Airbase <2013
     with --history) using pollutant vocabulary URIs, downloads only the nearby sampling points.
  3. Computes monthly means (≥50% coverage for hourly/daily series; ≥2 samples for variable-interval samplers such
     as metals) and annual means with coverage %, exceedance counts for the main pollutants.
Writes
  stations.json   [{name, code, lat, lon, type, area, pollutants[], units{key:unit}, monthly{key:{year:[12]}},
                    annual{key:{year:{mean, coverage_pct, n, ...}}}, period{key:[first,last]}, flag}]
  aq_inventory.json {key: {n_points_ever, n_stations_recent, nearest_km, first, last, names[]}}  -> feeds the
                    "pollutant coverage" table (a pollutant with n_points_ever == 0 is "not monitored here").
Requires: pandas, pyarrow (pip install pandas pyarrow). Outside Europe use other sources (see reference/data-sources.md)
and write the same schema.
"""
import argparse, csv, io, json, math, os, re, sys, time, urllib.request, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollutants import BY_KEY, EEA_CODE_TO_KEY, EXCLUDE_NAME

META_URL = "https://discomap.eea.europa.eu/App/AQViewer/download?fqn=Airquality_Dissem.b2g.measurements&f=csv"
API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
CACHE = os.path.expanduser("~/.cache/ecomap"); os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "eco-exposure-map/1.0"}
NAME_RULES = [  # EEA "Air Pollutant" notation -> registry key (fallback when the code is not in the registry)
 (r"^NO2$", "no2"), (r"^SO2$", "so2"), (r"^CO$", "co"), (r"^O3$", "o3"), (r"^NH3$", "nh3"), (r"^H2S$", "h2s"),
 (r"^PM10$", "pm10"), (r"^PM2\.5$", "pm25"), (r"^(BC|EC|Black ?Carbon|EC in PM|Elemental)", "bc"),
 (r"^As\b", "as"), (r"^Cd\b", "cd"), (r"^Ni\b", "ni"), (r"^Pb\b", "pb"), (r"^Hg", "hg"), (r"^Cr\b", "cr"), (r"^(Cu|Zn)\b", "cu_zn"),
 (r"^C6H6$", "benzene"), (r"^(C6H5-CH3|C6H5-C2H5|m,p-C6H4|o-C6H4|.*xylene|toluene)", "toluene_btex"),
 (r"^(BaP|Benzo\(a\)pyrene)", "bap"), (r"^HCHO|[Ff]ormaldehyde", "hcho")]

def key_for(name, code):
    # name first: numeric vocabulary codes are easy to confuse (e.g. 38 = NO, not H2S)
    if re.search(EXCLUDE_NAME, name or ""): return None      # deposition / other size fractions are not air concentrations
    for rx, k in NAME_RULES:
        if re.search(rx, (name or "").strip()): return k
    return None

def hav(a, b, c, d):
    t = math.pi / 180; x = math.sin((c - a) * t / 2) ** 2 + math.cos(a * t) * math.cos(c * t) * math.sin((d - b) * t / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))

FAILED = []
def get(url, path=None, data=None, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={**UA, **({"Content-Type": "application/json"} if data else {})})
            with urllib.request.urlopen(req, timeout=600) as r:
                b = r.read()
            if path: open(path, "wb").write(b)
            return b
        except Exception as e:
            print("  retry", url[:90], e, file=sys.stderr); time.sleep(3 * (i + 1))
    FAILED.append(url[:160]); return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True); ap.add_argument("--home", nargs=2, type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--margin-km", type=float, default=10)
    ap.add_argument("--years", nargs="*", default=None); ap.add_argument("--history", action="store_true")
    a = ap.parse_args(); S, W, N, E = a.bbox; hl, ho = a.home
    dl = a.margin_km / 111.0; dlo = a.margin_km / (111.0 * math.cos(math.radians((S + N) / 2)))
    meta = os.path.join(CACHE, "eea_meta.zip")
    if not os.path.exists(meta) or time.time() - os.path.getmtime(meta) > 30 * 86400:
        print("downloading EEA metadata …"); get(META_URL, meta)
    z = zipfile.ZipFile(meta); f = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8", errors="replace")
    pts = {}; stations = {}; inv = {}
    for x in csv.DictReader(f):
        try: la, lo = float(x["Latitude"]), float(x["Longitude"])
        except Exception: continue
        if not (S - dl < la < N + dl and W - dlo < lo < E + dlo): continue
        spid = x["Sampling Point Id"]; m = re.search(r"_(\d+)_", spid + "_"); code = int(m.group(1)) if m else None
        k = key_for(x["Air Pollutant"], code)
        if not k: continue
        st = x["Air Quality Station EoI Code"]
        stations.setdefault(st, {"name": x["Air Quality Station Name"].strip('" '), "code": st, "lat": la, "lon": lo,
                                 "type": f'{x["Air Quality Station Area"]} {x["Air Quality Station Type"]}'.strip(), "country": x["Country"]})
        pts[spid] = (st, k, code, x["Air Pollutant"], x["Operational Activity Begin"][:10], x["Operational Activity End"][:10])
        iv = inv.setdefault(k, {"n_points_ever": 0, "stations": set(), "names": set(), "nearest_km": 1e9})
        iv["n_points_ever"] += 1; iv["stations"].add(st); iv["names"].add(x["Air Pollutant"]); iv["nearest_km"] = min(iv["nearest_km"], round(hav(hl, ho, la, lo), 1))
    print("nearby sampling points:", len(pts), "stations:", len(stations), "pollutants:", sorted(inv))
    countries = sorted({s["country"] for s in stations.values()})
    iso = {"Portugal": "PT", "Spain": "ES", "France": "FR", "Italy": "IT", "Germany": "DE", "Belgium": "BE", "Netherlands": "NL", "Austria": "AT", "Poland": "PL", "Czechia": "CZ",
           "Denmark": "DK", "Sweden": "SE", "Finland": "FI", "Ireland": "IE", "Greece": "GR", "Hungary": "HU", "Slovakia": "SK", "Slovenia": "SI", "Croatia": "HR", "Romania": "RO",
           "Bulgaria": "BG", "Luxembourg": "LU", "Malta": "MT", "Cyprus": "CY", "Estonia": "EE", "Latvia": "LV", "Lithuania": "LT", "Norway": "NO", "Switzerland": "CH",
           "Iceland": "IS", "Monaco": "MC", "Liechtenstein": "LI", "United Kingdom": "GB", "Serbia": "RS", "Albania": "AL", "North Macedonia": "MK", "Montenegro": "ME", "Bosnia and Herzegovina": "BA", "Turkey": "TR", "Türkiye": "TR", "Andorra": "AD", "Kosovo": "XK"}
    cc = sorted({iso.get(c, c[:2].upper()) if len(c) > 2 else c for c in countries})
    codes = sorted({p[2] for p in pts.values() if p[2]}); want = {p for p in pts}
    stcodes = {p[0] for p in pts.values()}
    import pandas as pd
    frames = []; ddir = os.path.join(CACHE, "eea_parquet"); os.makedirs(ddir, exist_ok=True)
    datasets = [2, 1] + ([3] if a.history else [])
    y0 = int(time.strftime("%Y"))
    # The API silently drops sampling points when the requested period is long or overruns the dataset,
    # so ask one calendar year at a time and union the URLs.
    windows = {2: [(f"{y}-01-01", f"{y}-12-31") for y in range(2013, y0)], 1: [(f"{y0-1}-01-01", time.strftime("%Y-%m-%d"))],
               3: [(f"{y}-01-01", f"{y}-12-31") for y in range(1990, 2013)]}
    code_key = {p[2]: p[1] for p in pts.values() if p[2]}
    for code in codes:
        for ds in datasets:
            urls = set()
            for start, end in windows[ds]:
                body = json.dumps({"countries": cc, "cities": [], "pollutants": [f"http://dd.eionet.europa.eu/vocabulary/aq/pollutant/{code}"], "dataset": ds, "source": "Api",
                                   "dateTimeStart": start + "T00:00:00Z", "dateTimeEnd": end + "T23:59:59Z", "aggregationType": None}).encode()
                r = get(API, data=body)
                if r: urls |= {u.strip() for u in r.decode("utf-8-sig").splitlines() if u.strip().endswith(".parquet")}
            urls = [u for u in urls if any(f"SPO-{st}_" in u for st in stcodes)]
            for u in urls:
                p = os.path.join(ddir, f"ds{ds}_" + os.path.basename(u))
                if not os.path.exists(p) or time.time() - os.path.getmtime(p) > 7 * 86400: get(u, p)
                if not os.path.exists(p): continue
                try: d = pd.read_parquet(p)
                except Exception as e: print("  bad parquet", p, e); continue
                u0 = str(d["Unit"].iloc[0]) if "Unit" in d and len(d) else ""
                if re.search(r"m-2|m2|/day|day-1|mm", u0): continue      # deposition fluxes, not concentrations
                d = d[pd.to_numeric(d["Validity"], errors="coerce") >= 1]
                d["Value"] = pd.to_numeric(d["Value"], errors="coerce"); d = d[d.Value.notna() & (d.Value > -1)]
                if not len(d): continue
                spid = d["Samplingpoint"].iloc[0].split("/")[-1]
                st = spid.replace("SPO-", "").split("_")[0]
                k = next((pp[1] for sp, pp in pts.items() if pp[0] == st and pp[2] == code), code_key.get(code))
                if not k: continue
                spn = next((pp[3] for sp_, pp in pts.items() if pp[0] == st and pp[2] == code), str(code))
                d = d.assign(st=st, key=k, ds=ds, sp=spid.split("_")[0] + "_" + str(code), species=spn)
                frames.append(d[["st", "key", "sp", "species", "Start", "End", "Value", "Unit", "AggType", "ds"]])
        print(f"  code {code}: {sum(len(x) for x in frames)} rows total so far")
    out = {}
    if frames:
        d = pd.concat(frames); d["Start"] = pd.to_datetime(d["Start"])
        # one species/sampling series per station and key (never average toluene with xylene, or Cu with Zn):
        pick = d.groupby(["st", "key", "species"]).size().reset_index(name="n").sort_values("n").drop_duplicates(["st", "key"], keep="last")
        d = d.merge(pick[["st", "key", "species"]], on=["st", "key", "species"])
        d = d.sort_values("ds").drop_duplicates(["st", "key", "Start"], keep="first")
        for (st, k), g in d.groupby(["st", "key"]):
            s = out.setdefault(st, {**stations.get(st, {"name": st, "code": st}), "pollutants": [], "units": {}, "monthly": {}, "annual": {}, "period": {}})
            g = g.set_index("Start").sort_index(); v = g.Value; agg = str(g.AggType.iloc[0]).lower()
            s.setdefault("species", {})[k] = str(g.species.iloc[0]); s["pollutants"].append(k); s["units"][k] = {"ug.m-3": "µg/m³", "ng.m-3": "ng/m³", "mg.m-3": "mg/m³"}.get(str(g.Unit.iloc[0]), BY_KEY[k]["unit"] if str(g.Unit.iloc[0]) in ("nan", "None", "") else str(g.Unit.iloc[0])); s["period"][k] = [int(v.index.min().year), int(v.index.max().year)]
            mon = v.resample("MS").agg(["mean", "count"])
            per = mon.index.days_in_month * (24 if agg.startswith("hour") else 1)
            ok = (mon["count"] / per >= .5) if (agg.startswith("hour") or agg.startswith("day")) else (mon["count"] >= 2)
            for ts, row in mon.iterrows():
                if row["count"] == 0: continue
                s["monthly"].setdefault(k, {}).setdefault(str(ts.year), [None] * 12)[ts.month - 1] = round(float(row["mean"]), 2) if ok[ts] else None
            for y, gy in v.groupby(v.index.year):
                if a.years and str(y) not in a.years and y < 2013: continue
                full = (366 if y % 4 == 0 else 365) * (24 if agg.startswith("hour") else 1)
                e = {"mean": round(float(gy.mean()), 2), "n": int(len(gy)), "coverage_pct": round(100 * len(gy) / full) if (agg.startswith("hour") or agg.startswith("day")) else None}
                if k == "no2" and agg.startswith("hour"): e["hours_gt200"] = int((gy > 200).sum())
                if k == "so2" and agg.startswith("hour"): e["hours_gt350"] = int((gy > 350).sum()); e["max_hour"] = round(float(gy.max()), 1)
                if k in ("pm10", "pm25"):
                    dd = gy.resample("D").mean() if agg.startswith("hour") else gy
                    e["days_gt50" if k == "pm10" else "days_gt15_WHO"] = int((dd > (50 if k == "pm10" else 15)).sum())
                if k == "o3" and agg.startswith("hour"):
                    r8 = gy.rolling("8h", min_periods=6).mean(); e["days_8h_gt120"] = int((r8.resample("D").max() > 120).sum())
                s["annual"].setdefault(k, {})[str(y)] = e
    res = sorted(out.values(), key=lambda s: hav(hl, ho, s.get("lat", hl), s.get("lon", ho)))
    json.dump(res, open(a.out, "w"), ensure_ascii=False)
    yr = int(time.strftime("%Y"))
    invo = {}
    for k, iv in inv.items():
        recent = [s["code"] for s in res if k in s["pollutants"] and s["period"][k][1] >= yr - 3]
        per = [s["period"][k] for s in res if k in s.get("period", {})]
        invo[k] = {"n_points_ever": iv["n_points_ever"], "n_stations_ever": len(iv["stations"]), "n_stations_recent": len(recent), "nearest_km": iv["nearest_km"],
                   "first": min((p[0] for p in per), default=None), "last": max((p[1] for p in per), default=None), "names": sorted(iv["names"]),
                   "note": None if per else "sampling points listed in metadata but no data files returned"}
    invo["_meta"] = {"source": "EEA", "countries": cc, "failed_requests": len(FAILED), "failed_examples": FAILED[:5],
                     "queried_keys": sorted(BY_KEY), "note": "keys absent from this file were queried and never monitored nearby"}
    json.dump(invo, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "aq_inventory.json"), "w"), ensure_ascii=False, indent=1)
    if FAILED: print(f"WARNING: {len(FAILED)} requests failed — coverage may be incomplete (see aq_inventory.json _meta)")
    print("wrote", a.out, len(res), "stations; inventory:", {k: (v["n_stations_recent"], v["first"], v["last"]) for k, v in invo.items() if k != "_meta"})

if __name__ == "__main__":
    main()
