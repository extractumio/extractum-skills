#!/usr/bin/env python3
"""Wind rose (ERA5) + CAMS monthly air quality / pollen for a point, via Open-Meteo (no key).

Usage: python fetch_meteo.py --lat 41.18 --lon -8.68 --start 2023-01-01 --end 2025-12-31 --out WORKDIR/meteo.json [--global]

--global  use CAMS global domain (outside Europe; no pollen). Default: auto (Europe bbox check).
Output: {wind:{year,DJF,MAM,JJA,SON:{n,calm,bins[16][4]}}, cams:{var:{YYYY-MM:mean}}, cams_cell:[lat,lon]}
Wind bins: 16 sectors (N first, clockwise) x speed classes <10,10-20,20-30,>=30 km/h, % of hours. Calm <2 km/h.
Stdlib only.
"""
import argparse, json, urllib.request
from collections import defaultdict

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "eco-exposure-map/1.0"}), timeout=300) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True); ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", default="2023-01-01"); ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", required=True); ap.add_argument("--global", dest="glob", action="store_true")
    a = ap.parse_args()
    europe = -25 < a.lon < 45 and 34 < a.lat < 72 and not a.glob
    w = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={a.lat}&longitude={a.lon}&start_date={a.start}&end_date={a.end}&hourly=wind_speed_10m,wind_direction_10m&timezone=auto")
    h = w["hourly"]; seasons = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11], "year": list(range(1, 13))}
    # meteorological seasons by months (not hemisphere-aware): label them by months in the UI
    rose = {}
    for s, ms in seasons.items():
        bins = [[0] * 4 for _ in range(16)]; n = calm = 0
        for i, ts in enumerate(h["time"]):
            if int(ts[5:7]) not in ms: continue
            sp, d = h["wind_speed_10m"][i], h["wind_direction_10m"][i]
            if sp is None or d is None: continue
            n += 1
            if sp < 2: calm += 1; continue
            bins[int(((d + 11.25) % 360) // 22.5)][0 if sp < 10 else 1 if sp < 20 else 2 if sp < 30 else 3] += 1
        rose[s] = {"n": n, "calm": round(calm / max(n, 1) * 100, 1), "bins": [[round(x / max(n, 1) * 100, 2) for x in b] for b in bins]}
    vars_ = "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide,dust"
    extra = ",ammonia,formaldehyde,non_methane_volatile_organic_compounds,total_elementary_carbon,sea_salt_aerosol,secondary_inorganic_aerosol"
    if europe: vars_ += ",alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen"
    dom = "cams_europe" if europe else "cams_global"
    base_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={a.lat}&longitude={a.lon}&start_date={a.start}&end_date={a.end}&domains={dom}&timezone=auto&hourly="
    try:
        q = get(base_url + vars_ + extra)   # extra species exist for cams_europe; fall back if the domain lacks them
        if q.get("error"): raise ValueError(q.get("reason"))
    except Exception as e:
        print("extra CAMS species unavailable:", e); q = get(base_url + vars_)
    hq = q["hourly"]; m = defaultdict(lambda: defaultdict(list))
    for i, ts in enumerate(hq["time"]):
        for k in hq:
            if k != "time" and hq[k][i] is not None: m[k][ts[:7]].append(hq[k][i])
    cams = {k: {ym: round(sum(v) / len(v), 2) for ym, v in sorted(mm.items()) if len(v) > 24 * 15} for k, mm in m.items()}
    json.dump({"wind": rose, "cams": cams, "cams_cell": [q.get("latitude"), q.get("longitude")], "wind_cell": [w.get("latitude"), w.get("longitude")], "domain": dom}, open(a.out, "w"))
    print("wrote", a.out, "domain", dom, "wind year sectors %:", [round(sum(b), 1) for b in rose["year"]["bins"]])

if __name__ == "__main__":
    main()
