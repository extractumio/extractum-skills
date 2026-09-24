#!/usr/bin/env python3
"""Facility-level pollutant releases (air + water) near a place from the EU Industrial Reporting / E-PRTR database
(EEA discodata SQL API; EU-27 + Iceland, Liechtenstein, Norway, Switzerland, Serbia; UK only up to 2020).

Usage: python3 fetch_prtr_eu.py --bbox S W N E --out WORK/prtr.json [--margin-km 5] [--since 2017]
Output prtr.json:
 {"facilities":[{"id","name","lat","lon","activity","latest_year",
                 "air":{POLLUTANT:{"kg":..,"year":..,"method":"M|C|E"}}, "water":{...},
                 "history":{POLLUTANT:{year:kg}}}],
  "by_key": {registry_key: [{"id","name","kg","year"}]}, "source": url}
Registry keys come from pollutants.py (e.g. "ni", "dioxins", "nmvoc"). Pollutants not in the registry are kept under
their PRTR code. Stdlib only. Outside Europe: US EPA TRI (Envirofacts), Canada NPRI, Australia NPI — same schema.
"""
import argparse, json, math, os, sys, urllib.parse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollutants import PRTR_TO_KEY
SQL = "https://discodata.eea.europa.eu/sql"

def q(sql, n=20000):
    url = SQL + "?" + urllib.parse.urlencode({"query": sql, "p": 1, "nrOfHits": n})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "eco-exposure-map/1.0"}), timeout=300) as r:
        d = json.load(r)
    if d.get("errors"): sys.exit(f"discodata error: {d['errors']}")
    return d.get("results", [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--margin-km", type=float, default=10); ap.add_argument("--since", type=int, default=2017)
    a = ap.parse_args(); S, W, N, E = a.bbox
    dl = a.margin_km / 111.0; dlo = a.margin_km / (111.0 * math.cos(math.radians((S + N) / 2)))
    S, N, W, E = S - dl, N + dl, W - dlo, E + dlo
    rows = q(f"""SELECT f.localId, f.cc, f.facilityName, f.x_4326 AS lon, f.y_4326 AS lat, f.act, r.reportingYear AS yr, p.mediumCode AS med, p.pollutant AS pol,
      p.totalPollutantQuantityKg AS kg, p.methodCode AS meth
      FROM [IED].[latest].[PollutantRelease] p
      JOIN [IED].[latest].[ProductionFacilityReport] r ON p.facilityReportId = r.Id
      JOIN (SELECT localId, MAX(facilityName) facilityName, MAX(x_4326) x_4326, MAX(y_4326) y_4326, MAX(EPRTRAnnexIMainActivity) act, MAX(countryCode) cc
            FROM [IED].[latest].[ProductionFacility] WHERE y_4326 BETWEEN {S} AND {N} AND x_4326 BETWEEN {W} AND {E} GROUP BY localId) f
        ON f.localId = r.localId
      WHERE r.reportingYear >= {a.since}""")
    fac = {}
    for x in rows:
        f = fac.setdefault(x["localId"], {"id": x["localId"], "name": (x["facilityName"] or "").strip(), "lat": round(x["lat"], 5), "lon": round(x["lon"], 5),
                                           "activity": x["act"], "country": x.get("cc"), "latest_year": 0, "air": {}, "water": {}, "history": {}})
        med = "air" if x["med"] == "AIR" else "water" if x["med"] == "WATER" else None
        if not med or x["kg"] is None: continue
        f["latest_year"] = max(f["latest_year"], x["yr"])
        if med == "air": f["history"].setdefault(x["pol"], {})[str(x["yr"])] = x["kg"]
        cur = f[med].get(x["pol"])
        if not cur or x["yr"] > cur["year"]: f[med][x["pol"]] = {"kg": x["kg"], "year": x["yr"], "method": x["meth"]}
    by_key = {}
    for f in fac.values():
        for pol, v in f["air"].items():
            by_key.setdefault(PRTR_TO_KEY.get(pol, pol), []).append({"id": f["id"], "name": f["name"], "kg": v["kg"], "year": v["year"], "pollutant": pol})
    for k in by_key: by_key[k].sort(key=lambda r: -r["kg"])
    # last reporting year of the whole register per country: "not_reporting" = absent while its country kept reporting (closed OR below all thresholds)
    # while its country kept reporting (UK data end in 2020 -> UK facilities are "unknown after 2020", not closed)
    ccs = sorted({f["country"] for f in fac.values() if f.get("country")})
    last = {}
    if ccs:
        for r in q("SELECT countryCode AS cc, MAX(reportingYear) AS y FROM [IED].[latest].[ProductionFacilityReport] WHERE countryCode IN (" + ",".join(f"'{c}'" for c in ccs) + ") GROUP BY countryCode"):
            last[r["cc"]] = r["y"]
    for f in fac.values():
        ly = last.get(f.get("country"))
        f["register_last_year"] = ly
        f["status"] = "not_reporting" if ly and f["latest_year"] < ly - 2 else ("unknown_after_register_end" if ly and ly < int(__import__("time").strftime("%Y")) - 3 else "reporting")
    out = {"facilities": sorted(fac.values(), key=lambda f: f["name"]), "by_key": by_key, "register_last_year": last,
           "source": "EEA Industrial Reporting (E-PRTR successor), discodata [IED].[latest]", "since": a.since}
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print("wrote", a.out, len(fac), "facilities; air pollutants:", {k: len(v) for k, v in sorted(by_key.items())})

if __name__ == "__main__":
    main()
