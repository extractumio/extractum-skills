# Data schema

All coordinates are WGS84. Point lists use `[lat, lon]`, except GeoJSON, which uses `[lon, lat]`.
All files sit in `$WORK`, and `config.files` references them by relative path.

## config.json
```jsonc
{
  "title": "Riverside Eco Map", "subtitle": "Air, noise, water, hazards · data as of 2026-09",
  "slug": "riverside_ecomap", "lang": "en", "decimal": ".",
  "bbox": [S, W, N, E], "cell_m": 25,
  "home": {"name": "12 Example St", "lat": 0, "lon": 0},
  "areas": [{"k": "home", "name": "Example St", "short": "Home", "lat": 0, "lon": 0}, {"k": "b", "name": "Old Town", "short": "Old Town", "lat": 0, "lon": 0}],
  "radius_m": 400, "center": [lat, lon], "zoom": 14,
  "files": {
    "base": "base.json", "meteo": "meteo.json", "grids": "grids.json",
    "stations": "stations.json", "sources": "sources.json", "incidents": "incidents.json",
    "bath": "bath.json", "polys": "polys.json", "lines": "lines.json",
    "power": "power.geojson", "coast": "coast.geojson",
    "roads_named": "roads_named.json", "port": "port.json",
    "noise_lden": ["noise_lden.geojson"], "noise_ln": ["noise_ln.geojson"],
    "noise_lden_extra": ["airport_lden.geojson"], "noise_ln_extra": ["airport_ln.geojson"]
  },
  "noise": {"mode": "official" /* or "modelled" */, "category_field": "category", "k_lden": 27},
  "traffic": {"ef_g_per_vkm": 0.4, "class_aadt": {"motorway": 30000, "trunk": 12000, "primary": 12000, "secondary": 6000, "tertiary": 3000, "minor": 400},
              "named": [{"name": "A28", "aadt": 94300, "year": 2024, "ll": [[lat, lon], ...]}]},
  "port": {"polygon": [[lat, lon], ...], "nox_t_yr": 1600, "ground_frac": 0.35, "berths": [[lat, lon, 0.2]], "literature_uplift": 0.5},
  "stacks": [[lat, lon, effective_ground_t_per_yr]],
  "no2": {"background": 10.5, "calib_years": ["2024", "2025"], "exclude": ["STATIONCODE"], "default_coef": [650, 820], "default_port_scale": 0.06, "default_rmse": 3},
  "odour": [[lat, lon, weight0to1, "label"]], "odour_R": 450,
  "dust": [[lat, lon, weight0to1, "label"]], "dust_R": 350,
  "hazard": {"upper": [250, 750], "lower": [150, 400]},
  "privacy": {"forbid": ["<home street>", "<house number + street>"]},   // build aborts if any string appears in the page
  "current_year": 2026,                                           // for "recent" windows; default = today
  "index": {"measured": {"no2": 0.5, "lden": 0.5}, "extended": {"no2": 0.35, "lden": 0.35, "odour": 0.12, "dust": 0.08, "risk": 0.10}},

  // page content (written after the model has run; in the user's language)
  "summaryIntro": "…", "summaryNote": "The ranking is the same without heuristic layers (index 37 vs 20 vs 17).",
  "summaryRows": [{"label": "NO₂ µg/m³ (range)", "key": "no2", "fmt": "range", "warn": 15, "bad": 20}, {"label": "…", "key": "lden65", "digits": 0, "warn": 3, "bad": 10}],
  "findings": [{"h": "Key findings for Example St"}, {"level": "bad|warn|good", "title": "Bold lead sentence.", "text": "Detail with numbers and sources."}, {"p": "Plain paragraph"}],
  "windNote": "…", "airIntro": "…", "method_html": "<h2>…</h2><p class='small muted'>…</p>", "sources_html": "…", "water_html": "…",
  "defaultStation": "PT01030", "years": ["2023", "2024", "2025"], "camsYears": ["2023", "2024", "2025"], "lastYearPartial": true,
  "strings": {"tabs": {"sum": "Summary", "pol": "Pollutants"}, "pollutants": {"ni": "Nickel"}, "groups": {"metals": "Heavy metals"}, "massUnits": ["kt","t","kg","g","mg"]},
                                                                  // UI translation overrides (all keys: STR in template/app.js)
  "layer_texts": {"calibrated_n": "{label} (calibrated model)", "emissions_n": "{label} — emissions-based potential", "emissions_desc": "…", "metals_label": "…"},
  "pol_html": "<h2>What is missing and who has it</h2>…",           // extra text under the pollutant table
  "heat": {"no2": {"desc": "…"}, "dust": null},                            // legend overrides; null = hide layer
  "cats": {…}, "icats": {…}, "refs": {"NO2": [{"v": 10, "l": "WHO 10", "c": "#2F8F5B"}]},
  "overlays": [["src", "Pollution sources", 1], ["inc", "Incidents", 1]],    // optional: order, labels and default visibility
  "attribution": "© OpenStreetMap contributors · <agencies>"
}
```

## stations.json — list (written by fetch_eea_aq.py in Europe; by agent A elsewhere)
```json
{"name": "Station name", "code": "XX01021", "lat": 0, "lon": 0, "type": "suburban background",
 "pollutants": ["no2", "o3", "pm10", "ni"], "units": {"no2": "µg/m³", "ni": "ng/m³"},
 "monthly": {"no2": {"2024": [25.2, 19.0, null, …12 values]}},
 "annual": {"no2": {"2024": {"mean": 17.7, "coverage_pct": 94, "n": 8230, "hours_gt200": 0}}},
 "period": {"no2": [1997, 2026]}, "flag": "optional data-quality note"}
```
- Pollutant keys are the registry keys from `scripts/pollutants.py`: `no2 so2 co o3 nh3 h2s pm10 pm25 bc as cd ni pb hg cr cu_zn benzene toluene_btex nmvoc hcho bap dioxins pcb_hcb`.
- `coverage_pct` is null for variable-interval samplers (metals, BaP); `n` then counts the samples.
- Exceedance keys must contain `days` or `hours`.

## aq_inventory.json — {key: {...}} (what was EVER monitored nearby)
`{"ni": {"n_points_ever": 1, "n_stations_ever": 1, "n_stations_recent": 0, "nearest_km": 11.3, "first": 2009, "last": 2009, "names": ["Ni"]}}`
A pollutant absent from the inventory has never been monitored within the bbox plus margin.

## prtr.json (fetch_prtr_eu.py in the EU; TRI / NPRI / NPI / NAEI elsewhere, same shape)
```json
{"facilities": [{"id": "PT.APA….CI", "name": "Plant", "lat": 0, "lon": 0, "activity": "2(b)", "latest_year": 2024,
                 "air": {"CDANDCOMPOUNDS": {"kg": 66.2, "year": 2023, "method": "M"}}, "water": {…}, "history": {"CDANDCOMPOUNDS": {"2021": 60}}}],
 "by_key": {"cd": [{"id": "…", "name": "Plant", "kg": 66.2, "year": 2023, "pollutant": "CDANDCOMPOUNDS"}]}}
```
- `by_key` uses registry keys; codes that are not in the registry stay under their original code.
- `latest_year` is used to decide whether the facility still operates (it must have reported within the last 3 years).

## sources.json — list
```json
{"name": "Cepsa fuel depot", "category": "fuel & bitumen storage", "lat": 41.19, "lon": -8.6793, "approx": false,
 "status": "active|closed|decommissioning", "pollutants_impacts": ["risk", "air (VOC)"],
 "quantitative": {"capacity_m3": 47000, "PRTR_2024_t": {"NOx": 12}}, "hazard_tier": "upper|lower|null",
 "description": "1–3 sentences", "sources": ["https://…"], "cat": "optional: fuel|chem|port|waste|food|industry|transport"}
```
- `cat` is inferred from `category` when missing.
- Categories starting with `road` are treated as line sources: they are listed but have no marker.
- `seveso_tier` is accepted as an alias of `hazard_tier`.

## incidents.json — list
```json
{"t": "Title", "d": "2025-07-09", "y": 2025, "loc": "Beach X", "lat": 0, "lon": 0, "approx": true,
 "c": "air|odour|water|noise|risk|soil|health", "s": 1-5, "f": "Facts with numbers", "src": "Outlet", "u": "https://…"}
```

## bath.json — list
`{"id": "XX0001", "name": "Main Beach", "lat": 0, "lon": 0, "class": {"2024": "Poor", "2025": "Poor"}, "url": "profile"}`.
Class values: `Excellent | Good | Sufficient | Poor`.

## polys.json / lines.json
- `polys.json`: `[{"name": "Port of X", "ll": [[lat, lon], …], "color": "#1F7A99", "dashed": true}]`
- `lines.json` (flight paths, pipelines, …): `[{"name": "…", "tip": "tooltip", "ll": [[lat, lon], …], "color": "#40408F", "dash": "8 6"}]`

## GeoJSON inputs
- **noise_\*.geojson**: polygon features whose `properties[category_field]` contains the band, e.g. `Lden5559`, `LdenGreaterThan75`, `55-59`, `>75`. The lower number is parsed and +2 dB is assigned as the band centre.
- **power.geojson**: LineString or Polygon features with `voltage_V` or `voltage` ("60000;15000") and `name`.
- **coast.geojson**: polygons with a `gridcode` or `class` of 1–5. Only 4 and 5 are drawn.

## Generated
- **base.json**: from `fetch_basemap.py`.
- **meteo.json**: from `fetch_meteo.py`.
- **grids.json**: from `model.py`; also `coverage` (one record per registry pollutant: status, layer, obs[], emitters[], cams_mean, rmse, note_fit) and `field_meta`, containing `stats[area_key] = {no2, no2_lo, no2_hi, bg, road, port, port_hi, lden_mean, lden65, ln55, odour, dust, risk, index_m, index}`, plus `calib`, `loo`, `rmse`, `coef`, `k_hi`, `calibrated`.
