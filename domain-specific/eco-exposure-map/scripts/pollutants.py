"""Pollutant registry shared by fetch_eea_aq.py, fetch_prtr_eu.py, model.py and build_page.py.

Every pollutant the analysis must ALWAYS report on (measured, modelled, emissions-only or "no data").
Fields:
  key        stable id used in files and grids ("pm10", "ni", ...)
  label      display name (English; translate via config.strings.pollutants)
  group      gases | particles | metals | organics | dioxins
  unit       concentration unit used for station data (µg/m³ or ng/m³ or mg/m³)
  eea        EEA/EIONET vocabulary codes (http://dd.eionet.europa.eu/vocabulary/aq/pollutant/<code>)
  prtr       E-PRTR / EU Industrial Reporting pollutant codes (air releases)
  cams       Open-Meteo CAMS variable (background model), if any
  kind       predictors for a calibrated field: traffic | mixed | industrial | regional
  refs       reference values [{v, l, src}] in `unit`, annual unless stated
  ref_main   the value used to normalise emission-based layers / toxicity weights (same unit)
  notes      why it matters / typical sources
"""

WHO, EU, EU30, US = "WHO 2021", "EU 2008/50·2004/107", "EU 2024/2881 (2030)", "US NAAQS"

POLLUTANTS = [
 # ---- gases ----
 dict(key="no2", label="NO₂", group="gases", unit="µg/m³", eea=[8], prtr=["NOX"], cams="nitrogen_dioxide", kind="traffic",
      refs=[dict(v=10, l="WHO 10", src=WHO), dict(v=20, l="EU-2030 20", src=EU30), dict(v=40, l="EU 40", src=EU)], ref_main=20,
      notes="Traffic, shipping, combustion. Main modelled field (special traffic+port model)."),
 dict(key="so2", label="SO₂", group="gases", unit="µg/m³", eea=[1], prtr=["SOX"], cams="sulphur_dioxide", kind="industrial",
      refs=[dict(v=40, l="WHO 24h 40", src=WHO), dict(v=20, l="EU-2030 annual 20", src=EU30)], ref_main=20,
      notes="Refineries, ships (fuel sulphur), power plants, smelters."),
 dict(key="co", label="CO", group="gases", unit="mg/m³", eea=[10], prtr=["CO"], cams="carbon_monoxide", kind="traffic",
      refs=[dict(v=4, l="WHO 24h 4", src=WHO), dict(v=10, l="EU 8h 10", src=EU)], ref_main=4,
      notes="Traffic, incomplete combustion. CAMS given in µg/m³ (divide by 1000)."),
 dict(key="o3", label="O₃", group="gases", unit="µg/m³", eea=[7], prtr=[], cams="ozone", kind="regional",
      refs=[dict(v=60, l="WHO peak-season 60", src=WHO), dict(v=100, l="WHO 8h 100", src=WHO), dict(v=120, l="EU target 8h 120", src=EU)], ref_main=60,
      notes="Secondary, regional; spatially smooth — no local field."),
 dict(key="nh3", label="NH₃", group="gases", unit="µg/m³", eea=[], prtr=["NH3"], cams="ammonia", kind="industrial",
      refs=[], ref_main=8, notes="Agriculture, waste incinerators (SNCR slip), composting. PM2.5 precursor."),
 dict(key="h2s", label="H₂S", group="gases", unit="µg/m³", eea=[], prtr=[], cams=None, kind="industrial",
      refs=[dict(v=7, l="WHO odour 30-min 7", src=WHO)], ref_main=7, notes="WWTPs, landfills, refineries — odour."),
 dict(key="hcl_hf", label="HCl / HF", group="gases", unit="µg/m³", eea=[], prtr=["CHLORINEANDINORGANICCOMPOUNDS", "FLUORINEANDINORGANICCOMPOUNDS", "HCL", "HF"], cams=None, kind="industrial",
      refs=[], ref_main=None, notes="Incinerators, glass/ceramics, aluminium. Rarely monitored."),
 # ---- particles ----
 dict(key="pm10", label="PM10", group="particles", unit="µg/m³", eea=[5], prtr=["PM10"], cams="pm10", kind="mixed",
      refs=[dict(v=15, l="WHO 15", src=WHO), dict(v=20, l="EU-2030 20", src=EU30), dict(v=40, l="EU 40", src=EU)], ref_main=15,
      notes="Traffic (wear), industry, bulk handling, construction, sea salt, Saharan dust."),
 dict(key="pm25", label="PM2.5", group="particles", unit="µg/m³", eea=[6001], prtr=["PM2_5", "PM2.5"], cams="pm2_5", kind="mixed",
      refs=[dict(v=5, l="WHO 5", src=WHO), dict(v=10, l="EU-2030 10", src=EU30), dict(v=25, l="EU 25", src=EU), dict(v=9, l="US 9", src=US)], ref_main=5,
      notes="Largest health burden. Combustion, secondary aerosol, shipping."),
 dict(key="bc", label="Black carbon / EC", group="particles", unit="µg/m³", eea=[391, 771, 772], prtr=[], cams="total_elementary_carbon", kind="traffic",
      refs=[], ref_main=1, notes="Diesel, ships, wood burning. WHO: measure it; no numeric guideline."),
 # ---- metals (in PM10) ----
 dict(key="as", label="Arsenic (As)", group="metals", unit="ng/m³", eea=[5018, 18], prtr=["ASANDCOMPOUNDS"], cams=None, kind="industrial",
      refs=[dict(v=6, l="EU target 6", src=EU)], ref_main=6, notes="Smelters, coal/oil combustion, glass, refineries."),
 dict(key="cd", label="Cadmium (Cd)", group="metals", unit="ng/m³", eea=[5014, 14], prtr=["CDANDCOMPOUNDS"], cams=None, kind="industrial",
      refs=[dict(v=5, l="EU target 5", src=EU), dict(v=5, l="WHO 5", src=WHO)], ref_main=5, notes="Metal processing, incineration, batteries."),
 dict(key="ni", label="Nickel (Ni)", group="metals", unit="ng/m³", eea=[5015, 15], prtr=["NIANDCOMPOUNDS"], cams=None, kind="industrial",
      refs=[dict(v=20, l="EU target 20", src=EU)], ref_main=20, notes="Heavy fuel oil (ships, refineries), steel."),
 dict(key="pb", label="Lead (Pb)", group="metals", unit="ng/m³", eea=[5012, 12], prtr=["PBANDCOMPOUNDS"], cams=None, kind="industrial",
      refs=[dict(v=500, l="EU/WHO 500", src=EU), dict(v=150, l="US 3-mo 150", src=US)], ref_main=500, notes="Smelters, battery recycling, aviation gasoline, legacy soil."),
 dict(key="hg", label="Mercury (Hg)", group="metals", unit="ng/m³", eea=[4813, 653, 5013], prtr=["HGANDCOMPOUNDS"], cams=None, kind="industrial",
      refs=[dict(v=1000, l="WHO 1000", src=WHO)], ref_main=1000, notes="Coal combustion, cement, crematoria, chlor-alkali."),
 dict(key="cr", label="Chromium (Cr)", group="metals", unit="ng/m³", eea=[5016, 16], prtr=["CRANDCOMPOUNDS"], cams=None, kind="industrial",
      refs=[], ref_main=25, notes="Steel, plating, refineries. Cr(VI) is carcinogenic; no EU limit."),
 dict(key="cu_zn", label="Copper / Zinc", group="metals", unit="ng/m³", eea=[5073, 5063], prtr=["CUANDCOMPOUNDS", "ZNANDCOMPOUNDS"], cams=None, kind="industrial",
      refs=[], ref_main=None, notes="Brake/tyre wear, foundries, galvanising."),
 # ---- organics ----
 dict(key="benzene", label="Benzene", group="organics", unit="µg/m³", eea=[20], prtr=["BENZENE"], cams=None, kind="traffic",
      refs=[dict(v=1.7, l="WHO ref 1.7", src=WHO), dict(v=3.4, l="EU-2030 3.4", src=EU30), dict(v=5, l="EU 5", src=EU)], ref_main=1.7,
      notes="Petrol vapour (fuel depots, stations), traffic, refineries."),
 dict(key="toluene_btex", label="Toluene / xylenes", group="organics", unit="µg/m³", eea=[21, 431, 464, 482], prtr=["TOLUENE", "XYLENES", "ETHYLBENZENE"], cams=None, kind="industrial",
      refs=[dict(v=260, l="WHO toluene weekly 260", src=WHO)], ref_main=260, notes="Solvents, printing, fuel storage."),
 dict(key="nmvoc", label="NMVOC (total)", group="organics", unit="µg/m³", eea=[], prtr=["NMVOC"], cams="non_methane_volatile_organic_compounds", kind="industrial",
      refs=[], ref_main=None, notes="Solvent use, fuel handling, printing. Ozone precursor."),
 dict(key="hcho", label="Formaldehyde", group="organics", unit="µg/m³", eea=[], prtr=["FORMALDEHYDE"], cams="formaldehyde", kind="industrial",
      refs=[dict(v=100, l="WHO 30-min 100", src=WHO)], ref_main=100, notes="Wood panels, combustion, secondary formation."),
 dict(key="bap", label="Benzo[a]pyrene (PAH)", group="organics", unit="ng/m³", eea=[5029, 29], prtr=["PAHS", "BENZO(A)PYRENE"], cams=None, kind="industrial",
      refs=[dict(v=1, l="EU target 1", src=EU), dict(v=0.12, l="WHO ref 0.12", src=WHO)], ref_main=1, notes="Wood burning, coke/steel, asphalt, diesel."),
 # ---- dioxins etc. ----
 dict(key="dioxins", label="Dioxins/furans (PCDD/F)", group="dioxins", unit="fg TEQ/m³", eea=[], prtr=["PCDD+PCDF(DIOXINS+FURANS)", "PCDD+PCDF"], mass_unit="g TEQ", cams=None, kind="industrial",
      refs=[], ref_main=None, notes="Incinerators, metal recycling, open burning. Almost never monitored in ambient air."),
 dict(key="pcb_hcb", label="PCBs / HCB", group="dioxins", unit="pg/m³", eea=[], prtr=["PCBS", "HCB"], cams=None, kind="industrial",
      refs=[], ref_main=None, notes="Legacy industry, metal recycling, incineration."),
]

# EU E-PRTR Annex II release thresholds to AIR (kg/yr): a facility only reports a pollutant above this.
# "Absent from the register" therefore means "below threshold", NOT zero. Used as the absolute scale of emission layers
# (1.0 = one facility releasing exactly the threshold, 1 km away).
PRTR_THRESHOLD_KG = {'no2': 100000, 'so2': 150000, 'co': 500000, 'nh3': 10000, 'hcl_hf': 10000, 'pm10': 50000, 'pm25': None, 'as': 20, 'cd': 10, 'ni': 50, 'pb': 200, 'hg': 10, 'cr': 100, 'cu_zn': 100, 'benzene': 1000, 'toluene_btex': None, 'nmvoc': 100000, 'hcho': None, 'bap': 50, 'dioxins': 0.0001, 'pcb_hcb': 0.1}
# per-facility values that are implausibly high for most activities -> flag "check source data"
SUSPICIOUS_KG = {'dioxins': 0.002}
for _p in POLLUTANTS:
    _p["prtr_threshold_kg"] = PRTR_THRESHOLD_KG.get(_p["key"])
    _p["suspicious_kg"] = SUSPICIOUS_KG.get(_p["key"])
# names that must NOT be mapped to air-concentration keys (deposition, precipitation, other size fractions)
EXCLUDE_NAME = r"(?i)deposit|precip|wet|dry dep|in PM2\.5|in PM1\b|in TSP|soil|water"
BY_KEY = {p["key"]: p for p in POLLUTANTS}
EEA_CODE_TO_KEY = {c: p["key"] for p in POLLUTANTS for c in p["eea"]}
PRTR_TO_KEY = {c: p["key"] for p in POLLUTANTS for c in p["prtr"]}
GROUPS = {"gases": "Gases", "particles": "Particles", "metals": "Heavy metals", "organics": "Organic compounds", "dioxins": "Dioxins & persistent organics"}
