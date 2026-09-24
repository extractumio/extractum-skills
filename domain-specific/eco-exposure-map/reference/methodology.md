# Methodology, thresholds, sanity checks

## Grid
- Regular lat/lon grid, `cell_m` (25 m default), over `bbox`.
- Values are quantised to uint8 per layer (`lo..hi`) and packed as a grayscale PNG. The page decodes them to colour the overlay and to answer point-probe clicks.

## NO2 screening model (model.py)
The concentration is built as:

```
NO2 = B + a·(E_land ⊗ K1) + b·(E_land ⊗ K2) + p·PPH
```

- **B**: regional background. Use the lowest suburban or coastal background station nearby (e.g. 8–12 µg/m³ in W. Europe).
- **E_land**: road emissions plus point stacks.
  - Roads: AADT × EF × cell length. EF is about 0.4 g NOx per vehicle-km for the EU fleet; raise it for old or diesel-heavy fleets.
  - OSM motorway and trunk roads are drawn per carriageway, so they get a per-carriageway AADT. Named official-AADT centrelines override them.
  - Stacks: tall stacks get only 5–20% of their emission as an effective ground-level source.
- **Kernels**:
  - K1, near-road: exp(−r/120)/(r+30), cut at 1.5 km.
  - K2, urban accumulation: exp(−r/2500), normalised, cut at 4 km.
- **PPH**: the port field.
  - Emissions = NOx t/yr × ground fraction 0.35, spread over the port polygon and the berth hotspots.
  - Dispersed with a wide kernel (ships' stacks are high), using the same a and b.
- **Fitting**:
  - a, b and p are fitted with NNLS on stations with NO2 inside the bbox (mean of the calibration years).
  - Leave-one-out (LOO): refit without each station and predict it. The LOO RMSE is the model error.
  - If fewer than 3 stations are available, the model is uncalibrated: it uses `default_coef` and must be stated as such.
- **Ranges shown to the user**:
  - lo = central − RMSE.
  - hi = roads + max(central port, literature scenario) + RMSE. The literature scenario is the port term scaled so that NO2 in a 500 m ring around the port is +50% (`literature_uplift`, Nunes et al. ACP 2020 for Iberian ports).

**Why the port gets its own parameter and scenario.** When the physical port term is plugged in unscaled, it can overshoot a station 1.5 km inland by 10+ µg/m³. The stations therefore constrain the port contribution downward. Yet no station sits at the berths, so the upper scenario must remain visible.

**Limits (always state them):**
- no street canyons, heavy-duty share, diurnal cycle, chemistry or stack heights;
- narrow busy streets can be much higher than the model;
- year-to-year variability at stations often exceeds the model error.

## Noise
- **Official mode** (preferred): strategic noise contours rasterised with their band centre (55–59 → 57). Airport maps are combined by taking the maximum.
- **Modelled mode** (fallback only):
  - L(10 m) = 10·log10(AADT) + 27 − 15·log10(d/10).
  - Energetic sum over AADT buckets; Lnight ≈ Lden − 8.
  - About ±5 dB. It ignores barriers and buildings, and there are no industrial or port sources.
- **Missing coverage**: leave it empty (NaN), never fill it with a mean.

## Heuristic layers (label them as estimates)
- **Odour and dust**: Σ w·exp(−d/R)·wind_factor.
  - R = 450 m for odour, 350 m for dust. Cut at 2.5 km and 2 km.
  - wind_factor = 0.35 + 0.65·16·f(sector the wind must come FROM), using the annual rose.
  - Weights w (0–1) are expert judgement, based on complaints, size and type.
- **Major-hazard proximity**: 1 inside r0, falling linearly to 0 at r1.
  - Upper tier: 250/750 m. Lower tier: 150/400 m, scaled by 0.6.
  - These are typical thermal-radiation distances for tank or pool fires. They are **not** official emergency-planning zones; LPG or toxic sites can reach further. Say which applies.


## Pollutant coverage: every pollutant, always

`scripts/pollutants.py` lists about 24 pollutants in 5 groups: gases, particles, heavy metals, organics, dioxins/POPs. Each has units, reference values, EEA codes, PRTR codes and a CAMS name. For each one, `model.py` assigns exactly one status:

Precedence: field > stations > emissions > below_threshold > historic > background > none / not_assessed.
Measurements always outrank emission estimates. The emission layer is still attached as a secondary layer (`layer2`).

| Status | Condition | What the page shows |
|---|---|---|
| `field` | ≥3 stations inside the bbox with valid recent years (hourly/daily ≥50% coverage; variable samplers ≥10 samples). A **nested** leave-one-out error is computed: predictors (intercept + subset of road-local, road-urban, facilities, coast, dust sources, port) are chosen inside every fold. More than 1 predictor is allowed only with ≥5 stations. Accept if RMSE < 0.9 × the "same value everywhere" baseline, ≤ 35% of the mean, and the field varies more than its error | concentration map `c_<key>` + `field_meta` |
| `stations` | measured nearby, but no valid field | values in the table; `note_fit` says why no map |
| `emissions` | operating facilities released it above the register threshold in the last 5 years | `e_<key>` on an **absolute** scale: Σ (kg/yr ÷ EU reporting threshold) × K(d) × wind weight, where K(1 km) = 1, K ∝ exp(−(d−1 km)/1 km) · 1150/(d+150 m), cut at 10 km. So 1.0 = "one facility at exactly the threshold, 1 km away". Not a concentration |
| `below_threshold` | operating facilities reported it earlier, but not in the last 5 years | "below reporting threshold since Y" — **not zero** |
| `historic` | only facilities no longer in the register (closed OR below all thresholds), or monitoring that stopped | period + last emitters |
| `background` | only CAMS | CAMS mean |
| `none` | the station network AND the register were queried and found nothing | explicit "no data" |
| `not_assessed` | nothing was queried for this country (e.g. a non-EU place before agents filled stations/prtr) | "not assessed" — never "no data" |

Facility status comes from `fetch_prtr_eu.py`:
- `reporting`;
- `not_reporting`: absent while its country kept reporting;
- `unknown_after_register_end`: the country's register stops, e.g. UK after 2020. These are treated as operating, and the UK run must use national registers for recent years.

Per-facility values above `SUSPICIOUS_KG` get the flag "check source data" (e.g. dioxins > 2 g TEQ/yr).

Heavy metals get an extra combined layer, `e_metals`: the sum of each metal's absolute emission potential. Each metal is divided by its EU reporting threshold (As 20, Cd 10, Cr 100, Cu/Zn 100, Hg 10, Ni 50, Pb 200 kg/yr), which roughly tracks regulatory significance. Hg is mostly gaseous and long-range, so the local kernel understates its reach; say so.

Why so strict:
- PM10, O3 and metals are dominated by regional background, sea salt, dust or a few point sources. A field that cannot beat "same value everywhere" in leave-one-out would be invented detail.
- Emission layers are shown because they locate the risk, but they are never labelled as concentrations.
- Absence of monitoring is itself a result: "heavy metals not measured here since 2009" tells the user what to ask for.

Writing about toxics:
- **Metals, PAH and dioxins:**
  - give the emitter, the reported kg/yr with its year, the distance and whether it still operates;
  - give the last year and nearest location of any measurement;
  - name who could measure it: regional air agency, operator's stack monitoring, soil survey.
- **Units:** dioxins are reported in kg TEQ. 0.0135 kg = 13.5 g TEQ/yr is large; typical modern incinerators report < 0.1 g TEQ/yr.
- **Historic industry:** a closed refinery or smelter means soil and groundwater legacy (check the contaminated-sites register), not current air exposure.

## Indices
- **`index_m`**: 50% NO2 (10→40) + 50% Lden (47→75). Measured and calibrated factors only; this is the headline.
- **`index`** (extended): 35/35/12/8/10 NO2/noise/odour/dust/risk. Shown as a secondary layer.
- Always report whether the ranking of areas is the same under both.

## Reference values (edit `config.refs` per jurisdiction)
| Pollutant | WHO 2021 | EU until 2029 | EU from 2030 (Dir. 2024/2881) | US NAAQS |
|---|---|---|---|---|
| NO2 | annual 10, 24h 25 | annual 40, 1h 200 (18×) | annual 20 | annual 53 ppb (~100 µg/m³), 1h 100 ppb |
| PM10 | annual 15, 24h 45 | annual 40, 24h 50 (35×) | annual 20 | 24h 150 |
| PM2.5 | annual 5, 24h 15 | annual 25 | annual 10 | annual 9, 24h 35 |
| O3 | peak season 60, 8h 100 | target 120 (8h, 25 d/yr) | | 8h 70 ppb |
| SO2 | 24h 40 | 1h 350, 24h 125 | | 1h 75 ppb |
| Noise | WHO road Lden 53, Ln 45; aircraft Lden 45, Ln 40; rail 54/44 | national limits (e.g. PT mixed zones Lden 65 / Ln 55) | | |

## Claims: how to write findings
1. Quote the numbers from stats or sources with a range. Give no two-digit precision for modelled values.
2. Do not attribute a source share unless it is computed (roads / port / background from stats). Otherwise write "near X", not "X is the main source".
3. **Wind:** compute the bearing home→source and quote the frequency of the wind blowing FROM that bearing, for the season you name. Say which sources are upwind in which season. Strong winds disperse; the worst episodes are calm and stable nights.
4. **Hazard sites:** give the distance, tier, substances and the typical consequence distance. Frame it as emergency preparedness, not daily exposure.
5. Declare omissions, e.g. "the source search focused on X; zeros in Y may partly reflect that."
6. Give the user something actionable: which station or dataset would test the model best, and whom to ask for it.

## Review checklist (for you and the reviewer)
- [ ] Are any hand-fixed scalings left in the model, and does the page text describe what the code actually does?
- [ ] Calibration: number of stations, LOO RMSE, stations excluded and why; ranges shown instead of false precision.
- [ ] The traffic numbers in the model match those in the page text.
- [ ] Wind claims were checked against the actual bearings and the seasonal sector frequencies.
- [ ] Hazard kernel radii equal the drawn rings, and the rings are labelled as indicative.
- [ ] The index ranking is robust with and without the heuristic layers.
- [ ] Noise has no NaN-filled areas; the different map vintages are noted.
- [ ] Every incident has a URL; there are no private individuals; approximate coordinates are flagged.
- [ ] Every registry pollutant appears with a status; no pollutant silently dropped; emissions layers never called concentrations.
- [ ] Closed facilities treated as historic; stations with 0–49% coverage not used.
- [ ] Gaps and "who holds the missing data" are stated.
