# Where to find the data

For each topic, work down the list and use the first source that covers the place. Always prefer
official measured data, then official modelled data, then peer-reviewed studies, then press and NGOs.
Record the URL for every number.

## Global, works everywhere
| Need | Source | Notes |
|---|---|---|
| Basemap, industrial areas, tanks, silos, WWTPs, chimneys, power lines, landfills | OpenStreetMap Overpass `https://overpass-api.de/api/interpreter` | `scripts/fetch_basemap.py`. Tags: `man_made=storage_tank/silo/wastewater_plant/chimney/works`, `landuse=industrial/port/landfill/quarry/brownfield`, `power=plant/substation/line` |
| Geocoding | Nominatim `nominatim.openstreetmap.org/search` | send a User-Agent; at most 1 request per second |
| Wind and weather history | Open-Meteo archive (ERA5) | `scripts/fetch_meteo.py` |
| Modelled air quality (CAMS) and pollen | Open-Meteo air-quality API (`cams_europe` 0.1°, `cams_global` 0.4°) | background context, not street level; pollen only for Europe |
| Satellite NO2 | Sentinel-5P TROPOMI (Copernicus Browser, Google Earth Engine `COPERNICUS/S5P/OFFL/L3_NO2`) | qualitative hotspot check |
| Station measurements (fallback) | OpenAQ v3 `api.openaq.org` (free key), WAQI `aqicn.org` (key) | check provenance; many stations are low-cost sensors |
| Facility emissions (global) | Climate TRACE `climatetrace.org` (asset-level), Global Energy Monitor trackers (plants, refineries, terminals, steel) | mostly CO2 and coal/oil/gas; good for locating assets |
| Ports | port authority annual reports (ship calls, TEU, cargo), EMSA (EU), IMO GHG studies | per-port NOx is rarely published; use a regional inventory |
| Airports | airport noise action plans and contours, ACI/airport traffic statistics | runway heading and the share of landings per runway set the approach corridor |
| Radon | national maps; EU: JRC European Indoor Radon Map | |
| Flood and coastal hazard | JRC Global Flood Maps, national flood viewers | |
| Peer-reviewed studies | Google Scholar / ScienceDirect: "<city> port emissions", "<city> air pollution health" | source-apportionment studies are valuable |

## European Union and EEA countries
| Topic | Source |
|---|---|
| Air measurements, ALL pollutants (E1a verified 2013+, E2a recent, Airbase 1990–2012) | **`scripts/fetch_eea_aq.py`** does it all. Notes: metadata `https://discomap.eea.europa.eu/App/AQViewer/download?fqn=Airquality_Dissem.b2g.measurements&f=csv`; POST `…/ParquetFile/urls` with pollutant **URIs** `http://dd.eionet.europa.eu/vocabulary/aq/pollutant/<code>` (names fail for metals). The API silently returns nothing when the period overruns a dataset or is long, so query **one year at a time**. Map pollutants by **name**, not code (e.g. 38 = NO, not H2S) |
| Air-quality assessment reports | national or regional agencies: PT CCDR/APA QualAr, ES MITECO + regional networks, FR Atmo AASQA regions, DE UBA + Länder, IT ARPA regions, NL RIVM/Luchtmeetnet, PL GIOŚ |
| Industrial emissions (all pollutants, air + water) | **`scripts/fetch_prtr_eu.py`**: EEA discodata SQL `https://discodata.eea.europa.eu/sql`, tables `[IED].[latest].[ProductionFacility]` (x_4326, y_4326) ⋈ `[ProductionFacilityReport]` (localId, reportingYear) ⋈ `[PollutantRelease]` (pollutant, mediumCode, totalPollutantQuantityKg). National PRTRs add sub-threshold detail (PT APA PRTR, ES PRTR-España, FR Géorisques/IREP, DE thru.de) |
| Major-hazard (Seveso III) sites | national list with upper/lower tier: PT APA "estabelecimentos abrangidos DL 150/2015", ES Protección Civil, FR Géorisques (ICPE Seveso), DE Länder "Störfallbetriebe", IT MASE "inventario stabilimenti"; accident history in eMARS/eSPIRS (JRC) |
| Noise (END strategic maps, Lden/Lnight contours) | EEA noise data; national viewers: PT APA SNIAmb ArcGIS `sniambgeoogc.apambiente.pt/.../END_visualizador_mer/MapServer`, ES SICA, FR "cartes de bruit stratégiques" (préfectures/Cerema), DE Umgebungslärmkartierung (Länder WMS), NL Atlas Leefomgeving. Query the ArcGIS REST `/query?f=geojson&geometry=<bbox>&inSR=4326&outSR=4326` |
| Bathing water | EEA WISE bathing water (per-site classification, 4-season); national profiles (PT APA "perfis de águas balneares", ES Náyade, FR baignades.sante.gouv.fr) |
| Rivers and coastal status (WFD) | EEA WISE WFD, national River Basin Management Plans |
| Contaminated sites | national registers (FR CASIAS/SIS in Géorisques, DE Altlastenkataster, NL Bodemloket, PT APA "solos contaminados") |
| Traffic (AADT) | PT IMT "relatórios de tráfego", ES DGT/Ministerio "mapa de tráfico", FR "trafic moyen journalier annuel" (data.gouv), DE BASt Verkehrszählung, IT ANAS |
| Complaints | Ombudsman, parliamentary questions (e.g. parlamento.pt), municipal assembly minutes, petitions (peticaopublica.com, change.org), EU petitions (PETI) |

## United Kingdom
- **Air:** UK-AIR (DEFRA) data selector; London Air (LAQN); local authority Annual Status Reports (diffusion tubes!)
- **Emissions:** NAEI point sources; EA Pollution Inventory
- **Major-hazard sites:** COMAH establishments list (HSE)
- **Noise:** DEFRA strategic noise maps (Extrium England noise map)
- **Water:** EA bathing water explorer; storm overflow Event Duration Monitoring (EDM) data
- **Traffic:** DfT road traffic statistics (AADF API `roadtraffic.dft.gov.uk`)
- **Contaminated land:** local authority registers
- **Radon:** UKHSA radon map

## Metals, dioxins, PAH: where measurements usually exist
- **EU:** metals and BaP in PM10 are mandatory where levels may exceed the target values (Directive 2004/107/EC), so the EEA data are often sparse. Look also at regional agency campaigns (FR Atmo "métaux lourds" campaigns, ES regional networks, IT ARPA), deposition gauges, and moss-biomonitoring surveys (ICP Vegetation).
- **Dioxins:** almost never measured in ambient air. Use operator stack reports (incinerator permits / annual environmental reports), soil, egg or milk surveys near incinerators, and the national food-safety agency.
- **Soil legacy:** contaminated-site registers (see the country sections).
- **US:** NATTS/UATMP air toxics (AQS), TRI and NEI emissions, AirToxScreen modelled concentrations (census tract).

## United States
- **Air:** EPA AQS API (free key), AirNow; state DEP reports
- **Emissions:** EPA TRI (toxics; Envirofacts API `https://data.epa.gov/efservice/`), NEI point sources, FLIGHT (GHG), ECHO (compliance & violations); **AirToxScreen** = modelled ambient toxics by census tract (can serve as the "field" for air toxics)
- **Environmental justice:** EJScreen was removed in 2025; use archived mirrors (e.g. PEDP), or CEJST for context
- **Major-hazard sites:** EPA RMP facilities (RTK NET / Data Liberation); PHMSA pipelines and incidents
- **Noise:** BTS National Transportation Noise Map (road, aviation, rail)
- **Water:** EPA How's My Waterway / ATTAINS, BEACON beach advisories
- **Contaminated land:** Superfund/NPL, Brownfields ("Cleanups in My Community")
- **Traffic:** state DOT AADT (FHWA HPMS)
- **Radon:** EPA radon zones
- **Flood:** FEMA NFHL
- **Complaints:** state environmental complaint portals, local news, city council minutes

## Canada, Australia, Asia and others
- **Canada:** NAPS (air), NPRI (emissions), E2 regulations (hazard sites), provincial AADT.
- **Australia:** state EPAs (air stations, e.g. NSW DPE, EPA Victoria), NPI (emissions), Beachwatch.
- **Japan:** Soramame (air), PRTR. **Korea:** AirKorea. **China:** CNEMC (via aqicn/OpenAQ), IPE blue map (enterprise violations).
- **India:** CPCB CAAQMS portal, SPCB consent lists, CPCB "Grossly Polluting Industries", Sameer app.
- **Latin America:** national networks (e.g. SINCA Chile, RAMA/SEDEMA Mexico City, CETESB São Paulo), RETC (PRTR).
- **Where there is no official network:** OpenAQ, low-cost sensor networks (PurpleAir, sensor.community) with a stated bias caveat; CAMS global; Sentinel-5P.

## Search patterns for agents (translate them into the local language)
- `"<place>" <pollution | smell | noise | discharge | spill | fire | explosion | fine | complaint | petition>`
- `"<facility name>" <emissions | inspection | fine | accident | Seveso | licence>`
- `site:<national-agency-domain> "<municipality>"`, `filetype:pdf "<municipality>" "qualidade do ar"` (local terms)
- Local-language NGO names (e.g. PT ZERO/Quercus, ES Ecologistas en Acción, FR France Nature Environnement, DE BUND/DUH, UK Friends of the Earth / Surfers Against Sewage, US local Riverkeepers)
