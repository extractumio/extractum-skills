#!/usr/bin/env python3
"""Config-driven exposure model on a regular lat/lon grid.

Usage: python model.py WORKDIR/config.json
Reads files referenced in config (paths relative to the config's folder), writes WORKDIR/grids.json.

Layers produced (uint8-quantised, later PNG-packed by build_page.py):
  no2      central NO2 annual mean (µg/m3): background + roads + port + stacks, NNLS-calibrated on stations
  no2hi    high-shipping scenario (port term scaled to +X% near port, literature) + LOO RMSE
  lden/ln  official noise contours rasterised (noise.mode="official") or road-based screening estimate ("modelled")
  odour, dust   wind-skewed exponential kernels around expert-weighted sources (0..1)
  risk     distance decay around major-hazard (Seveso/RMP/...) sites using consequence distances (0..1)
  index_m  50/50 NO2+Lden (measured/calibrated factors only)
  index    extended (weights from config)
  land     land mask
Also area stats for config.areas (radius_m circles), calibration table, leave-one-out validation.
Requires numpy scipy pillow shapely.
"""
import json, math, os, re, sys, base64
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollutants import POLLUTANTS, BY_KEY, GROUPS

cfg_path = sys.argv[1]; W_DIR = os.path.dirname(os.path.abspath(cfg_path))
C = json.load(open(cfg_path)); F = C.get("files", {})
# research agents may deliver named roads / port as separate files
if F.get("roads_named") and not C.get("traffic", {}).get("named"):
    C.setdefault("traffic", {})["named"] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(cfg_path)), F["roads_named"])))
if F.get("port") and not C.get("port"):
    C["port"] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(cfg_path)), F["port"])))
P = lambda k: os.path.join(W_DIR, F[k]) if F.get(k) else None
def load(k, default=None):
    p = P(k); return json.load(open(p)) if p and os.path.exists(p) else default

S, Wb, N, E = C["bbox"]; CELL = C.get("cell_m", 25)
DLAT = CELL / 111320.0; DLON = CELL / (111320.0 * math.cos(math.radians((S + N) / 2)))
NY = int(round((N - S) / DLAT)); NX = int(round((E - Wb) / DLON))
print("grid", NX, "x", NY, "cells of", CELL, "m")
px = lambda lat, lon: ((lon - Wb) / DLON, (N - lat) / DLAT)
Y, X = np.mgrid[0:NY, 0:NX]

def poly_mask(rings):
    im = Image.new("L", (NX, NY), 0); d = ImageDraw.Draw(im)
    for r in rings:
        if len(r) >= 3: d.polygon([px(a, b) for a, b in r], fill=1)
    return np.array(im, dtype=bool)

base = load("base"); L = base["layers"]
land = poly_mask(L["land"]) | poly_mask(L.get("port", [])) | poly_mask(L.get("industrial", []))

# ---------------- roads / traffic ----------------
T = C.get("traffic", {})
CLS = {"motorway": 30000, "trunk": 12000, "primary": 12000, "secondary": 6000, "tertiary": 3000, "minor": 400}
CLS.update(T.get("class_aadt", {}))   # per carriageway for dual roads (motorway/trunk drawn twice in OSM)
im = Image.new("F", (NX, NY), 0); dr = ImageDraw.Draw(im)
for k in ["minor", "tertiary", "secondary", "primary", "trunk", "motorway"]:
    for line in L["road"].get(k, []): dr.line([px(a, b) for a, b in line], fill=CLS[k], width=1)
road = np.array(im)
named_im = Image.new("F", (NX, NY), 0); nd = ImageDraw.Draw(named_im)
for r in T.get("named", []):          # centrelines with official AADT (both directions)
    if r.get("ll") and r.get("aadt"): nd.line([px(a, b) for a, b in r["ll"]], fill=r["aadt"], width=1)
named = np.array(named_im)
road = np.where(named > 0, np.maximum(named, road * (named == 0)), road)  # named centreline wins in its cells
EF = T.get("ef_g_per_vkm", 0.40)     # g NOx per vehicle-km, fleet average (EU ~0.3-0.5; adjust for fleet)
E_road = road * EF * (CELL / 1000) * 365 / 1e6   # t/yr per cell

# ---------------- port / shipping ----------------
PT = C.get("port") or {}
E_port = np.zeros((NY, NX), np.float32); port = np.zeros((NY, NX), bool)
if PT.get("polygon") and PT.get("nox_t_yr"):
    port = poly_mask([PT["polygon"]]); tot = PT["nox_t_yr"] * PT.get("ground_frac", 0.35)
    bw = PT.get("berths", []); share_b = sum(b[2] for b in bw)
    if port.sum(): E_port[port] += tot * max(0, 1 - share_b) / port.sum()
    for la, lo, w in bw:
        x, y = map(int, px(la, lo)); E_port[max(0, y-4):y+5, max(0, x-4):x+5] += tot * w / 81
E_pt = np.zeros((NY, NX), np.float32)
for la, lo, t in C.get("stacks", []):       # effective ground-level t/yr (tall stacks: use 5-20% of emission)
    x, y = map(int, px(la, lo))
    if 0 <= x < NX and 0 <= y < NY: E_pt[y, x] += t

# ---------------- dispersion kernels ----------------
rr = int(4000 / CELL); yy, xx = np.mgrid[-rr:rr+1, -rr:rr+1]; r = np.hypot(yy, xx) * CELL
K1 = np.exp(-r / 120) / (r + 30); K1[r > 1500] = 0          # near-road
K2 = np.exp(-r / 2500); K2[r > 4000] = 0; K2 /= K2.sum()     # urban accumulation
Kp = np.exp(-r / 600); Kp[r > 3000] = 0; Kp /= Kp.sum()      # elevated ship stacks
E_land = E_road + E_pt
C1 = fftconvolve(E_land, K1, mode="same"); C2 = fftconvolve(E_land, K2, mode="same")
CPl = fftconvolve(E_port, Kp, mode="same") * K1.sum(); CP2 = fftconvolve(E_port, K2, mode="same")

# ---------------- calibration ----------------
NO2c = C.get("no2", {}); B = NO2c.get("background", 10.0)
years = NO2c.get("calib_years", ["2024", "2025"]); excl = set(NO2c.get("exclude", []))
obs = []
for s in load("stations", []) or []:
    a = (s.get("annual") or {}).get("no2") or (s.get("annual") or {}).get("NO2", {})
    vals = [a[y]["mean"] for y in years if y in a and a[y].get("mean") is not None]
    if not vals or s.get("code") in excl or not (S < s["lat"] < N and Wb < s["lon"] < E): continue
    x, y = px(s["lat"], s["lon"]); obs.append((s["name"], int(y), int(x), float(np.mean(vals))))
A0 = lambda idx: np.array([[C1[obs[i][1], obs[i][2]], C2[obs[i][1], obs[i][2]]] for i in idx])
calibrated = len(obs) >= 3
if calibrated:
    c0 = nnls(A0(range(len(obs))), np.array([o[3] - B for o in obs]))[0]
else:
    c0 = np.array(NO2c.get("default_coef", [650.0, 820.0])); print("WARNING: <3 NO2 stations — using default coefficients (uncalibrated)")
PPH = c0[0] * CPl + c0[1] * CP2      # physically-scaled port field
def fit(idx):
    A = np.array([[C1[obs[i][1], obs[i][2]], C2[obs[i][1], obs[i][2]], PPH[obs[i][1], obs[i][2]]] for i in idx])
    return nnls(A, np.array([obs[i][3] - B for i in idx]))[0]
if calibrated and len(obs) >= 4 and PPH.max() > 0:
    coef = fit(range(len(obs)))
    loo = []
    for j in range(len(obs)):
        cj = fit([i for i in range(len(obs)) if i != j]); n_, y, x, o = obs[j]
        loo.append({"name": n_, "obs": round(o, 1), "pred": round(float(B + cj[0]*C1[y, x] + cj[1]*C2[y, x] + cj[2]*PPH[y, x]), 1)})
else:
    coef = np.array([c0[0], c0[1], NO2c.get("default_port_scale", 0.06)]); loo = []
rmse = float(np.sqrt(np.mean([(l["obs"] - l["pred"]) ** 2 for l in loo]))) if loo else NO2c.get("default_rmse", 3.0)
ROADT = coef[0] * C1 + coef[1] * C2; PORTT = coef[2] * PPH
NO2 = B + ROADT + PORTT
k_hi = 0.0; PORT_HI = np.zeros_like(PORTT)
if port.any() and PPH.max() > 0:
    ring = binary_dilation(port, iterations=max(1, int(500 / CELL))) & ~port & land
    if ring.any():
        k_hi = PT.get("literature_uplift", 0.5) * float((B + ROADT)[ring].mean()) / float(PPH[ring].mean()); PORT_HI = k_hi * PPH
NO2_lo = B + ROADT + PORTT - rmse; NO2_hi = B + ROADT + np.maximum(PORT_HI, PORTT) + rmse
NO2, NO2_lo, NO2_hi = [gaussian_filter(a, 0.7) for a in (NO2, NO2_lo, NO2_hi)]
print("coef", coef, "rmse_loo", round(rmse, 2), "port max", round(float(PORTT.max()), 1), "port_hi max", round(float(PORT_HI.max()), 1))
for (n_, y, x, o) in obs: print(f"  {n_[:40]:40s} obs {o:5.1f}  model {NO2[y, x]:5.1f}")

# ---------------- noise ----------------
NZ = C.get("noise", {"mode": "modelled"})
def band_val(cat):
    c = str(cat); nums = [int(x) for x in re.findall(r"\d{2}", c)]
    if not nums: return None
    if re.search(r"greater|more|>|\+", c, re.I): return nums[0] + 2
    if re.search(r"lower|less|<", c, re.I): return nums[0] - 2
    return nums[0] + 2
def rast_noise(files, field):
    g = np.zeros((NY, NX), np.float32); feats = []
    for f in files: feats += json.load(open(os.path.join(W_DIR, f)))["features"]
    feats = [(band_val(f["properties"].get(field)), f) for f in feats]; feats = sorted([x for x in feats if x[0]], key=lambda x: x[0])
    def polys(geom):
        t = geom["type"]
        if t == "Polygon": yield geom["coordinates"]
        elif t == "MultiPolygon": yield from geom["coordinates"]
        elif t == "GeometryCollection":
            for gg in geom["geometries"]: yield from polys(gg)
    for v, f in feats:
        ext = Image.new("L", (NX, NY), 0); de = ImageDraw.Draw(ext); hol = Image.new("L", (NX, NY), 0); dh = ImageDraw.Draw(hol)
        for p in polys(f["geometry"]):
            de.polygon([px(la, lo) for lo, la in p[0]], fill=1)
            for h in p[1:]:
                if len(h) >= 3: dh.polygon([px(la, lo) for lo, la in h], fill=1)
        m = (np.array(ext) > 0) & ~(np.array(hol) > 0); g[m] = v
    return g
def modelled_noise(night=False):
    # screening: L(10 m) = 10*log10(AADT) + k ; -15*log10(d/10) (divergence + ground); energetic sum over AADT buckets
    k = NZ.get("k_lden", 27.0) - (8.0 if night else 0.0); tot = np.zeros((NY, NX))
    for lo_, hi_ in [(1, 2000), (2000, 8000), (8000, 25000), (25000, 60000), (60000, 1e9)]:
        m = (road >= lo_) & (road < hi_)
        if not m.any(): continue
        d = np.maximum(distance_transform_edt(~m) * CELL, 5.0); aadt = float(np.median(road[m]))
        Lv = 10 * np.log10(aadt) + k - 15 * np.log10(d / 10); tot += 10 ** (Lv / 10)
    return np.where(tot > 0, 10 * np.log10(tot + 1e-9), 0)
if NZ.get("mode") == "official" and F.get("noise_lden"):
    lden = rast_noise(F["noise_lden"], NZ.get("category_field", "category"))
    ln = rast_noise(F["noise_ln"], NZ.get("category_field", "category")) if F.get("noise_ln") else np.where(lden > 0, lden - 8, 0)
    for extra in F.get("noise_lden_extra", []): lden = np.maximum(lden, rast_noise([extra], NZ.get("category_field", "category")))
    for extra in F.get("noise_ln_extra", []): ln = np.maximum(ln, rast_noise([extra], NZ.get("category_field", "category")))
else:
    lden = modelled_noise(); ln = modelled_noise(True)
print("noise mode", NZ.get("mode"), "coverage", round(float((lden > 0).mean()), 2))

# ---------------- wind-skewed heuristic fields ----------------
met = load("meteo", {}) or {}
freq = np.array([sum(b) for b in met["wind"]["year"]["bins"]]) if met.get("wind") else np.ones(16)
freq = freq / freq.sum()
def kernel_field(src, R, cut):
    g = np.zeros((NY, NX), np.float32)
    for s in src:
        la, lo, I = s[0], s[1], s[2]; x, y = px(la, lo)
        dx = (X - x) * CELL; dy = (Y - y) * CELL; d = np.hypot(dx, dy)
        brg = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360; frm = (brg + 180) % 360
        ww = 0.35 + 0.65 * 16 * freq[(((frm + 11.25) % 360) // 22.5).astype(int)]
        v = I * np.exp(-d / R) * ww; v[d > cut] = 0; g += v
    return np.clip(g, 0, 1)
odour = kernel_field(C.get("odour", []), C.get("odour_R", 450), 2500)
dust = kernel_field(C.get("dust", []), C.get("dust_R", 350), 2000)
HZ = C.get("hazard", {"upper": [250, 750], "lower": [150, 400]})
risk = np.zeros((NY, NX), np.float32); srcs = load("sources", []) or []
for s in srcs:
    t = s.get("hazard_tier") or s.get("seveso_tier")
    if t not in ("upper", "lower"): continue
    x, y = px(s["lat"], s["lon"]); d = np.hypot(X - x, Y - y) * CELL; r0, r1 = HZ[t]
    risk = np.maximum(risk, (1.0 if t == "upper" else 0.6) * np.clip((r1 - d) / (r1 - r0), 0, 1))


# ---------------- every pollutant: calibrated field | emissions-based layer | stations only | background | no data ----------------
# Decision per pollutant (see reference/methodology.md "Pollutant coverage"):
#   field      >= 3 stations with recent annual means AND leave-one-out RMSE <= 35% of the mean -> concentration map
#   emissions  facilities in the reporting register (PRTR/TRI/NPRI) with recent air releases -> relative 0..1 layer
#   stations   1-2 stations (or a field that failed validation) -> values in the table, no map
#   historic   only closed facilities / only measurements older than 5 years -> table note
#   background only a regional model (CAMS) value -> table note
#   none       nothing -> "no data" (always listed, never silently dropped)
YR = int(C.get("current_year") or __import__("time").strftime("%Y"))
RECENT = [str(y) for y in range(YR - 3, YR + 1)]
PR = load("prtr", {}) or {}; INV = load("aq_inventory", {}) or {}; ST = load("stations", []) or []
CAMS = (met or {}).get("cams", {}) or {}
fac_by_id = {f["id"]: f for f in PR.get("facilities", [])}
sea_d = distance_transform_edt(land) * CELL                       # metres to the nearest sea cell (0 at sea)
coast = np.exp(-sea_d / 1000.0) if (~land).any() else np.zeros((NY, NX))
def point_field(rows):
    """wind-skewed dispersion of reported releases (kg/yr) from elevated stacks -> relative field"""
    g = np.zeros((NY, NX), np.float32)
    for r_ in rows:
        f = fac_by_id.get(r_["id"]);
        if not f: continue
        x, y = px(f["lat"], f["lon"]); dx = (X - x) * CELL; dy = (Y - y) * CELL; d = np.hypot(dx, dy)
        brg = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360; frm = (brg + 180) % 360
        ww = 0.35 + 0.65 * 16 * freq[(((frm + 11.25) % 360) // 22.5).astype(int)]
        v = r_["kg"] * np.exp(-d / 1000.0) / (d + 150.0) * ww; v[d > 6000] = 0; g += v
    return g
def norm_unit(u, default):
    u = str(u or "")
    if u in ("", "nan", "None"): return default
    return {"ug.m-3": "µg/m³", "ng.m-3": "ng/m³", "mg.m-3": "mg/m³", "ug/m3": "µg/m³", "ng/m3": "ng/m³", "mg/m3": "mg/m³"}.get(u, u)
def st_obs(key):
    out_ = []
    for s_ in ST:
        a_ = (s_.get("annual") or {}).get(key) or {}
        def ok_year(e):   # hourly/daily series: >=50% coverage; variable samplers (metals, BaP): >=10 samples
            cov = e.get("coverage_pct")
            return e.get("mean") is not None and ((cov is not None and cov >= 50) or (cov is None and (e.get("n") or 0) >= 10))
        vals = [(y, a_[y]["mean"]) for y in RECENT if y in a_ and ok_year(a_[y])]
        if not vals: continue
        y_, v_ = vals[-1]; inside = S < s_["lat"] < N and Wb < s_["lon"] < E
        out_.append({"name": s_["name"], "code": s_.get("code"), "lat": s_["lat"], "lon": s_["lon"], "year": y_, "mean": round(float(np.mean([v for _, v in vals])), 2),
                     "latest": round(v_, 2), "unit": norm_unit((s_.get("units") or {}).get(key), BY_KEY[key]["unit"]), "inside": inside,
                     "dist_km": round(math.hypot((s_["lat"] - C["home"]["lat"]) * 111, (s_["lon"] - C["home"]["lon"]) * 111 * math.cos(math.radians(S))), 1)})
    return sorted(out_, key=lambda o: o["dist_km"])
coverage = []; extra = {}; field_meta = {}
QUERIED_ST = bool(INV.get("_meta")) or bool(ST)          # was a station source queried at all?
QUERIED_PR = bool(PR.get("facilities") is not None and PR)  # was an emission register queried at all?
def fac_active(fid):
    st_ = fac_by_id.get(fid, {}).get("status")
    if st_: return st_ in ("reporting", "unknown_after_register_end")
    return (fac_by_id.get(fid, {}).get("latest_year") or 0) >= YR - 3
def kern_abs(rows, thr):
    """absolute emission potential: sum over facilities of (kg/yr / reporting threshold) x K(d), K(1 km) = 1, wind-weighted"""
    g = np.zeros((NY, NX), np.float32)
    for r_ in rows:
        f = fac_by_id.get(r_["id"])
        if not f: continue
        x, y = px(f["lat"], f["lon"]); dx = (X - x) * CELL; dy = (Y - y) * CELL; d = np.hypot(dx, dy)
        brg = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360; frm = (brg + 180) % 360
        ww = 0.35 + 0.65 * 16 * freq[(((frm + 11.25) % 360) // 22.5).astype(int)]
        v = (r_["kg"] / thr) * np.exp(-(d - 1000.0) / 1000.0) * (1150.0 / (d + 150.0)) * ww; v[d > 10000] = 0; g += v
    return g
def nested_loo(cand, pts_, yv, max_pred):
    """choose predictors inside each fold (honest error); returns (outer_rmse, chosen_on_all, coefs, cols)"""
    from itertools import combinations
    def fit_best(idx):
        best = None
        for m_ in range(0, max_pred + 1):
            for combo in combinations(sorted(cand), m_):
                cols = [np.ones((NY, NX))] + [cand[c_] for c_ in combo]
                Am = np.array([[c[pts_[i][0], pts_[i][1]] for c in cols] for i in idx]); yy_ = yv[idx]
                errs = []
                for jj in range(len(idx)):
                    ii = [t for t in range(len(idx)) if t != jj]
                    if len(ii) < 2: continue
                    cj = nnls(Am[ii], yy_[ii])[0]; errs.append(float(Am[jj] @ cj - yy_[jj]))
                r_ = float(np.sqrt(np.mean(np.square(errs)))) if errs else 1e9
                if best is None or r_ < best[0] - 1e-9: best = (r_, combo, cols)
        return best
    outer = []
    for j in range(len(pts_)):
        idx = [i for i in range(len(pts_)) if i != j]; _, combo, cols = fit_best(idx)
        Am = np.array([[c[pts_[i][0], pts_[i][1]] for c in cols] for i in idx]); cj = nnls(Am, yv[idx])[0]
        outer.append(float(np.array([c[pts_[j][0], pts_[j][1]] for c in cols]) @ cj - yv[j]))
    _, combo, cols = fit_best(list(range(len(pts_))))
    Am = np.array([[c[p_[0], p_[1]] for c in cols] for p_ in pts_]); cf = nnls(Am, yv)[0]
    return float(np.sqrt(np.mean(np.square(outer)))), combo, cf, cols
for P_ in POLLUTANTS:
    k = P_["key"]; obs_k = st_obs(k); inv = INV.get(k, {}); thr = P_.get("prtr_threshold_kg")
    rows = [r_ for r_ in PR.get("by_key", {}).get(k, []) if r_["kg"] and r_["kg"] > 0]
    cur_rows = [r_ for r_ in rows if r_["year"] >= YR - 5 and fac_active(r_["id"])]           # operating + recently above threshold
    below_rows = [r_ for r_ in rows if fac_active(r_["id"]) and r_ not in cur_rows]             # operating, but below threshold since
    gone_rows = [r_ for r_ in rows if not fac_active(r_["id"])]                                   # not in register any more
    cams_key = P_.get("cams"); cams_mean = None
    if cams_key and CAMS.get(cams_key):
        vv = [v for ym, v in CAMS[cams_key].items() if ym[:4] in RECENT[:-1]]
        if vv: cams_mean = round(float(np.mean(vv)) / (1000 if k == "co" else 1), 3)
    def em(r_):
        f = fac_by_id.get(r_["id"], {})
        return {"name": r_["name"], "kg": r_["kg"], "year": r_["year"], "fac_status": f.get("status") or ("reporting" if fac_active(r_["id"]) else "not_reporting"),
                "fac_last_year": f.get("latest_year"), "suspicious": bool(P_.get("suspicious_kg") and r_["kg"] > P_["suspicious_kg"]),
                "dist_km": round(math.hypot((f.get("lat", 0) - C["home"]["lat"]) * 111, (f.get("lon", 0) - C["home"]["lon"]) * 111 * math.cos(math.radians(S))), 1) if f else None}
    rec = {"key": k, "label": P_["label"], "group": P_["group"], "unit": P_["unit"], "refs": P_["refs"], "notes": P_["notes"], "threshold_kg": thr,
           "n_stations_recent": len(obs_k), "n_stations_ever": inv.get("n_stations_ever", 0), "period": [inv.get("first"), inv.get("last")],
           "obs": obs_k[:6], "emitters": [em(r_) for r_ in (cur_rows + below_rows + gone_rows)[:8]],
           "emissions_kg_recent": float(sum(r_["kg"] for r_ in cur_rows)), "cams_mean": cams_mean, "layer": None, "layer2": None, "status": "none", "rmse": None,
           "sources_queried": {"stations": QUERIED_ST, "register": QUERIED_PR}}
    if k == "no2":
        rec.update(status="field", layer="no2", rmse=round(rmse, 2))
        if not calibrated: rec["note_fit"] = f"UNCALIBRATED: only {len(obs)} NO2 station(s) inside the bbox — default coefficients, error ≈ ±{rmse:.0f} µg/m³ assumed"
        coverage.append(rec); continue
    # secondary layer: absolute emission potential of currently reported releases
    if cur_rows:
        if thr: pf = kern_abs(cur_rows, thr); extra["e_" + k] = (np.clip(pf, 0, 10), 0, 10)
        else:
            pf = kern_abs(cur_rows, max(r_["kg"] for r_ in cur_rows)); extra["e_" + k] = (np.clip(pf, 0, 10), 0, 10); rec["scale_note"] = "no EU threshold: scaled to the largest emitter"
        rec["layer2"] = "e_" + k
    else: pf = None
    inside = [o for o in obs_k if o["inside"]]
    if len(inside) >= 3 and P_["kind"] != "regional":
        cand = {}
        if P_["kind"] in ("traffic", "mixed"): cand.update(road_local=C1, road_urban=C2)
        if pf is not None and pf.max() > 0: cand["facilities"] = pf / pf.max()
        if P_["kind"] == "mixed" and (~land).any(): cand["coast"] = coast
        if P_["kind"] == "mixed" and dust.max() > 0: cand["dust_sources"] = dust
        if port.any() and P_["kind"] in ("mixed", "industrial"): cand["port"] = PPH / (PPH.max() or 1)
        pts_ = [(int(px(o["lat"], o["lon"])[1]), int(px(o["lat"], o["lon"])[0]), o["mean"]) for o in inside]
        yv = np.array([p_[2] for p_ in pts_]); mean_obs = float(yv.mean())
        max_pred = 1 if len(pts_) < 5 else min(3, len(pts_) - 3)          # >1 predictor only with >=5 stations
        r_, combo, cf, cols = nested_loo(cand, pts_, yv, max_pred)
        null_rmse = float(np.sqrt(np.mean([(yv[j] - np.delete(yv, j).mean()) ** 2 for j in range(len(yv))])))
        fld = gaussian_filter(sum(c * w for c, w in zip(cols, cf)), 0.7)
        if combo and r_ < 0.9 * null_rmse and r_ <= 0.35 * mean_obs and float(np.ptp(fld[land])) > r_:
            lo_, hi_ = float(np.percentile(fld[land], 1)), float(np.percentile(fld[land], 99.5))
            extra["c_" + k] = (fld, round(lo_, 3), round(hi_ * 1.05 + 1e-9, 3)); rec.update(status="field", layer="c_" + k, rmse=round(r_, 3), predictors=list(combo))
            field_meta["c_" + k] = {"calib": [{"name": o["name"], "obs": o["mean"], "mod": round(float(fld[y, x]), 3)} for (y, x, _), o in zip(pts_, inside)],
                                    "loo_rmse": round(r_, 3), "null_rmse": round(null_rmse, 3), "predictors": list(combo), "nested": True}
        else:
            rec["note_fit"] = f"no concentration map: nested leave-one-out RMSE {r_:.2f} ({'+'.join(combo) or 'background only'}) vs 'same value everywhere' {null_rmse:.2f}; station mean {mean_obs:.2f}"
    if rec["status"] != "field":
        # precedence: stations > emissions > below threshold > not in register > historic stations > background > none
        if obs_k: rec["status"] = "stations"
        elif cur_rows: rec["status"] = "emissions"
        elif below_rows: rec["status"] = "below_threshold"
        elif gone_rows or (inv.get("n_stations_ever") or 0) > 0: rec["status"] = "historic"
        elif cams_mean is not None: rec["status"] = "background"
        elif not QUERIED_ST and not QUERIED_PR: rec["status"] = "not_assessed"
        else:
            rec["status"] = "none"
            if not QUERIED_PR: rec["note_fit"] = "emission register not queried for this country"
            if not QUERIED_ST: rec["note_fit"] = "station network not queried for this country"
        if rec["status"] in ("stations", "emissions", "below_threshold") and rec["layer2"] and not rec["layer"]: rec["layer"] = rec["layer2"]
    coverage.append(rec)
# combined heavy-metal emission potential, each metal scaled by its EU reporting threshold (≈ regulatory significance)
raw = np.zeros((NY, NX), np.float32); n_metals = 0
for p_ in POLLUTANTS:
    rows_m = [x for x in PR.get("by_key", {}).get(p_["key"], []) if x["year"] >= YR - 5 and x["kg"] and fac_active(x["id"])]
    if p_["group"] == "metals" and p_.get("prtr_threshold_kg") and rows_m:
        raw += kern_abs(rows_m, p_["prtr_threshold_kg"]); n_metals += 1
if n_metals: extra["e_metals"] = (np.clip(raw, 0, 10), 0, 10)
print("pollutant coverage:", {c["key"]: c["status"] for c in coverage})

# ---------------- indices ----------------
n_no2 = np.clip((NO2 - 10) / 30, 0, 1); n_ld = np.where(lden > 0, np.clip((lden - 47) / 28, 0, 1), np.nan)
IX = C.get("index", {}); wm = IX.get("measured", {"no2": .5, "lden": .5}); we = IX.get("extended", {"no2": .35, "lden": .35, "odour": .12, "dust": .08, "risk": .10})
comps = {"no2": n_no2, "lden": n_ld, "odour": odour, "dust": dust, "risk": risk}
index_m = 100 * sum(w * comps[k] for k, w in wm.items()); index_e = 100 * sum(w * comps[k] for k, w in we.items())

# ---------------- stats ----------------
stats = {}
for a in C.get("areas", []):
    x, y = px(a["lat"], a["lon"]); m = (np.hypot(X - x, Y - y) * CELL <= C.get("radius_m", 400)) & land
    ld = lden[m]; lnn = ln[m]; nm = lambda arr: round(float(np.nanmean(arr[m])), 1) if m.any() else None
    stats[a["k"]] = {"no2": nm(NO2), "no2_lo": nm(NO2_lo), "no2_hi": nm(NO2_hi), "bg": B, "road": nm(ROADT), "port": nm(PORTT), "port_hi": nm(PORT_HI),
        "lden_mean": round(float(ld[ld > 0].mean()), 1) if (ld > 0).any() else None,
        "lden65": round(float((ld >= 65).mean() * 100)), "ln55": round(float((lnn >= 55).mean() * 100)),
        "odour": round(float(odour[m].mean()), 2), "dust": round(float(dust[m].mean()), 2), "risk": round(float(risk[m].mean()), 2),
        "index_m": round(float(np.nanmean(index_m[m]))) if np.isfinite(index_m[m]).any() else None,
        "index": round(float(np.nanmean(index_e[m]))) if np.isfinite(index_e[m]).any() else None,
        **{k_: round(float(np.nanmean(a_[0][m])), 3) for k_, a_ in extra.items()}}
print(json.dumps(stats, indent=1))

# ---------------- export ----------------
def enc(a, lo, hi):
    a = np.asarray(a, dtype=float); q = np.clip(np.nan_to_num((a - lo) / (hi - lo) * 254, nan=0), 0, 254).round().astype(np.uint8) + 1
    q[~np.isfinite(a)] = 0; return base64.b64encode(q.tobytes()).decode()
nz = lambda a: np.where(a > 0, a, np.nan)
grids = {"no2": (NO2, 5, 45), "no2hi": (NO2_hi, 5, 60), "lden": (nz(lden), 30, 80), "ln": (nz(ln), 30, 80), "odour": (odour, 0, 1), "dust": (dust, 0, 1),
         "risk": (risk, 0, 1), "index_m": (index_m, 0, 100), "index": (index_e, 0, 100), "land": (land.astype(float), 0, 1)}
if not C.get("odour"): grids.pop("odour")
if not C.get("dust"): grids.pop("dust")
if not risk.any(): grids.pop("risk")
if not port.any(): grids.pop("no2hi")
grids.update(extra)
out = {"grid": {"lat0": S, "lat1": N, "lon0": Wb, "lon1": E, "nx": NX, "ny": NY, "dlat": DLAT, "dlon": DLON},
       "grids": {k: {"lo": lo, "hi": hi, "d": enc(a, lo, hi)} for k, (a, lo, hi) in grids.items()},
       "stats": stats, "calib": [{"name": n_, "obs": round(o, 1), "mod": round(float(NO2[y, x]), 1)} for n_, y, x, o in obs],
       "loo": loo, "rmse": round(rmse, 2), "coef": [float(c) for c in coef], "k_hi": round(k_hi, 3), "calibrated": calibrated,
       "noise_mode": NZ.get("mode"), "coverage": coverage, "field_meta": field_meta, "groups": GROUPS, "port_max": round(float(PORTT.max()), 1), "port_hi_max": round(float(PORT_HI.max()), 1)}
json.dump(out, open(os.path.join(W_DIR, F.get("grids", "grids.json")), "w"))
Image.fromarray((np.clip(np.nan_to_num(index_m) / 80, 0, 1) * 255).astype(np.uint8)).save(os.path.join(W_DIR, "preview_index.png"))
print("wrote grids.json + preview_index.png")
