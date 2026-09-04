#!/usr/bin/env python
"""Generates ASEAN_DC_EIA_GeoEconometrics.ipynb (Colab-ready)."""
import nbformat as nbf, sys
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ---------------------------------------------------------------- 0. Title
md(r"""
# ASEAN-10 Data-Centre Environmental & Climate Risk — Geo-Economic + Econometric Proof-of-Concept for EIA / EIS Screening

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**What this notebook proves.** ASEAN environmental-assessment regimes (Thailand NEQA / EIA / EHIA, Philippines PD 1586 *EIS System*, Malaysia EIA Order 2015, Indonesia AMDAL, Vietnam ĐTM, ...) almost never list a data centre as a scheduled project. Facility-level PUE / WUE / energy / water are undisclosed for most ASEAN sites. This notebook shows, end-to-end and with known ground truth, that a **geo-economic + econometric pipeline built on alternative data** can (i) recover portfolio-level energy, carbon and water exposure without facility disclosure, (ii) quantify and *correct* the bias created by non-random (MNAR) disclosure, (iii) produce calibrated uncertainty, and (iv) turn that into a **resource-based EIA/EIS screening tier** that captures far more environmental load per assessed project than today's indirect triggers.

**How it proves it.** We generate an *illustrative* register of about 290 facilities across 22 real ASEAN data-centre hubs (real coordinates, real hub-level water-stress / flood / grid context, illustrative facility attributes) with **latent true** PUE, utilisation, WUE, energy, carbon and water. A realistic MNAR disclosure mechanism hides most of the truth (greener, larger, multinational operators disclose more). Every estimator is then scored **against the hidden truth**. Swap the synthetic register for real BOI / IEAT / market-database rows (template exported at the end) and every downstream cell keeps working.

**Pipeline (each section is runnable on its own after §1–§3):**

| § | Block | Method | Geo output |
|---|---|---|---|
| 2 | Legal framework as data | ASEAN-10 comparative table, Regulatory Capture Index (RCI), Thai + cross-jurisdiction rule engines | Choropleth of RCI |
| 3 | Facility register | 22 hubs, ~290 facilities, latent truth, MNAR disclosure, alt-data proxies (night-lights, thermal-IR, footprints, market DB) | Scatter-geo of register |
| 4 | Bottom-up engineering estimator | E = IT × PUE × util × 8760; CO₂ = E × EF; water = E_IT × WUE; Monte-Carlo propagation, tornado | Facility-level P5/P50/P95 |
| 5 | Coverage-aware totals | Naive gross-up vs Horvitz–Thompson / IPW vs Heckman two-step (MNAR) | Bias table |
| 6 | Hierarchical partial pooling | Empirical-Bayes shrinkage, MixedLM random intercepts by country, optional PyMC | Country shrinkage plot |
| 7 | Missing-data machinery | MCAR/MAR/MNAR diagnostics, MICE + Rubin, Heckman-in-MICE, δ-sensitivity, GBM + split-conformal, SoftImpute matrix completion | — |
| 8 | Data fusion / small-area | Fay–Herriot EBLUP at hub level, nowcasting from permits / jobs / satellite construction | Hub-level EBLUP map |
| 9 | Geo-econometrics | Haversine spatial weights, Moran's I (permutation), SLX & spatial-lag 2SLS, Poisson counts, TWFE difference-in-differences + event study (Singapore moratorium, Johor vetting) | Moran scatter, event study |
| 10 | Scenario overlays | NGFS-style carbon-price NPV, flood depth–damage EAD & VaR (baseline vs 2050), water-curtailment cost, heat-driven PUE uplift | Risk maps |
| 11 | PCAF data-quality | DQ1–5 assignment, emissions-weighted DQ, measured / estimated / proxied split | Stacked bars |
| 12 | Validation / backtest | MAE, RMSE, MAPE, bias, interval coverage, pinball loss, calibration curve, regime-switch rule | Calibration plot |
| 13 | **EIA / EIS improvement** | Capture rate of current regimes, threshold frontier (load captured vs projects assessed), harmonisation heat-map, Resource-Based Screening Index (RBSI) tiers | Screening-tier map |
| 14 | Geo dashboard | Multi-layer Folium map (RCI choropleth, facilities, water stress, flood EAD, EBLUPs, CO₂ heat-map) + Plotly maps | Interactive HTML |
| 15 | Export | CSVs, HTML map, real-data template, data-source pointers | — |

> **Legal status date:** mid-August 2026 (Thai PM's Office Regulation on the Data Center Business Policy Committee B.E. 2569, Royal Gazette 13 Aug 2026). Items marked *UNVERIFIED* in the rule engine are flagged, not guessed.
> **All facility numbers are illustrative.** Hub coordinates and hub-level context are real-world anchored; country parameters (grid EF, tariffs) are approximate placeholders to be replaced with IEA / Ember / Electricity Maps values.
""")

# ---------------------------------------------------------------- 1. Setup
md(r"""
## 1. Setup
Runs on Google Colab (installs the few missing libraries) or locally (Python ≥ 3.10 with pandas, numpy, scipy, statsmodels, scikit-learn, plotly, folium, geopandas).
""")
code(r'''
import sys, subprocess, os, warnings, json, math, time, io
warnings.filterwarnings("ignore")
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import importlib
_missing = [pkg for mod, pkg in [("geopandas","geopandas"),("folium","folium"),("plotly","plotly"),("statsmodels","statsmodels"),("sklearn","scikit-learn"),("shapely","shapely"),("pyproj","pyproj"),("branca","branca"),("requests","requests")]
            if importlib.util.find_spec(mod) is None]
if _missing:
    print("Installing:", _missing); subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + _missing, check=False)

import numpy as np, pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px, plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Point
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore"); warnings.simplefilter("ignore", ConvergenceWarning)   # statsmodels re-enables its own warnings on import

SEED = 2569          # B.E. 2569 = 2026
rng = np.random.default_rng(SEED)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 80); pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
PX_TEMPLATE = "plotly_white"; px.defaults.template = PX_TEMPLATE
if not IN_COLAB: pio.renderers.default = "notebook_connected"   # Colab uses its own renderer; elsewhere emit self-contained HTML so nbconvert exports show the figures
print(f"Colab={IN_COLAB} | numpy {np.__version__} | pandas {pd.__version__} | statsmodels {sm.__version__} | geopandas {gpd.__version__} | plotly {pio.__version__ if hasattr(pio,'__version__') else ''}")
''')

# ---------------------------------------------------------------- 2. Legal framework
md(r"""
## 2. The legal framework as data
The comparative table from the brief is encoded as a DataFrame so it can drive screening logic and maps. Five 0–5 scores summarise each regime (justified in the `rationale` column) and combine into a **Regulatory Capture Index (RCI)** — how strongly a jurisdiction's law *actually* gates a data-centre's environmental footprint (listed EIA trigger, indirect capture, resource gating on power/water, disclosure regime, enforcement capacity, public participation).
""")
code(r'''
LEGAL = pd.DataFrame([
 # iso3, country, primary_law, dc_listed, indirect_trigger, authority, timeline, indirect, resource_gate, disclosure, enforcement, participation, rationale
 ("THA","Thailand","NEQA B.E. 2535 (1992) am. (No.2) B.E. 2561 (2018); Constitution s.58 (EHIA)",
  "No - not a 'factory'; usually no EIA","Extra-large building >=10,000 m2; IEAT estate EIA; thermal >10 MW EIA (5-10 MW ESA); ERC licence >=1,000 kVA; BOI power confirmation (30 Mar 2026)",
  "ONEP / Expert Review Cttee / NEB; ERC; BOI; DC Policy Cttee (Aug 2026)","~60 d initial review; months to >1 yr full EIA",
  3.0,4.0,3.0,3.5,4.0,"IEE/EIA/EHIA tiers, indirect capture only; ERC power gate + BOI PUE tiers + 2026 committee; 56-1 One Report, ISSB pending"),
 ("SGP","Singapore","No general EIA statute; Planning Act, EPMA; DC-CFA capacity allocation",
  "No - allocation-based","DC-CFA call (2022 pilot ~80 MW; DC-CFA2 >=200 MW, closed 31 Mar 2026): PUE<=1.3 @100% IT, WUE<=2 m3/MWh, >=50% green",
  "EDB / IMDA / EMA / PUB / NEA","Selective CFA windows",
  1.0,5.0,4.0,5.0,1.0,"No EIA act but the strongest binding resource gate in ASEAN; MAS ERM guidelines, SGX ISSB-aligned disclosure"),
 ("MYS","Malaysia","Environmental Quality Act 1974 s.34A; EIA (Prescribed Activities) Order 2015",
  "No - not prescribed (situational EIA possible)","DOE situational EIA; state water/energy approvals; Johor DC committee (Jun 2024, ~30% rejected); water tariff RM5.33/m3; SPAN alt-water guidelines Sep 2025",
  "DOE (MNRES); state authorities; SPAN","Varies; state committee vetting",
  2.0,4.0,4.0,4.0,3.0,"Water/energy gating at state level is the effective constraint; BNM CRMSA + 2024 climate stress test; Bursa sustainability reporting"),
 ("IDN","Indonesia","Job Creation Law (11/2020 am. 6/2023); GR 5/2021; OSS-RBA; BKPM Reg 5/2025",
  "Risk-based (KBLI): AMDAL / UKL-UPL / SPPL","AMDAL (high risk) / UKL-UPL (medium) / SPPL (low) by KBLI, land area, power; AMDAL prerequisite for PBG, PKKPR",
  "KLHK / BKPM / local govt","30-90 days",
  4.0,2.0,2.0,3.0,3.0,"Every project gets *some* instrument; fines to IDR 3 bn; OJK sustainable finance taxonomy"),
 ("VNM","Vietnam","Law on Environmental Protection 2020 (72/2020); Decree 08/2022; Circular 02/2022",
  "Group-based (I-IV)","DTM (EIA) for Group I and some Group II (Annexes III-V) by land area / capacity / sensitivity",
  "MONRE / provincial DONRE","Several months",
  4.0,2.0,1.0,3.0,3.0,"Universal classification; EIA reports publicly disclosed; corporate ESG disclosure thin"),
 ("PHL","Philippines","PD 1586 (1978) Philippine EIS System; DAO 2003-30; EMB MC 2014-005",
  "ECP/ECA-based","ECC if Environmentally Critical Project or in Environmentally Critical Area; otherwise Certificate of Non-Coverage (CNC)",
  "DENR-EMB","Weeks to months",
  4.0,2.0,2.0,3.0,4.0,"The literal 'EIS System'; screening-by-category; SEC PH sustainability reporting (comply-or-explain)"),
 ("KHM","Cambodia","Environment & Natural Resources Code 2023; Prakas 72/1999; Prakas 3591/0525 (May 2025)",
  "Annex-based","Listed in EIA annex / 2025 classification","Ministry of Environment","Variable",
  3.0,1.0,0.0,1.5,2.0,"Broadened 2025 classification; enforcement historically weak"),
 ("LAO","Lao PDR","Decree 21/2019 on EIA (and Decree 291/2017)","Category-based","IEE/EIA by category","MONRE","Variable; weak enforcement",
  2.0,1.0,0.0,1.5,1.0,"Limited implementation and public engagement"),
 ("MMR","Myanmar","EIA Procedure, Notification 616/2015 under Environmental Conservation Law 2012","Category-based","IEE/EIA by type / size","MONREC","Variable; weak enforcement",
  2.0,1.0,0.0,1.0,1.0,"Capacity and enforcement limited"),
 ("BRN","Brunei","Environmental Protection & Management Act (Cap. 240, 2022); EIA Guidelines v2.0 (2012)","Prescribed-activity-based","Written notification + EIA/EMMP for prescribed activities","Dept of Environment, Parks & Recreation","Variable",
  2.0,1.0,1.0,3.0,2.0,"ESG disclosure encouraged, not mandated"),
], columns=["iso3","country","primary_law","dc_listed_trigger","indirect_trigger","authority","timeline",
            "s_indirect","s_resource_gate","s_disclosure","s_enforcement","s_participation","rationale"])

W_RCI = {"s_indirect":0.20,"s_resource_gate":0.25,"s_disclosure":0.20,"s_enforcement":0.20,"s_participation":0.15}
LEGAL["RCI"] = sum(LEGAL[k]*w for k,w in W_RCI.items())

# Country-level physical / economic context (ILLUSTRATIVE placeholders; replace with IEA / Ember / Electricity Maps / utility tariffs)
CTX = pd.DataFrame({
  "iso3":      ["THA","SGP","MYS","IDN","VNM","PHL","KHM","LAO","MMR","BRN"],
  "grid_ef":   [0.45, 0.40, 0.55, 0.70, 0.55, 0.65, 0.50, 0.25, 0.45, 0.65],   # tCO2/MWh, location-based average
  "elec_usd_kwh":[0.12,0.20, 0.10, 0.08, 0.08, 0.16, 0.15, 0.06, 0.07, 0.06],  # industrial tariff proxy
  "gdp_pc_kusd":[7.3, 88.0, 12.5, 5.0, 4.5, 4.0, 2.5, 2.1, 1.2, 35.0],
  "ef_decline_nz":[0.035,0.030,0.030,0.025,0.030,0.025,0.030,0.010,0.020,0.020],  # annual grid EF decline under Net-Zero path
})
LEGAL = LEGAL.merge(CTX, on="iso3")
display(LEGAL[["iso3","country","dc_listed_trigger","s_indirect","s_resource_gate","s_disclosure","s_enforcement","s_participation","RCI","grid_ef","elec_usd_kwh"]].sort_values("RCI", ascending=False).reset_index(drop=True))
''')
code(r'''
fig = px.choropleth(LEGAL, locations="iso3", color="RCI", hover_name="country",
                    hover_data={"dc_listed_trigger":True,"s_resource_gate":True,"s_disclosure":True,"RCI":":.2f","iso3":False},
                    color_continuous_scale="Viridis", range_color=(0,5), scope="asia",
                    title="Regulatory Capture Index (0-5): how strongly each ASEAN regime gates a data centre's footprint")
fig.update_geos(lataxis_range=[-12, 26], lonaxis_range=[91, 130], showcountries=True, showcoastlines=True, projection_type="mercator")
fig.update_layout(height=520, margin=dict(l=0,r=0,t=50,b=0))
fig.show()

comp = LEGAL.melt(id_vars=["iso3","country"], value_vars=list(W_RCI), var_name="dimension", value_name="score")
fig = px.bar(comp, x="country", y="score", color="dimension", barmode="group", title="RCI components by jurisdiction (0-5)")
fig.update_layout(height=380); fig.show()
''')
md(r"""
### 2.1 Rule engines
`thai_triggers()` encodes the Thai instrument stack (status mid-Aug 2026) instrument-by-instrument with a confidence flag; `screen_generic()` applies a *simplified* screening logic for the other nine regimes. Tier semantics: **0** = permits/registration only, **1** = light assessment (IEE / ESA / UKL-UPL / CNC / Group-III registration), **2** = full assessment or binding resource gate (EIA / AMDAL / ĐTM / ECC / DC-CFA). Thresholds for IDN / VNM / PHL / KHM / LAO / MMR / BRN are **illustrative placeholders** for the real schedules, which key on KBLI risk class, land area, capacity and sensitive-area siting.
""")
code(r'''
def thai_triggers(f, include_uncertain=True):
    """Instrument-level screening of one facility (dict/Series) under Thai law, status mid-Aug 2026."""
    T = []
    def add(inst, tier, conf, basis): T.append(dict(instrument=inst, tier=tier, confidence=conf, basis=basis))
    fa = float(f["floor_area_m2"])
    if fa >= 10000: add("Building Control Act B.E. 2522: extra-large building (>=10,000 m2) - Aor.1 permit, Aor.6 cert, s.32bis inspection", 0, "confirmed", f"floor {fa:,.0f} m2")
    elif fa > 2000: add("Building Control Act B.E. 2522: large building (>2,000 m2)", 0, "confirmed", f"floor {fa:,.0f} m2")
    if fa >= 4000 and include_uncertain: add("ONEP building EIA threshold (>=4,000 m2 usable) - applicability to a pure DC building UNVERIFIED (confirm with ONEP)", 2, "uncertain", f"floor {fa:,.0f} m2")
    if bool(f["in_industrial_estate"]): add("IEAT industrial estate: covered by the ESTATE-level EIA + IEAT operating permit (no facility-specific assessment)", 1, "confirmed", "sited in IEAT estate")
    th = float(f["onsite_thermal_mw"])
    if th > 10: add("NEQA EIA: thermal power plant >10 MW", 2, "confirmed", f"{th:.1f} MW on-site thermal")
    elif th >= 5: add("ERC Environmental & Safety Assessment (ESA): thermal 5-10 MW", 1, "confirmed", f"{th:.1f} MW on-site thermal")
    sol = float(f["onsite_solar_mwp"])
    if sol >= 10: add("MONRE 2023 solar notification: >=10 MWp -> EIA", 2, "confirmed", f"{sol:.0f} MWp")
    elif sol >= 5: add("MONRE 2023 solar notification: 5-10 MWp -> IEE", 1, "confirmed", f"{sol:.0f} MWp")
    kva = float(f["grid_conn_kva"])
    if kva >= 1000: add("Energy Industry Act B.E. 2550: ERC energy-business licence (>=1,000 kVA)", 0, "confirmed", f"{kva:,.0f} kVA")
    if kva >= 200: add("Energy Development & Promotion Act B.E. 2535: Por Kor 2 controlled-energy licence (>=200 kVA)", 0, "confirmed", f"{kva:,.0f} kVA")
    hp = float(f["backup_gen_mw"]) * 1341.0
    if hp >= 50 and include_uncertain: add("Factory Act (No.2) B.E. 2562: machinery >=50 HP -> Ror Ngor 4? Whether standby gensets count toward HP is UNVERIFIED (confirm with DIW)", 0, "uncertain", f"{hp:,.0f} HP standby")
    if bool(f["boi_promoted"]): add("BOI: ERC written confirmation of power supply (from 30 Mar 2026); PUE-tiered incentives (<=1.3 top tier); ERC collateral 4.5 MB/MW", 0, "confirmed", "BOI-promoted")
    if float(f["water_m3_day"]) > 0 and include_uncertain: add("Water Resources Act B.E. 2561 Type 2/3 use licence / Groundwater Act B.E. 2520 - volumetric thresholds UNVERIFIED (confirm ONWR / DGR)", 0, "uncertain", f"{float(f['water_m3_day']):,.0f} m3/day")
    add("NBTC licensing (where telecom services provided); PM's Office Regulation B.E. 2569 DC Policy Committee (coordination only, no licensing yet)", 0, "confirmed", "all DCs")
    return T

ILLUSTRATIVE_THRESHOLDS = {   # placeholders - replace with verified schedule values
    "MYS": dict(sensitive_flood_m=1.0, state_water_vetting_m3_day=500),
    "IDN": dict(amdal_mw=50, amdal_ha=5, uklupl_mw=5),
    "VNM": dict(g1_mw=100, g1_ha=50, g2_mw=20, g2_ha=10, g3_mw=5, sensitive_flood_m=1.0),
    "PHL": dict(ecc_ha=5, eca_flood_m=1.0),
    "KHM": dict(eia_mw=50, iee_mw=10), "LAO": dict(eia_mw=50, iee_mw=10), "MMR": dict(eia_mw=50, iee_mw=10), "BRN": dict(eia_mw=50, iee_mw=10),
}
def screen_generic(f, iso3, include_uncertain=False):
    """Return (tier, instrument) for facility f screened under jurisdiction iso3's (simplified) regime."""
    mw, ha, fl, w = float(f["it_mw"]), float(f["site_ha"]), float(f["flood100_m"]), float(f["water_m3_day"])
    if iso3 == "THA":
        T = thai_triggers(f, include_uncertain); tier = max([t["tier"] for t in T] + [0])
        return tier, ("NEQA indirect capture: " + "; ".join(sorted({t["instrument"].split(":")[0] for t in T if t["tier"] == tier}))) if tier > 0 else "Permits only (no EIA)"
    if iso3 == "SGP":
        return (2, "DC-CFA capacity allocation gate (PUE<=1.3, WUE<=2, >=50% green)") if int(f["year_online"]) >= 2022 else (0, "Pre-2022 capacity: no allocation gate")
    p = ILLUSTRATIVE_THRESHOLDS.get(iso3, {})
    if iso3 == "MYS":
        if fl >= p["sensitive_flood_m"]: return 2, "DOE situational EIA (sensitive site)"
        if w >= p["state_water_vetting_m3_day"]: return 1, "State water/energy vetting (Johor-type committee)"
        return 0, "Not a prescribed activity"
    if iso3 == "IDN":
        if mw >= p["amdal_mw"] or ha >= p["amdal_ha"]: return 2, "AMDAL (high-risk KBLI)"
        if mw >= p["uklupl_mw"]: return 1, "UKL-UPL (medium risk)"
        return 0, "SPPL (low risk)"
    if iso3 == "VNM":
        if mw >= p["g1_mw"] or ha >= p["g1_ha"]: return 2, "Group I -> DTM (EIA)"
        if mw >= p["g2_mw"] or ha >= p["g2_ha"]: return (2, "Group II sensitive -> DTM") if fl >= p["sensitive_flood_m"] else (1, "Group II -> environmental licence")
        if mw >= p["g3_mw"]: return 1, "Group III -> registration"
        return 0, "Group IV"
    if iso3 == "PHL":
        if ha >= p["ecc_ha"] or fl >= p["eca_flood_m"]: return 2, "ECC (ECP / ECA)"
        return 1, "CNC"
    if mw >= p["eia_mw"]: return 2, "EIA (category)"
    if mw >= p["iee_mw"]: return 1, "IEE (category)"
    return 0, "Registration"
print("Rule engines defined: thai_triggers(), screen_generic()")
''')

# ---------------------------------------------------------------- 3. Register
md(r"""
## 3. Illustrative facility register with latent ground truth, MNAR disclosure and alternative-data proxies
22 hubs with real coordinates and real-world-anchored hub context (Aqueduct-style baseline water stress 0–5, Fathom-style 100-year flood depth, distance to submarine-cable landing, wet-bulb climate, night-light index). Facility attributes are drawn from hub-conditional distributions. **Latent truth** (PUE, utilisation, WUE, energy, CO₂, water) is generated from engineering relationships and then hidden by a **Missing-Not-At-Random** disclosure process: larger, multinational, CDP-reporting and *greener* facilities are more likely to disclose — exactly the selection problem a modeller faces.
""")
code(r'''
HUBS = pd.DataFrame([
 # hub, iso3, lat, lon, n, mw_med, water_stress(0-5), flood100_m, dist_cable_km, wetbulb_c, nightlight_idx
 ("Bangkok Metro (Samut Prakan / Bang Na)","THA",13.65,100.65,14,12,3.2,0.8,90,25.8,8.5),
 ("Chonburi / Sriracha (EEC)","THA",13.15,100.95,10,30,3.4,0.3,60,25.9,6.5),
 ("Rayong (EEC)","THA",12.70,101.20,8,35,3.5,0.2,45,25.8,5.5),
 ("Johor (Sedenak / Nusajaya)","MYS",1.60,103.55,16,45,2.6,0.4,35,26.4,6.0),
 ("Klang Valley / Cyberjaya","MYS",2.92,101.65,12,15,2.8,0.5,60,26.0,8.0),
 ("Penang / Kulim","MYS",5.40,100.45,4,10,2.5,0.4,25,26.1,6.0),
 ("Kuching (Sarawak)","MYS",1.55,110.35,2,8,1.8,0.6,20,26.2,4.0),
 ("Singapore West (Jurong / Tuas)","SGP",1.32,103.68,10,25,4.0,0.2,15,26.5,9.5),
 ("Singapore East (Loyang / Tampines)","SGP",1.36,103.95,8,20,4.0,0.2,10,26.5,9.5),
 ("Jakarta / Bekasi / Cikarang","IDN",-6.30,107.10,14,20,3.6,1.2,80,26.3,8.0),
 ("Batam","IDN",1.08,104.03,5,25,2.3,0.4,15,26.4,5.0),
 ("Surabaya","IDN",-7.30,112.75,2,6,3.0,0.9,40,26.2,6.0),
 ("Ho Chi Minh City / Binh Duong","VNM",10.95,106.75,8,12,2.4,0.9,50,25.6,7.5),
 ("Hanoi / Bac Ninh","VNM",21.10,106.05,6,15,2.6,0.7,120,23.4,7.0),
 ("Da Nang","VNM",16.05,108.20,2,8,2.0,0.8,10,24.6,5.0),
 ("Manila / Cavite / Laguna","PHL",14.35,121.05,9,12,2.9,1.0,60,25.7,7.5),
 ("Clark / Pampanga","PHL",15.18,120.55,3,15,2.7,0.8,110,25.5,5.5),
 ("Cebu","PHL",10.32,123.90,2,5,2.5,0.6,20,26.0,5.0),
 ("Phnom Penh","KHM",11.55,104.90,3,6,2.0,1.1,150,25.9,5.0),
 ("Vientiane","LAO",17.97,102.62,2,4,1.6,0.9,500,24.2,3.5),
 ("Yangon","MMR",16.85,96.15,2,5,1.9,1.3,40,25.8,4.0),
 ("Bandar Seri Begawan","BRN",4.90,114.93,2,5,1.2,0.5,15,26.3,4.5),
], columns=["hub","iso3","lat","lon","n","mw_med","water_stress","flood100_hub_m","dist_cable_km","wetbulb_c","nightlight_idx"])
HUBS["n"] = HUBS.n*2                      # ~290 facilities, close to the real ASEAN count
HUBS["hub_id"] = np.arange(len(HUBS))
HUBS = HUBS.merge(LEGAL[["iso3","country","RCI","s_disclosure","grid_ef","elec_usd_kwh","gdp_pc_kusd"]], on="iso3")

TYPE_P   = {"hyperscale":0.22,"colocation":0.48,"enterprise":0.20,"edge":0.10}
TYPE_SC  = {"hyperscale":2.2,"colocation":1.0,"enterprise":0.45,"edge":0.12}
COOL_P   = {"hyperscale":[0.30,0.40,0.30],"colocation":[0.50,0.40,0.10],"enterprise":[0.70,0.25,0.05],"edge":[0.85,0.15,0.0]}
COOLS    = ["air","evaporative","liquid"]
PUE_BASE = {"air":1.58,"evaporative":1.42,"liquid":1.27}
WUE_BASE = {"air":0.30,"evaporative":1.90,"liquid":0.75}     # m3 per MWh of IT energy
YEARS    = np.arange(2012, 2027); YP = (1 + 0.25*(YEARS-2012)); YP = YP/YP.sum()

CTRY_PUE_OFF = {"SGP":-0.10,"MYS":0.02,"THA":0.00,"IDN":0.05,"VNM":-0.02,"PHL":0.04,"KHM":0.12,"LAO":0.15,"MMR":0.16,"BRN":0.08}   # latent country-level PUE offsets (regulatory pressure, kit vintage, grid quality)
HUB_PUE_EFF = {int(h): float(v) for h, v in zip(HUBS.hub_id, rng.normal(0, 0.06, len(HUBS)))}                             # latent hub-level effects (local climate, water availability, estate power quality)
rows = []
for h in HUBS.itertuples():
    jit = 0.035 if h.iso3 == "SGP" else 0.12
    for k in range(h.n):
        ftype = rng.choice(list(TYPE_P), p=list(TYPE_P.values()))
        it_mw = float(np.clip(rng.lognormal(np.log(h.mw_med*TYPE_SC[ftype]), 0.45), 0.5, 250))
        cooling = rng.choice(COOLS, p=COOL_P[ftype])
        op = rng.choice(["multinational","regional","local"], p=[0.75,0.2,0.05] if ftype=="hyperscale" else [0.40,0.38,0.22])
        cdp = rng.random() < {"multinational":0.80,"regional":0.35,"local":0.10}[op]
        year = int(rng.choice(YEARS, p=YP))
        floor = float(it_mw*rng.uniform(600,1200) + rng.uniform(800,2500))
        est_p = 0.7 if "EEC" in h.hub else (0.5 if h.iso3=="THA" else 0.45 if h.iso3=="MYS" else 0.4)
        wet = h.wetbulb_c + rng.normal(0, 0.3)
        # ----- latent truth -----
        pue = PUE_BASE[cooling] + 0.02*(wet-25) - (0.06 if ftype=="hyperscale" else 0) - 0.006*(year-2015) + CTRY_PUE_OFF[h.iso3] + HUB_PUE_EFF[int(h.hub_id)] + rng.normal(0, 0.12)
        pue = float(np.clip(pue, 1.12, 1.95))
        util = float(np.clip(rng.beta(7,3)*0.6 + 0.30, 0.35, 0.96))          # utilisation independent of type/disclosure so the S5 comparison isolates PUE selection
        wue = float(WUE_BASE[cooling] * np.exp(rng.normal(0, 0.25)) * (1 + 0.03*(wet-25)))
        rows.append(dict(fac_id=f"F{len(rows)+1:03d}", hub=h.hub, hub_id=h.hub_id, iso3=h.iso3, country=h.country,
            lat=h.lat + rng.uniform(-jit, jit), lon=h.lon + rng.uniform(-jit, jit),
            ftype=ftype, it_mw=it_mw, cooling=cooling, operator=op, parent_reports_cdp=int(cdp), year_online=year,
            floor_area_m2=floor, site_ha=floor*2.5/1e4, in_industrial_estate=int(rng.random() < est_p),
            onsite_thermal_mw=float(0.6*it_mw if rng.random() < 0.07 else 0.0), backup_gen_mw=float(1.25*it_mw),
            onsite_solar_mwp=float(rng.choice([0,2,6,12], p=[0.65,0.15,0.12,0.08])),
            boi_promoted=int(h.iso3=="THA" and rng.random() < 0.7),
            water_source=rng.choice(["potable","reclaimed","groundwater","surface"], p=[0.5,0.3,0.15,0.05]) if cooling=="evaporative" else "potable",
            water_stress=h.water_stress, flood100_m=float(h.flood100_hub_m*np.exp(rng.normal(0,0.5))) if rng.random()<0.6 else 0.0,
            dist_cable_km=h.dist_cable_km, wetbulb_c=wet, grid_ef=h.grid_ef, elec_usd_kwh=h.elec_usd_kwh, RCI=h.RCI, s_disclosure=h.s_disclosure,
            pue_true=pue, util_true=util, wue_true=wue))
df = pd.DataFrame(rows)
for c in ["ftype","cooling","operator","water_source"]: df[c] = df[c].map(str)   # numpy str_ -> str (sklearn feature-name check)
df["hyperscale"] = (df.ftype=="hyperscale").astype(int); df["multinational"] = (df.operator=="multinational").astype(int)
df["log_mw"] = np.log(df.it_mw); df["year_c"] = df.year_online - 2020
df["grid_conn_kva"] = df.it_mw*1.45*1000/0.9
df["it_energy_true_mwh"] = df.it_mw*df.util_true*8760
df["energy_true_mwh"]    = df.it_energy_true_mwh*df.pue_true
df["co2_true_t"]         = df.energy_true_mwh*df.grid_ef
df["water_true_m3"]      = df.it_energy_true_mwh*df.wue_true
# ----- MNAR disclosure mechanism (greener / bigger / multinational / CDP-reporting disclose more) -----
lin = (-3.3 + 1.8*df.multinational + 0.25*(df.log_mw-df.log_mw.mean()) + 1.6*df.parent_reports_cdp + 0.45*df.s_disclosure - 12.0*(df.pue_true-1.42))   # the -12 term is the MNAR component; CDP / multinational / disclosure regime are the exclusion restrictions
df["p_disclose_true"] = 1/(1+np.exp(-lin)); df["disclosed"] = (rng.random(len(df)) < df.p_disclose_true).astype(int)
for c in ["pue","util","wue"]: df[f"{c}_obs"] = np.where(df.disclosed==1, df[f"{c}_true"], np.nan)
df["energy_obs_mwh"] = np.where(df.disclosed==1, df.energy_true_mwh, np.nan)
df["co2_obs_t"] = np.where(df.disclosed==1, df.co2_true_t, np.nan); df["water_obs_m3"] = np.where(df.disclosed==1, df.water_true_m3, np.nan)
# ----- alternative-data proxies (always observed, noisy) -----
waste_heat = df.it_energy_true_mwh*(df.pue_true-1)
df["nightlight_rad"]  = 2.0 + 0.9*np.log(df.energy_true_mwh) + rng.normal(0, 0.35, len(df))
df["thermal_anom_k"]  = 0.6*np.log1p(waste_heat/1000) + rng.normal(0, 0.30, len(df))
df["footprint_m2"]    = df.floor_area_m2*(1+rng.normal(0,0.15,len(df)))
df["marketdb_mw"]     = df.it_mw*(1+rng.normal(0,0.12,len(df)))
df["permits_count"]   = rng.poisson(1+df.it_mw/20); df["job_posts"] = rng.poisson(2+df.it_mw/8)
print(f"{len(df)} facilities | {df.it_mw.sum():,.0f} MW IT | disclosed: {df.disclosed.mean():.0%} of facilities, {df.loc[df.disclosed==1,'it_mw'].sum()/df.it_mw.sum():.0%} of MW")
print("Mean true PUE  - disclosed: %.3f | undisclosed: %.3f  (selection bias visible)" % (df.loc[df.disclosed==1,'pue_true'].mean(), df.loc[df.disclosed==0,'pue_true'].mean()))
display(df.groupby("country").agg(n=("fac_id","size"), it_mw=("it_mw","sum"), disclosed=("disclosed","mean"), pue_true=("pue_true","mean")).sort_values("it_mw", ascending=False))
''')
code(r'''
fig = px.scatter_geo(df, lat="lat", lon="lon", size="it_mw", color="disclosed", hover_name="fac_id",
                     hover_data={"hub":True,"ftype":True,"cooling":True,"it_mw":":.1f","lat":False,"lon":False},
                     color_continuous_scale=[[0,"#c0392b"],[1,"#1f77b4"]], scope="asia", title="Illustrative ASEAN data-centre register (size = IT MW; blue = discloses PUE/WUE, red = undisclosed)")
fig.update_geos(lataxis_range=[-12, 26], lonaxis_range=[91, 130], showcountries=True, showland=True, landcolor="#f3f3f3")
fig.update_layout(height=560, margin=dict(l=0,r=0,t=50,b=0)); fig.show()
''')

# ---------------------------------------------------------------- 4. Bottom-up
md(r"""
## 4. Bottom-up engineering estimator with Monte-Carlo uncertainty
Core identities (per facility): **E = IT_MW × PUE × utilisation × 8 760**, **CO₂ = E × grid EF** (location-based), **Water = E_IT × WUE**. For undisclosed facilities, PUE / utilisation / WUE are drawn from cooling- and climate-conditional priors; for disclosed ones, from the reported value with a small measurement error. Grid EF carries a ±15 % triangular uncertainty everywhere (it is usually the single largest driver in gas-heavy ASEAN grids). Outputs are P5 / P50 / P95 per facility, plus a tornado variance decomposition. The worked example from the brief (100 MW, PUE 1.4, utilisation 0.70 → ≈ 858 GWh/yr) is reproduced first.
""")
code(r'''
PUE_PRIOR = {"air":(1.58,0.10),"evaporative":(1.42,0.08),"liquid":(1.27,0.07)}
WUE_PRIOR = {"air":(0.30,0.35),"evaporative":(1.80,0.35),"liquid":(0.75,0.35)}   # lognormal median, sigma
def bottom_up_mc(d, n_draws=3000, seed=SEED+1):
    """Vectorised Monte-Carlo of energy / CO2 / water for every row of d. Returns dict of (n_draws, N) arrays."""
    r = np.random.default_rng(seed); N = len(d)
    mu = np.array([PUE_PRIOR[c][0] for c in d.cooling]) + 0.02*(d.wetbulb_c.values-25) - 0.06*d.hyperscale.values - 0.006*(d.year_online.values-2015)
    sd = np.array([PUE_PRIOR[c][1] for c in d.cooling])
    pue = st.truncnorm.rvs((1.10-mu)/sd, (2.0-mu)/sd, loc=mu, scale=sd, size=(n_draws,N), random_state=r)
    a = np.full(N, 9.0); b = np.full(N, 3.5)
    util = r.beta(a, b, size=(n_draws,N))
    wm = np.array([WUE_PRIOR[c][0] for c in d.cooling])*(1+0.03*(d.wetbulb_c.values-25)); ws = np.array([WUE_PRIOR[c][1] for c in d.cooling])
    wue = np.exp(r.normal(np.log(wm), ws, size=(n_draws,N)))
    # disclosed: replace prior draws by reported value + 2% measurement noise
    disc = d.disclosed.values==1
    pue[:,disc]  = d.pue_obs.values[disc]*(1+r.normal(0,0.02,(n_draws,disc.sum())))
    util[:,disc] = np.clip(d.util_obs.values[disc]*(1+r.normal(0,0.02,(n_draws,disc.sum()))), 0.05, 1.0)
    wue[:,disc]  = d.wue_obs.values[disc]*(1+r.normal(0,0.03,(n_draws,disc.sum())))
    ef = d.grid_ef.values*r.triangular(0.85,1.0,1.15,size=(n_draws,N))
    it_e = d.it_mw.values*util*8760
    E = it_e*pue
    return dict(pue=pue, util=util, wue=wue, ef=ef, it_energy=it_e, energy=E, co2=E*ef, water=it_e*wue)

# worked example from the brief
ex = pd.DataFrame([dict(cooling="evaporative", wetbulb_c=25.9, hyperscale=1, year_online=2025, disclosed=0, pue_obs=np.nan, util_obs=np.nan, wue_obs=np.nan, grid_ef=0.45, it_mw=100.0)])
mc_ex = bottom_up_mc(ex, 20000)
print("Worked example, 100 MW hyperscale in the EEC (undisclosed):")
print("  deterministic (PUE 1.4, util 0.70): %.0f GWh/yr" % (100*1.4*0.70*8760/1000))
for k, unit, s in [("energy","GWh/yr",1e-3),("co2","ktCO2/yr",1e-3),("water","million m3/yr",1e-6)]:
    q = np.quantile(mc_ex[k][:,0], [0.05,0.5,0.95])*s
    print(f"  {k:6s}: P5 {q[0]:8.1f} | P50 {q[1]:8.1f} | P95 {q[2]:8.1f} {unit}")

MC = bottom_up_mc(df)
for k, col, s in [("energy","energy_est_mwh",1),("co2","co2_est_t",1),("water","water_est_m3",1),("pue","pue_est",1)]:
    q = np.quantile(MC[k], [0.05,0.5,0.95], axis=0)
    df[col] = q[1]*s; df[col+"_p05"] = q[0]*s; df[col+"_p95"] = q[2]*s
df["water_m3_day"] = df.water_est_m3/365
print(f"\nPortfolio P50: {df.energy_est_mwh.sum()/1e6:.2f} TWh/yr | {df.co2_est_t.sum()/1e6:.2f} MtCO2/yr | {df.water_est_m3.sum()/1e6:.1f} Mm3/yr   (truth: {df.energy_true_mwh.sum()/1e6:.2f} TWh | {df.co2_true_t.sum()/1e6:.2f} Mt | {df.water_true_m3.sum()/1e6:.1f} Mm3)")
''')
code(r'''
# Tornado: variance share of log(CO2) attributable to each input (one-at-a-time, holding others at their draw mean)
def tornado(mc, j):
    base = dict(pue=mc["pue"][:,j], util=mc["util"][:,j], ef=mc["ef"][:,j])
    itmw = df.it_mw.iloc[j]; out = {}
    for k in base:
        args = {kk: (base[kk] if kk==k else base[kk].mean()) for kk in base}
        y = np.log(itmw*args["util"]*8760*args["pue"]*args["ef"]); out[k] = np.var(y)
    tot = np.var(np.log(mc["co2"][:,j])); return {k: v/tot for k,v in out.items()}
j = int(df.loc[df.disclosed==0].sort_values("it_mw").index[-1])
t = tornado(MC, j)
fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{df.fac_id[j]} ({df.hub[j]}, {df.it_mw[j]:.0f} MW, undisclosed): CO2 posterior", "Variance share of log CO2 by input"))
fig.add_trace(go.Histogram(x=MC["co2"][:,j]/1e3, nbinsx=60, name="ktCO2/yr", marker_color="#636efa"), row=1, col=1)
fig.add_trace(go.Bar(x=list(t.values()), y=["PUE","Utilisation","Grid EF"], orientation="h", marker_color=["#ef553b","#00cc96","#ab63fa"]), row=1, col=2)
fig.update_layout(height=380, showlegend=False); fig.show()

fig = px.scatter(df, x="co2_true_t", y="co2_est_t", error_y=df.co2_est_t_p95-df.co2_est_t, error_y_minus=df.co2_est_t-df.co2_est_t_p05,
                 color="disclosed", log_x=True, log_y=True, hover_name="fac_id", hover_data=["hub","cooling","it_mw"],
                 title="Bottom-up P50 with P5-P95 bars vs hidden truth (tCO2/yr) - prior-only for undisclosed, near-exact for disclosed")
fig.add_shape(type="line", x0=df.co2_true_t.min(), y0=df.co2_true_t.min(), x1=df.co2_true_t.max(), y1=df.co2_true_t.max(), line=dict(dash="dash", color="gray"))
fig.update_layout(height=480); fig.show()
''')

# ---------------------------------------------------------------- 5. Coverage
md(r"""
## 5. Coverage-adjusted portfolio totals: naive gross-up vs Horvitz–Thompson vs Heckman
The disclosed subset covers a minority of facilities and is **systematically greener**. Applying disclosed-average PUE to the undisclosed fleet (the common practice) therefore under-states the portfolio. Inverse-probability weighting on *observable* covariates (Horvitz–Thompson) corrects the MAR part; the Heckman two-step (probit selection → inverse Mills ratio in the outcome equation, with CDP reporting, multinational status and the national disclosure regime as exclusion restrictions) corrects the **MNAR** part — partially in practice, because two-step estimates of ρσ are noisy with the ~100-facility disclosure samples typical of ASEAN; the engineering prior of §4, which is centred by construction, is the natural complement. Every estimator uses the same utilisation assumption and known MW / grid EF, so only the PUE imputation differs, and everything is scored against the hidden truth.
""")
code(r'''
truth_co2 = df.co2_true_t.sum(); disc = df.disclosed==1; u = ~disc
util_bar = df.loc[disc,"util_obs"].mean()          # one utilisation assumption for every estimator (utilisation is independent of disclosure in the DGP) -> only the PUE imputation differs
def assemble(pue_hat_undisc):
    """Portfolio CO2 = disclosed actuals + sum over undisclosed of MW x util x 8760 x PUE_hat x EF."""
    return df.co2_obs_t[disc].sum() + (df.it_mw[u]*util_bar*8760*pue_hat_undisc*df.grid_ef[u]).sum()
# (a) naive: pooled and cooling-specific disclosed means
pue_naive = pd.Series(df.pue_obs[disc].mean(), index=df.index[u]); pue_naive_cool = df.groupby("cooling").pue_obs.transform("mean")[u]
# (b) Horvitz-Thompson / IPW: logistic disclosure propensity on observables -> weighted cooling-specific means
Xp = sm.add_constant(df[["multinational","log_mw","parent_reports_cdp","s_disclosure"]].astype(float))
logit = sm.Logit(df.disclosed, Xp).fit(disp=0); df["pi_hat"] = np.clip(logit.predict(Xp), 0.02, 0.98)
w = 1/df.pi_hat[disc]; ipw = (df.pue_obs[disc]*w).groupby(df.cooling[disc]).sum()/w.groupby(df.cooling[disc]).sum(); pue_ipw = df.cooling[u].map(ipw)
# (c) Heckman two-step: probit selection -> inverse Mills ratio -> outcome OLS on disclosers (exclusion restrictions: CDP reporting, multinational, disclosure regime)
probit = sm.Probit(df.disclosed, Xp).fit(disp=0); zg = Xp.values @ probit.params.values
df["imr"] = np.where(disc, st.norm.pdf(zg)/st.norm.cdf(zg), -st.norm.pdf(zg)/(1-st.norm.cdf(zg)))     # E[eps | disclosed] = rho*sigma*lambda ; E[eps | not disclosed] = -rho*sigma*phi/(1-Phi)
Xo = pd.get_dummies(df[["cooling"]], drop_first=True).astype(float)
Xo["log_mw"]=df.log_mw; Xo["year_c"]=df.year_c; Xo["wetbulb_c"]=df.wetbulb_c-25; Xo["hyperscale"]=df.hyperscale; Xo = sm.add_constant(Xo)
Xo_h = Xo.copy(); Xo_h["imr"] = df.imr
ols_naive = sm.OLS(df.pue_obs[disc], Xo[disc]).fit(); ols_heck = sm.OLS(df.pue_obs[disc], Xo_h[disc]).fit()
df["pue_hat_ols"] = ols_naive.predict(Xo); df["pue_hat_heck"] = ols_heck.predict(Xo_h)
rs = ols_heck.params["imr"]
print("Probit selection equation: " + ", ".join(f"{k} {v:+.2f}" for k, v in probit.params.items()))
print(f"Heckman selection term rho*sigma = {rs:+.3f} (p = {ols_heck.pvalues['imr']:.3f}) -> " + ("disclosers have LOWER PUE than their observables predict: MNAR / greener-discloser selection" if rs < 0 else "no evidence of selection on the unobserved outcome"))
EST = {"Naive: disclosed mean PUE": pue_naive, "Naive: disclosed mean by cooling": pue_naive_cool, "Horvitz-Thompson / IPW by cooling": pue_ipw,
       "OLS on disclosers (X only)": df.pue_hat_ols[u], "Heckman two-step (IMR-corrected)": df.pue_hat_heck[u], "Bottom-up MC prior P50 (no disclosure data)": df.pue_est[u]}
tab = pd.DataFrame({name: {"PUE_bias_undisclosed": (p-df.pue_true[u]).mean(), "PUE_RMSE_undisclosed": np.sqrt(((p-df.pue_true[u])**2).mean()), "estimate_tCO2": assemble(p)} for name, p in EST.items()}).T
tab["truth_tCO2"] = truth_co2; tab["bias_%"] = 100*(tab.estimate_tCO2/truth_co2-1)
display(tab.round(3))
fig = make_subplots(rows=1, cols=2, subplot_titles=("True PUE: disclosed vs undisclosed (the selection problem)", "Portfolio CO2 bias vs hidden truth by estimator (%)"))
fig.add_trace(go.Histogram(x=df.pue_true[disc], name="disclosed", opacity=0.6, nbinsx=30), row=1, col=1); fig.add_trace(go.Histogram(x=df.pue_true[u], name="undisclosed", opacity=0.6, nbinsx=30), row=1, col=1)
fig.add_trace(go.Bar(x=list(tab.index), y=tab["bias_%"], marker_color=np.where(tab["bias_%"]<0, "#c0392b", "#2980b9"), showlegend=False), row=1, col=2)
fig.update_layout(barmode="overlay", height=460, xaxis2_tickangle=-25); fig.show()
''')

# ---------------------------------------------------------------- 6. Multilevel
md(r"""
## 6. Hierarchical partial pooling across jurisdictions
Laos, Brunei, Myanmar and Cambodia have only a handful of disclosed facilities each. Selection is handled upstream: the inputs are the Heckman-corrected residuals from §5, and the MixedLM carries the selection term as a fixed effect. A multilevel model (facility → country → region) shrinks thin-country estimates toward the regional mean *by an amount that depends on how thin they are*, and widens their intervals. Shown two ways: closed-form empirical-Bayes shrinkage of country means, and a `statsmodels` MixedLM (random country intercept, fixed effects for cooling, size, vintage, climate) used to predict PUE for **every** facility, disclosed or not. An optional PyMC block runs a full Bayesian version when the library is available (it is pre-installed on Colab).
""")
code(r'''
# Inputs are the Heckman-corrected residuals from S5 (selection handled upstream), so pooling is evaluated on its own terms.
d1 = df[disc].copy(); d1["resid"] = d1.pue_obs - ols_heck.predict(Xo_h[disc])           # residual after X and the selection term -> contains country + hub effects + noise
xpart_all = ols_heck.predict(Xo_h)                                                          # X-part for every facility (own IMR branch)
g = d1.groupby("country").resid.agg(["mean","count"]).rename(columns={"mean":"resid_mean","count":"n"})
sigma2 = d1.resid.var(); g["se2"] = sigma2/g.n
tau2 = max(g.resid_mean.var() - g.se2.mean(), 1e-5); g["B"] = tau2/(tau2+g.se2); g["eb_effect"] = g.B*g.resid_mean
g = g.reindex(sorted(df.country.unique())); g["n"] = g.n.fillna(0).astype(int); g["eb_effect"] = g.eb_effect.fillna(0.0); g["B"] = g.B.fillna(0.0)
xp_c = pd.Series(xpart_all, index=df.index).groupby(df.country).mean()
g["no_pool"] = xp_c + g.resid_mean.fillna(0.0); g["complete_pool"] = xp_c; g["partial_pool_EB"] = xp_c + g.eb_effect
g["truth_all_facilities"] = df.groupby("country").pue_true.mean()
# MixedLM: random country intercept + the Heckman selection term as a fixed effect (= selection-corrected multilevel model)
mlm = smf.mixedlm("pue_obs ~ C(cooling) + log_mw + year_c + I(wetbulb_c-25) + hyperscale + imr", data=d1, groups=d1["iso3"]).fit(reml=True, method=["lbfgs"])
re = {k: float(v.iloc[0]) for k, v in mlm.random_effects.items()}
df["pue_hat_mlm"] = mlm.predict(df) + df.iso3.map(re).fillna(0.0)
g["MixedLM_pred_all"] = df.groupby("country").pue_hat_mlm.mean()
print(f"EB shrinkage on country residual means: tau^2 = {tau2:.4f}, within sigma^2 = {sigma2:.4f} | MixedLM country RE variance = {float(mlm.cov_re.iloc[0,0]):.4f}, selection term = {mlm.params['imr']:+.3f}")
for c in ["no_pool","complete_pool","partial_pool_EB","MixedLM_pred_all"]:
    print(f"   RMSE vs truth (country mean PUE): {c:18s} {np.sqrt(((g[c]-g.truth_all_facilities)**2).mean()):.4f}")
display(g[["n","no_pool","complete_pool","partial_pool_EB","MixedLM_pred_all","truth_all_facilities","B"]].round(3))
gg = g.reset_index().rename(columns={"index":"country"}).melt(id_vars=["country","n"], value_vars=["no_pool","partial_pool_EB","MixedLM_pred_all","truth_all_facilities"], var_name="estimate", value_name="PUE")
fig = px.scatter(gg, x="country", y="PUE", color="estimate", symbol="estimate", size=np.where(gg.estimate=="truth_all_facilities", 14, 9), size_max=14,
                 title="Country-level PUE: no pooling vs EB partial pooling vs selection-corrected MixedLM vs hidden truth (labels = n disclosed)", text=np.where(gg.estimate=="no_pool", gg.n.astype(str), ""))
fig.update_traces(textposition="top center"); fig.update_layout(height=450); fig.show()
''')
code(r'''
# Optional full-Bayesian version (runs if PyMC is importable; Colab ships PyMC)
try:
    import pymc as pm, arviz as az
    d1 = df[disc].copy(); ci = pd.Categorical(d1.iso3); cool = pd.get_dummies(d1.cooling, drop_first=True).astype(float).values
    X = np.column_stack([cool, d1.log_mw-d1.log_mw.mean(), d1.year_c, d1.wetbulb_c-25, d1.hyperscale, d1.imr])
    with pm.Model() as m:
        mu_r = pm.Normal("mu_region", 1.45, 0.2); tau = pm.HalfNormal("tau_country", 0.1)
        a = pm.Normal("a_country", mu_r, tau, shape=len(ci.categories))
        b = pm.Normal("beta", 0, 0.3, shape=X.shape[1]); s = pm.HalfNormal("sigma", 0.1)
        pm.Normal("y", a[ci.codes] + X@b, s, observed=d1.pue_obs.values)
        idata = pm.sample(1000, tune=1000, chains=2, target_accept=0.9, progressbar=False, random_seed=SEED)
    summ = az.summary(idata, var_names=["a_country","tau_country"]).round(3); summ.index = list(ci.categories)+["tau_country"]
    display(summ)
except ImportError:
    print("PyMC not installed in this environment - skipping the full-Bayesian block (it runs on Colab). MixedLM above is the frequentist equivalent.")
''')

# ---------------------------------------------------------------- 7. Missing data
md(r"""
## 7. Missing-data machinery: diagnostics, MICE + Rubin's rules, Heckman-in-MICE, δ-sensitivity, ML + conformal, matrix completion
1. **Mechanism diagnostics** — standardised mean differences and a logistic missingness model reject MCAR; the Heckman ρσ term in §5 indicates MNAR.
2. **MICE** (`IterativeImputer` with Bayesian ridge and posterior sampling, *m* = 10) using alternative-data auxiliaries (night-lights, thermal anomaly, footprint, market-DB MW). Rubin's rules pool between-imputation variance.
3. **Heckman-in-MICE** (Galimard et al.): the inverse Mills ratio enters each imputation model as an auxiliary predictor; a **δ-adjustment** (pattern-mixture) sensitivity shifts imputed values by δ ∈ {0, +0.03, +0.06}.
4. **Gradient boosting + split-conformal** 90 % intervals — conformal coverage assumes exchangeability, which MNAR breaks; we check the empirical coverage on the hidden truth with and without a Heckman-implied shift.
5. **SoftImpute** low-rank matrix completion across the facility × metric matrix.
""")
code(r'''
# 1. mechanism diagnostics
cov = ["it_mw","year_online","multinational","parent_reports_cdp","hyperscale","s_disclosure","nightlight_rad","thermal_anom_k"]
smd = pd.DataFrame({"disclosed_mean": df.loc[disc,cov].mean(), "undisclosed_mean": df.loc[~disc,cov].mean()})
smd["SMD"] = (smd.disclosed_mean-smd.undisclosed_mean)/np.sqrt((df.loc[disc,cov].var()+df.loc[~disc,cov].var())/2)
print("Standardised mean differences (|SMD|>0.2 => not MCAR):"); display(smd.round(3))
print(f"Logistic missingness model pseudo-R2 = {logit.prsquared:.3f}; LR p-value = {logit.llr_pvalue:.2e}  -> disclosure depends on observables (not MCAR)")
print(f"Heckman rho*sigma on PUE = {ols_heck.params['imr']:.3f} (p={ols_heck.pvalues['imr']:.3f}) -> residual dependence on the unobserved outcome (MNAR)")
''')
code(r'''
# 2. MICE with alternative-data auxiliaries + Rubin's rules; 3. Heckman-in-MICE + delta sensitivity
aux = ["log_mw","year_c","hyperscale","multinational","nightlight_rad","thermal_anom_k","wetbulb_c","s_disclosure"]
Xc = pd.concat([df[["pue_obs","wue_obs","util_obs"]], df[aux], pd.get_dummies(df.cooling, drop_first=True).astype(float), np.log(df.footprint_m2).rename("log_footprint"), np.log(df.marketdb_mw.clip(0.3)).rename("log_marketdb")], axis=1)
def mice(X, m=10, extra=None):
    imps = []
    for k in range(m):
        Xk = X.copy() if extra is None else pd.concat([X, extra], axis=1)
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=20, random_state=SEED+k, skip_complete=True)
        imps.append(pd.DataFrame(imp.fit_transform(Xk), columns=Xk.columns, index=Xk.index)[["pue_obs","wue_obs","util_obs"]])
    return np.stack([i.values for i in imps])   # (m, N, 3)
M_plain = mice(Xc); M_heck = mice(Xc, extra=df[["imr"]])
def rubin(M):
    qbar = M.mean(0); B = M.var(0, ddof=1); T = (1+1/M.shape[0])*B   # within-var of a point imputation is 0 -> total = (1+1/m)B
    return qbar, np.sqrt(T)
pue_mice, se_mice = rubin(M_plain); pue_hm, se_hm = rubin(M_heck)
df["pue_mice"] = pue_mice[:,0]; df["pue_mice_se"] = se_mice[:,0]; df["pue_hmice"] = pue_hm[:,0]; df["pue_hmice_se"] = se_hm[:,0]
df["wue_mice"] = np.clip(pue_mice[:,1], 0.05, None); df["util_mice"] = np.clip(pue_mice[:,2], 0.2, 0.98)
u = ~disc
rows = []
for name, col in [("Disclosed-mean by cooling", None), ("MICE (alt-data aux)", "pue_mice"), ("Heckman-in-MICE", "pue_hmice"), ("Heckman two-step (S5)", "pue_hat_heck"), ("MixedLM (S6)", "pue_hat_mlm")]:
    pred = df.groupby("cooling").pue_obs.transform("mean") if col is None else df[col]
    err = pred[u]-df.pue_true[u]; rows.append(dict(method=name, bias=err.mean(), MAE=err.abs().mean(), RMSE=np.sqrt((err**2).mean())))
for dlt in [0.03, 0.06]:
    err = (df.pue_mice+dlt)[u]-df.pue_true[u]; rows.append(dict(method=f"MICE + delta {dlt:+.2f} (MNAR sensitivity)", bias=err.mean(), MAE=err.abs().mean(), RMSE=np.sqrt((err**2).mean())))
mice_tab = pd.DataFrame(rows).set_index("method"); print("PUE prediction error on UNDISCLOSED facilities (vs hidden truth):"); display(mice_tab.round(4))
''')
code(r'''
# 4. Gradient boosting + split-conformal intervals (and the Heckman shift)
feat = aux + ["log_footprint","log_marketdb","evaporative","liquid","imr"]
Xf = Xc.drop(columns=["pue_obs","wue_obs","util_obs"]).copy(); Xf["imr"] = df.imr
Xd, yd = Xf[disc].drop(columns="imr"), df.pue_obs[disc]
Xtr, Xca, ytr, yca = train_test_split(Xd, yd, test_size=0.40, random_state=SEED)
gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED).fit(Xtr, ytr)
alpha = 0.10; r_ca = np.abs(yca - gbm.predict(Xca)); n_ca = len(r_ca)
q_conf = np.quantile(r_ca, min(1.0, np.ceil((n_ca+1)*(1-alpha))/n_ca))
df["pue_gbm"] = gbm.predict(Xf.drop(columns="imr")); df["pue_gbm_lo"] = df.pue_gbm-q_conf; df["pue_gbm_hi"] = df.pue_gbm+q_conf
shift = ols_heck.params["imr"]*df.imr          # Heckman-implied conditional shift for non-disclosers (negative IMR branch)
df["pue_gbm_shift"] = df.pue_gbm + np.where(disc, 0.0, shift - shift[disc].mean())
cov_gbm = ((df.pue_true>=df.pue_gbm_lo)&(df.pue_true<=df.pue_gbm_hi))[u].mean()
cov_shift = ((df.pue_true>=df.pue_gbm_shift-q_conf)&(df.pue_true<=df.pue_gbm_shift+q_conf))[u].mean()
print(f"Split-conformal 90% interval half-width q = {q_conf:.3f} PUE")
print(f"Empirical coverage on undisclosed truth: GBM {cov_gbm:.0%} (exchangeability broken by MNAR)  |  GBM + Heckman shift {cov_shift:.0%}")
e1 = (df.pue_gbm-df.pue_true)[u]; e2 = (df.pue_gbm_shift-df.pue_true)[u]
print(f"Bias: GBM {e1.mean():+.3f} | GBM+shift {e2.mean():+.3f}   RMSE: GBM {np.sqrt((e1**2).mean()):.3f} | GBM+shift {np.sqrt((e2**2).mean()):.3f}")
# 5. SoftImpute matrix completion on standardised facility x metric matrix
def soft_impute(M, rank=3, lam=0.3, iters=300):
    mask = ~np.isnan(M); Z = np.where(mask, M, 0.0)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False); s = np.maximum(s-lam, 0)[:rank]
        L = (U[:,:rank]*s) @ Vt[:rank]; Zn = np.where(mask, M, L)
        if np.linalg.norm(Zn-Z) < 1e-7: Z = Zn; break
        Z = Zn
    return Z
Mraw = Xc[["pue_obs","wue_obs","util_obs","log_mw","nightlight_rad","thermal_anom_k","log_footprint","log_marketdb","year_c","wetbulb_c"]].astype(float)
mu_, sd_ = Mraw.mean(), Mraw.std(); Mstd = ((Mraw-mu_)/sd_).values
Mc = soft_impute(Mstd, rank=4, lam=0.1); df["pue_softimpute"] = Mc[:,0]*sd_["pue_obs"]+mu_["pue_obs"]
e3 = (df.pue_softimpute-df.pue_true)[u]; print(f"SoftImpute (rank 3): bias {e3.mean():+.3f} | RMSE {np.sqrt((e3**2).mean()):.3f}")
fig = px.scatter(df[u], x="pue_true", y="pue_gbm_shift", error_y=[q_conf]*int(u.sum()), color="cooling", hover_name="fac_id",
                 title="Undisclosed facilities: GBM + Heckman shift with split-conformal 90% bars vs hidden true PUE")
fig.add_shape(type="line", x0=1.1, y0=1.1, x1=2.0, y1=2.0, line=dict(dash="dash", color="gray")); fig.update_layout(height=450); fig.show()
''')

# ---------------------------------------------------------------- 8. Fay-Herriot / nowcast
md(r"""
## 8. Data fusion and small-area estimation (Fay–Herriot) + nowcasting
Hub-level mean PUE is needed for siting policy and for estate-level EIA (IEAT, Sedenak, Cikarang), but each hub has only a handful of disclosures. The **Fay–Herriot** area-level model combines the noisy direct estimate (here the hub mean of the Heckman-corrected residuals, added back to the facility X-part) with a regression on hub-level auxiliaries (night-light index, share evaporative-cooled, share hyperscale, climate, RCI) and returns an EBLUP with a shrinkage weight γ = A/(A+D). We then **nowcast** national Thai capacity from lagged disclosures plus high-frequency proxies (permits, job posts, satellite construction index).
""")
code(r'''
def fay_herriot(theta, D, X, iters=200):
    n, p = X.shape; A = max(np.var(theta), 1e-4)
    for _ in range(iters):
        w = 1/(A+D); beta = np.linalg.solve((X*w[:,None]).T@X, (X*w[:,None]).T@theta); r = theta-X@beta
        f = np.sum(r**2*w)-(n-p); fp = -np.sum(r**2*w**2); An = max(A - f/fp, 1e-6)
        if abs(An-A) < 1e-9: A = An; break
        A = An
    w = 1/(A+D); beta = np.linalg.solve((X*w[:,None]).T@X, (X*w[:,None]).T@theta)
    gamma = A/(A+D); eblup = gamma*theta+(1-gamma)*(X@beta); mse = gamma*D
    return eblup, gamma, beta, A, mse
df["resid_heck"] = np.where(disc, df.pue_obs - xpart_all, np.nan); df["xpart_heck"] = xpart_all
hub = df.groupby(["hub_id","hub","iso3","country"]).agg(n=("fac_id","size"), n_disc=("disclosed","sum"), it_mw=("it_mw","sum"), direct_resid=("resid_heck","mean"), xpart=("xpart_heck","mean"),
      truth=("pue_true","mean"), evap_share=("cooling", lambda s: (s=="evaporative").mean()), hyper_share=("hyperscale","mean"), lat=("lat","mean"), lon=("lon","mean"),
      co2_est=("co2_est_t","sum"), water_est=("water_est_m3","sum"), water_stress=("water_stress","first"), RCI=("RCI","first")).reset_index()
hub = hub.merge(HUBS[["hub_id","nightlight_idx","dist_cable_km","grid_ef","elec_usd_kwh","gdp_pc_kusd","wetbulb_c"]], on="hub_id")
s2_pool = df.resid_heck.var()
hub["D"] = s2_pool/hub.n_disc.clip(lower=1); hub.loc[hub.n_disc==0, "D"] = s2_pool*4     # no disclosed facility: direct estimate is the zero residual with a very large variance
hub["direct_resid"] = hub.direct_resid.fillna(0.0)
Xh = np.column_stack([np.ones(len(hub)), hub.nightlight_idx, hub.evap_share, hub.hyper_share, hub.wetbulb_c-25, hub.RCI])
eb_resid, hub["gamma"], beta_fh, A_fh, hub["mse"] = fay_herriot(hub.direct_resid.values, hub.D.values, Xh)
hub["direct"] = hub.xpart + hub.direct_resid; hub["eblup"] = hub.xpart + eb_resid          # add the facility-level X-part (with selection term) back
print(f"Fay-Herriot: model variance A = {A_fh:.5f}; beta = {np.round(beta_fh,3)}")
print("RMSE vs hub truth  - direct: %.4f | EBLUP: %.4f" % (np.sqrt(((hub.direct-hub.truth)**2).mean()), np.sqrt(((hub.eblup-hub.truth)**2).mean())))
display(hub[["hub","n","n_disc","direct","eblup","gamma","truth"]].round(3))
fig = go.Figure()
fig.add_trace(go.Scatter(x=hub.hub, y=hub.direct, mode="markers", name="direct (disclosed mean)", marker=dict(size=9, color="#ef553b"), error_y=dict(array=np.sqrt(hub.D), visible=True)))
fig.add_trace(go.Scatter(x=hub.hub, y=hub.eblup, mode="markers", name="Fay-Herriot EBLUP", marker=dict(size=9, color="#636efa"), error_y=dict(array=np.sqrt(hub.mse), visible=True)))
fig.add_trace(go.Scatter(x=hub.hub, y=hub.truth, mode="markers", name="hidden truth (all facilities)", marker=dict(size=11, color="black", symbol="x")))
fig.update_layout(title="Hub-level mean PUE: direct vs EBLUP (shrunk toward auxiliary regression) vs truth", height=450, xaxis_tickangle=-40); fig.show()
''')
code(r'''
# Nowcasting Thai national capacity from lagged disclosures + high-frequency proxies (illustrative quarterly panel)
qs = pd.period_range("2019Q1","2026Q2", freq="Q"); T = len(qs); rq = np.random.default_rng(SEED+7)
true_cap = 300*np.exp(0.055*np.arange(T)) + rq.normal(0, 12, T)                 # MW online (illustrative)
disclosed_lag2 = np.r_[np.nan, np.nan, true_cap[:-2]*0.62 + rq.normal(0, 25, T-2)]   # market/market-DB coverage ~62%, two-quarter lag
permits = np.r_[true_cap[2:]-true_cap[:-2], np.nan, np.nan]*0.9 + rq.normal(0, 8, T)  # permits lead capacity by ~2 quarters
jobs = 0.8*true_cap + rq.normal(0, 120, T); sat_constr = 0.15*np.r_[true_cap[1:]-true_cap[:-1], np.nan] + rq.normal(0, 5, T)
nc = pd.DataFrame(dict(q=qs.astype(str), true_cap=true_cap, disclosed_lag2=disclosed_lag2, permits_lead=permits, job_posts=jobs, sat_construction=sat_constr))
tr = nc.iloc[2:-4].dropna(); te = nc.iloc[-6:-2].dropna()
mdl = sm.OLS(tr.true_cap, sm.add_constant(tr[["disclosed_lag2","job_posts","sat_construction"]])).fit()
nc["nowcast"] = mdl.predict(sm.add_constant(nc[["disclosed_lag2","job_posts","sat_construction"]]))
print(mdl.summary().tables[1]); print("Out-of-sample MAPE on last held-out quarters: %.1f%%" % (100*np.mean(np.abs(nc.nowcast.iloc[-6:-2]-nc.true_cap.iloc[-6:-2])/nc.true_cap.iloc[-6:-2])))
fig = px.line(nc, x="q", y=["true_cap","disclosed_lag2","nowcast"], title="Thailand DC capacity nowcast (illustrative): lagged disclosures + job posts + satellite construction index", labels={"value":"MW online"}); fig.update_layout(height=380); fig.show()
''')

# ---------------------------------------------------------------- 9. Geo-econometrics
md(r"""
## 9. Geo-econometrics at hub level
Hub-level capacity and estimated emissions are spatially structured: clusters around Singapore–Johor–Batam, the Thai EEC and greater Jakarta. We build **haversine inverse-distance spatial weights**, test **Moran's I** with permutation inference, estimate an **SLX** model (spatially lagged covariates) and a **spatial-lag (SAR) model by 2SLS** (instruments WX, W²X), a **Poisson** model of facility counts, and finally a **two-way fixed-effects difference-in-differences** on a quarterly hub panel — Singapore's 2019–2022 moratorium as treatment, Johor/Batam spillover — with an event-study plot. The DiD panel is simulated with *known* effects so recovery can be checked; the design transfers directly to real quarterly capacity data.
""")
code(r'''
def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2); dphi = p2-p1; dl = np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2; return 2*6371.0*np.arcsin(np.sqrt(a))
LAT, LON = hub.lat.values, hub.lon.values
Dm = haversine(LAT[:,None], LON[:,None], LAT[None,:], LON[None,:])
W = np.where((Dm>0)&(Dm<=900), 1/np.maximum(Dm,20), 0.0)          # inverse distance within 900 km
for i in range(len(W)):                                            # guarantee >=3 neighbours (k-nearest fallback)
    if (W[i]>0).sum() < 3:
        nn = np.argsort(Dm[i])[1:4]; W[i, nn] = 1/np.maximum(Dm[i, nn], 20)
W = W/W.sum(1, keepdims=True)
def morans_I(y, W, nperm=999, seed=SEED):
    r = np.random.default_rng(seed); z = np.asarray(y, float)-np.mean(y); n = len(z); S0 = W.sum()
    I = (n/S0)*(z@W@z)/(z@z); perms = np.array([(n/S0)*(zp@W@zp)/(zp@zp) for zp in (r.permutation(z) for _ in range(nperm))])
    p = (np.sum(perms>=I)+1)/(nperm+1) if I > -1/(n-1) else (np.sum(perms<=I)+1)/(nperm+1)
    return I, -1/(n-1), p
hub["log_mw"] = np.log(hub.it_mw); hub["log_co2"] = np.log(hub.co2_est); hub["log_price"] = np.log(hub.elec_usd_kwh); hub["log_cable"] = np.log(hub.dist_cable_km)
for v in ["log_mw","log_co2","eblup"]:
    I, EI, p = morans_I(hub[v].values, W); print(f"Moran's I for {v:8s}: I = {I:+.3f} (E[I] = {EI:+.3f}), permutation p = {p:.3f}")
# facility-level Moran's I with k-nearest-neighbour weights (k = 8) on all facility points
Df = haversine(df.lat.values[:,None], df.lon.values[:,None], df.lat.values[None,:], df.lon.values[None,:]); np.fill_diagonal(Df, np.inf)
Wf = np.zeros_like(Df); nn8 = np.argsort(Df, axis=1)[:, :8]
for i in range(len(df)): Wf[i, nn8[i]] = 1.0
Wf = Wf/Wf.sum(1, keepdims=True)
I_fac, _, p_fac = morans_I(np.log(df.co2_est_t.values), Wf); I_fac2, _, p_fac2 = morans_I(df.pue_est.values, Wf)
print(f"Facility-level Moran's I (k=8 NN): log CO2 estimate I = {I_fac:+.3f} (p = {p_fac:.3f}) | estimated PUE I = {I_fac2:+.3f} (p = {p_fac2:.3f})")
z = (hub.log_mw-hub.log_mw.mean())/hub.log_mw.std(); wz = W@z
fig = px.scatter(x=z, y=wz, text=hub.hub.str.split(" (", regex=False).str[0], title="Moran scatterplot of log hub capacity (x: z-score, y: spatial lag)", labels={"x":"z(log MW)","y":"W z(log MW)"})
fig.add_shape(type="line", x0=z.min(), y0=np.polyval(np.polyfit(z,wz,1), z.min()), x1=z.max(), y1=np.polyval(np.polyfit(z,wz,1), z.max()), line=dict(color="crimson"))
fig.add_hline(y=0, line_dash="dot"); fig.add_vline(x=0, line_dash="dot"); fig.update_traces(textposition="top center", textfont_size=9); fig.update_layout(height=480); fig.show()
''')
code(r'''
# SLX and SAR (2SLS) models of hub capacity; Poisson model of facility counts
Xv = ["log_price","water_stress","RCI","log_cable","grid_ef"]
Xmat = hub[Xv].values; WX = W@Xmat
slx = sm.OLS(hub.log_mw, sm.add_constant(pd.concat([hub[Xv], pd.DataFrame(WX, columns=["W_"+v for v in Xv], index=hub.index)], axis=1))).fit(cov_type="HC1")
print("SLX (OLS with spatially lagged X), HC1 SE:"); print(slx.summary().tables[1])
def sar_2sls(y, X, W):
    Wy = W@y; Xs = np.column_stack([Wy, X]); WX = W@X[:,1:]; Z = np.column_stack([X, WX, W@WX])
    Pz = Z@np.linalg.pinv(Z.T@Z)@Z.T; Xh = Pz@Xs
    b = np.linalg.solve(Xh.T@Xs, Xh.T@y); e = y-Xs@b; s2 = e@e/(len(y)-Xs.shape[1]); se = np.sqrt(np.diag(s2*np.linalg.pinv(Xh.T@Xh)))
    return b, se
Xsar = np.column_stack([np.ones(len(hub)), Xmat]); b, se = sar_2sls(hub.log_mw.values, Xsar, W)
sar = pd.DataFrame({"coef": b, "se": se}, index=["rho (W*log_mw)","const"]+Xv); sar["t"] = sar.coef/sar.se
print("\nSpatial-lag (SAR) model by 2SLS, instruments [X, WX, W2X]:"); display(sar.round(3))
pois = smf.glm("n ~ log_price + water_stress + RCI + log_cable + gdp_pc_kusd", data=hub, family=sm.families.Poisson()).fit()
print("\nPoisson model of facility counts per hub:"); print(pois.summary().tables[1])
''')
code(r'''
# TWFE difference-in-differences + event study on a simulated quarterly hub panel (known effects)
rp = np.random.default_rng(SEED+11); qs = pd.period_range("2018Q1","2026Q2", freq="Q")
base = (hub.set_index("hub_id").it_mw/len(qs)*1.6).to_dict()
MORA = (pd.Period("2019Q2","Q"), pd.Period("2022Q2","Q")); JOHOR_VET = pd.Period("2024Q3","Q")
EFF_SG, EFF_SPILL, EFF_JOHOR = -0.70, +0.35, -0.30   # true effects on quarterly additions, in units of each hub's baseline quarterly addition
pan = []
for h in hub.itertuples():
    for t, qq in enumerate(qs):
        trend = 1 + 0.03*t; sg = int(h.iso3=="SGP" and MORA[0] <= qq <= MORA[1])
        spill = int(("Johor" in h.hub or "Batam" in h.hub) and MORA[0] <= qq <= MORA[1]); jv = int("Johor" in h.hub and qq >= JOHOR_VET)
        mu = base[h.hub_id]*(trend + EFF_SG*sg + EFF_SPILL*spill + EFF_JOHOR*jv)          # additive in baseline units -> TWFE correctly specified
        pan.append(dict(hub_id=h.hub_id, hub=h.hub, iso3=h.iso3, q=str(qq), t=t, add_mw=max(mu + rp.normal(0, 0.25*base[h.hub_id]), 0), sg_mora=sg, spill=spill, johor_vet=jv, base=base[h.hub_id]))
pan = pd.DataFrame(pan); pan["add_rel"] = pan.add_mw/pan.base
did = smf.ols("add_rel ~ sg_mora + spill + johor_vet + C(hub_id) + C(q)", data=pan).fit(cov_type="cluster", cov_kwds={"groups": pan.hub_id})
print("TWFE DiD on additions relative to hub baseline (cluster-robust by hub):")
print(pd.DataFrame({"coef": did.params[["sg_mora","spill","johor_vet"]], "se": did.bse[["sg_mora","spill","johor_vet"]], "true_effect": [EFF_SG, EFF_SPILL, EFF_JOHOR]}).round(3))
# event study around the Singapore moratorium
pan["rel"] = pan.t - list(qs).index(MORA[0]); pan["sg"] = (pan.iso3=="SGP").astype(int)
ev = pan[(pan.rel>=-6)&(pan.rel<=12)].copy()
for k in range(-6, 13):
    if k != -1: ev[f"ev_{k+6}"] = ((ev.rel==k)&(ev.sg==1)).astype(int)
evm = smf.ols("add_rel ~ " + " + ".join([f"ev_{k+6}" for k in range(-6,13) if k!=-1]) + " + C(hub_id) + C(q)", data=ev).fit(cov_type="cluster", cov_kwds={"groups": ev.hub_id})
es = pd.DataFrame({"rel_q": [k for k in range(-6,13) if k!=-1], "coef": [evm.params[f"ev_{k+6}"] for k in range(-6,13) if k!=-1], "se": [evm.bse[f"ev_{k+6}"] for k in range(-6,13) if k!=-1]})
es = pd.concat([es, pd.DataFrame({"rel_q":[-1],"coef":[0.0],"se":[0.0]})]).sort_values("rel_q")
fig = go.Figure(go.Scatter(x=es.rel_q, y=es.coef, mode="markers+lines", error_y=dict(array=1.96*es.se), name="SG x quarter"))
fig.add_hline(y=0, line_dash="dot"); fig.add_vrect(x0=-0.5, x1=12.5, fillcolor="orange", opacity=0.1, annotation_text="moratorium")
fig.update_layout(title="Event study: Singapore capacity additions around the 2019 moratorium (relative quarters; 95% CI)", height=400, xaxis_title="quarters relative to 2019Q2", yaxis_title="effect on additions / baseline"); fig.show()
''')

# ---------------------------------------------------------------- 10. Scenarios
md(r"""
## 10. Scenario and stress overlays (transition, physical, water, heat)
- **Transition:** four NGFS-style carbon-price paths (illustrative ASEAN levels, USD/tCO₂) × estimated location-based emissions with path-specific grid decarbonisation, discounted at 6 % to 2050.
- **Physical (flood):** facility 100-year depth (hub median × lognormal spread; 40 % of sites assumed elevated) → return-period depths → a data-centre depth–damage curve → **expected annual damage (EAD)** on replacement value (USD 10 M per IT-MW) plus business interruption; baseline vs a 2050 SSP3-7.0-type uplift; **VaR-95** from the Monte-Carlo.
- **Water:** basin stress → curtailment probability → expected curtailed MW-days plus the cost of switching potable withdrawals to reclaimed water (benchmark: Johor RM 5.33/m³).
- **Heat:** +1.5 °C wet-bulb by 2050 → cooling-technology-specific PUE uplift → extra energy and CO₂.
""")
code(r'''
CARBON_PATHS = {"Net Zero 2050": {2025:8,2030:60,2035:110,2040:160,2045:210,2050:260}, "Delayed Transition": {2025:5,2030:5,2035:100,2040:150,2045:200,2050:250},
                "Too Little Too Late": {2025:4,2030:6,2035:60,2040:100,2045:130,2050:160}, "Current Policies": {2025:3,2030:6,2035:9,2040:12,2045:15,2050:18}}
EF_DECLINE = {"Net Zero 2050": 1.0, "Delayed Transition": 0.5, "Too Little Too Late": 0.6, "Current Policies": 0.25}   # multiplier on ef_decline_nz
years = np.arange(2025, 2051); disc_f = 1/(1.06)**(years-2025)
efd = df.iso3.map(LEGAL.set_index("iso3").ef_decline_nz).values
npv = {}
for s, pth in CARBON_PATHS.items():
    price = np.interp(years, list(pth), list(pth.values()))
    ef_path = df.grid_ef.values[:,None]*(1-EF_DECLINE[s]*efd[:,None])**(years-2025)[None,:]
    cost = df.energy_est_mwh.values[:,None]*ef_path*price[None,:]
    npv[s] = (cost*disc_f[None,:]).sum(1)
NPV = pd.DataFrame(npv, index=df.index); df["carbon_npv_nz_musd"] = NPV["Net Zero 2050"]/1e6; df["carbon_npv_delayed_musd"] = NPV["Delayed Transition"]/1e6
byc = NPV.groupby(df.country).sum()/1e6
fig = px.bar(byc.reset_index().melt(id_vars="country", var_name="scenario", value_name="NPV_MUSD"), x="country", y="NPV_MUSD", color="scenario", barmode="group",
             title="Transition risk: NPV (6%, 2025-2050) of carbon cost on estimated location-based emissions, USD million"); fig.update_layout(height=400); fig.show()
# ---- physical: flood
RP = np.array([10,25,50,100,250,500]); DEPTH_MULT = np.array([0.35,0.60,0.80,1.00,1.25,1.45]); PEX = 1/RP
DD = np.array([[0,0],[0.15,0.05],[0.3,0.15],[0.6,0.35],[1.0,0.60],[2.0,0.85],[4.0,0.95]])
dmg = lambda d: np.interp(d, DD[:,0], DD[:,1])
def ead(depth100, uplift=1.0):
    d = depth100[:,None]*DEPTH_MULT[None,:]*uplift; D = dmg(d)                       # damage fraction by RP
    p = np.r_[PEX, 0.0]; Dfull = np.column_stack([D, D[:,-1]])                          # extend to p->0
    return np.sum(0.5*(Dfull[:,:-1]+Dfull[:,1:])*np.abs(np.diff(p))[None,:], axis=1)   # trapezoid over exceedance prob
repl = df.it_mw*10e6; bi_day = df.it_mw*4000
df["flood_ead_musd"] = (ead(df.flood100_m.values)*(repl + 60*bi_day))/1e6
df["flood_ead_2050_musd"] = (ead(df.flood100_m.values, 1.25)*(repl + 60*bi_day))/1e6
rq = np.random.default_rng(SEED+3); sim = df.flood100_m.values[None,:]*np.exp(rq.normal(0,0.35,(2000,len(df))))
df["flood_var95_musd"] = np.quantile(ead(sim.T.reshape(-1)).reshape(len(df),2000)*(repl.values[:,None]+60*bi_day.values[:,None]), 0.95, axis=1)/1e6
# ---- water
df["p_curtail"] = 0.02 + 0.25/(1+np.exp(-2*(df.water_stress-3.6)))
df["water_cost_musd"] = (df.p_curtail*10*df.it_mw*4000 + np.where(df.water_source=="potable", df.water_est_m3*0.6, 0))/1e6   # curtailment + switching premium (USD 0.6/m3)
# ---- heat
DPUE = {"air":0.05,"evaporative":0.04,"liquid":0.02}; df["pue_heat_2050"] = df.pue_est + df.cooling.map(DPUE)
df["co2_heat_delta_t"] = df.it_mw*np.where(disc, df.util_obs, util_bar)*8760*df.cooling.map(DPUE)*df.grid_ef
summ = df.groupby("country").agg(co2_kt=("co2_est_t", lambda s: s.sum()/1e3), carbon_npv_nz=("carbon_npv_nz_musd","sum"), flood_ead=("flood_ead_musd","sum"), flood_ead_2050=("flood_ead_2050_musd","sum"),
                                 water_cost=("water_cost_musd","sum"), heat_co2_kt=("co2_heat_delta_t", lambda s: s.sum()/1e3)).round(2)
print("Scenario overlays by country (USD million/yr for EAD & water; NPV for carbon):"); display(summ)
fig = px.scatter(df, x="flood_ead_musd", y="carbon_npv_nz_musd", size="it_mw", color="water_stress", hover_name="fac_id", hover_data=["hub","cooling"], log_x=True, log_y=True,
                 color_continuous_scale="YlOrRd", title="Facility risk landscape: flood EAD (x) vs carbon NPV Net-Zero (y), colour = basin water stress, size = IT MW"); fig.update_layout(height=480); fig.show()
''')

# ---------------------------------------------------------------- 11. PCAF
md(r"""
## 11. PCAF data-quality scoring and coverage reporting
Every facility estimate is tagged with a PCAF-style data-quality score: **DQ1** verified reported (multinational + CDP), **DQ2** reported unverified, **DQ3** physical-activity estimate (facility IT-MW and cooling technology known), **DQ4** economic-activity proxy (market-database MW only), **DQ5** country-average intensity. The portfolio reports an *emissions-weighted* DQ and the measured / estimated / proxied split — the disclosure a financier or regulator needs to judge the numbers.
""")
code(r'''
def dq(r):
    if r.disclosed and r.multinational and r.parent_reports_cdp: return 1
    if r.disclosed: return 2
    if r.ftype in ("hyperscale","colocation"): return 3
    return 4 if r.permits_count > 0 else 5
df["pcaf_dq"] = df.apply(dq, axis=1); df["provenance"] = df.pcaf_dq.map({1:"measured",2:"measured",3:"estimated",4:"proxied",5:"proxied"})
wdq = (df.pcaf_dq*df.co2_est_t).sum()/df.co2_est_t.sum()
print(f"Emissions-weighted PCAF DQ = {wdq:.2f}")
print(df.groupby("provenance").co2_est_t.sum().div(df.co2_est_t.sum()).mul(100).round(1).rename("% of estimated CO2").to_string())
pc = df.groupby(["country","pcaf_dq"]).co2_est_t.sum().reset_index()
fig = px.bar(pc, x="country", y="co2_est_t", color=pc.pcaf_dq.astype(str), title="Estimated CO2 by PCAF data-quality score (1 = verified reported ... 5 = economic proxy)", labels={"color":"DQ"}, color_discrete_sequence=px.colors.sequential.Viridis_r)
fig.update_layout(height=400); fig.show()
''')

# ---------------------------------------------------------------- 12. Validation
md(r"""
## 12. Validation and backtest against later-disclosed truth
We simulate the arrival of new disclosures (30 % of previously undisclosed facilities begin reporting — e.g. EED-style reports from multinationals, or a Thai committee data call) and score every estimator on those facilities: MAE, RMSE, MAPE, bias (mean signed error), 90 % interval coverage, and pinball loss at the 5th/95th percentiles. A calibration curve checks whether the Monte-Carlo intervals are honest at every nominal level. Finally the governance rule from the brief is operationalised: once measured coverage exceeds 50 % of exposure, the pipeline should switch from proxy-heavy imputation to calibration / measurement-error models.
""")
code(r'''
rv = np.random.default_rng(SEED+5); newly = df.index[(~disc) & (rv.random(len(df)) < 0.30)]
truth = df.loc[newly, "pue_true"]
def pinball(y, qhat, tau): d = y-qhat; return np.mean(np.maximum(tau*d, (tau-1)*d))
cands = {
  "Disclosed-mean by cooling": (df.groupby("cooling").pue_obs.transform("mean"), None, None),
  "Bottom-up MC prior (P50)":  (df.pue_est, df.pue_est_p05, df.pue_est_p95),
  "MICE (alt-data aux)":       (df.pue_mice, df.pue_mice-1.645*df.pue_mice_se, df.pue_mice+1.645*df.pue_mice_se),
  "Heckman-in-MICE":           (df.pue_hmice, df.pue_hmice-1.645*df.pue_hmice_se, df.pue_hmice+1.645*df.pue_hmice_se),
  "Heckman two-step":          (df.pue_hat_heck, None, None),
  "MixedLM partial pooling":   (df.pue_hat_mlm, None, None),
  "GBM + conformal":           (df.pue_gbm, df.pue_gbm_lo, df.pue_gbm_hi),
  "GBM + Heckman shift + conformal": (df.pue_gbm_shift, df.pue_gbm_shift-q_conf, df.pue_gbm_shift+q_conf),
  "SoftImpute":                (df.pue_softimpute, None, None),
}
rows = []
for name, (p, lo, hi) in cands.items():
    e = p.loc[newly]-truth; r = dict(method=name, n=len(newly), bias=e.mean(), MAE=e.abs().mean(), RMSE=np.sqrt((e**2).mean()), MAPE_pct=100*(e.abs()/truth).mean())
    if lo is not None:
        r["cov90"] = ((truth>=lo.loc[newly])&(truth<=hi.loc[newly])).mean(); r["pinball"] = 0.5*(pinball(truth, lo.loc[newly], 0.05)+pinball(truth, hi.loc[newly], 0.95))
    rows.append(r)
val = pd.DataFrame(rows).set_index("method"); print(f"Backtest on {len(newly)} newly-disclosing facilities:"); display(val.round(4))
# calibration curve for the Monte-Carlo bottom-up intervals (all undisclosed, vs truth)
noms = np.linspace(0.1, 0.95, 18); emp = []
for c in noms:
    lo_, hi_ = np.quantile(MC["pue"][:, u.values], [(1-c)/2, 1-(1-c)/2], axis=0); emp.append(np.mean((df.pue_true[u].values>=lo_)&(df.pue_true[u].values<=hi_)))
fig = go.Figure([go.Scatter(x=noms, y=emp, mode="lines+markers", name="MC bottom-up prior"), go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="gray"), name="perfect")])
fig.update_layout(title="Calibration of Monte-Carlo PUE intervals on undisclosed facilities (nominal vs empirical coverage)", xaxis_title="nominal", yaxis_title="empirical", height=400); fig.show()
meas_share = df.loc[df.provenance=="measured","co2_est_t"].sum()/df.co2_est_t.sum()
print(f"Measured share of exposure = {meas_share:.0%} -> regime: {'CALIBRATION / measurement-error models' if meas_share > 0.5 else 'PROXY-HEAVY imputation with Heckman correction'}")
''')

# ---------------------------------------------------------------- 13. EIA improvement
md(r"""
## 13. Improving EIA / EIS screening: from indirect triggers to resource-based tiers
This is the policy pay-off. We (a) measure how much of the register's *environmental load* (energy, CO₂, water) today's rules actually capture in an assessment instrument, jurisdiction by jurisdiction; (b) trace the **threshold frontier** — share of load captured versus number of projects that must be assessed — for candidate resource-based rules; (c) apply every regime's rule to the same register to expose the **harmonisation gap**; and (d) build a **Resource-Based Screening Index (RBSI)** that ranks facilities by estimated energy, water × basin stress, CO₂, flood EAD and heat exposure, assigning Tier A / B / C. Everything is computed from *estimated* values — i.e. exactly the information a regulator has *before* a proponent files anything.
""")
code(r'''
# (a) current capture under home jurisdiction (confirmed instruments only, then including UNVERIFIED Thai items)
df["tier_home"], df["instrument_home"] = zip(*[screen_generic(r, r.iso3, include_uncertain=False) for _, r in df.iterrows()])
df["tier_home_unc"], _ = zip(*[screen_generic(r, r.iso3, include_uncertain=True) for _, r in df.iterrows()])
def capture(mask):
    return pd.Series({"projects_assessed": int(mask.sum()), "share_projects": mask.mean(), "share_MW": df.it_mw[mask].sum()/df.it_mw.sum(),
                      "share_energy": df.energy_est_mwh[mask].sum()/df.energy_est_mwh.sum(), "share_CO2": df.co2_est_t[mask].sum()/df.co2_est_t.sum(), "share_water": df.water_est_m3[mask].sum()/df.water_est_m3.sum()})
cap = pd.DataFrame({"Full assessment (tier 2), confirmed": capture(df.tier_home==2), "Any assessment (tier>=1), confirmed": capture(df.tier_home>=1), "Full assessment incl. UNVERIFIED Thai items": capture(df.tier_home_unc==2)}).T
print("Capture of environmental load by CURRENT home-jurisdiction rules (whole register):"); display(cap.round(3))
byc = df.groupby("country").apply(lambda g: pd.Series({"n": len(g), "tier2_share_CO2": g.co2_est_t[g.tier_home==2].sum()/g.co2_est_t.sum(), "tier2_share_water": g.water_est_m3[g.tier_home==2].sum()/g.water_est_m3.sum(), "tier0_share_projects": (g.tier_home==0).mean()}))
display(byc.round(3))
th = df[df.iso3=="THA"]
print("\nThailand - instrument frequency (confirmed + uncertain) across %d facilities:" % len(th))
inst = pd.Series([t["instrument"].split(":")[0] + (" [UNVERIFIED]" if t["confidence"]=="uncertain" else "") for _, r in th.iterrows() for t in thai_triggers(r, True)]).value_counts()
display(inst.to_frame("facilities"))
''')
code(r'''
# (b) threshold frontier: candidate resource-based rules vs load captured
grid = np.r_[0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 150]
rows = []
for T in grid:
    m1 = df.it_mw >= T; rows.append(dict(rule="IT load >= T MW", T=T, **capture(m1)))
    m2 = (df.it_mw >= T) | (df.water_m3_day >= 300) ; rows.append(dict(rule="IT >= T MW OR water >= 300 m3/day", T=T, **capture(m2)))
    m3 = (df.it_mw >= T) | ((df.water_stress >= 3.4) & (df.water_m3_day >= 100)) | (df.flood100_m >= 1.0); rows.append(dict(rule="IT >= T MW OR (stressed basin & water>=100) OR flood>=1 m", T=T, **capture(m3)))
fr = pd.DataFrame(rows)
fig = px.line(fr, x="projects_assessed", y="share_CO2", color="rule", markers=True, hover_data=["T","share_water","share_energy"],
              title="Threshold frontier: share of estimated CO2 captured vs number of projects assessed (labels = T in MW)", text=fr["T"].astype(str))
cur = cap.loc["Full assessment (tier 2), confirmed"]; cur_u = cap.loc["Full assessment incl. UNVERIFIED Thai items"]
fig.add_trace(go.Scatter(x=[cur.projects_assessed], y=[cur.share_CO2], mode="markers+text", text=["current rules (confirmed)"], textposition="bottom right", marker=dict(size=14, color="black", symbol="star"), name="current regimes"))
fig.add_trace(go.Scatter(x=[cur_u.projects_assessed], y=[cur_u.share_CO2], mode="markers+text", text=["incl. unverified Thai items"], textposition="top left", marker=dict(size=12, color="gray", symbol="star"), name="current + unverified"))
fig.update_traces(textposition="top center", selector=dict(mode="lines+markers+text")); fig.update_layout(height=500, xaxis_title="projects requiring full assessment", yaxis_title="share of estimated CO2 captured"); fig.show()
fw = fr[fr.rule=="IT load >= T MW"]
print("Water capture of a pure IT-MW threshold is weak because water scales with cooling technology, not size:")
display(fw[["T","projects_assessed","share_CO2","share_water"]].round(3).set_index("T").T)
''')
code(r'''
# (c) harmonisation gap: apply every regime to the same register
H = pd.DataFrame({iso: [screen_generic(r, iso)[0] for _, r in df.iterrows()] for iso in LEGAL.iso3}, index=df.index)
harm = pd.DataFrame({iso: {"tier2_share_CO2": df.co2_est_t[H[iso]==2].sum()/df.co2_est_t.sum(), "tier>=1_share_CO2": df.co2_est_t[H[iso]>=1].sum()/df.co2_est_t.sum(),
                            "tier2_share_water": df.water_est_m3[H[iso]==2].sum()/df.water_est_m3.sum(), "share_projects_tier2": (H[iso]==2).mean()} for iso in LEGAL.iso3}).T
fig = px.imshow(harm.T, text_auto=".2f", color_continuous_scale="Blues", aspect="auto", title="Harmonisation gap: share of the SAME register's load captured if each regime's rule applied (columns = regime)")
fig.update_layout(height=380); fig.show()
# (d) Resource-Based Screening Index
Z = lambda s: (s-s.mean())/s.std()
df["RBSI"] = (0.30*Z(np.log(df.energy_est_mwh)) + 0.25*Z(np.log(df.water_est_m3*(1+df.water_stress/5))) + 0.20*Z(np.log(df.co2_est_t)) + 0.15*Z(np.log1p(df.flood_ead_musd)) + 0.10*Z(df.co2_heat_delta_t))
df["rbsi_tier"] = pd.cut(df.RBSI.rank(pct=True), [0, 0.40, 0.75, 1.0], labels=["C: registration + resource plan","B: IEE/ESA-lite + water & power plan","A: full EIA + resource-consumption plan"])
tierc = df.groupby("rbsi_tier", observed=True).agg(n=("fac_id","size"), share_CO2=("co2_est_t", lambda s: s.sum()/df.co2_est_t.sum()), share_water=("water_est_m3", lambda s: s.sum()/df.water_est_m3.sum()), mean_MW=("it_mw","mean"))
print("RBSI tiers (computed from ESTIMATED values only):"); display(tierc.round(3))
xt = pd.crosstab(df.rbsi_tier, df.tier_home.map({0:"permits only",1:"light",2:"full"})); print("\nRBSI tier vs current home-jurisdiction tier (facilities):"); display(xt)
miss = df[(df.rbsi_tier.astype(str).str.startswith("A")) & (df.tier_home < 2)]
print(f"\n{len(miss)} Tier-A facilities ({miss.co2_est_t.sum()/df.co2_est_t.sum():.0%} of estimated CO2, {miss.water_est_m3.sum()/df.water_est_m3.sum():.0%} of water) currently face NO full assessment.")
display(miss[["fac_id","country","hub","ftype","it_mw","cooling","water_stress","co2_est_t","water_est_m3","instrument_home"]].sort_values("co2_est_t", ascending=False).head(12).round(1))
''')

# ---------------------------------------------------------------- 14. Maps
md(r"""
## 14. Geo dashboard
An interactive multi-layer **Folium** map (toggle layers top-right): RCI choropleth (Natural Earth boundaries fetched at run time, skipped if offline), facilities coloured by RBSI screening tier and sized by IT-MW with full trigger pop-ups, hub water-stress rings, flood-EAD markers, Fay–Herriot hub EBLUPs, and a CO₂ heat-map. Followed by Plotly tile maps. The HTML is saved to `outputs/asean_dc_risk_map.html`.
""")
code(r'''
import requests
ASEAN = set(LEGAL.iso3)
m = folium.Map(location=[8.5, 108], zoom_start=5, tiles="OpenStreetMap", control_scale=True)   # OSM: no API key needed
try:
    gj = requests.get("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson", timeout=40).json()
    feats = [f for f in gj["features"] if f["properties"].get("ADM0_A3") in ASEAN or f["properties"].get("ISO_A3") in ASEAN]
    for f in feats:
        iso = f["properties"].get("ADM0_A3") if f["properties"].get("ADM0_A3") in ASEAN else f["properties"].get("ISO_A3")
        row = LEGAL.set_index("iso3").loc[iso]; f["properties"].update(dict(iso3=iso, RCI=round(float(row.RCI),2), dc_listed=row.dc_listed_trigger, law=row.primary_law))
    gjd = {"type":"FeatureCollection","features":feats}
    folium.Choropleth(geo_data=gjd, data=LEGAL, columns=["iso3","RCI"], key_on="feature.properties.iso3", fill_color="YlGnBu", fill_opacity=0.55, line_opacity=0.6, legend_name="Regulatory Capture Index (0-5)", name="RCI choropleth").add_to(m)
    folium.GeoJson(gjd, name="Country legal summary (hover)", style_function=lambda x: {"fillOpacity":0,"weight":0},
                   tooltip=folium.GeoJsonTooltip(fields=["iso3","RCI","dc_listed","law"], aliases=["ISO3","RCI","DC listed EIA trigger?","Primary law"])).add_to(m)
    print(f"Choropleth: {len(feats)} ASEAN polygons loaded")
except Exception as e:
    print("Country polygons unavailable (offline?) - choropleth skipped:", str(e)[:80])
TIER_COL = {"A": "#c0392b", "B": "#e67e22", "C": "#27ae60"}
fg_fac = folium.FeatureGroup(name="Facilities (colour = RBSI tier, size = IT MW)").add_to(m)
for _, r in df.iterrows():
    trig = thai_triggers(r, True) if r.iso3=="THA" else None
    trig_html = "<br>".join(f"&bull; {t['instrument'][:90]} <i>({t['confidence']})</i>" for t in trig) if trig else f"&bull; {r.instrument_home}"
    html = (f"<b>{r.fac_id}</b> - {r.hub}<br>{r.ftype}, {r.cooling}-cooled, {r.it_mw:.1f} MW IT, online {r.year_online}<br>"
            f"<b>RBSI tier:</b> {r.rbsi_tier}<br><b>Current home tier:</b> {r.tier_home} | PCAF DQ {r.pcaf_dq} | disclosed={bool(r.disclosed)}<br>"
            f"Energy P50 {r.energy_est_mwh/1e3:,.0f} GWh [{r.energy_est_mwh_p05/1e3:,.0f}-{r.energy_est_mwh_p95/1e3:,.0f}] | CO2 {r.co2_est_t/1e3:,.0f} kt | water {r.water_est_m3/1e3:,.0f} k m3<br>"
            f"Water stress {r.water_stress:.1f} | flood EAD {r.flood_ead_musd:.2f} MUSD/yr | carbon NPV(NZ) {r.carbon_npv_nz_musd:.1f} MUSD<br><b>Triggers:</b><br>{trig_html}")
    folium.CircleMarker([r.lat, r.lon], radius=3+2.2*np.sqrt(r.it_mw), color=TIER_COL[str(r.rbsi_tier)[0]], fill=True, fill_opacity=0.65, weight=1,
                        popup=folium.Popup(html, max_width=420), tooltip=f"{r.fac_id} {r.it_mw:.0f} MW - tier {str(r.rbsi_tier)[0]}").add_to(fg_fac)
fg_ws = folium.FeatureGroup(name="Hub baseline water stress (Aqueduct-style 0-5)", show=False).add_to(m)
for h in HUBS.itertuples():
    folium.Circle([h.lat, h.lon], radius=35000, color="#1f77b4", weight=1, fill=True, fill_color=["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#2c7fb8","#253494"][min(int(h.water_stress),5)], fill_opacity=0.35,
                  tooltip=f"{h.hub}: water stress {h.water_stress}").add_to(fg_ws)
fg_fl = folium.FeatureGroup(name="Flood EAD (MUSD/yr, baseline)", show=False).add_to(m)
for _, r in df[df.flood_ead_musd>0.01].iterrows():
    folium.CircleMarker([r.lat, r.lon], radius=3+6*np.sqrt(r.flood_ead_musd), color="#8e44ad", fill=True, fill_opacity=0.4, weight=1, tooltip=f"{r.fac_id}: EAD {r.flood_ead_musd:.2f} -> 2050 {r.flood_ead_2050_musd:.2f} MUSD/yr").add_to(fg_fl)
fg_fh = folium.FeatureGroup(name="Hub Fay-Herriot PUE EBLUP", show=False).add_to(m)
for h in hub.itertuples():
    folium.Marker([h.lat, h.lon], icon=folium.DivIcon(html=f'<div style="font-size:11px;font-weight:bold;color:#2c3e50;background:rgba(255,255,255,.8);padding:2px 4px;border-radius:3px">PUE {h.eblup:.2f}<br>(γ={h.gamma:.2f}, n={h.n_disc})</div>')).add_to(fg_fh)
plugins.HeatMap(df[["lat","lon","co2_est_t"]].values.tolist(), name="Estimated CO2 heat-map", radius=28, blur=22, show=False, min_opacity=0.3).add_to(m)
plugins.MiniMap(toggle_display=True).add_to(m); plugins.Fullscreen().add_to(m); plugins.MeasureControl(primary_length_unit="kilometers").add_to(m)
legend = '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 10px;border:1px solid #999;font-size:12px"><b>RBSI screening tier</b><br>' + "".join(f'<span style="color:{c}">&#9679;</span> Tier {k}<br>' for k,c in TIER_COL.items()) + "</div>"
m.get_root().html.add_child(folium.Element(legend)); folium.LayerControl(collapsed=False).add_to(m)
m.save(f"{OUT}/asean_dc_risk_map.html"); print(f"Saved {OUT}/asean_dc_risk_map.html"); m
''')
code(r'''
df["tier_letter"] = df.rbsi_tier.astype(str).str[0]
kw = dict(lat="lat", lon="lon", size="it_mw", color="tier_letter", color_discrete_map=TIER_COL, category_orders={"tier_letter":["A","B","C"]}, hover_name="fac_id",
          hover_data={"hub":True,"cooling":True,"it_mw":":.1f","co2_est_t":":,.0f","water_est_m3":":,.0f","instrument_home":True,"lat":False,"lon":False}, size_max=22, zoom=4.3, center=dict(lat=8, lon=108), height=600)
fig = px.scatter_map(df, map_style="open-street-map", **kw) if hasattr(px, "scatter_map") else px.scatter_mapbox(df, mapbox_style="open-street-map", **kw)
fig.update_layout(title="RBSI screening tier by facility (tile map)", margin=dict(l=0,r=0,t=40,b=0)); fig.show()
fig = px.density_map(df, lat="lat", lon="lon", z="water_est_m3", radius=25, map_style="open-street-map", zoom=4.3, center=dict(lat=8, lon=108), height=520, title="Estimated water withdrawal density (m3/yr)") if hasattr(px, "density_map") else px.density_mapbox(df, lat="lat", lon="lon", z="water_est_m3", radius=25, mapbox_style="open-street-map", zoom=4.3, center=dict(lat=8, lon=108), height=520, title="Estimated water withdrawal density (m3/yr)")
fig.update_layout(margin=dict(l=0,r=0,t=40,b=0)); fig.show()
fig = px.treemap(df, path=[px.Constant("ASEAN-10"), "country", "hub", "fac_id"], values="co2_est_t", color="pcaf_dq", color_continuous_scale="Viridis_r", range_color=(1,5),
                 title="Estimated CO2 by country > hub > facility, coloured by PCAF data quality"); fig.update_layout(height=520); fig.show()
fig = px.sunburst(df.assign(tier=df.tier_home.map({0:"permits only",1:"light assessment",2:"full assessment"})), path=["country","tier","rbsi_tier"], values="co2_est_t",
                  title="Current assessment tier vs RBSI tier, weighted by estimated CO2"); fig.update_layout(height=520); fig.show()
''')

# ---------------------------------------------------------------- 15. Export
md(r"""
## 15. Exports, real-data template and data-source pointers
""")
code(r'''
cols_out = ["fac_id","country","iso3","hub","lat","lon","ftype","cooling","operator","year_online","it_mw","floor_area_m2","site_ha","in_industrial_estate","onsite_thermal_mw","onsite_solar_mwp","boi_promoted",
            "water_source","water_stress","flood100_m","disclosed","pue_obs","wue_obs","util_obs","pue_est","pue_est_p05","pue_est_p95","energy_est_mwh","energy_est_mwh_p05","energy_est_mwh_p95",
            "co2_est_t","co2_est_t_p05","co2_est_t_p95","water_est_m3","water_m3_day","pue_hmice","pue_gbm_shift","pcaf_dq","provenance","carbon_npv_nz_musd","flood_ead_musd","flood_ead_2050_musd","flood_var95_musd","water_cost_musd",
            "tier_home","instrument_home","RBSI","rbsi_tier"]
df[cols_out].to_csv(f"{OUT}/facility_estimates_and_screening.csv", index=False)
hub.to_csv(f"{OUT}/hub_level_estimates.csv", index=False); LEGAL.to_csv(f"{OUT}/asean10_legal_framework.csv", index=False); val.to_csv(f"{OUT}/validation_metrics.csv"); fr.to_csv(f"{OUT}/threshold_frontier.csv", index=False)
template_cols = ["fac_id","country_iso3","hub","lat","lon","ftype","cooling","operator","parent_reports_cdp","year_online","it_mw","floor_area_m2","in_industrial_estate","onsite_thermal_mw","onsite_solar_mwp","boi_promoted",
                 "water_source","pue_reported","wue_reported","util_reported","water_stress_aqueduct","flood100_m_fathom","grid_ef_tco2_mwh","nightlight_rad_viirs","thermal_anom_k","footprint_m2","marketdb_mw","permits_count","job_posts"]
pd.DataFrame(columns=template_cols).to_csv(f"{OUT}/REAL_DATA_TEMPLATE.csv", index=False)
print("Written:", sorted(os.listdir(OUT)))
if IN_COLAB:
    print("Download from the Files pane (left) or run: from google.colab import files; files.download('outputs/asean_dc_risk_map.html')")
''')
md(r"""
### Plugging in real data
Fill `outputs/REAL_DATA_TEMPLATE.csv` and load it in place of the synthetic register in §3 (`df = pd.read_csv(...)`); columns `*_true` simply become unavailable and the validation section (§12) should be re-pointed at whatever later disclosures you obtain.

| Need | Source | Notes |
|---|---|---|
| Facility list, IT MW, location | BOI approvals (36 projects / THB 728 bn in 2025), IEAT estate registers, NBTC licences; datacenterHawk, Structure Research, Cloudscene, Data Center Map, Baxtel | market DBs give MW and coordinates → DQ4 unless cooling known |
| Baseline water stress | WRI Aqueduct 4.0 (bws_cat / bws_score; 2030/2050 SSP variants) | join on facility point → basin polygon |
| Flood depth | Fathom Global 3, JBA, WWF Water Risk Filter (free basin layer) | 100-year fluvial + pluvial depth at the site |
| Grid EF | Ember yearly, IEA, Electricity Maps (flow-traced hourly), WattTime (marginal, for siting only) | location-based Scope 2 uses average EF |
| Night-lights / thermal / footprints | VIIRS DNB (NASA Black Marble), Landsat TIRS / ECOSTRESS, Microsoft & Google Open Buildings, OSM | construction progress from Sentinel-2 |
| Disclosures | EU EED Delegated Reg. 2024/1364 database (multinationals), sustainability reports, CDP, SEC 56-1 One Report | seeds the Heckman selection model |
| Scenarios | NGFS Phase V, CMIP6 downscaled (CORDEX-SEA), IPCC AR6 | replace illustrative price paths |
| Standards | PCAF, GHG Protocol Scope 2, ISSB IFRS S2, Thailand Taxonomy Phase 2 (27 May 2025), BOT Feb-2023 policy statement, MAS ERM, BNM CRMSA | reporting layer |

### What the run demonstrated
The cell below assembles the headline numbers from this run (every comparison is against the hidden truth of the synthetic register).
""")

code(r'''
fr20 = fr[(fr.rule=="IT >= T MW OR water >= 300 m3/day") & (fr["T"]==20)].iloc[0]; cur = cap.loc["Full assessment (tier 2), confirmed"]
print("=== KEY FINDINGS FROM THIS RUN (all comparisons vs the hidden truth) ===")
print(f"1. Disclosure: {df.disclosed.mean():.0%} of facilities ({df.loc[disc,'it_mw'].sum()/df.it_mw.sum():.0%} of MW) report PUE/WUE. Disclosers' mean true PUE {df.loc[disc,'pue_true'].mean():.3f} vs {df.loc[~disc,'pue_true'].mean():.3f} for non-disclosers (MNAR).")
print(f"2. Portfolio CO2 bias: naive disclosed-mean PUE {tab.loc['Naive: disclosed mean PUE','bias_%']:+.1f}% | Horvitz-Thompson/IPW {tab.loc['Horvitz-Thompson / IPW by cooling','bias_%']:+.1f}% | OLS-on-disclosers {tab.loc['OLS on disclosers (X only)','bias_%']:+.1f}% | Heckman two-step {tab.loc['Heckman two-step (IMR-corrected)','bias_%']:+.1f}% | bottom-up MC P50 {tab.loc['Bottom-up MC prior P50 (no disclosure data)','bias_%']:+.1f}%   (rho*sigma = {ols_heck.params['imr']:+.3f}, p = {ols_heck.pvalues['imr']:.3f})")
print(f"3. Facility PUE backtest ({len(newly)} newly-disclosing sites): lowest RMSE = '{val.RMSE.idxmin()}' at {val.RMSE.min():.3f} vs disclosed-mean baseline {val.loc['Disclosed-mean by cooling','RMSE']:.3f}; 90% coverage: MC prior {val.loc['Bottom-up MC prior (P50)','cov90']:.0%}, MICE {val.loc['MICE (alt-data aux)','cov90']:.0%}, Heckman-in-MICE {val.loc['Heckman-in-MICE','cov90']:.0%}, GBM+conformal {val.loc['GBM + conformal','cov90']:.0%}.")
print(f"4. Partial pooling: EB shrinkage weights range {g.B[g.n>0].min():.2f}-{g.B.max():.2f}; country-mean PUE RMSE vs truth: no-pool {np.sqrt(((g.no_pool-g.truth_all_facilities)**2).mean()):.3f} | EB {np.sqrt(((g.partial_pool_EB-g.truth_all_facilities)**2).mean()):.3f} | MixedLM {np.sqrt(((g.MixedLM_pred_all-g.truth_all_facilities)**2).mean()):.3f}.")
print(f"5. Fay-Herriot hub PUE: RMSE direct {np.sqrt(((hub.direct-hub.truth)**2).mean()):.3f} -> EBLUP {np.sqrt(((hub.eblup-hub.truth)**2).mean()):.3f} (model variance A = {A_fh:.4f}). Nowcast MAPE {100*np.mean(np.abs(nc.nowcast.iloc[-6:-2]-nc.true_cap.iloc[-6:-2])/nc.true_cap.iloc[-6:-2]):.1f}%.")
print(f"6. Spatial: facility-level Moran's I on log CO2 = {I_fac:+.3f} (p = {p_fac:.3f}); hub-level SAR rho = {sar.loc['rho (W*log_mw)','coef']:+.2f} (t = {sar.loc['rho (W*log_mw)','t']:.2f}); water stress and RCI are the significant hub covariates in the Poisson count model (p = {pois.pvalues['water_stress']:.3f}, {pois.pvalues['RCI']:.3f}).")
print(f"7. DiD recovery: Singapore moratorium {did.params['sg_mora']:+.2f} (true {EFF_SG:+.2f}), Johor/Batam spillover {did.params['spill']:+.2f} (true {EFF_SPILL:+.2f}), Johor vetting {did.params['johor_vet']:+.2f} (true {EFF_JOHOR:+.2f}).")
print(f"8. EIA/EIS capture today: a FULL assessment covers {cur.share_projects:.0%} of projects but only {cur.share_CO2:.0%} of estimated CO2 and {cur.share_water:.0%} of water. Rule 'IT >= 20 MW OR water >= 300 m3/day' would capture {fr20.share_CO2:.0%} of CO2 and {fr20.share_water:.0%} of water with {int(fr20.projects_assessed)} projects. {len(miss)} RBSI Tier-A facilities ({miss.co2_est_t.sum()/df.co2_est_t.sum():.0%} of CO2, {miss.water_est_m3.sum()/df.water_est_m3.sum():.0%} of water) currently face no full assessment.")
print(f"9. PCAF: emissions-weighted DQ = {wdq:.2f}; measured {df.loc[df.provenance=='measured','co2_est_t'].sum()/df.co2_est_t.sum():.0%} / estimated {df.loc[df.provenance=='estimated','co2_est_t'].sum()/df.co2_est_t.sum():.0%} / proxied {df.loc[df.provenance=='proxied','co2_est_t'].sum()/df.co2_est_t.sum():.0%} of exposure.")
print(f"10. Scenarios: carbon NPV (Net Zero 2050) {NPV['Net Zero 2050'].sum()/1e6:,.0f} MUSD vs Current Policies {NPV['Current Policies'].sum()/1e6:,.0f}; flood EAD {df.flood_ead_musd.sum():,.0f} -> {df.flood_ead_2050_musd.sum():,.0f} MUSD/yr by 2050; water curtailment/switching {df.water_cost_musd.sum():,.0f} MUSD/yr; heat uplift +{df.co2_heat_delta_t.sum()/1e3:,.0f} ktCO2/yr.")
''')

nb["cells"] = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"},
               "colab": {"name": "ASEAN_DC_EIA_GeoEconometrics.ipynb", "toc_visible": True, "provenance": []}}
out = sys.argv[1] if len(sys.argv) > 1 else "ASEAN_DC_EIA_GeoEconometrics.ipynb"
nbf.write(nb, out); print("wrote", out, "cells:", len(cells))
