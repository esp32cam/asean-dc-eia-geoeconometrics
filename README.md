# ASEAN-10 Data-Centre EIA / EIS — Geo-Economic & Econometric Proof-of-Concept (Colab)

`ASEAN_DC_EIA_GeoEconometrics.ipynb` is a self-contained Google Colab notebook that turns the
"Environmental & Climate Risk for Data Centres in ASEAN-10" brief into a runnable, geo-visual,
econometric pipeline — and scores every estimator against a hidden ground truth so the
methodology is *demonstrated*, not just described.

## Open in Colab

1. Upload the `.ipynb` to Google Drive or GitHub, or drag it into <https://colab.research.google.com/> (File → Upload notebook).
2. Runtime → Run all. The first cell installs anything missing (Colab already ships pandas, statsmodels, scikit-learn, plotly, folium, geopandas, PyMC). Full run takes well under a minute on a CPU runtime.
3. Outputs (CSVs, interactive HTML map, real-data template) are written to `outputs/` in the Colab file pane.

The notebook was also executed locally end to end (Python 3.12, statsmodels 0.15, geopandas 1.1, plotly 7) with zero cell errors; `ASEAN_DC_EIA_GeoEconometrics_executed_preview.html` is that run rendered to HTML.

## What is inside

| § | Block | Method |
|---|---|---|
| 2 | Legal framework as data | ASEAN-10 comparative table → **Regulatory Capture Index**; Thai instrument-level rule engine (NEQA / Building Control Act / IEAT / ERC / Por Kor 2 / Factory Act / BOI / water licences, with *UNVERIFIED* flags) + simplified cross-jurisdiction screening; choropleth |
| 3 | Facility register | 22 real ASEAN hubs, ~290 illustrative facilities with latent true PUE / utilisation / WUE / energy / CO₂ / water, a **Missing-Not-At-Random** disclosure process, and alternative-data proxies (night-lights, thermal-IR anomaly, building footprint, market-DB MW, permits, job posts) |
| 4 | Bottom-up estimator | E = IT × PUE × util × 8760; CO₂ = E × EF; water = E_IT × WUE; vectorised Monte-Carlo, P5/P50/P95, tornado |
| 5 | Coverage-aware totals | Naive disclosed-mean vs Horvitz–Thompson / IPW vs Heckman two-step vs engineering prior, all vs truth |
| 6 | Hierarchical pooling | Empirical-Bayes shrinkage and MixedLM random country intercepts on Heckman-corrected residuals; optional PyMC |
| 7 | Missing-data machinery | MCAR/MAR/MNAR diagnostics, MICE + Rubin's rules, Heckman-in-MICE, δ-sensitivity, GBM + split-conformal, SoftImpute |
| 8 | Small-area estimation | Fay–Herriot EBLUP at hub level; capacity nowcasting from permits / job posts / satellite construction |
| 9 | Geo-econometrics | Haversine spatial weights, Moran's I (permutation), SLX, spatial-lag 2SLS, Poisson counts, TWFE difference-in-differences + event study (Singapore moratorium, Johor vetting) with known effects recovered |
| 10 | Scenario overlays | NGFS-style carbon-price NPV, flood depth–damage EAD / VaR (baseline vs 2050), water-curtailment cost, heat-driven PUE uplift |
| 11 | PCAF data quality | DQ1–5, emissions-weighted DQ, measured / estimated / proxied split |
| 12 | Validation | Backtest on newly-disclosing facilities: MAE, RMSE, MAPE, bias, 90 % coverage, pinball loss, calibration curve, regime-switch rule |
| 13 | **EIA / EIS improvement** | Capture rate of current regimes, threshold frontier (load captured vs projects assessed), harmonisation heat-map, Resource-Based Screening Index tiers A/B/C |
| 14 | Geo dashboard | Multi-layer Folium map (RCI choropleth, facilities by tier, water stress, flood EAD, hub EBLUPs, CO₂ heat-map) + Plotly tile maps, treemap, sunburst |
| 15 | Export | CSVs, HTML map, `REAL_DATA_TEMPLATE.csv`, data-source pointers, computed findings summary |

## Plugging in real data

Fill `outputs/REAL_DATA_TEMPLATE.csv` (BOI / IEAT / market-database rows, Aqueduct water stress, Fathom flood depth, Ember / Electricity Maps grid EF, VIIRS night-lights, building footprints) and load it in §3 in place of the synthetic register. Every downstream section keeps working; the `*_true` columns simply disappear and §12 should be re-pointed at whatever later disclosures you obtain.

## Caveats

* Legal status is mid-August 2026 (Thai PM's Office Regulation on the Data Center Business Policy Committee B.E. 2569, Royal Gazette 13 Aug 2026). Items the rule engine marks *UNVERIFIED* (ONEP building-EIA applicability to a pure DC, standby gensets under the Factory Act HP test, water-licence volumetric thresholds) are flagged, not guessed.
* All facility numbers are illustrative. Country parameters (grid EF, tariffs), the NGFS-style price paths and the non-Thai screening thresholds are placeholders to replace with IEA / Ember / NGFS / official schedules.
* `outputs/` contains one local run's artefacts for reference; Colab regenerates them.

## Regenerating the notebook

`tools/build_notebook.py` is the generator used to author the notebook (cells are defined as strings and written with nbformat). To rebuild after editing it:

```bash
python3 tools/build_notebook.py ASEAN_DC_EIA_GeoEconometrics.ipynb
```
