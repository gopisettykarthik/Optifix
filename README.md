# OptiFix: Spatial-ROI Allocation Engine
Datathon 2026 · Track 3 – Smart City · Team Larpdeeznuh

**Question:** "Where should we fix first?" OptiFix ranks *one-trip fixes* by **Return on Effort (RoE)** and explains every pick in plain text.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The three CSVs are bundled in `data/` (`locations.csv`, `incidents.csv`, `public_complaints.csv`). You can also upload replacements in the sidebar.

## How it works (Scikit-Learn only, no LLM/API)
1. **Join** the three tables on `location_id` (19,000 rows, 0 orphan IDs).
2. **Features (31)** per site from incidents, complaints and site context.
3. **Random Forest** predicts severity-weighted incident load (severity 1–4, +2 per injury). Validated on a **time split**: features from Jan–Jun, target Jul–Dec, 5-fold CV across sites. Forecasts use the Jul–Dec window.
4. **Risk = predicted load × exposure** (density, school, hospital, pedestrian activity; tunable policy weights).
5. **DBSCAN (haversine, 0.5 km, min 3)** groups nearby sites into one-trip clusters; isolated sites are single-site trips.
6. **RoE = cluster risk ÷ crew-hours** (set-up + on-site + nearest-neighbour travel). Trips are picked greedily by RoE within the crew-hour budget.
7. **Justification queue**: plain-text reasons from each site's values vs the city median.

## Files
- `engine.py` – data join, features, validation, clustering, RoE, budget pick, justifications
- `app.py` – Streamlit dashboard (plan map, justification queue, budget comparison, model validation, limitations)

## Results on the bundled data (80 crew-hours)
OptiFix services 66 sites in 9 trips vs 28 sites in 28 trips for worst-first single-site dispatch (assumes 1 h set-up/trip, 1 h on site, 25 km/h).

## Limitations (be upfront with judges)
- Data is **synthetic**; incident counts per site look like random noise. The Random Forest does **not** beat a simple mean baseline (MAE 5.13 vs 5.03, R² –0.03), so the value shown comes from the allocation layer, not prediction accuracy.
- Straight-line travel from one depot; no road network, crew skills or repair costs.
- Effort assumptions and exposure weights are adjustable policy choices, not learned.
- Static snapshot; no live updates.
