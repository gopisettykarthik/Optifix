"""OptiFix: Spatial-ROI Allocation Engine  (Datathon 2026, Track 3 - Smart City)
Run:  streamlit run app.py
"""
from pathlib import Path
import numpy as np, pandas as pd, pydeck as pdk, streamlit as st
import engine as E

st.set_page_config(page_title="OptiFix", page_icon="🛠️", layout="wide")
DATA = Path(__file__).parent / "data"
CYAN = [25, 227, 245]

st.title("OptiFix: Spatial-ROI Allocation Engine")
st.caption("Where should we fix first? Rank one-trip fixes by Return on Effort (RoE) and see the reason behind every pick. "
           "Scikit-Learn only (Random Forest + DBSCAN), no LLM or external API.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Controls")
    budget = st.slider("Crew-hour budget", 20, 200, 80, 10, help="Total crew-hours available for this planning round")
    st.subheader("One-trip clustering (DBSCAN)")
    eps = st.slider("Cluster radius (km, haversine)", 0.2, 1.0, 0.5, 0.05)
    min_s = st.slider("Min sites per cluster", 2, 5, 3)
    st.subheader("Effort assumptions")
    setup_h = st.number_input("Set-up hours per trip", 0.0, 5.0, 1.0, 0.5)
    onsite_h = st.number_input("On-site hours per site", 0.25, 5.0, 1.0, 0.25)
    speed = st.number_input("Travel speed (km/h)", 5.0, 60.0, 25.0, 5.0)
    with st.expander("Exposure weights (policy, tunable)"):
        w_den = st.slider("Population density", 0.0, 1.0, 0.5, 0.05)
        w_sch = st.slider("Near school", 0.0, 1.0, 0.25, 0.05)
        w_hos = st.slider("Near hospital", 0.0, 1.0, 0.25, 0.05)
        w_ped = st.slider("High pedestrian activity", 0.0, 1.0, 0.5, 0.05)
    st.subheader("Data")
    ups = st.file_uploader("Replace the bundled CSVs (locations, incidents, public_complaints)", type="csv", accept_multiple_files=True)


# ------------------------------------------------------------------ data + model (cached)
@st.cache_data(show_spinner="Loading and joining the three tables...")
def get_data(uploaded: tuple):
    files = {n: pd.read_csv(DATA / f"{n}.csv") for n in ("locations", "incidents", "public_complaints")}
    for name, raw in uploaded:  # (filename, bytes)
        key = Path(name).stem
        if key in files:
            import io
            files[key] = pd.read_csv(io.BytesIO(raw))
    return E.load_data(files["locations"], files["incidents"], files["public_complaints"])


@st.cache_data(show_spinner="Validating on a time split and training the Random Forest...")
def get_model(uploaded: tuple):
    loc, inc, com = get_data(uploaded)
    rf, X2, metrics, imp = E.validate_and_fit(loc, inc, com)
    pred = pd.Series(rf.predict(X2), index=loc.location_id, name="pred_load")
    return pred, X2, metrics, imp


uploaded = tuple((f.name, f.getvalue()) for f in ups) if ups else ()
loc, inc, com = get_data(uploaded)
pred, X2, metrics, imp = get_model(uploaded)

# ------------------------------------------------------------------ risk -> clusters -> RoE -> budget pick
sites = loc.set_index("location_id").copy()
sites["pred_load"] = pred
sites["exposure"] = E.exposure_index(loc, w_den, w_sch, w_hos, w_ped)
sites["risk"] = sites.pred_load * sites.exposure
sites["cl"] = E.cluster_sites(loc, eps, min_s)
depot = (sites.latitude.mean(), sites.longitude.mean())
units = E.build_units(sites, depot, setup_h, onsite_h, speed)
chosen, used = E.pick_within_budget(units, budget)
naive = E.worst_first(sites, depot, budget, setup_h, onsite_h, speed)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sites fixed (OptiFix)", int(chosen.n_sites.sum()), f"+{int(chosen.n_sites.sum()) - naive['sites']} vs worst-first")
c2.metric("Trips needed", len(chosen), f"{len(chosen) - naive['trips']} vs worst-first", delta_color="inverse")
c3.metric("Risk covered", f"{chosen.risk.sum():,.0f}", f"{chosen.risk.sum() / max(naive['risk'], 1e-9) - 1:+.0%} vs worst-first")
c4.metric("Crew-hours used", f"{used:.1f} / {budget}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Priority plan", "Justification queue", "Budget comparison", "Model validation", "Data & limitations"])

# ------------------------------------------------------------------ tab 1: map + plan
with tab1:
    plan = []
    for rank, (_, u) in enumerate(chosen.iterrows(), start=1):
        for i in u.ids:
            r = sites.loc[i]
            plan.append(dict(location_id=i, trip=rank, zone=r.zone, latitude=r.latitude, longitude=r.longitude,
                             risk=round(r.risk, 1), trip_RoE=round(u.roe, 2)))
    plan = pd.DataFrame(plan)
    others = sites.reset_index()[["location_id", "latitude", "longitude", "zone"]]
    others = others[~others.location_id.isin(plan.location_id)] if len(plan) else others
    layers = [pdk.Layer("ScatterplotLayer", others, get_position="[longitude, latitude]", get_radius=45,
                        get_fill_color=[170, 170, 170, 130], pickable=True)]
    if len(plan):
        layers.append(pdk.Layer("ScatterplotLayer", plan, get_position="[longitude, latitude]", get_radius=110,
                                get_fill_color=CYAN + [230], get_line_color=[0, 0, 0], line_width_min_pixels=1,
                                stroked=True, pickable=True))
        layers.append(pdk.Layer("TextLayer", plan.groupby("trip")[["latitude", "longitude"]].mean().reset_index().assign(t=lambda d: d.trip.astype(str)),
                                get_position="[longitude, latitude]", get_text="t", get_size=15, get_color=[0, 0, 0],
                                get_pixel_offset=[0, -16]))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=depot[0], longitude=depot[1], zoom=9.6),
                             tooltip={"text": "{location_id}\n{zone}"}), height=480)
    st.caption("Cyan = sites in the selected one-trip plan (numbers are trip ranks). Grey = other sites.")
    show = chosen.assign(trip=range(1, len(chosen) + 1))[["trip", "n_sites", "risk", "crew_hours", "km", "roe"]]
    show.columns = ["Trip", "Sites", "Risk", "Crew-hours", "Travel km", "RoE"]
    st.dataframe(show.round(2), hide_index=True, width="stretch")
    if len(plan):
        st.download_button("Download plan (CSV)", plan.to_csv(index=False), "optifix_plan.csv", "text/csv")

# ------------------------------------------------------------------ tab 2: justification queue
with tab2:
    st.subheader("Plain-text justification queue")
    st.caption("Reasons are generated from the data (each site's values vs the city median). No LLM is used.")
    for rank, (_, u) in enumerate(chosen.iterrows(), start=1):
        head, lines = E.justify(u, sites, X2, rank, len(units))
        kind = f"{u.n_sites}-site cluster" if u.cluster >= 0 else "single site"
        with st.expander(f"#{rank}  {kind}: RoE {u.roe:.1f}, {u.crew_hours:.1f} crew-hours", expanded=rank <= 2):
            st.write(head)
            st.markdown("**Evidence**\n" + "\n".join(f"- {l}" for l in lines))
            st.caption("Sites: " + ", ".join(u.ids))

# ------------------------------------------------------------------ tab 3: budget comparison
with tab3:
    rows = []
    for b in range(20, 201, 20):
        ch, _ = E.pick_within_budget(units, b)
        nv = E.worst_first(sites, depot, b, setup_h, onsite_h, speed)
        rows.append({"Budget (crew-hours)": b, "OptiFix": int(ch.n_sites.sum()), "Worst-first (one site per trip)": nv["sites"]})
    comp = pd.DataFrame(rows).set_index("Budget (crew-hours)")
    st.subheader("Sites serviced for a given crew-hour budget")
    st.bar_chart(comp, color=["#19E3F5", "#555555"])
    st.caption(f"Assumes {setup_h:g} h set-up per trip, {onsite_h:g} h on site per site, {speed:g} km/h straight-line travel from a central depot. "
               "The gain comes from sharing set-up and travel inside a cluster, not from better prediction.")
    st.dataframe(comp, width="stretch")

# ------------------------------------------------------------------ tab 4: model validation
with tab4:
    st.subheader("Leakage-free validation (time split)")
    st.write("Features come from Jan-Jun; the target is severity-weighted incident load in Jul-Dec "
             "(severity 1-4, +2 per injury). Scores are 5-fold cross-validated across sites.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RF MAE", f"{metrics['mae_rf']:.2f}")
    m2.metric("Mean-baseline MAE", f"{metrics['mae_mean']:.2f}")
    m3.metric("RF R²", f"{metrics['r2_rf']:.3f}")
    m4.metric("Top-10% capture", f"{metrics['top10_capture']:.1%}", "random = 10%", delta_color="off")
    if metrics["r2_rf"] <= 0.02:
        st.warning("The Random Forest does not beat a simple average on this dataset: past load barely predicts future load "
                   f"(correlation {metrics['h1_h2_corr']:.2f}). The data is synthetic. OptiFix's value here comes from the allocation layer "
                   "(one-trip clusters + RoE), and the model should be re-validated on real city data.")
    err = pd.DataFrame({"MAE (lower is better)": [metrics["mae_mean"], metrics["mae_rf"], metrics["mae_carry"]]},
                       index=["Mean baseline", "Random Forest", "Carry-forward"])
    a, b = st.columns(2)
    a.bar_chart(err, color="#19E3F5")
    b.bar_chart(imp.head(8).iloc[::-1].rename("Importance"), horizontal=True, color="#19E3F5")
    st.caption(f"{metrics['n_features']} features. Right: Random Forest feature importance (top 8).")

# ------------------------------------------------------------------ tab 5: data + limitations
with tab5:
    st.subheader("Data quality")
    q = E.quality_report(loc, inc, com)
    st.dataframe(pd.DataFrame({"Check": q.keys(), "Value": [str(v) for v in q.values()]}), hide_index=True, width="stretch")
    st.subheader("Limitations")
    st.markdown(
        "- **Synthetic data.** Counts per site look like random noise, so rankings are demonstrations of the method, not real priorities.\n"
        "- **No road network.** Travel is straight-line (haversine) from one central depot; no one-way streets, traffic or multiple depots.\n"
        "- **Assumed effort.** Set-up, on-site time and speed are adjustable assumptions, not measured; repair cost and crew skills are not modelled.\n"
        "- **Exposure weights are policy choices**, not learned from outcomes.\n"
        "- **Static snapshot.** No live data feed or re-planning as new incidents arrive.")
