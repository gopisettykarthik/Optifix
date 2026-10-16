"""OptiFix engine: Spatial-ROI allocation (Scikit-Learn only, no LLM/API)."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import spearmanr

EARTH_KM = 6371.0088
SEV = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
OPEN_STATUS = ["Open", "Assigned", "In progress"]
INCIDENT_TYPES = ["Pothole", "Collision", "Streetlight outage", "Signal failure", "Blocked drain", "Flooding"]


# ---------------------------------------------------------------- data
def load_data(loc, inc, com):
    loc, inc, com = loc.copy(), inc.copy(), com.copy()
    inc["timestamp"] = pd.to_datetime(inc["timestamp"])
    com["created_at"] = pd.to_datetime(com["created_at"])
    inc["load"] = inc["severity"].map(SEV) + 2 * inc["injuries"]  # severity 1-4, +2 per injury
    return loc, inc, com


def quality_report(loc, inc, com):
    return {
        "Sites": len(loc), "Incidents": len(inc), "Complaints": len(com),
        "Orphan incident IDs": int((~inc.location_id.isin(loc.location_id)).sum()),
        "Orphan complaint IDs": int((~com.location_id.isin(loc.location_id)).sum()),
        "Null cells (all tables)": int(loc.isna().sum().sum() + inc.isna().sum().sum() + com.isna().sum().sum()),
        "Duplicate rows": int(loc.duplicated().sum() + inc.duplicated().sum() + com.duplicated().sum()),
        "Unresolved incidents %": round(100 * (1 - inc.resolved.mean()), 1),
        "High/Critical incidents %": round(100 * inc.severity.isin(["High", "Critical"]).mean(), 1),
        "Open complaints %": round(100 * com.status.isin(OPEN_STATUS).mean(), 1),
    }


# ---------------------------------------------------------------- features
def build_features(loc, inc, com, start, end):
    """Aggregate incidents + complaints in [start, end) and join site context on location_id."""
    i = inc[(inc.timestamp >= start) & (inc.timestamp < end)]
    c = com[(com.created_at >= start) & (com.created_at < end)]
    g = i.groupby("location_id")
    A = pd.DataFrame({
        "inc_n": g.size(), "inc_load": g["load"].sum(), "inc_injuries": g.injuries.sum(),
        "inc_high_crit": g.severity.apply(lambda s: s.isin(["High", "Critical"]).sum()),
        "inc_unresolved": g.resolved.apply(lambda s: (1 - s).sum()),
        "inc_res_hours": g.resolution_hours.mean()})
    for t in INCIDENT_TYPES:
        A["n_" + t] = i[i.incident_type == t].groupby("location_id").size()
    gc = c.groupby("location_id")
    B = pd.DataFrame({
        "cmp_n": gc.size(), "cmp_high_prio": gc.priority_reported.apply(lambda s: (s == "High").sum()),
        "cmp_open": gc.status.apply(lambda s: s.isin(OPEN_STATUS).sum()),
        "cmp_days": gc.days_to_resolution.mean()})
    S = loc.set_index("location_id").drop(columns=["zone", "latitude", "longitude"])
    S = pd.get_dummies(S, columns=["road_type", "road_condition", "pedestrian_activity"])
    X = S.join(A).join(B).reindex(loc.location_id)
    dyn = [c_ for c_ in X.columns if c_.startswith(("inc_", "n_", "cmp_"))]
    X[dyn] = X[dyn].fillna(0)
    return X.astype(float)


def make_rf(seed=42):
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)


# ---------------------------------------------------------------- model + honest validation
def validate_and_fit(loc, inc, com, split="2025-07-01"):
    """Features from before `split`, target = severity-weighted load after it (no leakage).
    Returns fitted model, forecast features (post-split window) and validation metrics."""
    t0 = inc.timestamp.min().normalize()
    t1 = inc.timestamp.max().normalize() + pd.Timedelta(days=1)
    tm = pd.Timestamp(split)
    X1 = build_features(loc, inc, com, t0, tm)
    X2 = build_features(loc, inc, com, tm, t1)
    y = inc[inc.timestamp >= tm].groupby("location_id")["load"].sum().reindex(loc.location_id).fillna(0)
    rf = make_rf()
    oof = cross_val_predict(rf, X1, y, cv=KFold(5, shuffle=True, random_state=42))
    top = np.argsort(-oof)[: max(1, len(y) // 10)]
    m = {
        "mae_mean": mean_absolute_error(y, np.full(len(y), y.mean())),
        "mae_rf": mean_absolute_error(y, oof),
        "mae_carry": mean_absolute_error(y, X1.inc_load),
        "r2_rf": r2_score(y, oof),
        "spearman_rf": float(spearmanr(y, oof)[0]),
        "top10_capture": float(y.values[top].sum() / y.sum()),
        "h1_h2_corr": float(np.corrcoef(X1.inc_load, y)[0, 1]),
        "n_features": X1.shape[1],
    }
    rf.fit(X1, y)
    imp = pd.Series(rf.feature_importances_, X1.columns).sort_values(ascending=False)
    return rf, X2, m, imp


# ---------------------------------------------------------------- risk, clusters, RoE
def exposure_index(loc, w_density=0.5, w_school=0.25, w_hospital=0.25, w_ped=0.5):
    """Transparent policy multiplier (tunable, not learned): how much harm a failure would cause."""
    d = loc.set_index("location_id")
    ped = d.pedestrian_activity.map({"Low": 0.0, "Medium": 0.5, "High": 1.0})
    return (1 + w_density * d.population_density_per_km2.rank(pct=True)
            + w_school * d.near_school + w_hospital * d.near_hospital + w_ped * ped)


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(h))


def tour_km(depot, pts):
    """Nearest-neighbour tour length (km) depot -> sites -> depot (straight-line, no road network)."""
    cur, left, tot = depot, list(pts), 0.0
    while left:
        j = int(np.argmin([haversine(cur, p) for p in left]))
        tot += haversine(cur, left[j]); cur = left[j]; left.pop(j)
    return tot + haversine(cur, depot)


def cluster_sites(loc, eps_km=0.5, min_samples=3):
    xy = np.radians(loc[["latitude", "longitude"]].values)
    return DBSCAN(eps=eps_km / EARTH_KM, min_samples=min_samples, metric="haversine", algorithm="ball_tree").fit_predict(xy)


def _unit(sub, ids, c, depot, setup_h, onsite_h, speed):
    km = tour_km(depot, list(zip(sub.latitude, sub.longitude)))
    return dict(cluster=int(c), ids=ids, n_sites=len(ids), risk=float(sub.risk.sum()), km=km,
                crew_hours=setup_h + onsite_h * len(ids) + km / speed,
                lat=float(sub.latitude.mean()), lon=float(sub.longitude.mean()))


def build_units(sites, depot, setup_h=1.0, onsite_h=1.0, speed=25.0):
    """A unit = one DBSCAN cluster (one trip) or one isolated site (its own trip). RoE = risk / crew-hours."""
    units = []
    for c, g in sites.groupby("cl"):
        if c >= 0:
            units.append(_unit(g, list(g.index), c, depot, setup_h, onsite_h, speed))
        else:
            for i in g.index:
                units.append(_unit(g.loc[[i]], [i], -1, depot, setup_h, onsite_h, speed))
    u = pd.DataFrame(units)
    u["roe"] = u.risk / u.crew_hours
    return u.sort_values("roe", ascending=False).reset_index(drop=True)


def pick_within_budget(units, budget_h):
    used, chosen = 0.0, []
    for k, r in units.iterrows():  # already sorted by RoE
        if used + r.crew_hours <= budget_h:
            used += r.crew_hours; chosen.append(k)
    return units.loc[chosen], used


def worst_first(sites, depot, budget_h, setup_h=1.0, onsite_h=1.0, speed=25.0):
    """Naive baseline: highest-risk sites first, one trip each."""
    used, n, risk = 0.0, 0, 0.0
    for i, r in sites.sort_values("risk", ascending=False).iterrows():
        cost = setup_h + onsite_h + tour_km(depot, [(r.latitude, r.longitude)]) / speed
        if used + cost <= budget_h:
            used += cost; n += 1; risk += r.risk
    return dict(sites=n, trips=n, risk=risk, crew_hours=used)


# ---------------------------------------------------------------- plain-text justification
def justify(unit, sites, X, rank, n_units):
    ids = unit["ids"]
    s = sites.loc[ids]
    cols = ["inc_load", "inc_unresolved", "cmp_open", "cmp_high_prio"]
    med, cur = X[cols].median(), X.loc[ids, cols].mean()
    label = {"inc_load": "severity-weighted incident load", "inc_unresolved": "unresolved incidents",
             "cmp_open": "open complaints", "cmp_high_prio": "high-priority complaints"}
    lines = [f"{label[k]} average {cur[k]:.1f} per site vs a city median of {med[k]:.1f}"
             for k in cols if med[k] > 0 and cur[k] >= 1.25 * med[k]]
    n_ped = int((s.pedestrian_activity == "High").sum())
    if n_ped: lines.append(f"{n_ped} site(s) with high pedestrian activity")
    if s.near_school.sum(): lines.append(f"{int(s.near_school.sum())} site(s) near a school")
    if s.near_hospital.sum(): lines.append(f"{int(s.near_hospital.sum())} site(s) near a hospital")
    n_bad = int(s.road_condition.isin(["Poor", "Critical"]).sum())
    if n_bad: lines.append(f"{n_bad} site(s) with Poor/Critical road condition")
    if not lines:
        lines.append("no single factor stands out; the rank comes from combined risk and low effort")
    head = (f"Ranked #{rank} of {n_units} by Return on Effort ({unit['roe']:.1f} risk per crew-hour). "
            f"{unit['n_sites']} site(s) fixed in one trip: about {unit['crew_hours']:.1f} crew-hours "
            f"including {unit['km']:.1f} km of travel.")
    return head, lines
