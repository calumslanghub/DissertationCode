import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---- config ----
TRIPS_CSV      = "glatrips.csv"          # origin, destination, started_at
PRECOVID_START = pd.Timestamp("2017-01-01", tz="Europe/London")
PRECOVID_END   = pd.Timestamp("2020-03-20", tz = "Europe/London")     # Scottish lockdown ~23 Mar 2020; classify strictly before
AM_PEAK        = (7, 10)                          # [07:00, 10:00)
PM_PEAK        = (16, 19)                         # [16:00, 19:00)
MIDDAY         = (10, 16)
MIN_TRIPS      = 30                               # min pre-COVID trips on a pair to classify it
K              = 2
EPS            = 1e-6


def canon_cols(origin, destination):
    """Canonical unordered key (a<=b) as two aligned arrays."""
    a = np.where(origin <= destination, origin, destination)
    b = np.where(origin <= destination, destination, origin)
    return a, b


def build_commuter_labels(trips, precovid_start=PRECOVID_START, precovid_end=PRECOVID_END,
                          am_peak=AM_PEAK, pm_peak=PM_PEAK, midday=MIDDAY,
                          min_trips=MIN_TRIPS, k=K, random_state=0):
    t = trips.copy()
    
    if t["started_at"].dt.tz is None:
        t["started_at"] = t["started_at"].dt.tz_localize("UTC")
    t["started_at"] = t["started_at"].dt.tz_convert("Europe/London")
    
    t = t[(t["started_at"] >= precovid_start) & (t["started_at"] < precovid_end)]
    t = t[t["origin"] != t["destination"]]
    if len(t) == 0:
        raise ValueError("no trips in the pre-COVID window — check dates / trip span")

    
    hour = t["started_at"].dt.hour
    t["am"]  = ((hour >= am_peak[0]) & (hour < am_peak[1])).astype(int)
    t["pm"]  = ((hour >= pm_peak[0]) & (hour < pm_peak[1])).astype(int)
    t["mid"] = ((hour >= midday[0])  & (hour < midday[1])).astype(int)
    t["wd"]  = (t["started_at"].dt.dayofweek < 5).astype(int)

    oc, dc = canon_cols(t["origin"].values, t["destination"].values)
    t["origin_c"], t["dest_c"] = oc, dc
    fwd = (t["origin"].values == oc)                # True if this trip runs a->b (canonical direction)
    t["fwd_am"] = (fwd & t["am"].values.astype(bool)).astype(int)
    t["rev_am"] = (~fwd & t["am"].values.astype(bool)).astype(int)
    t["fwd_pm"] = (fwd & t["pm"].values.astype(bool)).astype(int)
    t["rev_pm"] = (~fwd & t["pm"].values.astype(bool)).astype(int)

    g = t.groupby(["origin_c", "dest_c"]).agg(
        total=("wd", "size"), weekday_trips=("wd", "sum"),
        am=("am", "sum"), pm=("pm", "sum"), mid=("mid", "sum"),
        fwd_am=("fwd_am", "sum"), rev_am=("rev_am", "sum"),
        fwd_pm=("fwd_pm", "sum"), rev_pm=("rev_pm", "sum"),
    ).reset_index()
    g["weekend_trips"] = g["total"] - g["weekday_trips"]

    # features
    g["peak_share"]   = (g["am"] + g["pm"]) / g["total"]
    g["midday_share"] = g["mid"] / g["total"]
    # weekday vs weekend per-day rate
    wk = (g["weekday_trips"] + 1) / 5.0
    we = (g["weekend_trips"] + 1) / 2.0
    g["weekday_rate_ratio"] = wk / we
    g["log_wkratio"] = np.log(g["weekday_rate_ratio"])

    def asym(f, r):
        d = (f + r).astype(float)
        out = np.zeros_like(d)
        nz = d > 0
        out[nz] = (f[nz] - r[nz]) / d[nz]        # only divide where trips exist
        return out
    am_asym = asym(g["fwd_am"].values, g["rev_am"].values)
    pm_asym = asym(g["fwd_pm"].values, g["rev_pm"].values)
    g["reversal"] = np.abs(am_asym - pm_asym)

    # classify only pairs with enough pre-COVID trips
    cl = g[g["total"] >= min_trips].copy()
    excluded = int((g["total"] < min_trips).sum())
    if len(cl) < k:
        raise ValueError(f"only {len(cl)} classifiable pairs (>= {min_trips} trips) — lower MIN_TRIPS")

    feats = ["peak_share", "log_wkratio", "reversal"]
    X = cl[feats].values
    Xs = (X - X.mean(0)) / (X.std(0) + EPS)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xs)
    cl["cluster"] = km.labels_

    # commuter cluster = highest COMBINED centroid score across all three features
    means = cl.groupby("cluster")[feats].mean()
    z = (means - means.mean()) / (means.std(ddof=0) + EPS)
    commuter_cluster = z.sum(axis=1).idxmax()
    cl["is_commuter"] = (cl["cluster"] == commuter_cluster)

    diagnostics = {
        "classified": len(cl), "excluded_low_volume": excluded,
        "pct_commuter": round(100 * cl["is_commuter"].mean(), 1),
        "cluster_means": means.assign(n=cl.groupby("cluster").size(),
                                      commuter=means.index == commuter_cluster),
    }
    out = cl[["origin_c", "dest_c", "total", "peak_share",
              "weekday_rate_ratio", "reversal", "midday_share",
              "cluster", "is_commuter"]].reset_index(drop=True)
    return out, diagnostics


def main():
    trips = pd.read_csv(TRIPS_CSV, parse_dates=["started_at"])
    assert {"origin", "destination", "started_at"}.issubset(trips.columns), trips.columns.tolist()
    labels, diag = build_commuter_labels(trips)
    print(f"classified {diag['classified']} pairs | "
          f"excluded (low volume) {diag['excluded_low_volume']} | "
          f"{diag['pct_commuter']}% commuter\n")
    print("cluster centroids (commuter cluster should have the higher values):")
    print(diag["cluster_means"].to_string())
    labels.to_csv("precovidcommuter_labels.csv", index=False)
    print("\nsaved precovidcommuter_labels.csv  (merge onto exposure/DiD tables on [origin_c, dest_c])")


if __name__ == "__main__":
    main()