"""
analyze_fills.py -- per-fill P&L attribution: find out WHICH fills lose money.

Put next to model.py + diagnose.py (with the PROVEN model.py in place) and run:

    python analyze_fills.py

Runs the replay once, then slices weighted markout (size_m * markout_pips,
the exact quantity profit_factor and execution quality are built from) by:
  hour, client size, spread state at fill, and same-tick same-side burst depth.

The output tells us whether toxic flow is concentrated (=> a targeted filter
can remove it cheaply) or diffuse (=> it cannot, and 58.99 is the ceiling).
"""

import glob, os
import pandas as pd

from openmic.projects.fx_algo_replay.replay import Replay
from model import SubmittedPricingStrategy

DATA_PATH = None  # set explicitly if auto-find fails, e.g. "../data/training_dataset.csv"
if DATA_PATH is None:
    for pattern in ("*.csv", "../*.csv", "../data/*.csv", "data/*.csv"):
        hits = [p for p in glob.glob(pattern) if "train_score" not in p and "diagnose" not in p]
        if hits:
            DATA_PATH = hits[0]
            break
if DATA_PATH is None or not os.path.exists(DATA_PATH):
    raise SystemExit("Set DATA_PATH at the top of analyze_fills.py")

print(f"loading {DATA_PATH}")
data = Replay.load_data(DATA_PATH)
result = Replay.replay_combined_data(data, SubmittedPricingStrategy())

f = result.fills.copy()
print(f"\n{len(f):,} fills.  columns: {list(f.columns)}")

f["wm"] = f["size_m"] * f["markout_pips"]          # weighted markout (pips*m)
f["hour"] = f["utc_time"].astype(str).str[:2].astype(int)

def slice_report(name, key):
    g = f.groupby(key).agg(
        fills=("wm", "size"),
        volume_m=("size_m", "sum"),
        wm_sum=("wm", "sum"),
        wm_per_m=("wm", lambda s: s.sum()),
    )
    g["wm_per_m"] = g["wm_sum"] / g["volume_m"]
    g["pct_of_volume"] = 100 * g["volume_m"] / f["size_m"].sum()
    g["pct_of_neg_wm"] = 0.0
    neg_total = f.loc[f["wm"] < 0, "wm"].sum()
    if neg_total < 0:
        neg = f[f["wm"] < 0].groupby(key)["wm"].sum()
        g.loc[neg.index, "pct_of_neg_wm"] = 100 * neg / neg_total
    print(f"\n{'='*76}\n{name}\n{'='*76}")
    print(g[["fills", "volume_m", "wm_sum", "wm_per_m",
             "pct_of_volume", "pct_of_neg_wm"]].round(3).to_string())

# ---- 1. by hour -----------------------------------------------------------
slice_report("BY HOUR  (wm_sum<0 rows are where the losses live)", "hour")

# ---- 2. by client size bucket --------------------------------------------
f["size_bucket"] = pd.cut(f["client_size_m"],
                          [0, 2, 5, 8, 12, 100],
                          labels=["0-2m", "2-5m", "5-8m", "8-12m", ">12m"])
slice_report("BY CLIENT SIZE", "size_bucket")

# ---- 3. by spread state at fill ------------------------------------------
f["spread_bucket"] = pd.cut(f["market_neutral_spread_pip"] if
                            "market_neutral_spread_pip" in f.columns else
                            f["quote_width_pips"],
                            [0, 0.10, 0.25, 0.60, 1.5, 100],
                            labels=["<0.10", "0.10-0.25", "0.25-0.60",
                                    "0.60-1.5", ">1.5"])
slice_report("BY SPREAD AT FILL (dislocation = wide spread)", "spread_bucket")

# ---- 4. by same-tick same-side burst depth -------------------------------
f = f.sort_values(["step"]).reset_index(drop=True)
f["burst_n"] = f.groupby(["step", "side"])["side"].transform("size")
f["burst_bucket"] = pd.cut(f["burst_n"], [0, 1, 2, 3, 100],
                           labels=["single", "2", "3", "4+"])
slice_report("BY SAME-TICK SAME-SIDE BURST DEPTH (sweeps = informed?)",
             "burst_bucket")

# ---- 5. the money table: hour x size for negative wm ---------------------
print(f"\n{'='*76}\nHOUR x SIZE: sum of weighted markout (negative cells = the toxic pockets)\n{'='*76}")
pv = f.pivot_table(index="hour", columns="size_bucket", values="wm",
                   aggfunc="sum", observed=False).round(0)
print(pv.to_string())

tot_pos = f.loc[f["wm"] > 0, "wm"].sum()
tot_neg = f.loc[f["wm"] < 0, "wm"].sum()
print(f"\nTOTAL: positive wm = {tot_pos:,.0f}   negative wm = {tot_neg:,.0f}"
      f"   profit_factor = {tot_pos/abs(tot_neg):.3f}")
f.to_csv("fills_attribution.csv", index=False)
print("per-fill dump written to fills_attribution.csv")