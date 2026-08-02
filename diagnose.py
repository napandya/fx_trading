"""
diagnose.py -- run the replay locally and print the per-component score breakdown.

Put this file in the SAME folder as your model.py (e.g. .../fx_algo/Game_Of_Trades/)
and run:

    python diagnose.py

It does NOT submit or upload anything -- it only replays the training data and
prints the score breakdown that build_summary computes.

If your training CSV lives somewhere else, change DATA_PATH below.
"""

import glob
import os

from openmic.projects.fx_algo_replay.replay import Replay

# your strategy -- this imports the model.py sitting next to this script
from model import SubmittedPricingStrategy


# ---------------------------------------------------------------------------
# 1. Find the training data CSV.
#    Adjust this path if needed -- it's the same file the normal replay uses.
# ---------------------------------------------------------------------------
DATA_PATH = None  # e.g. r"/Users/kd90400/Citi-Projects/open-mic/.../training_data.csv"

if DATA_PATH is None:
    # try to auto-find a csv in this folder or one level up
    for pattern in ("*.csv", "../*.csv", "../data/*.csv", "data/*.csv"):
        hits = [p for p in glob.glob(pattern) if "train_score" not in p]
        if hits:
            DATA_PATH = hits[0]
            break

if DATA_PATH is None or not os.path.exists(DATA_PATH):
    raise SystemExit(
        "Could not find the training CSV automatically.\n"
        "Edit DATA_PATH at the top of diagnose.py to point at it."
    )

print(f"loading data: {DATA_PATH}")
data = Replay.load_data(DATA_PATH)

# ---------------------------------------------------------------------------
# 2. Run the replay (same as the submission flow, minus the upload).
# ---------------------------------------------------------------------------
result = Replay.replay_combined_data(data, SubmittedPricingStrategy())

# ---------------------------------------------------------------------------
# 3. Print everything that matters.
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SCORE BREAKDOWN (this is the table we need)")
print("=" * 80)
print(result.score_breakdown.to_string(index=False))

print("\n" + "=" * 80)
print("KEY RAW METRICS")
print("=" * 80)
keys = [
    "score",
    "final_pnl_usd",
    "max_drawdown_usd",
    "sharpe_proxy",
    "calmar_proxy",
    "market_share" if "market_share" in result.summary else "volume_fill_rate",
    "fill_rate",
    "avg_abs_inventory_m",
    "max_abs_inventory_m",
    "terminal_inventory_m",
    "terminal_liquidation_cost_usd",
    "avg_public_markout_pips",
    "avg_spread_capture_pips",
    "public_adverse_selection_cost_pips",
    "pnl_per_volume_pips",
    "profit_factor",
    "profitable_volume_share",
    "avg_relative_quote_width",
    "avg_quote_width_pips",
    "quote_uptime_rate",
    "avg_quote_latency_ms",
    "p99_quote_latency_ms",
    "filled_volume_m",
    "client_volume_m",
    "fills",
    "client_trades",
]
for k in keys:
    if k in result.summary:
        v = result.summary[k]
        print(f"  {k:<38} {v}")

# also dump the full summary to a file so nothing is lost
with open("diagnose_summary.txt", "w") as f:
    f.write("SCORE BREAKDOWN\n")
    f.write(result.score_breakdown.to_string(index=False))
    f.write("\n\nFULL SUMMARY\n")
    for k, v in result.summary.items():
        f.write(f"{k}: {v}\n")
print("\nfull dump written to diagnose_summary.txt")