#!/usr/bin/env python3
"""
Estimate remaining time for empc/sysid sweep from start_time.txt and
val_test_losses.csv in work/il.*empc* and work/il.*sysid* run dirs.
Run from imitation_nonconvex: python estimate_sweep_time.py
"""
import os
import time
import glob

WORK = "work"
TARGET_EPOCHS = 10000
TOTAL_JOBS = 96  # empc + sysid for pendulum + cartpole
MAX_PARALLEL = 32  # 16 per stream * 2 streams

def main():
    now = time.time()
    runs = []
    for pattern in ["il.*.empc.*", "il.*.sysid.*"]:
        for path in glob.glob(os.path.join(WORK, pattern)):
            if not os.path.isdir(path):
                continue
            for seed_dir in os.listdir(path):
                save_dir = os.path.join(path, seed_dir)
                if not os.path.isdir(save_dir):
                    continue
                start_file = os.path.join(save_dir, "start_time.txt")
                val_file = os.path.join(save_dir, "val_test_losses.csv")
                end_file = os.path.join(save_dir, "end_time.txt")
                if not os.path.exists(start_file):
                    continue
                with open(start_file) as f:
                    start_ts = float(f.read().strip())
                epochs = 0
                if os.path.exists(val_file):
                    with open(val_file) as f:
                        epochs = max(0, sum(1 for _ in f) - 1)  # minus header
                if epochs <= 0:
                    runs.append({"dir": save_dir, "start": start_ts, "epochs": 0, "end": None})
                    continue
                end_ts = None
                if os.path.exists(end_file):
                    with open(end_file) as f:
                        end_ts = float(f.read().strip())
                runs.append({"dir": save_dir, "start": start_ts, "epochs": epochs, "end": end_ts})

    completed = sum(1 for r in runs if r["end"] is not None)
    in_progress = [r for r in runs if r["end"] is None and r["epochs"] > 0]
    not_started = sum(1 for r in runs if r["epochs"] == 0 and r["end"] is None)

    print("Sweep progress")
    print("  Completed (10k epochs): {} / {}".format(completed, TOTAL_JOBS))
    print("  In progress (≥1 epoch): {}".format(len(in_progress)))
    print("  Not started / no epochs yet: {}".format(not_started))

    if not in_progress and completed == 0:
        if not runs:
            print("  No run dirs with start_time.txt (re-run sweep with updated il_exp.py to get timing).")
        else:
            print("  No run has reached 1 epoch yet; estimate after some progress.")
        return

    # Epochs per hour from in-progress or completed runs
    rates = []
    for r in runs:
        if r["epochs"] <= 0:
            continue
        elapsed = (r["end"] or now) - r["start"]
        if elapsed <= 0:
            continue
        rates.append(r["epochs"] / (elapsed / 3600.0))

    if not rates:
        print("  No epochs-per-hour data yet.")
        return

    import statistics
    rate_median = statistics.median(rates)
    rate_mean = statistics.mean(rates)
    hours_per_run = TARGET_EPOCHS / rate_median
    remaining = TOTAL_JOBS - completed
    # Assume remaining jobs run with MAX_PARALLEL parallelism
    est_remaining_hours = (remaining / MAX_PARALLEL) * hours_per_run if remaining > 0 else 0

    print("")
    print("Timing (from runs with ≥1 epoch)")
    print("  Epochs per hour: median {:.1f}, mean {:.1f}".format(rate_median, rate_mean))
    print("  Hours per run (to 10k epochs): {:.1f}".format(hours_per_run))
    print("  Remaining jobs: {}".format(remaining))
    print("  Estimated remaining wall time: {:.1f} hours ({:.1f} h total for full sweep)".format(
        est_remaining_hours, (TOTAL_JOBS / MAX_PARALLEL) * hours_per_run))

if __name__ == "__main__":
    main()
