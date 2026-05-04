#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

Path("results/summary").mkdir(parents=True, exist_ok=True)

kstats_path = Path("results/raw/03_k_stats_raw.csv")
sweep_path = Path("results/raw/05_k_sweep_raw.csv")

if kstats_path.exists():
    df = pd.read_csv(kstats_path)

    summary = (
        df.groupby(["dataset", "n"])
          .agg(
              count=("k_final", "count"),
              mean_k=("k_final", "mean"),
              median_k=("k_final", "median"),
              min_k=("k_final", "min"),
              max_k=("k_final", "max"),
              mean_k_over_n=("k_over_n", "mean"),
          )
          .reset_index()
    )

    summary.to_csv("results/summary/04_k_stats_summary.csv", index=False)
    print("Wrote results/summary/04_k_stats_summary.csv")
    print(summary)
else:
    print("Skipping K stats summary: results/raw/03_k_stats_raw.csv not found")

if sweep_path.exists():
    df = pd.read_csv(sweep_path)

    summary = (
        df.groupby(["n", "target_k", "algo", "variant"])
          .agg(
              measured_k_mean=("measured_k", "mean"),
              measured_k_min=("measured_k", "min"),
              measured_k_max=("measured_k", "max"),
              mean_sec=("time_sec", "mean"),
              std_sec=("time_sec", "std"),
              min_sec=("time_sec", "min"),
              max_sec=("time_sec", "max"),
              sorted_ok=("sorted_ok", "min"),
              fallback_rate=("used_fallback", "mean"),
          )
          .reset_index()
    )

    timsort = (
        summary[summary["algo"] == "TimSort"]
        [["n", "target_k", "mean_sec"]]
        .rename(columns={"mean_sec": "timsort_mean_sec"})
    )

    summary = summary.merge(timsort, on=["n", "target_k"], how="left")
    summary["speedup_vs_timsort"] = summary["timsort_mean_sec"] / summary["mean_sec"]

    summary.to_csv("results/summary/06_k_sweep_summary.csv", index=False)
    print("Wrote results/summary/06_k_sweep_summary.csv")
    print(summary)
else:
    print("Skipping K sweep summary: results/raw/05_k_sweep_raw.csv not found")
