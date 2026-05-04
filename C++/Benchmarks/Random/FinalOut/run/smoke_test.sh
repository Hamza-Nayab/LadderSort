#!/usr/bin/env bash
set -e

mkdir -p results/raw
GIT_HASH=$(./run/get_git_hash.sh)

./bin/bench_laddersort_ablation \
  --csv results/raw/smoke_ablation.csv \
  --git-hash "$GIT_HASH" \
  --dataset social_feed \
  --seed 12648430

./bin/bench_postinsert_multi \
  --csv results/raw/smoke_postinsert.csv \
  --git-hash "$GIT_HASH" \
  --dataset post_insert_case \
  --seed 12648430