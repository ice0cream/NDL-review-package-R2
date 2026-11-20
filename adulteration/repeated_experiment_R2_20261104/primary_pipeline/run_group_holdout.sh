#!/usr/bin/env bash
# Manuscript pipeline with a whole measurement batch or a whole supplier held out (Reviewer 6, comment 26).
# Plans: inputs/outer_plan_by_batch.csv (3 folds), inputs/outer_plan_by_supplier.csv (fold 1 = F1-F17, fold 2 = F18-F22).
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
BV="../../NIR-CLO/REVISE2_benchmark/code/benchmark_variants.py"
for plan in batch supplier; do
  for v in fusion_stack nir_stack nir_full feature_fusion color_only; do
    out="runs_holdout/$plan/$v"
    if [ -f "$out/metrics_summary.csv" ]; then echo "skip $out"; continue; fi
    echo "=== $plan $v $(date '+%F %T')"
    python -u "$BV" --variant "$v" --task quantitative_with_zero105 --input inputs/NIR_color_merged_dataset_redo110.csv \
      --group-map inputs/parent_map_redo.csv --outer-plan "inputs/outer_plan_by_$plan.csv" --workers "${WORKERS:-5}" --output "$out"
  done
done
echo "=== ALL DONE $(date '+%F %T')"
