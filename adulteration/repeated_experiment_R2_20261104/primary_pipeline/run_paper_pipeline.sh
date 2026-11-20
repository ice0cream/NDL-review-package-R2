#!/usr/bin/env bash
# Manuscript pipeline (decision_nested_83.py via REVISE2 benchmark_variants.py) on the repeated experiment.
# Tasks: positive mixtures only (88; task name kept from the original code), all 110 states incl. authentic,
#        binary authentic vs adulterated, four-class Authentic/Low(<=20)/Medium(<=40)/High.
# Fixed parent-to-fold plan in inputs/outer_fold_plan_redo.csv. Re-running skips finished runs.
set -e
cd "$(dirname "$0")"
BV="../../NIR-CLO/REVISE2_benchmark/code/benchmark_variants.py"
ARGS=(--input inputs/NIR_color_merged_dataset_redo110.csv --group-map inputs/parent_map_redo.csv
      --outer-plan inputs/outer_fold_plan_redo.csv --workers "${WORKERS:-14}")
export PYTHONIOENCODING=utf-8
for v in color_only nir_full feature_fusion nir_stack fusion_stack; do
  for task in quantitative_positive83 quantitative_with_zero105 Task2_binary Task1_4class; do
    out="runs/$task/$v"
    if [ -f "$out/metrics_summary.csv" ]; then echo "skip $out"; continue; fi
    echo "=== $v $task $(date +%T)"
    python -u "$BV" --variant "$v" --task "$task" "${ARGS[@]}" --output "$out"
  done
done
echo "=== ALL DONE $(date +%T)"
