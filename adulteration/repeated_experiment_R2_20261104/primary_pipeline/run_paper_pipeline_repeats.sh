#!/usr/bin/env bash
# Partition sensitivity of the manuscript pipeline on the repeated experiment (Reviewer 6, comments 27/28/33/39).
# No --outer-plan: benchmark_variants.py draws random parent-grouped 5-fold partitions with seeds 2026 + repeat*10000,
# so every variant is evaluated on the same partitions and the comparisons are paired.
#   single-model variants (color_only, nir_full, feature_fusion): 10 partitions, all 110 states and the 88 mixtures
#   stacked variants (nir_stack, fusion_stack):                    5 partitions, all 110 states (about 30 min per partition)
# Output: runs_repeats/<task>/<variant>. Re-running resumes finished folds and skips finished runs.
set -e
cd "$(dirname "$0")"
BV="../../NIR-CLO/REVISE2_benchmark/code/benchmark_variants.py"
ARGS=(--input inputs/NIR_color_merged_dataset_redo110.csv --group-map inputs/parent_map_redo.csv --workers "${WORKERS:-12}")
export PYTHONIOENCODING=utf-8
run() {  # variant task repeats
  out="runs_repeats/$2/$1"
  if [ -f "$out/metrics_summary.csv" ]; then echo "skip $out"; return; fi
  echo "=== $1 $2 repeats=$3 $(date +%T)"
  python -u "$BV" --variant "$1" --task "$2" --repeats "$3" "${ARGS[@]}" --output "$out"
}
for v in color_only nir_full feature_fusion; do
  run "$v" quantitative_with_zero105 10
  run "$v" quantitative_positive83 10
done
for v in nir_stack fusion_stack; do
  run "$v" quantitative_with_zero105 5
done
echo "=== ALL DONE $(date +%T)"
