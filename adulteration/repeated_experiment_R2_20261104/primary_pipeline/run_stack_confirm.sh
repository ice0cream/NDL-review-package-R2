#!/usr/bin/env bash
# Confirmatory runs for the stacking pipeline as primary analysis (Reviewer 6, comments 37, 39, 41).
# Order: driver self-check -> stacked variants to 10 random partitions -> permutation test (99) -> refit bootstrap (100).
# Every step resumes from its checkpoints, so the script can be restarted after an interruption.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
W="${WORKERS:-14}"
BV="../../NIR-CLO/REVISE2_benchmark/code/benchmark_variants.py"
ARGS=(--input inputs/NIR_color_merged_dataset_redo110.csv --group-map inputs/parent_map_redo.csv --workers "$W")

echo "=== check $(date '+%F %T')"
python -u stack_confirm.py check --workers "$W"

for v in nir_stack fusion_stack; do
  echo "=== $v repeats=10 $(date '+%F %T')"
  python -u "$BV" --variant "$v" --task quantitative_with_zero105 --repeats 10 "${ARGS[@]}" \
         --output "runs_repeats/quantitative_with_zero105/$v"
done
python collect_results.py > /dev/null
echo "=== repeats collected $(date '+%F %T')"

echo "=== permutation $(date '+%F %T')"
python -u stack_confirm.py perm --workers "$W"
python stack_confirm.py summary

echo "=== bootstrap $(date '+%F %T')"
python -u stack_confirm.py boot --workers "$W"
python stack_confirm.py summary
echo "=== ALL DONE $(date '+%F %T')"
