# Data and code — SEIN-D-26-03096, Revision 2

Data, code and results behind the revised manuscript. All paths are relative to this folder, and every script reads and
writes only inside it, so the analyses can be re-run as they are. Integrity: `MANIFEST_sha256.csv` gives the size,
SHA-256 hash and class of every file.

## 1. Contents

| Folder or file | Content |
|---|---|
| `NIR/botanical_samples/converted_csv/` | NIR spectra of the 58 botanical samples, one CSV per batch with its three readings (`batch_spectra/`), with acquisition metadata, repeatability check and batch means |
| `script/data/` | Modelling tables of the 58 samples: mean NIR spectra, CIELAB, assay values; the 24 preprocessed spectral tables |
| `phytochemical_assays/` | Reference values (`DZ.CSV`); calibration readings and the three absorbance readings of each sample (`raw_records/`) |
| `moisture_weighings.csv` | Weighings of the moisture determination (W0 bottle, W1 with sample, W2 after 5 h, W3 after a further hour) |
| `script/` | Analysis code of the first submission (exploratory analyses, colorimetry, phytochemical parameters) |
| `R1_revision/` | Nested PLS-DA and nested PLSR of the first revision, which give the classification and regression results of the manuscript |
| `R2_revision/` | Analyses of this revision for the botanical and phytochemical comments (`scripts/` and one result folder per comment), supplementary tables and figures |
| `adulteration/NIR-CLO/` | Code of the decision-level fusion pipeline (`R1_primary_analysis_20260911/code/`, used unchanged) and of the five approaches compared in this revision (`REVISE2_benchmark/code/`) |
| `repeated_experiment_record_sheet.csv` | Repeated adulteration experiment, 6–7 Nov 2026: one row per state with the weighings, the concentration on the sheet and the nine CIELAB readings (`c1`–`c9` = L*, a*, b* of readings 1–3) |
| `adulteration/repeated_experiment_R2_20261104/` | Repeated experiment: NIR spectra of the 339 readings (`spectra_dataset/NIR_spectra_by_reading.csv`), design (`design_*`), data checks (`data_checks/`), primary analysis (`primary_pipeline/`), supporting PLSR analysis (`supporting_analysis/`) |

The spectra are given as extracted from the instrument files: apparent absorbance, log10(1/R), 1,843 points from 11,545
to 3,954 cm⁻¹, one row or file per reading, with the values read from the OPUS files. The instrument files themselves are
not included and are available from the corresponding authors. Weighings, colour readings, absorbances and VIP values
are given as tables extracted from the laboratory workbooks, with the values as recorded (a dash marks an empty field;
one colour reading was recorded as "97. 28" and is reported by the data checks as non-numeric).

## 2. Requirements

Python 3.10 with numpy 2.2, pandas 2.3, scipy 1.15, scikit-learn 1.7, joblib 1.5, matplotlib, openpyxl 3.1 and
brukeropusreader 1.3 (`pip install -r script/requirements.txt brukeropusreader scikit-learn joblib openpyxl`). The `.sh`
files need a bash shell. All random seeds are fixed in the scripts.

## 3. How to run

Run from this folder. Each script writes into the folder it reports.

**Botanical samples: classification and phytochemical regression (manuscript Sections 3.1–3.2)**

```bash
python R1_revision/scripts/comment03_nested_cv.py
python R1_revision/comment03_validation/nested_PLSR_20260910/nested_plsr.py --workers 8 --output R1_revision/comment03_validation/nested_PLSR_20260910/results_rerun
```

`nested_plsr.py` writes to a new folder and refuses one that holds results of other code or data, so the saved results
are kept for comparison.

**Analyses of the first submission** (`script/`; results go to `script/results/`): each script runs on its own, except
that `03_general_parameters/plot_regression_coefficients.py` needs the output of `03_general_parameters/ks_plsr.py`.

```bash
python script/01_exploratory/pls_da.py      # for example
```

**Analyses of this revision for the botanical and phytochemical comments**

```bash
python R2_revision/R6-3_sample_traceability/build_traceability_check.py
python R2_revision/scripts/r6_9_moisture_dry_basis.py
python R2_revision/scripts/r6_9_moisture_nir_association.py
python R2_revision/scripts/r6_12_repeatability_preprocessed.py
python R2_revision/scripts/r6_19_calibration_residuals.py
python R2_revision/scripts/r6_35_phytochem_baselines.py
python R2_revision/scripts/r6_35_phytochem_alternatives.py
python R2_revision/scripts/r6_36_nested_confusion_figure.py
python R2_revision/scripts/r6_37_full_procedure_permutation.py
python R2_revision/scripts/make_supplement_r2.py       # supplementary tables of the botanical analyses
python R2_revision/scripts/make_print_figures_R2.py    # Figs. 1, 3, 4, 5 at print size
```

**Repeated adulteration experiment (manuscript Section 3.3)**

```bash
cd adulteration/repeated_experiment_R2_20261104
python supporting_analysis/redo_analysis.py
python supporting_analysis/redo_analysis_extended.py
python supporting_analysis/confirm_analysis.py
python primary_pipeline/build_inputs.py
bash primary_pipeline/run_paper_pipeline.sh
bash primary_pipeline/run_paper_pipeline_repeats.sh
bash primary_pipeline/run_stack_confirm.sh
bash primary_pipeline/run_group_holdout.sh
python primary_pipeline/collect_results.py
python primary_pipeline/collect_primary.py      # every quoted number -> primary_pipeline/summary/primary_numbers.json
python primary_pipeline/make_figures.py
```

Times on a 16-core desktop: the nested PLS-DA, the nested PLSR and the R6-37 permutation test take hours each; for the
repeated experiment from `redo_analysis.py` onwards 4 min, 5 min, 30 min, under 1 min, 5 h, 6 h, about 3 days
(`run_stack_confirm.sh`) and 3 h (`run_group_holdout.sh`), then under 1 min each. The pipeline resumes from the saved
folds, so `collect_results.py` and `collect_primary.py` run directly on the results in this package.

## 4. Data handling in the repeated experiment

- **Unit.** A state is one parent powder at one starch level: 22 FTR parents × 5 levels (0, 5, 15, 30, 50 % w/w) = 110
  states, each with three repacked NIR readings and three colour readings. The starch content is calculated from the
  weighings, not taken as the nominal level. The three spectra of a state are averaged; the colour value is the median.
- **File-level cleaning** (366 raw files → 339): 4 files relabelled to the step shown by their acquisition time; 21
  extra readings removed (a state with four readings loses the earliest, with five the two earliest); 4 files of a
  superseded set of F3 at 0 % and 2 repeat files of F6 at 5 % removed. Every file has a row in
  `data_checks/correction_log.csv`. No state and no parent is excluded from the primary analysis.
- **Partitions.** All states of a parent stay on the same side of every split (outer, inner, stacking, bootstrap);
  `split_audit` files list every split, and the scripts stop if a parent is shared.
- **Status of the analyses.** Primary: `primary_pipeline/` (the pipeline of the original submission, code unchanged).
  Supporting: `supporting_analysis/redo_analysis*.py`, `confirm_analysis.py`. Exploratory only, used for no claim:
  `supporting_analysis/explore_*.py` and their `results_*_explore/` folders.
- **Checks.** `confirm_analysis.py` recomputes the primary predictions and stops unless they match the saved ones to
  10⁻⁶; `stack_confirm.py check` verifies that the driver reproduces the saved pipeline predictions.

## 5. Where each reviewer comment is answered

Paths of the repeated experiment are relative to `adulteration/repeated_experiment_R2_20261104/`.

| Comment | Topic | Files |
|---|---|---|
| R3-1, R6-14 | Same mass and packing for authentic and mixed states | `repeated_experiment_record_sheet.csv`; `design_summary.txt` |
| R3 unresolved 1 | OPLS-DA settings | `R2_revision/R3-U1_OPLSDA_SIMCA/` |
| R6-3, R6-4 | Sample identifiers and suppliers | `R2_revision/R6-3_sample_traceability/` |
| R6-6 | Total sulfur | `R2_revision/R6-6_elemental_analysis/` |
| R6-9 | Moisture and dry-matter sensitivity | `R2_revision/R6-9_moisture/` |
| R6-11, R6-15 | Acquisition parameters, background, timing | `R2_revision/R6-11_15_acquisition_metadata/` |
| R6-12 | Replicate variability, botanical spectra | `R2_revision/R6-12_15_spectral_repeatability/` |
| R6-12, R6-43 | Repacked replicate variability, repeated experiment | `supporting_analysis/results/replicate_predictions_M1.csv`, `replicate_repeatability_M1.csv`; `supporting_analysis/results_confirm/domain_flags_per_state.csv` |
| R6-19 | Assay calibration | `R2_revision/R6-19_calibration/` |
| R6-25, R6-43 | Weighing resolution propagated to concentration | `supporting_analysis/results_confirm/weighing_uncertainty_by_state.csv`, `weighing_uncertainty_summary.csv`, `weighing_consistency_checks.csv` |
| R6-26 | Batch- and supplier-level validation | `supporting_analysis/results_confirm/partition_summary.csv`, `split_audit.csv`; `primary_pipeline/runs_holdout/` |
| R6-27, R6-28 | Five approaches on the same states and folds; ablation of colour | `primary_pipeline/summary/metrics_all.csv`, `paired_differences.csv`, `repeats_summary.csv` |
| R6-29 | No subset is used | `supporting_analysis/results/states_used.csv` |
| R6-30 to R6-32 | Low-level performance; thresholds from training data | `supporting_analysis/results/per_level_errors.csv`; `supporting_analysis/results_confirm/detection_by_level.csv`, `detection_auc_thresholds.csv` |
| R6-33 | With and without authentic states | `supporting_analysis/results_extended/regression_with_without_authentic.csv`; `primary_pipeline/summary/paired_differences.csv` |
| R6-35 | Phytochemical baselines and alternatives | `R2_revision/R6-35_phytochem_baselines/` |
| R6-36 | Nested confusion matrices | `R2_revision/R6-36_nested_confusion/`; `supporting_analysis/results_extended/plsda_confusion.csv`; `primary_pipeline/summary/fourclass_recall.csv` |
| R6-37 | Permutation test of the complete procedure | `R2_revision/R6-37_full_procedure_permutation/`; `supporting_analysis/results_confirm/permutation_full_procedure.csv`; `primary_pipeline/stack_confirm/permutation_null.csv` |
| R6-39 | Random partitions; poorest folds and parents | `primary_pipeline/summary/repeats_by_partition.csv`; `supporting_analysis/results_confirm/partition_fold_errors.csv`, `error_by_parent.csv` |
| R6-40 | Predictions outside 0–100 % | `supporting_analysis/results_confirm/clipping_effect.csv`; `primary_pipeline/summary/clipping_counts.csv` |
| R6-41 | Conditional and refit intervals; stability of selections | `supporting_analysis/results/bootstrap_intervals.csv`, `fold_choices.csv`; `supporting_analysis/results_confirm/bootstrap_refit_summary.csv`, `bootstrap_refit_selection_frequency.csv` |
| R6-42, R6-53 | Record of every fold of the stacking pipeline | `primary_pipeline/runs/<task>/<variant>/<task>/repeat_01/fold_XX/`, `selected_bands_by_outer_training_fold.csv`, `outer_OOF_predictions.csv` |
| R6-44 | Order, batches and reference drift | `supporting_analysis/results_confirm/order_correlations.csv`, `order_parent_sequence_between_rounds.csv`, `batch_times.csv`, `starch_reference_by_session.csv` |
| Supplementary tables and figures | | `R2_revision/Supplement_R2/`, `R2_revision/Figures_R2/` |

## 6. Notes

- Four scripts work on the instrument files and are included as the record of how the spectral datasets and their
  metadata were obtained: `data_checks/check_redo_data.py` (file inventory), `data_checks/fix_labels_by_time.py` (the
  cleaning rule, applied file by file in `correction_log.csv`), `data_checks/export_spectra_dataset.py` (the spectral
  dataset of the repeated experiment) and `R2_revision/scripts/r6_11_15_opus_acquisition_metadata.py` (acquisition
  parameters; its results are in `R2_revision/R6-11_15_acquisition_metadata/`). In this package the analyses start from
  the spectral datasets: `supporting_analysis/redo_analysis.py` reads `spectra_dataset/NIR_spectra_by_reading.csv`, whose
  values are bit-identical to those read from the instrument files.
- Not included because the scripts regenerate them: plots of intermediate runs, fitted-model arrays and per-fold
  checkpoints of the nested pipeline. Each finished fold keeps `completed.json`, `locked_selection_before_outer_evaluation.json`,
  `inner_candidate_results.csv` and `split_audit.csv`.
- Folder and file names are English translations of the working names, and paths recorded in run manifests are given
  relative to this folder. Scripts therefore differ from the hashes recorded in the run manifests only in these path
  names and, for `redo_analysis.py` and `confirm_analysis.py`, in reading the spectral dataset instead of the instrument
  files.
