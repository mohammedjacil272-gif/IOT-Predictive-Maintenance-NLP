# Phase 4 — Azure PdM Temporal Dataset and Target Foundation

Status: Completed foundation generation; no model training or inference

## 1. Contract changes and resolved temporal semantics

Created [Experiment Contract v1.1](../contracts/azure_pdm_experiment_contract_v1.1.md) without overwriting v1.0.

Resolved rules:

- Three-hour anchors are actual telemetry observations, phased from each machine's first observed timestamp. No timestamps are invented and no interpolation is used.
- Lookbacks use half-open elapsed-time intervals `[anchor-lookback, anchor)` for 3-hour and 24-hour windows.
- A window is eligible only when every expected hourly timestamp in that interval exists for that machine. No silent fill is used.
- Targets use `(anchor, anchor+24h]`.
- Target classes are `none`, `comp1`, `comp2`, `comp3`, `comp4`, and `multi_failure`.
- Distinct components in one target interval produce `multi_failure`; repeated same-component failures do not.
- Duplicate failure keys fail closed.
- Nominal split cutpoints are `2015-08-08 06:00:00` and `2015-10-20 06:00:00`.
- Accepted anchors use a two-sided 24-hour purge/embargo:
  - train: `anchor < 2015-08-07 06:00:00`;
  - validation: `2015-08-09 06:00:00 <= anchor < 2015-10-19 06:00:00`;
  - test: `anchor >= 2015-10-21 06:00:00`.
- Anchors whose 24-hour horizon exceeds telemetry coverage are excluded.
- Past lookback context is causal and may use observations before the anchor; no observation at or after the anchor is allowed.
- Errors and maintenance are excluded from features because availability remains unknown.
- Failures are target-only.

## 2. Per-machine data audit

Source audit results:

- Telemetry rows: 876,100.
- Machines: 100.
- Expected hourly rows per machine: 8,761.
- Telemetry coverage: `2015-01-01T06:00:00` to `2016-01-01T06:00:00`.
- Missing hourly intervals: 0 for all machines.
- Duplicate telemetry keys: 0.
- Invalid telemetry rows: 0.
- Source rows are not globally ordered, so generation sorted timestamps per machine.
- Failure rows: 761.
- Failure duplicate keys: 0.
- Unsupported failure components: 0.
- Invalid failure rows: 0.
- Failure events cover 98 machines; machines 6 and 77 have no recorded failures.

The audit distinguishes missing records from confirmed event absence. In particular, a machine with no failure row is not treated as proven healthy.

## 3. Anchor-generation statistics

The canonical v2 audit recorded:

- Actual observation anchors before filtering: 292,100.
- Candidate anchors after split and complete-horizon filtering, per lookback: 288,100.
- Excluded for first boundary purge/embargo: 1,600 anchor instances.
- Excluded for second boundary purge/embargo: 1,600 anchor instances.
- Excluded for incomplete 24-hour horizon: 800 anchor instances.
- Excluded for incomplete 24-hour lookback: 800 anchor instances.
- Excluded for incomplete 3-hour lookback: 100 anchor instances.

The source telemetry is complete hourly for every machine, so lookback losses are boundary losses rather than observed internal gaps.

## 4. Lookback coverage

Two metadata-only temporal samples are emitted for eligible anchors:

- 3-hour lookback: `[t-3h,t)` with 3 expected hourly observations.
- 24-hour lookback: `[t-24h,t)` with 24 expected hourly observations.

The temporal dataset contains window metadata and observation counts, not model feature values. No interpolation, forward-fill, backward-fill, or event feature branch was generated.

## 5. Target-label distribution

Canonical v2 output counts across both lookback variants:

| Class | Rows |
|---|---:|
| `none` | 564,111 |
| `comp1` | 2,638 |
| `comp2` | 3,596 |
| `comp3` | 1,837 |
| `comp4` | 2,450 |
| `multi_failure` | 668 |
| Total | 575,300 |

All six classes are present in the generated train, validation, and test partitions. This is a support observation for this generated benchmark dataset, not a performance claim.

## 6. Split sizes and class distribution

Rows include both lookback variants:

| Split | Rows |
|---|---:|
| Train | 347,900 |
| Validation | 113,600 |
| Test | 113,800 |

Per-split class counts:

| Split | none | comp1 | comp2 | comp3 | comp4 | multi_failure |
|---|---:|---:|---:|---:|---:|---:|
| Train | 340,919 | 1,806 | 2,140 | 1,149 | 1,490 | 396 |
| Validation | 111,440 | 368 | 768 | 352 | 544 | 128 |
| Test | 111,752 | 464 | 688 | 336 | 416 | 144 |

The nominal split is 60/20/20 by time coverage. Purge, lookback eligibility, horizon completeness, and the doubled lookback variants mean usable row counts are not exactly 60/20/20.

## 7. Embargo and leakage validation

Canonical dataset validation status: `PASS`.

Automated checks covered:

- unique sample IDs;
- unique `(machineID, anchor, lookback)` keys;
- accepted split assignment;
- target event count/reference consistency;
- target event timestamps strictly greater than anchor and at or before anchor + 24 hours;
- train targets below the first split boundary;
- validation targets below the second split boundary;
- target-only failure provenance;
- no event or maintenance features in the output;
- no preprocessing or model artifacts.

The generated dataset contains no telemetry feature values, so future-feature leakage is structurally avoided at this foundation stage. Future feature generation must independently validate that each feature window contains only observations strictly before its anchor.

## 8. Excluded and rejected samples

| Reason | Count |
|---|---:|
| First boundary purge/embargo | 1,600 |
| Second boundary purge/embargo | 1,600 |
| Incomplete 24-hour horizon | 800 |
| Incomplete 24-hour lookback | 800 |
| Incomplete 3-hour lookback | 100 |
| Invalid failure rows | 0 |
| Unsupported failure components | 0 |
| Duplicate failure keys | 0 |
| Duplicate telemetry keys | 0 |

## 9. Output paths and manifest hashes

Canonical output directory:

`${AZURE_PDM_ROOT}\derived_phase4_v3`

Files:

- `phase4_temporal_dataset.csv` — 575,300 rows; SHA-256 recorded in `phase4_manifest.json`
- `phase4_audit.json`
- `phase4_dataset_validation.json` — status `PASS`
- `phase4_manifest.json`

The manifest records hashes and row counts for all five source CSVs, contract version, configuration, generation timestamp, output hash, output row count, split/class counts, and exclusion counts.

Earlier preliminary outputs exist at `${AZURE_PDM_ROOT}\derived_phase4` and `${AZURE_PDM_ROOT}\derived_phase4_v2`. They were not overwritten or deleted. The v3 directory is canonical because it includes target-event timestamps, direct target horizon/boundary validation, and raw failure counts by component and machine.

## 10. Test results

Focused Phase 4 tests:

```text
7 passed
```

Focused Phase 4 plus existing Azure validator tests:

```text
23 passed
```

Regression suite excluding the pre-existing blocked training-contract test:

```text
149 passed, 1 warning
```

The warning is the existing pytest `PytestReturnNotNoneWarning` from `tests/test_influx_ingestion.py::test_config`.

The requested broader run was separately attempted and was blocked during collection by the pre-existing missing module `training.train_multimodal` imported by `tests/test_training_contract.py`. No files were changed to hide or bypass that failure.

## 11. Architecture boundary

No model was implemented. The future Azure architecture remains:

1. telemetry encoder;
2. optional machine metadata branch;
3. optional availability-approved structured-event branch;
4. fusion layer;
5. six-class classification head.

The existing FD001 fusion module remains isolated. It is an RUL regression model expecting 64-dimensional sensor and 768-dimensional text embeddings, not an Azure timestamped six-class interface. No BERT embeddings or technician-narrative representations were generated.

## 12. Remaining limitations

- Event availability is still unknown; errors and maintenance remain blocked as features.
- Timezone-naive timestamps remain benchmark-only values.
- Machine metadata validity over time is undocumented.
- `none` means no recorded failure in the target interval, not machine health.
- The generated foundation is not a model-ready feature matrix.
- The benchmark is simulated and supports no production performance claim.
- The existing `training.train_multimodal` module referenced by one regression test is absent; that test remains blocked.

## 13. Safety and authorization statement

Raw CSVs, the acquisition manifest, FD001 artifacts, existing IoT production modules, and existing Phase 3.1 reports were not modified. No existing untracked files were deleted or overwritten. No Git state-changing operations occurred.

Labels and temporal metadata were generated under the newly created v1.1 contract. No model training, inference, embeddings, model artifacts, or predictive-performance claims were produced.
