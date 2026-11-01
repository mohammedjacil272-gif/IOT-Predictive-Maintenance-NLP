# Phase 3.6 — Azure PdM Implementation Readiness Review

Status: Read-only audit

## 1. Executive verdict

**NOT READY for full Azure PdM implementation or training.**

The contract is directionally consistent with the approved Phase 3.5 decisions, and the acquired files are reproducible as a simulated benchmark. However, implementation must not begin until the temporal split/embargo semantics and target-audit interfaces are made precise. Error and maintenance modalities are blocked because independent event availability metadata is absent. The existing fusion model is an FD001 RUL regression model and is not a safe Azure PdM implementation starting point.

Safe preparatory work is limited to contract/schema test design and review. This audit generated no labels, features, alignments, embeddings, models, training runs, or inference outputs.

## 2. Audit scope and evidence

Reviewed:

- `experiments/azure_pdm/contracts/azure_pdm_experiment_contract_v1.0.md`
- `experiments/azure_pdm/contracts/azure_pdm_experiment_contract_v1.0_approval_summary.md`
- `experiments/azure_pdm/reports/phase3_1_validation_02.json`
- `${AZURE_PDM_ROOT}\acquisition_manifest_02.json`
- Phase 3.2 audit of Microsoft’s pinned notebook at commit `e836f4e866028db7fd7d1a66b88f79664a1f4eac`
- Existing Azure validation modules under `experiments/azure_pdm`
- Existing fusion, NLP, training, and provenance modules/tests

Pinned Microsoft notebook:

`SQLServerAndAzureMachineLearning/ML Services for SQL Server/notebooks/Predictive Maintenance in Python Notebook.ipynb`

## 3. Contract consistency findings

### Consistent and approved

The contract correctly preserves:

- 24-hour prediction horizon.
- Target interval `(t, t+24h]`.
- Six target classes: `none`, `comp1`, `comp2`, `comp3`, `comp4`, `multi_failure`.
- Benchmark-only interpretation of telemetry timestamps.
- Timezone-naive source values with no UTC or IST assumption.
- Conservative exclusion of errors and maintenance without independent availability evidence.
- Failure records as target-only inputs.
- Machine-specific explicit elapsed-time target construction.
- Prohibition on Microsoft’s global forward-fill/backward-fill label behavior.
- Global chronological 60/20/20 split intent.
- 24-hour embargo intent.
- Training-only fitting of learned preprocessing.
- Distinction between missing records and proof that no event occurred.
- Macro/per-class/component/multi-failure metrics and non-trivial baselines.

### Ambiguities or missing definitions

1. **Split interval closure is unspecified.** The contract gives nominal cutpoints and embargo windows but does not define whether boundaries are open or closed for anchors.

2. **The embargo orientation is insufficiently precise.** Excluding the 24 hours after each cutpoint does not, by itself, prevent training anchors immediately before a cutpoint from having target intervals that extend into the next partition. A purge rule for anchors before each boundary is required.

3. **The relationship between nominal 60/20/20 coverage and embargoed usable samples is not fully defined.** The contract states that embargo rows are excluded but does not specify whether the 60/20/20 proportions refer to raw time coverage, eligible anchors, or post-embargo samples.

4. **Lookback endpoint semantics are incomplete.** The contract requires elapsed-time windows but does not specify whether a lookback is `[t-24h,t]`, `[t-24h,t)`, or another convention.

5. **Three-hour cadence alignment is not finalized.** It is unclear whether anchors are selected from source timestamps, generated from fixed elapsed bins, or defined by resampling labels. No observations may be invented.

6. **Class-support requirements are not defined.** The report cannot establish before target construction whether every split contains all six classes, especially `multi_failure`.

7. **Telemetry completeness is reported globally, not as a formal per-machine boundary audit.** The row count is consistent with hourly coverage for 100 machines, but implementation should verify per-machine counts and gaps before creating anchors.

These are contract-definition blockers, not findings to be silently resolved by code.

## 4. Data and provenance readiness

### Verified facts

Phase 3.1 confirms:

- Exact five-table schemas passed.
- Telemetry: 876,100 rows.
- Errors: 3,919 rows.
- Maintenance: 3,286 rows.
- Failures: 761 rows.
- Machines: 100 rows.
- No duplicate keys, blanks, invalid numeric values, or malformed timestamps were detected.
- Telemetry covers `2015-01-01T06:00:00` through `2016-01-01T06:00:00`.
- Telemetry and errors/maintenance cover all 100 machine IDs.
- Failures cover 98 machines; IDs 6 and 77 have no failure records.
- All recorded event/telemetry machine IDs are present in `PdM_machines.csv`.
- All acquired file sizes, SHA-256 hashes, and Git blob SHA-1 values match the acquisition manifest.
- The pinned source repository and commit match the manifest.
- The dataset is explicitly classified as simulated benchmark data.

### Limitations

- Timestamps are timezone-naive.
- Event tables contain no `available_at` or equivalent availability metadata.
- The source does not document whether event timestamps mean occurrence, detection, recording, or ingestion time.
- Machine model and age are treated as benchmark-static only; validity over time is not documented.
- Failure records are recorded failures, not evidence of complete failure history.
- Raw rows are not globally chronological; processing must sort within machine and timestamp.

### Readiness conclusion

Provenance is sufficient for a reproducible benchmark audit. It is not sufficient to claim operational event availability or production readiness.

## 5. Temporal and target readiness

### Three-hour cadence

An hourly source can support a three-hour cadence without inventing observations by selecting existing timestamps or defining fixed three-hour bins and emitting anchors only where the required source coverage exists. The implementation must not interpolate or synthesize anchor observations merely to complete a cadence.

The exact anchor rule remains unresolved:

- source timestamp selection versus fixed-bin resampling;
- bin closure and label placement;
- treatment of the first and last partial bins;
- minimum hourly coverage required inside a three-hour bin.

### Lookback windows

The contract’s 3-hour and 24-hour candidates are reasonable candidates, but both must be explicit elapsed-time intervals. A future implementation should specify a convention such as `[t-24h,t)` or `[t-24h,t]` and test it directly. A 24-row rolling window must not be used as a substitute for a 24-hour duration unless hourly completeness has been verified for that machine and interval.

### Global split feasibility

All telemetry machines are represented in the validated table and share the reported overall coverage, so global time cutpoints are structurally possible. However, per-machine gap/completeness checks at the cutpoints are still required before anchors are eligible. The failure table omits machines 6 and 77, so those machines cannot receive recorded-failure targets unless their target interval contains no recorded failure, which must not be interpreted as health.

### Embargo and target leakage

The approved 24-hour embargo is consistent with the current maximum proposed 24-hour lookback and 24-hour horizon, but the contract needs a purge definition.

For a boundary `B`, an implementation must prevent:

- training anchors whose target interval extends into validation/test time;
- validation anchors whose target interval overlaps test time;
- feature windows that cross a partition in a way prohibited by the final leakage policy;
- target intervals crossing machine boundaries.

The current contract only states that the 24 hours after `B` are excluded. That is not enough to prove that anchors before `B` cannot have overlapping future target intervals. The anchor purge side of the boundary must be approved before implementation.

### Simultaneous and overlapping failures

The six-class policy is deterministic in principle:

- one distinct component in `(t,t+24h]` → that component;
- two or more distinct components → `multi_failure`;
- same-component duplicate key → audit failure, not silent deduplication.

The implementation still requires an audit interface that reports duplicate keys, ambiguous component values, missing machine/timestamp fields, simultaneous distinct components, and overlapping target windows. No such Azure target implementation currently exists.

### Evaluation support

It is not yet possible to establish whether every split contains all six target classes without generating or auditing targets. `multi_failure` may be rare or absent. Evaluation must support absent-class warnings and must not fabricate scores for unsupported classes.

## 6. Modality and architecture readiness

### Existing model

`models/multimodal_fusion.py` defines `MultimodalFusionModel`, whose contract is:

- 64-dimensional sensor embeddings;
- 768-dimensional text embeddings;
- optional `has_log` input;
- scalar RUL regression output.

It is not reusable as the Azure six-class head without a new task-specific adapter/model contract. It must remain isolated from the frozen FD001 baseline and must not inherit FD001 preprocessing assumptions.

### Existing dataset/alignment stack

`training/fusion_dataset.py` assumes:

- `(unit_id, cycle)` keys;
- sensor embeddings;
- RUL targets;
- technician-log records or text embeddings;
- FD001-style split/provenance fields.

Those assumptions do not match Azure `(machineID, datetime)` anchors, six-class targets, or blocked event availability. It is not a safe Azure adapter without a separately approved interface.

`training/fusion_metrics.py` is regression-oriented and is not sufficient for the approved multiclass metrics.

`training/fusion_provenance.py` contains reusable ideas for explicit provenance and point-in-time validation, but its existing timestamp and genuine-data contracts must not be applied blindly to the benchmark’s timezone-naive timestamps.

### Recommended future architecture

Only after the contract blockers are resolved, a suitable future design would be:

1. Telemetry encoder for machine-specific causal windows.
2. Optional static metadata branch for model and age.
3. Optional structured-event branch using availability-eligible error counts/categories, not technician narrative embeddings.
4. Explicit fusion layer with modality-presence indicators.
5. Six-class classification head.

The structured-event branch could use categorical/count representations and a small MLP or temporal aggregation. BERT is not justified: the available records are structured error events, not natural-language technician narratives, and their availability is currently blocked.

## 7. Leakage and evaluation risks

High-risk unresolved items:

- global row-based fill crossing machine boundaries;
- target intervals crossing split boundaries;
- ambiguous embargo orientation;
- fitting categorical encoders or imputers outside training data;
- treating absence of a failure row as machine health;
- using event occurrence timestamps as availability timestamps;
- assuming all machines have identical valid hourly coverage at boundaries;
- reporting metrics for classes absent from a split;
- reusing FD001 RUL preprocessing or regression metrics;
- representing derived structured text as genuine technician narrative.

The Microsoft notebook’s historical approach used global backward-fill for labels and global forward-fill for maintenance-derived features. The current contract correctly prohibits copying that behavior.

## 8. Unresolved decisions and blockers

Implementation blockers:

1. Approve exact half-open/closed anchor and split interval semantics.
2. Approve a purge/embargo rule that prevents target overlap on both sides of every boundary.
3. Approve the precise three-hour anchor/cadence construction without inventing observations.
4. Approve lookback endpoint conventions and minimum coverage requirements.
5. Perform per-machine hourly completeness and boundary-gap audit.
6. Define the target audit output for duplicates, ambiguous failures, simultaneous failures, and overlapping intervals.
7. Establish how unsupported target classes will be reported in validation/test metrics.
8. Define the Azure-specific dataset/provenance interface separately from FD001 `(unit_id, cycle, RUL)` interfaces.
9. Keep errors and maintenance blocked unless independent availability evidence is supplied.

Non-blocking benchmark limitations:

- timezone semantics are unavailable but intentionally preserved under the approved benchmark policy;
- machine metadata temporal validity is undocumented;
- failure absence is not equivalent to health;
- simulated benchmark results cannot support production claims.

## 9. Phased implementation roadmap

The following is a proposal only. No phase is authorized by this report.

### Phase A — Contract and schema test coverage

Scope: encode the approved policy as tests without touching raw files.

Expected files/modules: new Azure-specific tests under `experiments/azure_pdm/tests` or `tests`; no production-module edits.

Required tests:

- six target-class vocabulary;
- `(t,t+24h]` inclusivity;
- timezone-naive preservation;
- event availability rejection;
- duplicate failure rejection;
- machine-specific grouping;
- explicit split and embargo semantics.

Dependencies: final decisions on interval closure and purge orientation.

Acceptance criteria: all policy edge cases fail closed and no FD001 test contract changes.

Risks: encoding an unresolved policy prematurely.

### Phase B — Machine-specific temporal anchor design

Scope: define, but do not yet materialize, cadence and lookback interfaces.

Expected files/modules: Azure temporal-policy module and read-only unit tests.

Required tests:

- hourly-to-3-hour anchor selection;
- no invented observations;
- elapsed-time lookback boundaries;
- per-machine sorting and gap handling;
- coverage at split boundaries.

Dependencies: cadence, bin closure, lookback endpoint, and minimum-coverage approvals.

Acceptance criteria: deterministic anchor specification with explicit boundary behavior.

Risks: hidden regularity assumptions from row counts.

### Phase C — Target construction and audit

Scope: implement only after approval, using failure records target-only.

Expected files/modules: Azure target-policy module, target audit report/tests.

Required tests:

- component and `multi_failure` mapping;
- simultaneous failures;
- overlapping intervals;
- duplicate and ambiguous failures;
- missing failure rows;
- split-boundary purge.

Dependencies: Phase B anchors and approved purge semantics.

Acceptance criteria: machine-specific explicit interval logic; no global fill; complete audit counts.

Risks: rare/absent `multi_failure` or component classes in validation/test.

### Phase D — Leakage-safe feature generation

Scope: create causal telemetry/static features only; event features remain disabled unless availability is approved.

Expected files/modules: Azure feature-policy module and provenance metadata.

Required tests:

- no future telemetry in lookbacks;
- no failure columns in features;
- training-only imputation/scaling/encoding;
- no cross-machine fill;
- stable feature schema.

Dependencies: finalized temporal policy and training split.

Acceptance criteria: feature lineage identifies source rows and cutoff rules.

Risks: accidental reuse of FD001 preprocessing or Microsoft global fill.

### Phase E — Dataset and provenance validation

Scope: validate the constructed dataset package without training.

Expected files/modules: Azure dataset manifest/validator and report-only validation tests.

Required tests:

- source hash and pinned-commit match;
- split membership;
- machine coverage;
- class-support diagnostics;
- missingness and duplicate audit;
- blocked event-modality enforcement.

Dependencies: target and feature contracts.

Acceptance criteria: reproducible package with explicit limitations and no unapproved modalities.

Risks: treating benchmark assumptions as source facts.

### Phase F — Model architecture and training preparation

Scope: design a separate Azure six-class architecture; do not modify FD001 modules.

Expected files/modules: Azure-specific model, adapter, multiclass metrics, and configuration modules.

Required tests:

- input/output shape contract;
- six-class head;
- modality-presence handling;
- provenance and split compatibility;
- no RUL regression dependency.

Dependencies: validated dataset package and architecture approval.

Acceptance criteria: architecture is task-specific and isolated from FD001.

Risks: accidental BERT/text integration or reuse of RUL interfaces.

### Phase G — Training and evaluation

Scope: training and final evaluation only after separate written authorization.

Expected files/modules: Azure training/evaluation entry points and new artifact paths.

Required tests:

- train/validation/test isolation;
- metric handling for unsupported classes;
- baseline comparisons;
- reproducibility and artifact provenance;
- no test-set fitting.

Dependencies: all prior acceptance gates and explicit training authorization.

Acceptance criteria: reported metrics are limited to the simulated benchmark and approved scope.

Risks: class imbalance, absent classes, and overinterpretation as production performance.

## 10. Authorization gates

Before any implementation:

1. Resolve split closure and two-sided purge/embargo semantics.
2. Resolve three-hour anchor and lookback endpoint definitions.
3. Approve per-machine completeness/gap criteria.
4. Approve the target-audit interface and unsupported-class reporting.
5. Confirm event modalities remain blocked without independent availability evidence.
6. Approve a separate Azure data/model interface isolated from FD001.

Before training:

1. Approve the generated dataset/provenance validation report.
2. Confirm all learned preprocessing is training-only.
3. Confirm target construction passed duplicate, ambiguity, overlap, and split-boundary audits.
4. Approve the six-class architecture and metrics.
5. Provide separate written authorization for training and evaluation.

Approval of the contract and this readiness report does not authorize implementation, training, inference, or performance claims.
