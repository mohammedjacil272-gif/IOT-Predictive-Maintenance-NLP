# Azure PdM Experiment Contract v1.1

Status: Phase 4 implementation contract

This version refines v1.0 only where Phase 3.6 identified temporal ambiguity. It does not authorize model training, inference, or predictive-performance claims.

## 1. Unchanged approved decisions

- Dataset classification: simulated benchmark.
- Telemetry timestamps: benchmark observation times only; this is not real ingestion or availability evidence.
- Timezone-naive timestamps are preserved; no UTC or IST assumption is made.
- Prediction horizon: 24 hours.
- Target interval: `(t, t+24h]`.
- Target classes: `none`, `comp1`, `comp2`, `comp3`, `comp4`, `multi_failure`.
- Errors and maintenance are excluded from features because availability is unknown.
- Failure records are target-only.
- No genuine technician narratives or synthetic narrative replacement are used.
- No global forward-fill or backward-fill is used for targets or temporal features.
- Preprocessing is fit on training data only.

## 2. Prediction anchors

Anchors are generated independently per machine from timestamps that actually occur in that machine's telemetry table.

- Cadence: three hours.
- The first observed timestamp for each machine establishes its cadence phase.
- An anchor is eligible only if its timestamp is present in the source telemetry and its elapsed offset from that machine's first observed timestamp is an exact multiple of three hours.
- No timestamp is invented and no interpolation is performed to create an anchor.
- A missing cadence timestamp produces no anchor at that timestamp.
- Source rows are sorted by `(machineID, datetime)` before temporal processing.
- A source timestamp may be used as an anchor even if the preceding lookback is incomplete; that anchor is then rejected separately for each lookback window.

## 3. Lookback windows

The supported candidate windows are three hours and 24 hours. Both use the half-open interval:

```text
[anchor - lookback_hours, anchor)
```

The anchor observation itself is excluded from its lookback. Windows are measured by elapsed naive benchmark time, not by a fixed row count.

Minimum coverage requirement: every expected hourly timestamp in the half-open window must exist for that machine in telemetry. Missing measurements or missing hourly timestamps do not receive silent fill or interpolation. A sample is emitted separately for each eligible lookback; a 3-hour sample may exist even when the 24-hour sample is rejected.

The current foundation stores telemetry window metadata and observation counts. It does not generate model features.

## 4. Target labels

For a machine-specific anchor `t`, failure events are selected only when:

```text
t < failure_time <= t + 24 hours
```

Mapping:

- zero recorded components: `none`;
- one distinct component: that component class;
- two or more distinct components: `multi_failure`.

Same-component failures do not create `multi_failure` merely because they occur more than once. Simultaneous distinct components do create `multi_failure`.

Duplicate `(machineID, failure_time, component)` records are audit failures. They are not silently deduplicated or discarded. Invalid machine IDs, timestamps, and unsupported/missing components fail the target audit and block dataset generation. A missing failure row means no recorded failure evidence, not proof of machine health.

Target-event references and target provenance are separate from any future model feature columns.

## 5. Chronological split, purge, and embargo

Verified telemetry coverage:

- start: `2015-01-01 06:00:00`;
- end: `2016-01-01 06:00:00`;
- first nominal cutpoint: `2015-08-08 06:00:00`;
- second nominal cutpoint: `2015-10-20 06:00:00`.

The nominal cutpoints represent 60% and 80% of the observed coverage duration. Operational anchor assignment is explicit and conservative:

- train: `anchor < B1 - 24h`;
- validation: `B1 + 24h <= anchor < B2 - 24h`;
- test: `anchor >= B2 + 24h`.

Anchors in `[B1-24h, B1+24h)` and `[B2-24h, B2+24h)` are excluded as boundary purge/embargo anchors. This is a two-sided 24-hour exclusion around each nominal boundary. The nominal 60/20/20 proportions refer to raw time coverage; usable anchor counts after purge, lookback rejection, and horizon truncation are reported rather than forced.

An anchor is additionally rejected when `anchor + 24h` exceeds the verified telemetry coverage end, because its complete target horizon is unavailable.

This rule prevents accepted training targets from reaching the validation period and accepted validation targets from reaching the test period. Overlapping target intervals within the same split are allowed; target intervals across accepted splits are prevented by the two-sided purge.

Lookback context is causal: only observations strictly before the anchor may be read. A past observation may be used as context for a later partition because it is historically available under the benchmark timestamp policy; no observation at or after the anchor may enter its lookback. Partition assignment is by anchor, not by copying rows between machines.

## 6. Timestamp and modality policy

The benchmark timestamp policy permits telemetry timestamps to act as observation times for this simulated benchmark only. It does not establish operational availability. Error and maintenance event timestamps are never used as availability timestamps and their modalities remain excluded from features. Failures remain target-only.

## 7. Missing data and audit outputs

- Missing telemetry hourly timestamps reject the affected lookback window.
- Missing telemetry values, duplicates, or invalid numeric values fail the source audit.
- Missing event rows are not interpreted as confirmed event absence outside the recorded data.
- Missing failure events produce `none` only in the limited sense of no recorded failure in the target interval.
- Duplicate or ambiguous failures block target generation.
- Exclusion counts, per-machine coverage, split counts, class counts, and output hashes are mandatory manifest fields.

## 8. Implementation boundary

The temporal foundation may generate anchors, target labels, target provenance, split assignments, and window metadata. It must not generate model features, embeddings, models, training outputs, or inference outputs. A future Azure model must be separate from the FD001 RUL/regression modules.
