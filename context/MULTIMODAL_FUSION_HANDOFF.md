# Multimodal Fusion Handoff

**State:** Provisional interface and validation work only. No fusion training or
performance evaluation on genuine data has been performed or authorized.

## Implemented safeguards

- Fusion alignment uses canonical candidate key `(unit_id, cycle)` and rejects
  missing/ambiguous namespaces, duplicate equipment-cycle windows, invalid
  cycles, split leakage/mismatch, duplicate log IDs/source IDs, unmatched text,
  malformed or non-finite embeddings, and entity provenance conflicts.
- Sensor cycles remain the sensor `label_cycle` / window-end cycle. Genuine
  equipment mapping and shared cycle semantics remain **PENDING**.
- The baseline embedding writer's non-shuffled inference batches produce an
  explicit artifact-local `embedding_index`; the fusion loader validates
  uniqueness, range, count, and coverage before indexing the sensor array.
  Cross-modal alignment remains exclusively `(unit_id, cycle)`. The checked-in
  metadata lacks an explicit namespace and required source/artifact lineage
  fields, so genuine loading is currently **BLOCKED** until an authoritative
  metadata version is supplied.
- Genuine data requires an explicit approval register and run metadata. The
  point-in-time validator preserves `event_time`, `record_creation_time`,
  `ingestion_time`, and `availability_time` distinctly and requires a selected
  approved timezone, cutoff inclusivity, and late-arrival policy. These values
  are not supplied by the known genuine technician pipeline, so genuine use
  currently fails closed.
- Missing-log zero vectors and sorted-log-ID mean aggregation are explicit
  selectable provisional policies with no builder default. The model requires
  an explicit `use_has_log` configuration; production inclusion/exclusion is
  **PENDING**.
- Future genuine runs require an explicit cutoff policy and operational
  horizon, owner-supplied sensor checkpoint, and passing verification report
  tied to the same commit/interpreter. Metadata records file hashes, manifest
  identity/hash, complete model/training configuration, Python/dependency
  versions, random seeds, preprocessing version, NLP model/tokenizer revisions
  and checksums when available, test results, exact command, and checkpoint
  lineage.
- The baseline verifier no longer calls baseline data preparation. It fails
  read-only when required protected artifacts are absent.

## Fixture boundary

Compatibility and provenance verifiers use the five-record synthetic NLP
fixture in a separate `synthetic_fixture` namespace. Synthetic sensor vectors
remain interface probes only. No FD001-to-technician mapping is fabricated.

> The synthetic fixture validates interface/provenance behavior only. It does
> not establish genuine sensor-technician pairing or real-world predictive
> performance.

## Baseline artifact note

At inspection time, the frozen baseline model directory contained configuration
files but did not contain `best_model.pt` or `last_model.pt`. Neither checkpoint
was regenerated. The baseline verifier now reports this missing-artifact state
and exits without writing to baseline paths. Baseline checkpoint recovery, if
needed, requires direction from the baseline owner and is outside this task.
The repository ignores `*.pt`, so a checkpoint may be supplied outside Git;
the repository does not establish that it has been delivered or who owns it.
The baseline verifier explicitly requires both checkpoints. Until the legitimate
checkpoint is supplied, baseline verification and future fusion checkpoint
lineage remain **BLOCKED**; this is separate from IoT hardware readiness.

## Remaining decisions

See [the decision register](MULTIMODAL_APPROVAL_CHECKLIST.md). Genuine dataset
permissions, authoritative equipment IDs/mapping, shared cycle semantics,
timestamp/timezone/late-arrival policy, cutoff/horizon, uncapped target
confirmation, missing/multiple-log policy, `has_log` feature selection,
equipment split manifest, evaluation protocol, and separate training/final-test
authorization are **PENDING**.
Dependency manifest/lock strategy and IoT hardware acceptance are also
unresolved in their supporting readiness documents.

No genuine training or evaluation should occur until all applicable approvals
and data requirements are satisfied.
