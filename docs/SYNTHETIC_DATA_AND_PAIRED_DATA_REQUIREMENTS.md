# Synthetic-Data Limitations and Paired-Data Requirements

## What the current fixture proves

The current NLP fixture contains five synthetic maintenance-log records, 16 rule-based entities, heuristic severity scores, and 768-dimensional text embeddings. The compatibility verifier uses synthetic 64-dimensional sensor vectors. These artifacts are suitable for checking parsing, provenance, tensor dimensions, alignment code paths, and missing-text behavior.

They do **not** prove:

- that the text came from a technician or an industrial system;
- that any log corresponds to a real sensor window;
- that a text field was available at prediction time;
- that severity labels are validated ground truth;
- that fusion improves prediction quality;
- that the system is production-ready.

The verifier's `rul=0.0` values are placeholders required by the present interface and are never valid labels for a performance experiment.

## Requirements before performance evaluation

An owner-approved dataset must pair sensor telemetry with technician-written text associated with the same equipment and defensible time windows. Record all of the following before training or evaluation:

1. **Source and permissions** — dataset owner, source URL/system, license, privacy restrictions, approved use, and any retention/access conditions.
2. **Equipment identity** — stable `unit_id` mapping, uniqueness, asset coverage, and whether identifiers can safely be used across sources.
3. **Time semantics** — timezone, timestamp precision, cycle definition, sensor-window start/end, log creation/open/close times, and which timestamps are event time versus ingestion time.
4. **Pairing rule** — explicit matching window and rationale; treatment of delayed, edited, duplicated, or post-event work orders; audit trail for every match.
5. **Point-in-time availability** — only include text fields that were genuinely available at the prediction cutoff. A timestamp equality or work-order open timestamp alone is not sufficient evidence.
6. **Missing text** — measure missing-text rate by split/equipment/time, characterize why text is absent, and predefine missing-text behavior without zero-filling labels or inventing notes.
7. **Target provenance** — exact label definition, source, generation method, prediction horizon, cutoff time, censoring/terminal-event policy, and whether it is observed ground truth or a generated/predicted score.
8. **Split policy** — equipment-level train/validation/test isolation, no overlap between equipment groups, and time ordering where needed for the intended deployment scenario.
9. **Leakage audit** — exclude target-derived fields, failure outcomes, terminal/future cycles, future work-order fields, and post-cutoff text. Record automated checks and their results.
10. **Data quality and provenance** — schema validation, duplicate checks, missingness report, lineage from each fused sample to its source rows, and a reproducible dataset version/hash.

## OIL-039 caution

Keep OIL-039 separate from FD001. Do not fabricate an equipment/cycle mapping between their schemas. OIL-039 is described in the current project notes as synthetic. In particular, do not treat `failure_probability_7d` as approved ground truth until its meaning, generation process, provenance, and prediction-time availability are established and the project team explicitly approves it.

## Approval gate

Until the dataset source, pairing semantics, target definition, split plan, and leakage checks are approved, the project may run interface tests only. Fusion training and predictive-performance evaluation remain **BLOCKED / NOT AUTHORIZED**.
