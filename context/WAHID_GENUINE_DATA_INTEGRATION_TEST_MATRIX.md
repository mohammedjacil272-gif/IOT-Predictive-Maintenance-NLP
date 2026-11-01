# Genuine-Data Integration Test Matrix

**Owner:** Wahid
**Project:** IOT-Predictive-Maintenance-NLP

## Test Matrix

| ID | Test scenario | Expected behavior | Existing coverage to inspect | Dependency | Priority |
|---|---|---|---|---|---|
| GIT-01 | Sensor–NLP alignment | Exact `(unit_id, cycle)` matching | test_fusion_alignment.py | Genuine paired data | Critical |
| GIT-02 | Embedding index | Reject invalid and duplicate indices | test_training_contract.py | Genuine artifact mapping | Critical |
| GIT-03 | Equipment crosswalk | Reject missing or conflicting mappings | test_fusion_alignment.py | Jesse's approved crosswalk | Critical |
| GIT-04 | Cycle mapping | Apply approved cycle mapping | test_fusion_alignment.py | Cycle semantics | Critical |
| GIT-05 | Missing logs | Apply explicit missing-log policy | test_fusion_alignment.py | Jesse's approval | High |
| GIT-06 | Multiple logs | Apply deterministic approved aggregation | test_fusion_alignment.py | Jesse's approval | High |
| GIT-07 | Entity ownership | Reject invalid parent log IDs | test_fusion_alignment.py | Genuine entity records | Critical |
| GIT-08 | Duplicate records | Reject conflicting duplicates | test_nlp_artifact_validation.py | Genuine source IDs | High |
| GIT-09 | Timestamp validation | Validate timezone and timestamp semantics | test_fusion_provenance.py | Timestamp mapping | Critical |
| GIT-10 | Prediction cutoff | Reject or exclude late records according to policy | test_fusion_provenance.py | Approved cutoff | Critical |
| GIT-11 | Split leakage | Reject equipment overlap | test_training_contract.py | Approved split manifest | Critical |
| GIT-12 | Artifact provenance | Verify artifact identity and hashes | test_nlp_artifact_validation.py | Trusted provenance | Critical |
| GIT-13 | Genuine adapter | Reject unsupported or unverified packages | test_nlp_fusion_adapter.py | Genuine adapter implementation | Critical |
| GIT-14 | Fusion integration | Validate before fusion sample construction | test_nlp_fusion_pipeline.py | Genuine integration | Critical |
| GIT-15 | Training authorization | Reject unauthorized training | test_fusion_provenance.py | Jesse's approval | Critical |
| GIT-16 | Baseline protection | Preserve frozen FD001 baseline | Existing baseline tests | Original checkpoints | Critical |

**Note:** Existing coverage references are tests to inspect, not proof that every scenario is covered.

## Execution Plan

### Phase 1: Synthetic validation
- Test alignment and dimensions.
- Test provenance and entity ownership.
- Test split mechanics.
- Record actual results.
- Do not claim genuine-data validation.

### Phase 2: Genuine-data acceptance
- Obtain permissioned datasets.
- Verify equipment crosswalk and cycle mapping.
- Validate timestamps and provenance.
- Obtain owner approvals.

### Phase 3: Pre-training validation
- Test cutoff and late-arrival behavior.
- Verify split isolation.
- Verify missing and multiple-log policies.
- Verify baseline checkpoint lineage.

### Phase 4: Authorized experiments
- Obtain separate fusion training authorization.
- Obtain separate held-out evaluation authorization.

## Outstanding Dependencies

1. Genuine paired sensor and technician-log datasets.
2. Authoritative equipment-ID crosswalk.
3. Approved cycle mapping.
4. Timestamp semantics and prediction cutoff.
5. Missing-log and multiple-log policies.
6. Approved split and RUL horizon.
7. Genuine NLP artifacts and trusted provenance.
8. Recovery of original FD001 checkpoints.
9. Explicit training and evaluation approvals.

## Restrictions

- Synthetic fixtures are not genuine-data evidence.
- Do not retrain or modify the frozen FD001 baseline.
- Do not invent mappings or production policies.
- Do not train or evaluate genuine data without authorization.

**Current status: Genuine-data integration remains BLOCKED.**
