# NLP Missing Source Fields and Dependency Report

**Owner:** Rafaz
**Status:** BLOCKED — genuine-data readiness
**Last updated:** 31 October 2026

## 1. Current position

The NLP pipeline has been exercised using five synthetic technician-log records.

Genuine paired technician and sensor data has not been established.

## 2. Missing source information

| Required information | Current status | Required from |
|---|---|---|
| Genuine paired technician logs | BLOCKED | Dataset/data owner |
| Data usage permission | PENDING | Data owner/project lead |
| Stable source log identifiers | UNCONFIRMED | Technician-log owner |
| Original equipment identifiers | UNCONFIRMED | Sensor/maintenance owners |
| Authoritative equipment crosswalk | BLOCKED | Data owners |
| Eligible equipment manifest | BLOCKED | Data owner/project lead |
| Cycle definition | BLOCKED | Sensor data owner |
| Technician-log to cycle relationship | BLOCKED | Maintenance data owner |
| Event timestamp | UNCONFIRMED | Technician-log owner |
| Creation timestamp | UNCONFIRMED | Source system owner |
| Ingestion timestamp | UNCONFIRMED | Source system owner |
| available_at timestamp | BLOCKED | Data/platform owner |
| Timestamp timezone | UNCONFIRMED | Source system owner |
| Approved train/validation/test split | PENDING APPROVAL | ML/project lead |
| Prediction target, cutoff and horizon | PENDING APPROVAL | ML/domain approver |
| Missing-log policy | PENDING APPROVAL | Project/ML lead |
| Multiple-log aggregation policy | PENDING APPROVAL | Project/ML lead |
| has_log model-input decision | PENDING APPROVAL | Project/ML lead |
| Original baseline checkpoints | BLOCKED | Baseline artifact owner |
| Fusion training authorization | PENDING APPROVAL | Authorized project approver |

## 3. Existing synthetic-only evidence

- Five synthetic fixture records.
- Rule-based NER implementation.
- Heuristic severity scoring.
- 768-dimensional DistilBERT embeddings.
- Fixture-specific artifact output directories.
- Existing NLP verification script.

These results demonstrate interface development only.

## 4. Actions required

1. Obtain genuine, permissioned paired technician and sensor data.
2. Obtain an authoritative equipment crosswalk and eligible-equipment manifest.
3. Confirm cycle definitions and their relationship to technician logs.
4. Establish trustworthy available_at timestamps and timezone.
5. Obtain decisions on target, prediction cutoff, horizon and evaluation split.
6. Confirm missing-log and multiple-log handling.
7. Obtain original baseline checkpoints and approved provenance.
8. Obtain explicit fusion training and evaluation authorization.

## 5. Restrictions

- Do not fabricate equipment mappings or timestamps.
- Do not label synthetic records as genuine.
- Do not modify the frozen sensor-only baseline.
- Do not recreate missing baseline checkpoints.
- Do not run multimodal fusion training.
- Do not claim predictive improvement.
- Do not perform unauthorized official test evaluation.

## 6. Current readiness decision

**NLP interface development:** SYNTHETIC-ONLY

**Genuine-data integration:** BLOCKED

**Fusion training:** NOT AUTHORIZED

**Predictive-performance evaluation:** NOT AUTHORIZED
