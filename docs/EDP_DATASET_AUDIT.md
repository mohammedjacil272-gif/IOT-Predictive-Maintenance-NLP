# EDP wind-farm dataset audit

Audit date: 2026-11-09. Raw workbooks were **not obtained in this run**, so no
claim is made about actual workbook sheets, row counts, text contents, or
failure joins. The audit is therefore **BLOCKED_PENDING_RAW_FILES**.

## Verified source metadata

The requested Mendeley record identifies DOI `10.17632/zjxjnjp3xs.1`, version
1, source EDP, CC BY 4.0, four turbines (T01, T06, T07, T11), 2016–2017
coverage, and Excel files. The record describes SCADA, meteorological mast,
event, and failure tables. These are publisher metadata, not local inspection
results. A later version 2 record (`10.17632/zjxjnjp3xs.2`) is now visible with
a changed file layout/size; the version must be selected explicitly before
acquisition.

The Mendeley public metadata page is reachable, but the documented API/download
routes returned an authorization error or unavailable file listing from this
environment. The related EDP data portal is reachable and exposes wind SCADA and
failure resources, but its download pages did not provide a reliable automated
archive link during this run. No access restriction was bypassed.

## Required local audit

The publisher metadata for version 1 describes eight Excel files (219 MB),
10-minute UTC timestamps, SCADA, event logs with free-text remarks, and 28
short-description failure records. This does not prove that the remarks are
technician-authored logs or that they pair with FD001 engines.

The repository-level inventory can be reproduced without network access:

```powershell
python scripts/validate_repository_datasets.py --report reports/dataset_validation/repository_inventory.json
```

After lawful acquisition, install a compatible Excel engine if needed and run:

```powershell
python scripts/audit_edp_dataset.py Data/external_datasets/edp/raw --report reports/dataset_validation/edp_audit.json
```

The read-only auditor records workbook hashes, sheet names, rows, columns,
dtypes, missing values, samples, sheet classification, and explicit policies
that timestamps, IDs, and labels are never invented. It does not train models.

## Suitability decision

The EDP release remains a **candidate for genuine sensor-event-text fusion**,
not an accepted training dataset. Suitability depends on confirming actual
remarks, stable turbine IDs, event timestamp semantics, duplicate handling, and
whether the 28 failure records can be joined without future-information leak.
The event reset field reportedly has many structural blanks; one-sided alarms
must not be treated as resolved maintenance outcomes without evidence.

Source v1: <https://data.mendeley.com/datasets/zjxjnjp3xs/1>
Current source record v2: <https://data.mendeley.com/datasets/zjxjnjp3xs/2>
Related publisher portal: <https://edp.com/en/innovation/data>
