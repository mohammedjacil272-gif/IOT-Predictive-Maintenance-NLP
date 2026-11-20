# Multimodal data contract

The EDP adapter uses an isolated namespace: `edp_wind_portugal`. Raw files live
under `Data/external_datasets/edp/raw/`; processed outputs and reports are
separate. FD001 paths are never used for EDP data.

Canonical sensor observations contain `dataset_id`, `machine_id`, `timestamp`,
`sensor_name`, `sensor_value`, `unit`, `source_record_id`, and `provenance`.
Canonical event records contain `dataset_id`, `machine_id`,
`event_timestamp`, `event_type`, `description`, optional `severity`,
`source_record_id`, and `provenance`.

The source timestamp and identifier are retained alongside normalized values.
Missing optional fields remain missing. No timestamps, IDs, labels, or text are
created to repair a source. Every ingestion run records source URL/version,
license, file hashes, and validation status.
