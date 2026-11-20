# Paired sensor/log data specification

Status: specification ready; authorized paired dataset blocked.

Required pair key: `(canonical_unit_id, cycle)` after an owner-approved machine
crosswalk and cycle semantics. Every record must retain source IDs, dataset
version, artifact hashes, preprocessing version, event/creation/ingestion/
availability timestamps, and split assignment.

Missing logs retain the sensor row with an explicit zero vector and
`has_log=False` only when the approved policy selects that behavior. Multiple
logs are sorted by stable `log_id` and aggregated only under the approved
policy. Equipment-level split manifests must be disjoint. Logs available after
the prediction cutoff are excluded or rejected according to the approved
policy, never silently used.

No C-MAPSS engine and synthetic technician-log fixture may be described as a
paired real observation.
