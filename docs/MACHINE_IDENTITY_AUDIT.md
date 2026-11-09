# Machine identity audit

FD001 uses dataset-local integer engine IDs. The synthetic technician fixture uses integer `unit_id` values, but no evidence links them to FD001 engines. Azure fixtures use `machineID`, also without an approved project crosswalk. Therefore no pairing is valid and no mapping is inferred.

The blank intake template is `templates/machine_identity_crosswalk_template.csv`. Required fields are source system, source machine ID, canonical unit ID, mapping version, evidence reference, and approval status. Duplicate or ambiguous source assignments must be rejected before use.
