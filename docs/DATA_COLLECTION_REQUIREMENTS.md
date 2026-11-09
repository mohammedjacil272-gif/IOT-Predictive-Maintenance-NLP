# Data collection requirements

To authorize fusion training, provide permissioned sensor windows and technician logs joined through an approved crosswalk. Sensor records need canonical unit ID, window start/end, representation lineage and target landmark. Logs need log ID, source machine ID, original/normalized text, event/created/ingested/available timestamps with timezone, provenance and approval status. Each pair needs cutoff, target definition/version and entity-disjoint split assignment.

Expected model inputs are finite 64-D sensor vectors and finite 768-D transformer vectors. Train/validation/test splits must be entity-disjoint and official FD001 test labels must remain outside training and validation. Use the empty templates in `templates/`; their headers and schema are examples only, not records.
