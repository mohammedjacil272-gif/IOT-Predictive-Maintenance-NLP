# Paired data readiness

**Decision: NOT READY FOR GENUINE FUSION.**

The repository contains NASA CMAPSS FD001 sensor data and a five-row technician-log fixture. The fixture is explicitly synthetic. No SCANIA data, approved technician-log export, authoritative machine crosswalk, shared cycle/timestamp semantics, availability policy, target definition, or training approval was found. FD001 engine IDs therefore must not be treated as technician-machine IDs.

The software can validate a future paired package, but genuine training is blocked until an approved paired export supplies source IDs, canonical entity mapping/version, event and availability timestamps, target/horizon, provenance hashes, embedding artifact lineage, cutoff policy, and entity-disjoint split approval.
