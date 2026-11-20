# Temporal alignment and leakage policy

Status: implemented as a contract/readiness specification; EDP execution is
blocked until raw workbooks are available.

For each turbine, sensor windows are eligible only when their source timestamps
and the event's information-availability timestamp are no later than the
prediction cutoff. Event occurrence, log creation, and log availability are
separate fields when supplied. A future failure description is a label or
post-cutoff outcome, never an input to an earlier window.

Ambiguous turbine IDs, duplicate `(machine_id, timestamp)` observations,
missing/naive timestamps without an approved timezone, and conflicting event
records are rejected into a report rather than silently joined. Splits must be
equipment-disjoint or use a justified chronological policy; test outcomes are
not used for preprocessing, tuning, or checkpoint selection.
