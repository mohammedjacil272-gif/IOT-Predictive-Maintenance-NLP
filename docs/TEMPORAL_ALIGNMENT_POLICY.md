# Temporal alignment policy

FD001 sensor time is an operating-cycle index, not wall-clock time. The technician fixture has no event or availability timestamps. A future approved policy must define window start/end, prediction cutoff, event time, availability time, timezone, and an allowed alignment interval.

Only information available at or before the cutoff may be used. A log created earlier but ingested/available later than the cutoff is late-arriving and must be excluded, retained in an exclusion audit, or handled by an explicitly approved policy. Missing, ambiguous, naive, or out-of-order timestamps must fail validation. No timestamps were fabricated in this audit.
