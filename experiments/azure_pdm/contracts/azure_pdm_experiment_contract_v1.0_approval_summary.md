# Azure PdM Experiment Contract v1.0 — Approval Summary

Status: Finalized for review

## Approved decisions

| Area | Final decision |
|---|---|
| Telemetry timestamp policy | `benchmark`: telemetry timestamps may be treated as observation times for this simulated benchmark only. |
| Timezone | Preserve timezone-naive timestamps; do not assume UTC or IST. |
| Prediction horizon | 24 hours using `(t, t+24h]`. |
| Multiple failures | Use `multi_failure` for two or more distinct components in the interval. |
| Target classes | `none`, `comp1`, `comp2`, `comp3`, `comp4`, `multi_failure`. |
| Event availability | Conservative: errors and maintenance remain excluded without independent availability evidence. |
| NLP modality | Structured events only; no genuine technician narratives are present. |
| Failure records | Target construction only. |
| Target construction | Machine-specific explicit elapsed-time intervals; no global forward-fill/backward-fill. |
| Chronological split | Global 60/20/20 by verified time coverage. |
| Embargo | 24 hours for the current 24-hour maximum lookback and 24-hour target horizon. |
| Preprocessing | Fit only on training data. |

## Verified date boundaries

Coverage verified from the Phase 3.1 report:

- Start: `2015-01-01T06:00:00`
- End: `2016-01-01T06:00:00`
- Duration: 365 days
- 60% cutpoint: `2015-08-08T06:00:00`
- 80% cutpoint: `2015-10-20T06:00:00`

Embargo windows:

- `2015-08-08T06:00:00`–`2015-08-09T06:00:00`
- `2015-10-20T06:00:00`–`2015-10-21T06:00:00`

The cutpoints are nominal 60%/80% coverage boundaries. Embargo rows are excluded from partition use, so usable row counts will not be exactly 60%/20%/20% after embargo.

## Remaining limitations and gates

- Error and maintenance availability remains unresolved and blocks those modalities as prediction features.
- Telemetry’s benchmark observation-time treatment is not evidence of real-world ingestion or availability.
- No timezone semantics are available.
- Static validity of machine model and age over time is undocumented.
- Exact implementation of target construction still requires audit checks for duplicate and ambiguous failure records.
- The 3-hour prediction cadence and 3-hour/24-hour lookback candidates require implementation-stage approval.

## Explicit non-authorization

This contract approval does not authorize data processing, label generation, feature generation, temporal alignment, embeddings, model creation, training, inference, or predictive-performance claims.
