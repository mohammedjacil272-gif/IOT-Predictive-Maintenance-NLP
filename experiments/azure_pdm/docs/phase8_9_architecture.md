# Azure PdM Phase 8–9 architecture and data flow

```mermaid
flowchart LR
    A[Raw PdM_telemetry.csv<br/>timezone-naive benchmark observations] --> B[Offline replay reader]
    M[PdM_machines.csv<br/>approved static metadata] --> C[Machine metadata validator]
    B --> D[Schema, finite-value,<br/>duplicate and gap checks]
    C --> E[24-hour contiguous window adapter]
    D --> E
    E --> F[Saved train-fitted preprocessing]
    F --> G[Engineered telemetry aggregates<br/>mean/std/min/max/range/delta/slope/variation]
    G --> H[Saved Phase 7 Extra Trees model]
    H --> I[Validation-frozen class thresholds]
    I --> J[Six-class prediction + probabilities<br/>model version + provenance + warning]
    K[PdM_failures.csv<br/>target-only audit; not runtime input] -. excluded .-> H
    L[PdM_errors.csv / PdM_maint.csv<br/>availability unknown; excluded] -. excluded .-> H
```

MQTT and InfluxDB are future ingestion sources only. No live connection is made by this demonstration, and the adapter remains independent of the existing FD001 contract.
