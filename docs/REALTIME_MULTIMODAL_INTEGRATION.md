# Real-time multimodal integration

The current MQTT/Influx contracts validate sensor payloads and preserve
timezone-aware event timestamps. FD001 replay timestamps remain explicitly
synthetic. Genuine EDP fusion inference is not enabled because no validated EDP
processed dataset or fusion checkpoint exists.

The intended flow is MQTT → schema validation → Influx telemetry → eligible
event retrieval → sensor/NLP inference → fusion readiness check → prediction
storage. Missing services, missing modalities, duplicate messages, and invalid
timestamps must produce explicit unavailable/error states rather than fallback
predictions.
