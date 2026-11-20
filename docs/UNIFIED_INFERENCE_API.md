# Unified inference API

`POST /api/infer` dispatches explicit `mode` values `sensor_only`, `nlp_only`, or `multimodal_fusion`. Sensor and NLP modes call their existing validated services and return structured mode/model metadata. Multimodal mode returns `status: not_validated` and `prediction: null` until approved paired data, policy, checkpoint and evaluation evidence exist. Existing dedicated endpoints remain supported.
