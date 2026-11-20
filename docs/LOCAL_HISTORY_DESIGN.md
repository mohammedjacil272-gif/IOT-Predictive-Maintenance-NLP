# Local history design

Successful `sensor_only` and `nlp_only` results are stored in a small local JSON file outside tracked artifacts. Each record includes UTC timestamp, mode, model ID, request/response, provenance and synthetic flag. Fusion results are never stored. The API exposes list and clear operations; malformed history fails safely and clearing removes it. Local history may contain submitted text, so users should clear it according to their retention policy.
