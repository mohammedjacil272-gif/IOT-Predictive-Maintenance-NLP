# Text embedding pipeline

`nlp.embeddings.TextEmbedder` is an offline-only, lazy DistilBERT service. It accepts normalized non-empty strings, uses a configurable local model path, CPU fallback, deterministic evaluation mode, attention-mask mean pooling, batching, and validates finite 768-dimensional output. Missing `transformers` or missing local weights raises a clear `RuntimeError`; no network download is attempted. Rule-based NER and severity remain separate outputs.

Actual transformer inference is blocked in this checkout unless the optional dependency and approved local model/tokenizer files are supplied.
