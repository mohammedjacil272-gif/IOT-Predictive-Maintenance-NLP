# NLP artifact provenance

`artifacts/embeddings/fd001_nlp_fixture` is a synthetic fixture derived from `tests/fixtures/technician_logs.csv`. Its tensor is 5x768 and the accompanying report identifies DistilBERT mean pooling, but transformer model weights and immutable revisions are not present locally. The existing builder can create `artifact_manifest.json` with payload hashes without overwriting artifacts; this is package-consistency evidence only, not proof of real-world origin or authenticity.

No genuine NLP artifact manifest can be constructed from current evidence. The package remains synthetic and must not enter genuine fusion training.
