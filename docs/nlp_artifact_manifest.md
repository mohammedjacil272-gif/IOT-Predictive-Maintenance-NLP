# NLP Artifact Manifest

**Owner:** Rafaz
**Status:** Synthetic fixture validation only

## 1. Input

| Artifact | Path | Purpose |
|---|---|---|
| Technician log fixture | tests/fixtures/technician_logs.csv | Five synthetic maintenance logs |

## 2. Processed data artifacts

| Artifact | Path | Format |
|---|---|---|
| Processed logs | artifacts/data/fd001_nlp_fixture/processed_logs.csv | CSV |
| Extracted entities | artifacts/data/fd001_nlp_fixture/entities.jsonl | JSONL |
| Severity results | artifacts/data/fd001_nlp_fixture/severity.csv | CSV |
| Verification report | artifacts/data/fd001_nlp_fixture/verification_report.json | JSON |

## 3. Embedding artifacts

| Artifact | Path | Format |
|---|---|---|
| Text embeddings | artifacts/embeddings/fd001_nlp_fixture/embeddings.pt | PyTorch tensor |
| Embedding metadata | artifacts/embeddings/fd001_nlp_fixture/embedding_metadata.csv | CSV |

## 4. Embedding configuration

- Model: distilbert-base-uncased
- Maximum sequence length: 128
- Embedding dimension: 768
- Pooling: attention-mask mean pooling
- Embedding input: normalized_text

## 5. Current fixture results

Previously reported pipeline output:

- Logs processed: 5
- Entities extracted: 16
- Embedding shape: (5, 768)
- Embeddings finite: True

These results apply only to the synthetic fixture.

## 6. Provenance and limitations

- Dataset classification: synthetic_fixture.
- No genuine paired technician-sensor dataset is established.
- Embedding metadata contains row_index and log_id for row mapping.
- Exact immutable model and tokenizer revisions are not recorded.
- Comprehensive embedding-to-metadata alignment verification remains outstanding.
- No predictive improvement or genuine-data performance claim is supported.

## 7. Baseline protection

All NLP outputs remain in fixture-specific directories.

Do not overwrite frozen sensor-only FD001 baseline artifacts, checkpoints, preprocessing or evaluation results.

## 8. Integration status

**Status: SYNTHETIC-ONLY**

Genuine-data integration remains blocked pending authoritative source data, permissions, equipment mapping, cycle definitions and point-in-time availability timestamps.
