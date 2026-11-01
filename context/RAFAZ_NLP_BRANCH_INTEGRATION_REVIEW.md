# Rafaz NLP Branch Integration Review

**Review date:** 2026-10-31
**Active branch preserved:** `feature/wahid-multimodal-fusion`
**Reported commit:** `9cce00559ffe83731be2a4fb231334a80479b981` (`docs: add NLP artifact manifest and data blockers`)
**Remote branch:** `origin/feature/rafaz-nlp-pipeline`, fetched to update its tracking ref; current tip is `f22e8efd71e7ee3866677a8d124738c928abac55`. No checkout, merge, cherry-pick, or Rafaz file copy was performed.

## Remote refs and commit contents

At inspection, `origin/feature/rafaz-nlp-pipeline` existed locally at `5e8d915` but did not contain object `9cce005`. I fetched only `refs/heads/feature/rafaz-nlp-pipeline` into that remote-tracking ref. The active branch remained `feature/wahid-multimodal-fusion` and the pre-existing untracked Wahid report was preserved.

Commit `9cce005` changes only:

- `docs/nlp_artifact_manifest.md` (added)
- `docs/nlp_missing_source_fields.md` (added)

The current remote branch has one later commit, `f22e8ef` (`test: add NLP embedding alignment validation`), changing:

- `docs/nlp_artifact_manifest.md`
- `scripts/verify_nlp.py`
- `tests/test_nlp_alignment.py`

## Files inspected

At commit `9cce005`:

- `docs/nlp_output_schema.md`
- `docs/nlp_provenance_contract.md`
- `docs/nlp_artifact_manifest.md`
- `docs/nlp_missing_source_fields.md`

At the current remote branch tip, also inspected:

- Updated `docs/nlp_artifact_manifest.md`
- `scripts/verify_nlp.py`
- `tests/test_nlp_alignment.py`
- `nlp/embeddings.py`
- `scripts/build_nlp_artifacts.py`
- Complete remote tree for generated artifact availability

## Findings

| Topic | Documented finding | Fusion integration assessment |
|---|---|---|
| Embedding | 768D, `distilbert-base-uncased`, max length 128, attention-mask mean pooling over `normalized_text`. | Dimension and pooling interface are compatible. Exact immutable model and tokenizer revisions/checksums are absent; fusion genuine provenance requires model/revision and artifact lineage. **BLOCKED for genuine use.** |
| Stable ID and row mapping | Unique non-empty `log_id`; `row_index` sequential from zero and aligned to the metadata/embedding tensor. Current remote validator checks tensor/metadata/processed row counts, unique IDs, sequential indexes, and metadata/processed ID order. | Suitable artifact-local lookup, with explicit mapping to `log_id`; never a fusion join key. The verified mapping applies only to the synthetic fixture output, not genuine source records. |
| Source record identity | `source_system`, `source_record_id`, source dataset/version and hashes are listed as unavailable/unconfirmed. | Genuine fusion requires authoritative source-system and source-record IDs plus dataset/version and artifact provenance. **Gap / BLOCKED.** |
| Equipment mapping | Original `source_asset_id` and canonical `unit_id` crosswalk are unavailable; docs require an authoritative mapping. | Fusion requires positive integer `unit_id`, equipment mapping ID, and source provenance. Current builder treats `unit_id` as optional and cannot produce an approved genuine crosswalk. **Gap / BLOCKED.** |
| Cycle | `cycle` is optional in the builder. Contract proposes mapping NLP cycle to sensor `label_cycle`, but says the genuine definition and relationship remain unverified. | Fusion requires positive integer cycle and joins by `(unit_id, cycle)`. Mapping to `label_cycle` is the intended interface but not evidence of genuine semantic equivalence. **Gap / BLOCKED.** |
| Split / namespace | Fixture `source_split` is `fixture`; source split is optional in builder. Dataset report marks `synthetic_fixture`. | Fusion requires per-record `namespace`, `split`, and matched sensor/log split equality. An adapter must map and validate the fixture/source fields. Genuine records require approved source splits and namespace. |
| Timestamps | Docs name `event_timestamp`, `created_at`, `ingested_at`, `available_at`, and timezone; all genuine values are unavailable/unconfirmed. | Fusion requires offset-aware `event_time`, `record_creation_time`, `ingestion_time`, `availability_time`, an approved IANA timezone/policy, and per-window cutoff. Semantic equivalents may exist, but field mapping and source meaning need approval/evidence. No cutoff eligibility can be determined now. **Gap / BLOCKED.** |
| Synthetic/genuine status | Explicitly five synthetic fixture logs; no genuine paired sensor/technician dataset. | Compatible with fixture-only validation. It cannot establish pairing, mapping, timing, model performance, or genuine provenance. **SYNTHETIC-ONLY.** |
| Missing logs | Proposed preserve sensor windows; zero 768D vector; `has_log=False`; report unmatched NLP records. `has_log` model-input decision is pending. | Matches current fusion choices, which require explicit `missing_log_policy` and separately preserve `has_log`. Jesse approval remains required. |
| Multiple logs | Proposed group by `(unit_id, cycle)`, stable sort by `log_id`, mean embeddings. Entity/severity aggregation remains to be confirmed. | Embedding aggregation matches current `mean_sorted_log_id` option. Fusion retains per-log entities/severity/provenance; policy approval and agreement on metadata aggregation remain pending. |

## Artifact availability

The remote tree contains the synthetic input `tests/fixtures/technician_logs.csv`, NLP code, documentation, validator, and tests. It does **not** contain the generated NLP output paths listed by the manifest:

- `artifacts/data/fd001_nlp_fixture/processed_logs.csv`
- `artifacts/data/fd001_nlp_fixture/entities.jsonl`
- `artifacts/data/fd001_nlp_fixture/severity.csv`
- `artifacts/data/fd001_nlp_fixture/verification_report.json`
- `artifacts/embeddings/fd001_nlp_fixture/embeddings.pt`
- `artifacts/embeddings/fd001_nlp_fixture/embedding_metadata.csv`

Thus the branch includes documentation and validation/build code, plus a synthetic input fixture, but no generated NLP embedding tensor or row metadata artifact. The manifest’s five-log / `(5, 768)` results are documented fixture results, not an artifact payload independently inspected in this branch.

## Compatibility gaps and status

**Interface shape: PARTIAL / SYNTHETIC-ONLY.** The embedding dimension and proposed `log_id`-sorted row mapping align conceptually. The current fixture is not committed as generated embeddings and cannot be loaded as genuine input.

**Genuine integration: BLOCKED.** The NLP builder does not require or establish all fusion fields: genuine `namespace`, approved `split`, source dataset/version, source system/record ID, equipment mapping ID, preprocessing version, artifact ID/SHA-256, exact model/tokenizer revisions, and four trusted timestamps. `unit_id`, `cycle`, and source split are optional; their genuine semantics and mapping evidence are unresolved. Current `available_at` and timestamp field names need an approved mapping to the fusion loader’s `availability_time` and other required names.

**Policy compatibility: PENDING APPROVAL.** Missing and multiple-log embedding choices match implemented fusion options, but they are proposals rather than approvals. `has_log` feature selection remains a separate pending model-contract decision. Entity/severity aggregation over multiple logs needs a decision if those outputs are to be consumed.

**Required before genuine integration:** Rafaz/data owners must provide the actual generated artifact package and an auditable per-record source/mapping/timestamp contract; owners must establish permissions, canonical equipment mapping, cycle semantics, timestamp availability/cutoff rules, exact model/tokenizer revisions and artifact hashes. Jesse/project owners must approve the missing/multiple-log and `has_log` policies and the broader genuine-data/fusion gates. Do not infer these values from the synthetic fixture.

## Scope and changes

No Rafaz files were copied or modified; no implementation files were touched and no commit was created. This review report is the only new file: `context/RAFAZ_NLP_BRANCH_INTEGRATION_REVIEW.md`.
