# Rafaz — Technician Logs, NLP, NER, Severity & Text Embeddings

## Project context and completed work
The project predicts RUL from sensor time-series and investigates adding technician maintenance text. Current verified sensor baseline uses FD001, 30-cycle windows, 17 features, an 80/20 engine split (seed 42), and a frozen sensor-only LSTM. Best validation metrics: MAE 20.884394, RMSE 31.364907, MSE 983.757406. Official test evaluation has NOT been performed. Do not change the sensor baseline or split.

## Git setup
```powershell
git clone https://github.com/mohammedjacil272-gif/IOT-Predictive-Maintenance-NLP.git
cd IOT-Predictive-Maintenance-NLP
git switch -c feature/rafaz-nlp-pipeline
```

## Input contract
Expected log fields: `log_id`, `unit_id`, `cycle`, `text`, `event_type`, `component`, `severity`, `generation_rule_id`, `template_id`, `source_split`. Confirm actual paths/schema with Jacil. Preserve traceability to each source log.

## Your independent workstream
Build the text-processing branch. If production synthetic logs are not available yet, use a tiny test fixture under `tests/fixtures/`, clearly labelled as a fixture. Do not wait for the log generator.

1. Implement deterministic text validation/normalization; preserve original text and `log_id`.
2. Implement transparent rule-based NER (e.g. spaCy EntityRuler or patterns). Labels: `COMPONENT`, `FAULT`, `SYMPTOM`, `ACTION`, `SEVERITY_TERM`, `MEASUREMENT`.
3. Save log ID, entity text/label, character offsets and rule/source ID. Test offsets against original text.
4. Implement heuristic severity score 0–4: 0 none/unknown, 1 low, 2 medium, 3 high, 4 critical. Define conflicts and negation. State clearly these are synthetic heuristics, not real-world ground truth.
5. Generate embeddings using frozen `distilbert-base-uncased`, matching tokenizer, max length 128 and mean pooling over non-padding tokens using the attention mask.
6. Expected embedding shape: `[number_of_logs, 768]`. Save metadata with `log_id`, `unit_id`, `cycle`, `source_split`, and model/config details.
7. Keep NER/severity as metadata initially; don't silently concatenate them into the text embedding.
8. Add build and verification scripts plus configuration/report files.

## Suggested files
```text
nlp/preprocessing.py
nlp/ner.py
nlp/severity.py
nlp/embeddings.py
scripts/build_nlp_artifacts.py
scripts/verify_nlp.py
tests/test_nlp_preprocessing.py
tests/test_ner.py
tests/test_severity.py
tests/test_embeddings.py
artifacts/data/fd001_nlp/
artifacts/embeddings/fd001_nlp/
```

## Boundaries
- Do not modify sensor preprocessing, the LSTM, its split/scaler/checkpoints/metrics, or official test evaluation.
- Do not use RUL, terminal cycles, future readings or test labels.
- Do not import Wahid's fusion code.
- No labeled real-world NER dataset exists yet; do not claim learned NER accuracy.
- Coordinate artifact paths with Jacil; do not overwrite another member's files.

## Tests
- Empty/malformed text handled explicitly.
- Valid entity labels and correct offsets.
- Severity in `[0,4]`, including conflict/negation cases.
- Token length ≤128 and attention mask correct.
- Embedding shape `[n,768]`, no NaN/Inf.
- Metadata row count/order matches embeddings.
- Repeat run is consistent within numerical tolerance.
- No RUL/test labels or future information loaded.

## Done when
A supplied CSV can be processed into cleaned text, entity results, severity scores, embeddings and a verification report. Document setup, model download requirements, limitations and test results.

## Git handoff
```powershell
git add <your-files>
git commit -m "Add NLP pipeline and text embeddings"
git push -u origin feature/rafaz-nlp-pipeline
```
Open a PR and report entity counts, severity distribution, model/config, embedding shape, tests and limitations.
