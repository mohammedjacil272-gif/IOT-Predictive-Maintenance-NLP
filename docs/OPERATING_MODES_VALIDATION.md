# Operating modes validation

Sensor-only uses the existing FD001 LSTM and is labeled as FD001 validation-trained RUL inference. NLP-only uses rule-based normalization, NER and severity; those outputs are not transformer or trained predictions. Multimodal fusion is unavailable unless paired representations and a verified checkpoint exist. Missing and invalid input tests return explicit validation errors; no mode silently substitutes a fabricated output.
