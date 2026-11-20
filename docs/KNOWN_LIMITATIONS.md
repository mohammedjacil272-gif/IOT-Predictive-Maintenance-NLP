# Known limitations

- The official FD001 test evaluation is not performed by the recorded baseline.
- Local ignored checkpoints are external artifact state and are not reproduced by a clean clone.
- Transformer embedding inference is optional and requires pinned local model/tokenizer assets.
- Rule NER and heuristic severity are not trained or calibrated real-world models.
- Synthetic NLP/fusion fixtures do not prove real pairing, mapping, leakage safety, or predictive improvement.
- No approved genuine fusion training authorization, paired data, target/cutoff policy, or trained fusion checkpoint is present.
- Live MQTT, InfluxDB, and other service health is not checked by default.
- The SCANIA task is documented repair-endpoint classification, not FD001 RUL.
