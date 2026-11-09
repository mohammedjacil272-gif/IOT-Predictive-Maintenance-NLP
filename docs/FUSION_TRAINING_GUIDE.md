# Fusion training guide

`training/train_multimodal.py` and the fusion validators provide a fail-closed training boundary. Genuine training requires an approval register, paired artifacts, mapping, timestamps/cutoffs, target, split manifest and reproducibility metadata. Current data does not satisfy that gate. Synthetic forward/contract checks are permitted only as smoke tests and are not predictive evidence. New approved runs must write to a dedicated integration artifact directory and compare sensor-only, NLP-only and fusion models on identical entity splits.
