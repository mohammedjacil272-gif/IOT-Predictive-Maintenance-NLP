\# IoT Predictive Maintenance Using NLP and Multimodal Fusion



\## Project Overview



This project explores Natural Language Processing (NLP) for predictive maintenance, with the longer-term goal of combining sensor-derived features and maintenance text for Remaining Useful Life (RUL) prediction.



The project currently includes:

\- A frozen sensor-only FD001 baseline.

\- An NLP pipeline for processing maintenance descriptions.

\- Rule-based entity extraction and heuristic severity scoring.

\- DistilBERT-based text embeddings.

\- A fusion dataset interface and multimodal model implementation, pending suitable paired data for meaningful evaluation.



\## Current Status



\### 1. Sensor-Only FD001 Baseline



The existing sensor-only baseline is preserved as the reference experiment.



Previously recorded validation metrics:



| Metric | Value |

|---|---:|

| MAE | 20.884394 |

| RMSE | 31.364907 |

| MSE | 983.757406 |



These are existing baseline validation results, not results from multimodal fusion. Official FD001 test evaluation has not been performed.



\### 2. NLP Pipeline



The NLP pipeline includes:

\- Text preprocessing and normalization.

\- Rule-based named entity extraction.

\- Heuristic severity scoring.

\- DistilBERT text embeddings with 768 dimensions.



\### 3. Nestor Maintenance-Text Validation



The existing NLP pipeline was validated on the NIST Nestor excavator maintenance-work-order toy dataset.



| Result | Value |

|---|---:|

| Source records processed | 5,485 |

| Extracted entities | 3,712 |

| Embedding dimensions | 768 |

| Invalid or empty text records | 0 |

| All embeddings finite | Yes |



The run completed using Python 3.11.9, PyTorch 2.14.1+cpu, and Transformers 5.19.0.



The generated validation report is available locally at:



`artifacts/data/nestor\_nlp/verification\_report.json`



The raw dataset and generated artifacts are excluded from Git.



\### Important Limitations



\- The Nestor dataset is used for independent NLP validation, not FD001 sensor-text fusion.

\- Entity extraction is rule-based and has not been evaluated against labelled NER ground truth in this experiment.

\- Severity scores are heuristic and are not ground-truth labels.

\- The maintenance type (`PMType`) is not a severity label.

\- No matching sensor telemetry and maintenance-text records have been established for the fusion experiment.

\- No fusion performance improvement has been demonstrated.



\## Multimodal Fusion: Next Research Milestone



Before training or evaluating the fusion model, an appropriate paired dataset is required. Sensor measurements and maintenance text must be linked using trustworthy machine/unit and cycle or timestamp information.



The fusion experiment must:

1\. Preserve the existing sensor-only baseline.

2\. Use defensible sensor-text alignment.

3\. Prevent future information or target leakage.

4\. Keep training and validation splits isolated.

5\. Compare against the frozen baseline using the same evaluation protocol.



The canonical fusion dataset interface is `training/fusion\_dataset.py::build\_fusion\_samples()`.



The existence of fusion code does not establish that multimodal prediction improves performance.



\## Repository Structure



\- `nlp/` — Text preprocessing, entity extraction, severity scoring, and embeddings.

\- `training/` — Training utilities and fusion dataset construction.

\- `models/` — Model implementations.

\- `preprocessing/` — Sensor preprocessing code.

\- `scripts/` — Pipeline and validation scripts.

\- `tests/` — Automated tests.

\- `data/` — Local datasets and data resources.

\- `artifacts/` — Generated experiment outputs, excluded where configured by `.gitignore`.



\## Reproduce the Nestor NLP Validation



From the repository root, run:



```powershell

python -m py\_compile scripts\\validate\_nestor.py

python scripts\\validate\_nestor.py

```



The local Nestor CSV must be available at:



`data/external/nestor/excavators.csv`



The script regenerates processed logs, extracted entities, heuristic severity results, text embeddings, metadata, and a verification report.



The validation script does not train a model, perform sensor-text fusion, or evaluate the official FD001 test set.



\## Research Integrity



Results from separate datasets and experiments must remain distinct. Synthetic fixtures and unrelated maintenance logs must not be presented as paired FD001 observations. Any future claim of improved RUL prediction must be supported by a controlled experiment on genuinely paired data.



\## Current Development Branch



`feature/rafaz-nlp-pipeline`



