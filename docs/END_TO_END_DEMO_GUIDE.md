# End-to-end demo guide

Demo A uses a verified FD001 30-cycle sample and calls sensor-only inference. Demo B uses `tests/fixtures/technician_logs.csv` and calls NLP-only analysis; its entities/severity are synthetic fixture/rule-based evidence. Demo C calls `/api/fusion/readiness` and displays blocked dependency, mapping, temporal, paired-data and checkpoint conditions. The demos never combine unrelated records and never produce a fabricated multimodal prediction.
