# Multimodal fusion

Status: software interfaces implemented; genuine training blocked.

The fusion contract accepts 64-D sensor embeddings and the verified text
embedding dimension, validates canonical metadata joins, supports explicit
missing/multiple-log policies and `has_log`, and fails closed without approved
genuine provenance, timestamp cutoffs, paired data, and a trained checkpoint.

Synthetic forward passes and fixture checks do not establish predictive
performance. No fusion checkpoint or fusion metric is currently claimed.
