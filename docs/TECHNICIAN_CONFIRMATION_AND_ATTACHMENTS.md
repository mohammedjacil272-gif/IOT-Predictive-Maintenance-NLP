# Confirmation and attachments

`assistant.confirmation` keeps the original transcript, creates an interpretation requiring confirmation, and produces explainable follow-up questions without inferring measurements. `assistant.attachments` accepts JPEG/PNG/PDF/TXT/CSV/JSON under the shared `/api/attachments/contract` contract, limits files to 10 MiB, validates content and extension/MIME agreement, sanitizes names, hashes content, stores outside static assets, and marks files `stored_unanalyzed`. No image or document analysis is claimed.
