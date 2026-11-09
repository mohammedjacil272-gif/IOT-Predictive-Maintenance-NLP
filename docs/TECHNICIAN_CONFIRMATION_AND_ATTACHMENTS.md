# Confirmation and attachments

`assistant.confirmation` keeps the original transcript, creates an interpretation requiring confirmation, and produces explainable follow-up questions without inferring measurements. `assistant.attachments` accepts only JPEG/PNG/PDF, limits files to 10 MiB, sanitizes names, hashes content, stores outside static assets, and marks files `stored_unanalyzed`. No image analysis is claimed.
