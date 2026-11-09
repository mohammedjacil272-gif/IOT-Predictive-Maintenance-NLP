# User data storage

Development storage remains in the existing Python SQLite/JSON paths. A production desktop build must redirect SQLite, JSON history, attachments, logs and caches into Tauri application-data directories using platform-aware configuration, while preserving existing data through an explicit backup/migration. No automatic migration or deletion was performed in this phase.
