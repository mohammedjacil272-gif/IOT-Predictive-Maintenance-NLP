# Storage and provenance design

The new assistant backend uses transactional SQLite for machines, sessions, messages, observations and attachment metadata. Existing JSON local history remains compatible for the current dashboard and is not silently migrated or deleted. Confirmed, pending and rejected observation states remain distinct; provenance, origin and timestamps are retained. A future migration can import existing history after an explicit schema review.
