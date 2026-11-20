# Final project handoff

Start with `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`. Use `/api/machines`, `/api/assistant/sessions`, `/api/assistant/sessions/{id}/analyze`, `/api/assistant/confirmations/{id}/confirm|reject|cancel`, `/api/attachments`, `/api/infer`, `/api/fusion/readiness`, and `/api/machines/{id}/health`. The backend uses SQLite for assistant records and preserves pending versus confirmed observations. Readiness is **NO-GO** for genuine fusion. Speech, model assets, approval and production security remain outstanding.
