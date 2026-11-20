# Frontend API integration

The React frontend calls `/api/machines`, `/api/fusion/readiness`, and `/api/assistant/analyze` against a configurable `VITE_API_BASE`, defaulting to loopback FastAPI. It labels synthetic typed input and unavailable speech/fusion states. Full route coverage, authentication, sidecar token negotiation and installed-app testing remain pending.
