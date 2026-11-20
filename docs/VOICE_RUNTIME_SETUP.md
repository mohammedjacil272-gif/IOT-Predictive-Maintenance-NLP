# Voice runtime setup

The current runtime intentionally provides typed input and explicit unavailable states. No microphone, speech-to-text, or text-to-speech provider is installed or claimed operational.

Verify the boundary:

```powershell
python -m pytest tests/test_assistant_backend.py tests/test_assistant_gateway_retrieval.py -q
Invoke-RestMethod http://127.0.0.1:8000/api/assistant/voice-status
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/assistant/transcribe -ContentType application/json -Body '{"typed_text":"bearing vibration"}'
```

Typed responses are labeled `typed_input`, never speech transcription. Wahid may select an approved local audio provider later, but must add permission, cancellation, timeout, audio-format, and mocked lifecycle tests before exposing it in the UI. Do not download speech models or send audio to cloud services without authorization.
