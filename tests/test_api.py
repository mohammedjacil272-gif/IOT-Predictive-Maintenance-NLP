from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_status_are_local_and_secret_free():
    assert client.get('/api/health').json()['status'] == 'ok'
    payload = client.get('/api/status').json()
    assert payload['iot']['mqtt'] == 'not_checked'
    assert 'token' not in str(payload).lower()
    assert payload['sensor']['official_test_evaluation'] == 'not_run'
    assert payload['sensor']['validation_metrics']['split'] == 'validation'
    assert payload['sensor']['official_test_evaluation_detail']['status'] == 'not_run'
    assert payload['iot']['local_replay'] == 'available'
    assert payload['iot']['configuration']['mqtt_configured'] is False
    assert 'token' not in str(payload['iot']).lower()


def test_official_test_evaluation_cannot_be_inferred_from_validation_metrics(monkeypatch, tmp_path):
    from app import main

    monkeypatch.setattr(main, 'OFFICIAL_TEST_METRICS_PATH', tmp_path / 'missing-test-metrics.json')
    monkeypatch.setattr(main, 'VALIDATION_METRICS_PATH', main.ROOT / 'artifacts/metrics/fd001_lstm_baseline/validation_metrics.json')

    result = main.official_test_evaluation()

    assert result['status'] == 'not_run'
    assert result['split'] == 'official_test'
    assert 'validation metrics are not official-test metrics' in result['reason']


def test_attachment_contract_is_explicit_and_shared():
    body = client.get('/api/attachments/contract').json()

    assert body['max_bytes'] == 10 * 1024 * 1024
    assert set(body['formats']) >= {'.txt', '.csv', '.json', '.pdf'}
    assert body['analysis'] == 'not_performed'


def test_log_analysis_returns_structured_rule_results():
    response = client.post('/api/technician-log/analyze', json={'log_id': 'x', 'text': 'Critical vibration near bearing'})
    assert response.status_code == 200
    body = response.json()
    assert body['mode'] == 'nlp_only'
    assert body['severity']['severity_label'] == 'critical'
    assert any(item['label'] == 'COMPONENT' for item in body['entities'])


def test_fusion_is_explicitly_unavailable_without_approved_checkpoint():
    body = client.post('/api/fusion/infer', json={}).json()
    assert body['prediction'] is None
    assert body['status'] == 'not_validated'


def test_fusion_route_validates_embedding_contract_before_fail_closed_response():
    response = client.post('/api/fusion/infer', json={
        'sensor_embedding': [0.0] * 64,
        'text_embedding': [0.0] * 768,
        'unit_id': 'fixture-unit',
        'log_id': 'fixture-log',
    })
    assert response.status_code == 200
    body = response.json()
    assert body['validated_inputs']['sensor_embedding_dim'] == 64
    assert body['validated_inputs']['text_embedding_dim'] == 768

    partial = client.post('/api/fusion/infer', json={'sensor_embedding': [0.0] * 64})
    assert partial.status_code == 422



def test_fusion_readiness_reports_evidence_backed_blockers():
    body = client.get('/api/fusion/readiness').json()
    assert body['ready_for_genuine_fusion'] is False
    assert body['conditions']['sensor_checkpoint']['status'] == 'READY'
    assert body['conditions']['paired_dataset']['status'] == 'BLOCKED'
    assert body['conditions']['machine_identity_mapping']['status'] == 'BLOCKED'
    assert body['conditions']['fusion_checkpoint']['status'] == 'BLOCKED'


def test_sensor_prediction_semantics_distinguish_separate_official_evaluation():
    from app import main

    class FakeResult:
        unit_id = 1
        window_start_cycle = 1
        window_end_cycle = 30
        prediction = 42.0

    original_model_path, original_scaler_path = main.MODEL_PATH, main.SCALER_PATH
    main.MODEL_PATH = main.ROOT / 'artifacts/models/fd001_lstm_baseline/best_model.pt'
    main.SCALER_PATH = main.ROOT / 'artifacts/data/fd001_lstm_baseline/scaler.joblib'
    try:
        from unittest.mock import patch

        records = [{'unit_id': 1, 'cycle': cycle} for cycle in range(1, 31)]
        with patch('iot.lstm_inference.predict_records', return_value=FakeResult()):
            response = client.post('/api/sensor/infer', json={'records': records})
        assert response.status_code == 200
        semantics = response.json()['prediction_semantics']
        assert 'validation-trained baseline' in semantics
        assert 'official-test evaluation status is reported separately' in semantics
        assert 'not an official test result' in semantics
    finally:
        main.MODEL_PATH, main.SCALER_PATH = original_model_path, original_scaler_path


def test_sensor_ui_explains_direct_window_submission_and_separate_replay():
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / 'app/static/index.html').read_text(encoding='utf-8')
    assert 'Paste a 30-cycle JSON window directly' in html
    assert 'replay tooling is available separately' in html
    assert 'Replay data must be prepared outside the UI' not in html


def test_unified_inference_rejects_unknown_mode():
    response = client.post('/api/infer', json={'mode': 'unknown'})
    assert response.status_code == 422


def test_unified_nlp_mode_and_history_are_explicit(monkeypatch, tmp_path):
    from app import main
    original = main.HISTORY
    main.HISTORY = main.LocalHistoryStore(tmp_path / 'history.json')
    try:
        response = client.post('/api/infer', json={
            'mode': 'nlp_only', 'log_id': 'history-1', 'text': 'Critical bearing vibration',
            'is_synthetic': True, 'source_provenance': 'synthetic_fixture',
        })
        assert response.status_code == 200
        assert response.json()['mode'] == 'nlp_only'
        records = client.get('/api/history').json()['records']
        assert len(records) == 1
        assert records[0]['is_synthetic'] is True
        assert client.delete('/api/history').json()['cleared'] is True
        assert client.get('/api/history').json()['records'] == []
    finally:
        main.HISTORY = original


def test_unified_fusion_never_returns_prediction():
    response = client.post('/api/infer', json={'mode': 'multimodal_fusion'})
    assert response.status_code == 200
    assert response.json()['prediction'] is None
    assert response.json()['status'] == 'not_validated'


def test_assistant_typed_transcription_and_confirmation_contract():
    transcription = client.post('/api/assistant/transcribe', json={'typed_text': 'bearing vibration', 'provenance': 'typed_demo'})
    assert transcription.status_code == 200
    assert transcription.json()['transcription']['status'] == 'typed_input'
    analysis = client.post('/api/assistant/analyze', json={'session_id': 's1', 'machine_id': '12', 'text': 'The bearing has vibration'})
    assert analysis.status_code == 200
    body = analysis.json()
    assert body['state'] == 'RESPONDING'
    assert body['confirmation']['confirmation_question'] == 'Is this what you meant?'


def test_assistant_machine_health_is_explicit_when_unregistered():
    response = client.get('/api/machines/not-registered/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'unavailable'


def test_machine_session_and_confirmation_lifecycle(monkeypatch, tmp_path):
    from app import main
    original = main.ASSISTANT_STORAGE
    main.ASSISTANT_STORAGE = main.AssistantStorage(tmp_path / 'assistant.sqlite3')
    try:
        machine = client.post('/api/machines', json={'machine_id': 'm-api', 'display_name': 'Fixture machine', 'is_synthetic': True})
        assert machine.status_code == 201
        session = client.post('/api/assistant/sessions', json={'machine_id': 'm-api'})
        assert session.status_code == 201
        sid = session.json()['session_id']
        analyzed = client.post(f'/api/assistant/sessions/{sid}/analyze', json={'text': 'The bearing has vibration'})
        assert analyzed.status_code == 200
        assert len(client.get(f'/api/assistant/sessions/{sid}/messages').json()['messages']) == 2
        confirmation = analyzed.json()['confirmation_id']
        pending = main.ASSISTANT_STORAGE.list_observations(session_id=sid)
        assert pending[0]['status'] == 'pending'
        decided = client.post(f'/api/assistant/confirmations/{confirmation}/confirm', json={'action': 'confirm'})
        assert decided.status_code == 200
        assert main.ASSISTANT_STORAGE.list_observations(session_id=sid)[0]['status'] == 'confirmed'
        again = client.post(f'/api/assistant/confirmations/{confirmation}/confirm', json={'action': 'confirm'})
        assert again.status_code == 200
    finally:
        main.ASSISTANT_STORAGE = original


def test_machine_patch_rejects_identity_change_and_updates_allowed_fields(tmp_path):
    from app import main
    original = main.ASSISTANT_STORAGE
    main.ASSISTANT_STORAGE = main.AssistantStorage(tmp_path / 'assistant.sqlite3')
    try:
        created = client.post('/api/machines', json={'machine_id': 'm-patch', 'display_name': 'Before'})
        assert created.status_code == 201
        changed = client.patch('/api/machines/m-patch', json={'display_name': 'After', 'location': 'Bay 1'})
        assert changed.status_code == 200
        assert changed.json()['display_name'] == 'After'
        assert changed.json()['location'] == 'Bay 1'
        identity_attempt = client.patch('/api/machines/m-patch', json={'machine_id': 'other'})
        assert identity_attempt.status_code == 422
    finally:
        main.ASSISTANT_STORAGE = original
