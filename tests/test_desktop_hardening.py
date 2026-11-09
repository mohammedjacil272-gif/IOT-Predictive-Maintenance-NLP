from fastapi.testclient import TestClient


def test_protected_api_requires_bearer_token(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "API_TOKEN", "test-launch-token")
    client = TestClient(main.app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/status", headers={"authorization": "Bearer test-launch-token"}).status_code == 200


def test_data_root_is_separate_from_read_only_resources(tmp_path, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(main, "HISTORY", main.LocalHistoryStore(tmp_path / ".local_history.json"))
    client = TestClient(main.app)
    response = client.post("/api/technician-log/analyze", json={"log_id": "data-root", "text": "bearing vibration"})
    assert response.status_code == 200
    assert (tmp_path / ".local_history.json").is_file()
    assert not (tmp_path / "artifacts").exists()


def test_runtime_resource_paths_are_not_repository_working_directory_paths():
    from app import main

    assert main.MODEL_PATH.is_absolute()
    assert main.SCALER_PATH.is_absolute()
    assert main.SENSOR_EMBEDDING_PATH.is_absolute()
