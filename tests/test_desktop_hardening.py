from fastapi.testclient import TestClient


def test_protected_api_requires_bearer_token(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "API_TOKEN", "test-launch-token")
    client = TestClient(main.app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/status", headers={"authorization": "Bearer test-launch-token"}).status_code == 200


def test_desktop_cors_preflight_and_authenticated_requests(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "API_TOKEN", "test-launch-token")
    client = TestClient(main.app)
    preflight_headers = {
        "origin": "http://tauri.localhost",
        "access-control-request-method": "GET",
        "access-control-request-headers": "Authorization, Content-Type",
    }

    response = client.options("/api/status", headers=preflight_headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://tauri.localhost"
    assert "authorization" in response.headers["access-control-allow-headers"].lower()
    assert "content-type" in response.headers["access-control-allow-headers"].lower()

    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"authorization": "Bearer wrong"}).status_code == 401
    authenticated = client.get(
        "/api/status",
        headers={"authorization": "Bearer test-launch-token", "origin": "http://tauri.localhost"},
    )
    assert authenticated.status_code == 200
    assert authenticated.headers["access-control-allow-origin"] == "http://tauri.localhost"

    unapproved = client.options(
        "/api/status",
        headers={
            **preflight_headers,
            "origin": "https://unapproved.example",
        },
    )
    assert "access-control-allow-origin" not in unapproved.headers


def test_desktop_cors_allows_configured_development_origin(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "API_TOKEN", "test-launch-token")
    client = TestClient(main.app)
    response = client.options(
        "/api/status",
        headers={
            "origin": "http://localhost:1420",
            "access-control-request-method": "GET",
            "access-control-request-headers": "Authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:1420"


def test_health_remains_public_with_desktop_cors(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "API_TOKEN", "test-launch-token")
    client = TestClient(main.app)
    response = client.get("/api/health", headers={"origin": "tauri://localhost"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "tauri://localhost"


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
