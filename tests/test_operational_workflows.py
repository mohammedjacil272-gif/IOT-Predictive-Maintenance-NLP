from fastapi.testclient import TestClient


def test_equipment_component_maintenance_alert_and_audit_workflow(tmp_path, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "ASSISTANT_STORAGE", main.AssistantStorage(tmp_path / "ops.sqlite3"))
    client = TestClient(main.app)
    headers = {}
    machine = client.post("/api/machines", json={"machine_id": "asset-1", "display_name": "Pump 1"}, headers=headers)
    assert machine.status_code == 201
    component = client.post("/api/machines/asset-1/components", json={"component_id": "bearing-1", "name": "Drive bearing"})
    assert component.status_code == 201
    event = client.post("/api/machines/asset-1/maintenance", json={"event_type": "preventive", "component_id": "bearing-1", "notes": "Lubrication inspection"})
    assert event.status_code == 201
    alert = client.post("/api/machines/asset-1/alerts", json={"category": "sensor_risk", "severity": "high", "message": "Bearing vibration requires review", "source": "sensor_model"})
    assert alert.status_code == 201 and alert.json()["status"] == "open"
    duplicate = client.post("/api/machines/asset-1/alerts", json={"category": "sensor_risk", "severity": "high", "message": "Bearing vibration requires review", "source": "sensor_model"})
    assert duplicate.status_code == 201 and duplicate.json()["duplicate_suppressed"] is True
    alert_id = alert.json()["alert_id"]
    resolved = client.patch(f"/api/alerts/{alert_id}", json={"status": "resolved"})
    assert resolved.status_code == 200 and resolved.json()["status"] == "resolved"
    audit = client.get("/api/audit", params={"resource_type": "machine"})
    assert audit.status_code == 200
    assert any(row["action"] == "machine.created" for row in audit.json()["events"])


def test_component_cannot_cross_machine_boundary(tmp_path, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "ASSISTANT_STORAGE", main.AssistantStorage(tmp_path / "ops.sqlite3"))
    client = TestClient(main.app)
    for machine_id in ("m1", "m2"):
        assert client.post("/api/machines", json={"machine_id": machine_id, "display_name": machine_id}).status_code == 201
    assert client.post("/api/machines/m1/components", json={"component_id": "c1", "name": "Bearing"}).status_code == 201
    response = client.post("/api/machines/m2/maintenance", json={"event_type": "corrective", "component_id": "c1"})
    assert response.status_code == 422
